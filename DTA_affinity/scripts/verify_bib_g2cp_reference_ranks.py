#!/usr/bin/env python3
"""Verify annotation-retrieval diagnostics against frozen scores and libraries."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from bib_g2cp_common import DATA, OUT, PROJECT, check_records, record, save_json


def main():
    directory = PROJECT / "reports/bib_g2cp_reference_diagnostic_20261019"
    analysis = json.loads((directory / "analysis.json").read_text())
    check_records(analysis["outputs"])
    query = pd.read_csv(OUT / "query_table.csv", usecols=["row_id", "Canonical_SMILES", "protein_id"])
    affinity = query.merge(pd.read_csv(OUT / "affinity_predictions.csv"), on="row_id", validate="one_to_one")
    candidates = pd.read_csv(OUT / "candidates.csv", keep_default_na=False).set_index("canonical_identity")
    ranks = pd.read_csv(directory / "target_ranks.csv", keep_default_na=False)
    count = 0
    for cell in analysis["config"]["cells"]:
        with np.load(DATA / f"{cell}.npz", allow_pickle=False) as archive:
            proteins = set(archive["proteins"].astype(str))
        meta = pd.read_csv(directory / f"{cell}_development_rows.csv").set_index("canonical_identity")
        current = ranks.loc[ranks.cell.eq(cell)]
        if len(current) != len(meta)*3 or current.duplicated(["canonical_identity", "method"]).any():
            raise RuntimeError("Incorrect number of target-ranking rows")
        for row in current.itertuples():
            if row.split != meta.loc[row.canonical_identity, "split"] or row.split not in ("train", "val"):
                raise RuntimeError("Target-ranking split mismatch")
            annotated = set(candidates.loc[row.canonical_identity, "mapped_kiba_targets"].split("|")) & proteins
            scores = affinity.loc[affinity.Canonical_SMILES.eq(row.canonical_identity) & affinity.protein_id.isin(proteins)]
            ordered = scores.sort_values([row.method, "protein_id"], ascending=[False, True]).protein_id.tolist()
            expected = sorted(ordered.index(protein)+1 for protein in annotated)
            if row.top5_targets != "|".join(ordered[:5]) or row.annotated_target_ranks != "|".join(map(str, expected)):
                raise RuntimeError("Saved target ranks differ from DTA predictions")
            if row.available_annotation_count != len(expected) or row.library_targets != len(proteins):
                raise RuntimeError("Target count mismatch")
            if expected:
                values = [int(min(expected) == 1), int(min(expected) <= 5), sum(r <= 5 for r in expected)/len(expected), 1/min(expected)]
                actual = np.asarray([row.hit_at_1, row.hit_at_5, row.recall_at_5, row.reciprocal_rank], dtype=float)
                if not np.allclose(values, actual, atol=1e-12):
                    raise RuntimeError("Annotation-retrieval metric mismatch")
            elif any(getattr(row, key) != "" for key in ("hit_at_1", "hit_at_5", "recall_at_5", "reciprocal_rank")):
                raise RuntimeError("Missing annotations converted to negatives")
            count += 1
    valid = ranks.loc[ranks.split.eq("val") & ranks.davis_external & ranks.available_annotation_count.gt(0)].copy()
    columns = ["hit_at_1", "hit_at_5", "recall_at_5", "reciprocal_rank"]
    valid[columns] = valid[columns].astype(float)
    summary = valid.groupby(["method", "canonical_identity"])[columns].mean().groupby("method").mean()
    result = {"state": "passed", "rows_checked": count, "validation_drugs": valid.canonical_identity.nunique(),
              "primary_validation_macro": summary.to_dict("index"), "source": record(directory / "target_ranks.csv"),
              "verifier": record(Path(__file__)), "limits": "Incomplete annotation recovery, not ground-truth selectivity; development data only"}
    save_json(directory / "rank_verification.json", result, immutable=True)
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
