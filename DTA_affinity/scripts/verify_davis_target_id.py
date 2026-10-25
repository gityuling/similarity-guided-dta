#!/usr/bin/env python3
"""Audit target isolation and independently reconstruct saved Davis neighbors."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from bib_common import PROJECT, ROOT, manifest_path, record, sha256, table_path


def main():
    folder = PROJECT / "reports/davis_target_id_20261025"
    amendment = json.loads((folder / "amendment.json").read_text())
    for item in [*amendment["code"], *amendment["neural_predictions_unchanged"], *amendment["outputs"]]:
        assert sha256(ROOT / item["path"]) == item["sha256"], item["path"]
    table = pd.read_csv(table_path("davis"))
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = {s: generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in table.Canonical_SMILES.unique()}
    similarity = {(s, t): DataStructs.TanimotoSimilarity(fingerprints[s], fingerprints[t])
                  for s in fingerprints for t in fingerprints}
    rows = []
    for fold in range(5):
        manifest = pd.read_csv(manifest_path("davis", fold))
        joined = manifest.merge(table, on="row_id", validate="one_to_one")
        train = joined.loc[joined.split.eq("train")]
        queries = joined.loc[joined.split.ne("train")]
        stem = f"bib_davis_target_id_f{fold}_frozen_prior"
        result = json.loads((PROJECT / f"results/{stem}.json").read_text())
        assert result["reference_key"] == "protein_id" and result["reference_groups"] == 442
        saved = pd.read_csv(PROJECT / f"results/predictions/{stem}_predictions.csv").set_index("row_id")
        k = result["selected_hyperparameters"]["top_k"]
        gamma = result["selected_hyperparameters"]["gamma"]
        errors, examples = [], {}
        excluded_cross_id_rows = 0
        for target, query in queries.groupby("protein_id", sort=False):
            support = train.loc[train.protein_id.eq(target)]
            assert support.protein_id.nunique() == 1 and set(support.row_id).isdisjoint(query.row_id)
            assert set(support.drug_id).isdisjoint(query.drug_id)
            sim = np.array([[similarity[s, t] for t in support.Canonical_SMILES] for s in query.Canonical_SMILES])
            ix = np.argsort(-sim, axis=1, kind="quicksort")[:, :k]
            weights = np.take_along_axis(sim, ix, axis=1) ** gamma
            values = support.pKd_floor.to_numpy()[ix]
            calculated = np.sum(weights * values, axis=1) / weights.sum(axis=1)
            error = float(np.max(np.abs(calculated - saved.loc[query.row_id, "prediction_raw"])))
            errors.append(error)
            assert error < 1e-12, (fold, target, error)
            siblings = train.loc[train.Sequence.eq(support.Sequence.iloc[0]) & train.protein_id.ne(target)]
            excluded_cross_id_rows += len(siblings)
            if target in ("BRAF", "BRAF(V600E)"):
                examples[target] = {"reference_rows": len(support), "excluded_sibling_rows": len(siblings),
                                    "support_label_mean": float(support.pKd_floor.mean())}
        rows.append({"fold": fold, "verified_queries": len(queries), "target_pools": 442,
                     "max_prediction_error": max(errors), "excluded_cross_id_rows_across_pools": excluded_cross_id_rows,
                     "examples": examples})
    kiba = pd.read_csv(table_path("kiba"))[["protein_id", "Sequence"]].drop_duplicates()
    assert len(kiba) == kiba.protein_id.nunique() == kiba.Sequence.nunique() == 229
    current = json.loads((folder / "analysis.json").read_text())
    original = json.loads((PROJECT / "reports/bib_fusion_20261017/analysis.json").read_text())
    assert current["datasets"]["kiba"] == original["datasets"]["kiba"]
    output = {"passed": True, "code": record(Path(__file__)), "folds": rows,
              "neural_prediction_files_unchanged": len(amendment["neural_predictions_unchanged"]),
              "kiba_id_sequence_bijection": True, "kiba_analysis_unchanged": True}
    (folder / "verification.json").write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
