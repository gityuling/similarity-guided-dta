#!/usr/bin/env python3
"""Create a strict, reproducible pKd source table from BindingDB articles.

Only exact positive Kd measurements for single-chain targets are retained.
Inequality-censored values and Ki/IC50 rows are deliberately excluded so the
source target has the same physical definition as Davis pKd.  For a Davis
outer fold, every validation/test drug is excluded before aggregation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_nexusdti_regressor import PROJECT, standardize_smiles


RAW_ZIP = PROJECT / "data/raw/bindingdb_articles/BindingDB_BindingDB_Articles_202610_tsv.zip"
OUT_DIR = PROJECT / "data/processed/bindingdb_articles_kd"
SMILES = "Ligand SMILES"
KD = "Kd (nM)"
CHAINS = "Number of Protein Chains in Target (>1 implies a multichain complex)"
SEQUENCE = "BindingDB Target Chain Sequence 1"


def canonical_or_none(value: str) -> str | None:
    try:
        return standardize_smiles(str(value))
    except (TypeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--davis-manifest", required=True, help="Outer-fold manifest relative to DTA_affinity.")
    parser.add_argument("--chunksize", type=int, default=100_000)
    parser.add_argument("--run-tag", required=True)
    args = parser.parse_args()
    manifest_path = Path(args.davis_manifest)
    if not manifest_path.is_absolute():
        manifest_path = PROJECT / manifest_path
    davis = pd.read_csv(manifest_path).merge(
        pd.read_csv(PROJECT / "data/processed/davis_dtiam_protocol/table.csv"),
        on="row_id",
        how="left",
        validate="one_to_one",
    )
    held_drugs = {
        canonical
        for canonical in davis.loc[davis.split != "train", "Canonical_SMILES"].map(canonical_or_none)
        if canonical is not None
    }
    frames = []
    audit = {"rows_read": 0, "positive_exact_kd": 0, "single_chain_with_sequence": 0, "valid_smiles": 0, "held_drug_rows_removed": 0}
    for chunk in pd.read_csv(
        RAW_ZIP,
        sep="\t",
        compression="zip",
        usecols=[SMILES, KD, CHAINS, SEQUENCE],
        chunksize=args.chunksize,
        low_memory=False,
    ):
        audit["rows_read"] += len(chunk)
        kd_text = chunk[KD].astype(str).str.strip()
        # pandas numeric coercion rejects the > / < censored records.
        kd = pd.to_numeric(kd_text, errors="coerce")
        frame = chunk.loc[kd.gt(0)].copy()
        frame["kd_nM"] = kd.loc[frame.index]
        audit["positive_exact_kd"] += len(frame)
        chain_count = pd.to_numeric(frame[CHAINS], errors="coerce")
        sequence = frame[SEQUENCE].fillna("").astype(str).str.strip()
        frame = frame.loc[(chain_count == 1) & sequence.ne("")].copy()
        frame["sequence"] = sequence.loc[frame.index]
        audit["single_chain_with_sequence"] += len(frame)
        frame["canonical_smiles"] = frame[SMILES].map(canonical_or_none)
        frame = frame.dropna(subset=["canonical_smiles"])
        audit["valid_smiles"] += len(frame)
        held_mask = frame.canonical_smiles.isin(held_drugs)
        audit["held_drug_rows_removed"] += int(held_mask.sum())
        frame = frame.loc[~held_mask, ["canonical_smiles", "sequence", "kd_nM"]]
        frames.append(frame)
    source = pd.concat(frames, ignore_index=True)
    source["pKd"] = -np.log10(source.kd_nM.to_numpy(dtype=np.float64) * 1e-9)
    source = (
        source.groupby(["canonical_smiles", "sequence"], as_index=False, sort=False)
        .agg(pKd=("pKd", "median"), measurements=("pKd", "size"))
        .reset_index(drop=True)
    )
    audit.update(
        {
            "davis_manifest": str(manifest_path),
            "downstream_held_drugs": len(held_drugs),
            "rows_after_filter_before_replicate_aggregation": int(sum(len(frame) for frame in frames)),
            "unique_pairs_after_median_aggregation": len(source),
            "unique_drugs": int(source.canonical_smiles.nunique()),
            "unique_targets": int(source.sequence.nunique()),
            "pKd_summary": {key: float(value) for key, value in source.pKd.describe().items()},
        }
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    table_path = OUT_DIR / f"bindingdb_articles_kd_{args.run_tag}.csv"
    audit_path = OUT_DIR / f"bindingdb_articles_kd_{args.run_tag}_audit.json"
    source.to_csv(table_path, index=False)
    audit_path.write_text(json.dumps(audit, indent=2))
    print(json.dumps({"table": str(table_path), "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
