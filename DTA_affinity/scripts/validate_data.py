#!/usr/bin/env python3
"""Run inexpensive integrity and split-leakage checks."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def validate_dataset(name: str, table_path: Path, drug_col: str, target_col: str, label_col: str) -> None:
    df = pd.read_csv(table_path)
    if df[[drug_col, target_col]].duplicated().any():
        raise AssertionError(f"{name}: duplicate drug-target pairs")
    if df[label_col].isna().any():
        raise AssertionError(f"{name}: missing labels")

    split_root = ROOT / "data/processed" / name / "splits"
    for mode in ("random", "cold_drug", "cold_target", "double_cold"):
        manifest = pd.read_csv(split_root / mode / "manifest.csv")
        if mode != "double_cold" and len(manifest) != len(df):
            raise AssertionError(f"{name}/{mode}: row count mismatch")
        retained = manifest[manifest["split"] != "unused"]
        if retained["row_id"].duplicated().any():
            raise AssertionError(f"{name}/{mode}: duplicate row ids")
        for left in ("train", "val", "test"):
            for right in ("train", "val", "test"):
                if left >= right:
                    continue
                a = set(manifest.loc[manifest["split"] == left, "row_id"])
                b = set(manifest.loc[manifest["split"] == right, "row_id"])
                if a & b:
                    raise AssertionError(f"{name}/{mode}: split overlap {left}/{right}")
        if mode == "cold_drug":
            sets = [set(manifest.loc[manifest["split"] == s, "drug_id"]) for s in ("train", "val", "test")]
            if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
                raise AssertionError(f"{name}/{mode}: drug leakage")
        if mode == "cold_target":
            sets = [set(manifest.loc[manifest["split"] == s, "target_id"]) for s in ("train", "val", "test")]
            if any(sets[i] & sets[j] for i in range(3) for j in range(i + 1, 3)):
                raise AssertionError(f"{name}/{mode}: target leakage")
        if mode == "double_cold":
            for split in ("train", "val", "test"):
                sub = manifest[manifest["split"] == split]
                if sub.empty:
                    raise AssertionError(f"{name}/{mode}: empty {split}")

    print(f"PASS {name}: {len(df)} rows")


def main() -> None:
    validate_dataset(
        "davis_clean",
        ROOT / "data/processed/davis/davis_entity_affinity_clean.csv",
        "Drug_Index",
        "Protein_Index",
        "pKd_floor",
    )
    validate_dataset(
        "davis_floor",
        ROOT / "data/processed/davis/davis_entity_affinity_floor.csv",
        "Drug_Index",
        "Protein_Index",
        "pKd_floor",
    )
    validate_dataset(
        "kiba_standard",
        ROOT / "data/processed/kiba/kiba_deepdta_affinity_long.csv",
        "drug_id",
        "protein_id",
        "kiba_score",
    )
    print(json.dumps({"status": "ok"}))


if __name__ == "__main__":
    main()
