#!/usr/bin/env python3
"""Materialize the exact outer Davis splits used by DTIAM (Lu et al., 2025).

DTIAM's official splitter uses all 68 x 442 Davis matrix entries, including
the conventional pKd=5 floor, and applies KFold(shuffle=True, random_state=0)
to pairs, drugs, or proteins.  This script reproduces those *outer* folds and
adds a deterministic, disjoint inner validation set for our early-stopping
training loop.  Test rows are unchanged from the official DTIAM split.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


PROJECT = Path(__file__).resolve().parents[1]
REFERENCE_DATA = PROJECT / "baselines/DTIAM/data/dta/davis"
OUT_DIR = PROJECT / "data/processed/davis_dtiam_protocol"
SETTINGS = ("warm_start", "drug_coldstart", "protein_coldstart")


def inner_validation(
    train: pd.DataFrame, setting: str, fold: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return row ids for a 90/10 inner train/validation split.

    Entity-cold outer folds retain the same property in validation: validation
    drugs/proteins are excluded from inner training as well.
    """
    rng = np.random.default_rng(seed + fold)
    if setting == "warm_start":
        ids = train.row_id.to_numpy(copy=True)
        rng.shuffle(ids)
        n_val = max(1, int(round(0.10 * len(ids))))
        return ids[n_val:], ids[:n_val]

    entity = "drug_idx" if setting == "drug_coldstart" else "protein_idx"
    groups = train[entity].drop_duplicates().to_numpy(copy=True)
    rng.shuffle(groups)
    n_val = max(1, int(round(0.10 * len(groups))))
    val_groups = set(groups[:n_val].tolist())
    is_val = train[entity].isin(val_groups).to_numpy()
    return train.loc[~is_val, "row_id"].to_numpy(), train.loc[is_val, "row_id"].to_numpy()


def outer_folds(table: pd.DataFrame, setting: str, n_splits: int, seed: int):
    """Yield official DTIAM outer train/test masks in its original ordering."""
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    if setting == "warm_start":
        universe = np.arange(len(table))
        for train_index, test_index in kfold.split(universe):
            yield np.isin(table.row_id.to_numpy(), train_index), np.isin(table.row_id.to_numpy(), test_index)
        return

    entity = "drug_idx" if setting == "drug_coldstart" else "protein_idx"
    universe = np.sort(table[entity].unique())
    for train_index, test_index in kfold.split(universe):
        train_entities, test_entities = universe[train_index], universe[test_index]
        values = table[entity].to_numpy()
        yield np.isin(values, train_entities), np.isin(values, test_entities)


def build_table(reference_dir: Path) -> pd.DataFrame:
    with (reference_dir / "ligands_can.txt").open() as handle:
        ligands = json.load(handle, object_pairs_hook=OrderedDict)
    with (reference_dir / "proteins.txt").open() as handle:
        proteins = json.load(handle, object_pairs_hook=OrderedDict)
    with (reference_dir / "Y").open("rb") as handle:
        affinity = np.asarray(pickle.load(handle, encoding="latin1"), dtype=np.float64)

    if affinity.shape != (len(ligands), len(proteins)):
        raise ValueError(
            f"DTIAM Davis matrix mismatch: {affinity.shape}, "
            f"expected {(len(ligands), len(proteins))}"
        )

    drug_ids, drug_smiles = list(ligands.keys()), list(ligands.values())
    protein_ids, sequences = list(proteins.keys()), list(proteins.values())
    rows, cols = np.where(np.isfinite(affinity))
    labels = -np.log10(affinity[rows, cols] / 1e9)
    table = pd.DataFrame(
        {
            "row_id": np.arange(len(rows), dtype=np.int64),
            "drug_idx": rows.astype(np.int64),
            "protein_idx": cols.astype(np.int64),
            "drug_id": [drug_ids[index] for index in rows],
            "protein_id": [protein_ids[index] for index in cols],
            "Canonical_SMILES": [drug_smiles[index] for index in rows],
            "Sequence": [sequences[index] for index in cols],
            "pKd_floor": labels,
        }
    )
    table["is_censored_floor"] = table.pKd_floor.le(5.0 + 1e-8)
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, default=REFERENCE_DATA)
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0, help="Official DTIAM outer-split seed.")
    parser.add_argument("--inner-seed", type=int, default=20261012)
    args = parser.parse_args()

    table = build_table(args.reference_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / "table.csv", index=False)
    summary: dict[str, object] = {
        "reference": "DTIAM official Davis data_split_dta.py",
        "outer_splitter": f"KFold(n_splits={args.n_splits}, shuffle=True, random_state={args.seed})",
        "inner_validation": "deterministic 10% of outer training rows/entities",
        "pairs": int(len(table)),
        "drugs": int(table.drug_idx.nunique()),
        "proteins": int(table.protein_idx.nunique()),
        "censored_floor_pairs": int(table.is_censored_floor.sum()),
        "settings": {},
    }

    for setting in SETTINGS:
        setting_dir = args.out_dir / "folds" / setting
        setting_dir.mkdir(parents=True, exist_ok=True)
        fold_summaries = []
        for fold, (outer_train, outer_test) in enumerate(
            outer_folds(table, setting, args.n_splits, args.seed)
        ):
            outer_train_df = table.loc[outer_train]
            train_ids, val_ids = inner_validation(outer_train_df, setting, fold, args.inner_seed)
            manifest = table[["row_id", "drug_idx", "protein_idx"]].copy()
            manifest["split"] = "test"
            manifest.loc[manifest.row_id.isin(train_ids), "split"] = "train"
            manifest.loc[manifest.row_id.isin(val_ids), "split"] = "val"
            if not np.array_equal(manifest.split.eq("test").to_numpy(), outer_test):
                raise RuntimeError(f"{setting} fold {fold}: test manifest differs from DTIAM outer split")
            manifest.to_csv(setting_dir / f"fold_{fold}.csv", index=False)
            fold_summaries.append(
                {
                    "fold": fold,
                    "rows": {key: int(value) for key, value in manifest.split.value_counts().items()},
                    "train_drugs": int(manifest.loc[manifest.split == "train", "drug_idx"].nunique()),
                    "val_drugs": int(manifest.loc[manifest.split == "val", "drug_idx"].nunique()),
                    "test_drugs": int(manifest.loc[manifest.split == "test", "drug_idx"].nunique()),
                    "train_proteins": int(manifest.loc[manifest.split == "train", "protein_idx"].nunique()),
                    "val_proteins": int(manifest.loc[manifest.split == "val", "protein_idx"].nunique()),
                    "test_proteins": int(manifest.loc[manifest.split == "test", "protein_idx"].nunique()),
                }
            )
        summary["settings"][setting] = fold_summaries

    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
