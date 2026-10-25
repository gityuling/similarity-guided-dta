#!/usr/bin/env python3
"""Evaluate a training-only chemical-neighbor prior for drug-cold DTA.

For a query (drug, target), the prior uses only *training* drugs measured on
the same target. Their pKd values are averaged with Morgan-fingerprint
Tanimoto weights. Hyperparameters and affine calibration are selected on the
validation split, then frozen for the untouched test split.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from train_nexusdti_regressor import PROJECT, regression_metrics


def canonical_smiles(smiles: str) -> str:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(molecule)


def fit_affine(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([y_pred, np.ones_like(y_pred)])
    slope, intercept = np.linalg.lstsq(design, y_true, rcond=None)[0]
    if slope <= 0:
        raise RuntimeError(f"Non-positive calibration slope ({slope:.4f}) would reverse rankings")
    return float(slope), float(intercept)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-table", required=True)
    parser.add_argument("--manifest-file", required=True)
    parser.add_argument("--output-stem", default="dtiam_drugcold_f0_morgan_neighbor_prior")
    parser.add_argument("--target-column", default="pKd_floor")
    parser.add_argument("--stable-neighbors", action="store_true", help="Resolve equal-similarity neighbors by stable input order.")
    args = parser.parse_args()

    table = pd.read_csv(Path(args.data_table))
    manifest = pd.read_csv(Path(args.manifest_file))
    joined = manifest.merge(table, on="row_id", how="left", validate="one_to_one")
    if joined[["Sequence", "Canonical_SMILES", args.target_column]].isna().any().any():
        raise RuntimeError("Split manifest and affinity table are not aligned")
    joined = joined.copy()
    joined["canonical_smiles"] = joined.Canonical_SMILES.map(canonical_smiles)
    train = joined[joined.split == "train"].reset_index(drop=True)
    val = joined[joined.split == "val"].reset_index(drop=True)
    test = joined[joined.split == "test"].reset_index(drop=True)
    if train.empty or val.empty or test.empty:
        raise RuntimeError("Expected non-empty train, val, and test rows")

    fp_generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    all_smiles = sorted(joined.canonical_smiles.unique())
    fingerprints = {
        smiles: fp_generator.GetFingerprint(Chem.MolFromSmiles(smiles))
        for smiles in all_smiles
    }
    supports: dict[str, tuple[list, np.ndarray, float]] = {}
    for sequence, rows in train.groupby("Sequence", sort=False):
        support_smiles = rows.canonical_smiles.tolist()
        supports[sequence] = (
            [fingerprints[smiles] for smiles in support_smiles],
            rows[args.target_column].to_numpy(dtype=np.float64),
            float(rows[args.target_column].mean()),
        )

    max_k = 40

    def cache_neighbors(frame: pd.DataFrame) -> list[tuple[np.ndarray, np.ndarray, float]]:
        cached = []
        for row in frame.itertuples(index=False):
            support_fps, support_y, target_mean = supports[row.Sequence]
            similarities = np.asarray(
                DataStructs.BulkTanimotoSimilarity(fingerprints[row.canonical_smiles], support_fps),
                dtype=np.float64,
            )
            order = np.argsort(-similarities, kind="stable" if args.stable_neighbors else "quicksort")[:max_k]
            cached.append((similarities[order], support_y[order], target_mean))
        return cached

    val_cache = cache_neighbors(val)
    test_cache = cache_neighbors(test)

    def predict_cached(cache: list[tuple[np.ndarray, np.ndarray, float]], top_k: int, gamma: float) -> np.ndarray:
        predictions = np.empty(len(cache), dtype=np.float64)
        for index, (similarities, values, fallback) in enumerate(cache):
            usable = min(top_k, len(values))
            weights = np.power(similarities[:usable], gamma)
            weight_sum = weights.sum()
            predictions[index] = float(np.dot(weights, values[:usable]) / weight_sum) if weight_sum > 0 else fallback
        return predictions

    y_val = val[args.target_column].to_numpy(dtype=np.float64)
    grid = []
    for top_k in (1, 3, 5, 10, 20, 40):
        for gamma in (0.5, 1.0, 2.0, 4.0):
            predictions = predict_cached(val_cache, top_k, gamma)
            grid.append({"top_k": top_k, "gamma": gamma, "val_mse_raw": float(np.mean((y_val - predictions) ** 2))})
    selected = min(grid, key=lambda item: item["val_mse_raw"])
    raw_val = predict_cached(val_cache, selected["top_k"], selected["gamma"])
    raw_test = predict_cached(test_cache, selected["top_k"], selected["gamma"])
    slope, intercept = fit_affine(y_val, raw_val)
    affine_val = slope * raw_val + intercept
    affine_test = slope * raw_test + intercept
    y_test = test[args.target_column].to_numpy(dtype=np.float64)

    result = {
        "protocol": "Morgan neighbor prior uses only training rows at the same target; top-k/gamma and affine calibration selected on validation rows, then frozen for test",
        "target_column": args.target_column,
        "stable_neighbors": args.stable_neighbors,
        "input_sha256": {
            "data_table": hashlib.sha256(Path(args.data_table).read_bytes()).hexdigest(),
            "manifest": hashlib.sha256(Path(args.manifest_file).read_bytes()).hexdigest(),
        },
        "rows": {"train": len(train), "val": len(val), "test": len(test)},
        "selected_hyperparameters": {
            **selected,
            "affine_slope": slope,
            "affine_intercept": intercept,
        },
        "validation": {"raw": regression_metrics(y_val, raw_val), "affine": regression_metrics(y_val, affine_val)},
        "test": {"raw": regression_metrics(y_test, raw_test), "affine": regression_metrics(y_test, affine_test)},
        "validation_grid": grid,
    }

    def result_frame(split: str, frame: pd.DataFrame, cache, raw: np.ndarray, affine: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "split": split,
                "row_id": frame.row_id.to_numpy(),
                "pKd_true" if args.target_column == "pKd_floor" else "affinity_true": frame[args.target_column].to_numpy(),
                "prediction_raw": raw,
                "prediction_affine": affine,
                "max_train_drug_tanimoto": [float(item[0][0]) for item in cache],
            }
        )

    results_dir = PROJECT / "results"
    prediction_dir = results_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    out_json = results_dir / f"{args.output_stem}.json"
    out_csv = prediction_dir / f"{args.output_stem}_predictions.csv"
    out_json.write_text(json.dumps(result, indent=2))
    pd.concat(
        [
            result_frame("val", val, val_cache, raw_val, affine_val),
            result_frame("test", test, test_cache, raw_test, affine_test),
        ],
        ignore_index=True,
    ).to_csv(out_csv, index=False)
    print(json.dumps(result, indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
