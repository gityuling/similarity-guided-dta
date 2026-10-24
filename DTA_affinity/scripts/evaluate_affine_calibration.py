#!/usr/bin/env python3
"""Evaluate validation-only affine calibration for a saved DTA checkpoint.

The calibrator is fit exclusively on the split's validation predictions:
    calibrated_prediction = slope * raw_prediction + intercept
It is then frozen and evaluated once on the untouched test split.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from train_nexusdti_regressor import (
    AffinityDataset,
    NexusDTIAffinityRegressor,
    PROJECT,
    build_pooled_stats,
    collate_fn,
    predict,
    regression_metrics,
)


def make_loader(dataset: AffinityDataset, batch_size: int, workers: int, pin_memory: bool) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=workers > 0,
    )


def fit_affine(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([y_pred, np.ones_like(y_pred)])
    slope, intercept = np.linalg.lstsq(design, y_true, rcond=None)[0]
    if slope <= 0:
        raise RuntimeError(
            f"Validation-derived calibration slope is non-positive ({slope:.4f}); "
            "refusing to invert the model's ranking."
        )
    return float(slope), float(intercept)


def subset_metrics(y_true: np.ndarray, y_pred: np.ndarray, target_column: str = "pKd_floor") -> dict[str, dict[str, float]]:
    floor_mask = np.isclose(y_true, 5.0)
    output = {"all": regression_metrics(y_true, y_pred)}
    if target_column != "pKd_floor":
        return output
    for label, mask in (("pKd_floor_5", floor_mask), ("pKd_above_5", ~floor_mask)):
        if int(mask.sum()) >= 2:
            output[label] = regression_metrics(y_true[mask], y_pred[mask])
    return output


def prediction_frame(
    split_name: str,
    split_df: pd.DataFrame,
    y_true: np.ndarray,
    raw_pred: np.ndarray,
    mean_shift_pred: np.ndarray,
    affine_pred: np.ndarray,
    target_column: str = "pKd_floor",
) -> pd.DataFrame:
    if len(split_df) != len(y_true):
        raise RuntimeError("Prediction order does not match the split dataframe")
    output = pd.DataFrame(
        {
            "split": split_name,
            "row_id": split_df.row_id.to_numpy(),
            "canonical_smiles": split_df.Canonical_SMILES.to_numpy(),
            "target_sequence": split_df.Sequence.to_numpy(),
            "pKd_true": y_true,
            "is_pKd_floor_5": np.isclose(y_true, 5.0),
            "prediction_raw": raw_pred,
            "prediction_mean_shift": mean_shift_pred,
            "prediction_affine": affine_pred,
            "residual_raw": raw_pred - y_true,
            "residual_affine": affine_pred - y_true,
        }
    )
    if target_column != "pKd_floor":
        output.rename(columns={"pKd_true": "affinity_true"}, inplace=True)
        output.drop(columns="is_pKd_floor_5", inplace=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Saved .pt checkpoint from train_nexusdti_regressor.py")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output-stem", default=None, help="Optional filename stem under DTA_affinity/results")
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint).resolve()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    payload = torch.load(checkpoint_path, map_location=device, weights_only=False)
    run_args = payload["args"]
    run_result = payload["result"]

    table_path = Path(run_result["data_table"])
    manifest_path = Path(run_result["manifest_file"])
    table = pd.read_csv(table_path)
    manifest = pd.read_csv(manifest_path)
    joined = manifest.merge(table, on="row_id", how="left", validate="one_to_one")
    if joined[["Sequence", "Canonical_SMILES"]].isna().any().any():
        raise RuntimeError("Split manifest and Davis affinity table are not aligned")

    train_df = joined[joined.split == "train"].reset_index(drop=True)
    val_df = joined[joined.split == "val"].reset_index(drop=True)
    test_df = joined[joined.split == "test"].reset_index(drop=True)
    pocket_fusion = run_args.get("pocket_fusion", False)
    pocket_radius = run_args.get("pocket_radius", 48)
    token_root = run_result.get("token_dir") or run_args.get("token_dir")
    token_dir = Path(token_root) if token_root else None
    if token_dir is not None and not token_dir.is_absolute():
        token_dir = PROJECT / token_dir
    target_column = run_result.get("target_column", run_args.get("target_column", "pKd_floor"))
    dataset_options = dict(kinase_pocket=pocket_fusion, pocket_radius=pocket_radius,
                           token_dir=token_dir, target_column=target_column)
    train_ds = AffinityDataset(train_df, **dataset_options)
    val_ds = AffinityDataset(val_df, **dataset_options)
    test_ds = AffinityDataset(test_df, **dataset_options)
    pin_memory = device.type == "cuda"
    val_loader = make_loader(val_ds, args.batch_size, args.workers, pin_memory)
    test_loader = make_loader(test_ds, args.batch_size, args.workers, pin_memory)

    pooled_mean, pooled_std = (build_pooled_stats(train_df, token_dir=token_dir) if run_args["hybrid"] else (None, None))
    censor_floor = run_args.get("censor_floor", 5.0)
    prediction_floor = (
        (censor_floor - run_result["target_mean"]) / run_result["target_std"]
        if run_args["target_normalize"]
        else censor_floor
    )
    model = NexusDTIAffinityRegressor(
        latent_dim=run_args["latent_dim"],
        pool_fusion=run_args["pool_fusion"],
        drug_attention=run_args["drug_attention"],
        pocket_fusion=pocket_fusion,
        pocket_residual=run_args.get("pocket_residual", False),
        hybrid=run_args["hybrid"],
        censor_aware=run_args.get("censor_aware", False),
        hard_censor_output=not run_args.get("activity_aux_only", False),
        prediction_floor=prediction_floor,
        pooled_mean=pooled_mean,
        pooled_std=pooled_std,
        dropout=run_args["dropout"],
    ).to(device)
    model.load_state_dict(payload["model"])

    y_val, raw_val = predict(model, val_loader, device, run_args["k_ratio"])
    y_test, raw_test = predict(model, test_loader, device, run_args["k_ratio"])
    if run_args["target_normalize"]:
        raw_val = raw_val * run_result["target_std"] + run_result["target_mean"]
        raw_test = raw_test * run_result["target_std"] + run_result["target_mean"]

    mean_shift = float(y_val.mean() - raw_val.mean())
    slope, intercept = fit_affine(y_val, raw_val)
    shifted_val = raw_val + mean_shift
    shifted_test = raw_test + mean_shift
    affine_val = slope * raw_val + intercept
    affine_test = slope * raw_test + intercept

    calibration_result = {
        "checkpoint": str(checkpoint_path),
        "device": str(device),
        "token_dir": str(token_dir) if token_dir else None,
        "target_column": target_column,
        "protocol": "fit only on validation predictions; evaluate once on untouched test predictions",
        "rows": {"train": len(train_ds), "val": len(val_ds), "test": len(test_ds)},
        "calibrator": {
            "mean_shift": mean_shift,
            "affine_slope": slope,
            "affine_intercept": intercept,
        },
        "validation": {
            "raw": subset_metrics(y_val, raw_val, target_column),
            "mean_shift": subset_metrics(y_val, shifted_val, target_column),
            "affine": subset_metrics(y_val, affine_val, target_column),
        },
        "test": {
            "raw": subset_metrics(y_test, raw_test, target_column),
            "mean_shift": subset_metrics(y_test, shifted_test, target_column),
            "affine": subset_metrics(y_test, affine_test, target_column),
        },
    }

    results_dir = PROJECT / "results"
    prediction_dir = results_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_stem or checkpoint_path.stem
    out_json = results_dir / f"{stem}_affine_calibration.json"
    out_csv = prediction_dir / f"{stem}_affine_calibration_predictions.csv"
    out_json.write_text(json.dumps(calibration_result, indent=2))
    pd.concat(
        [
            prediction_frame("val", val_df, y_val, raw_val, shifted_val, affine_val, target_column),
            prediction_frame("test", test_df, y_test, raw_test, shifted_test, affine_test, target_column),
        ],
        ignore_index=True,
    ).to_csv(out_csv, index=False)

    print(json.dumps(calibration_result, indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
