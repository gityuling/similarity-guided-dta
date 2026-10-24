#!/usr/bin/env python3
"""Validation-only similarity-gated blend of deep DTA and chemical-neighbor prior.

The deep-model weight is sigmoid(a + b * standardized_max_tanimoto). Both
gate parameters and an optional affine calibration are fitted only on the
validation split and then frozen for test evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

from train_nexusdti_regressor import PROJECT, regression_metrics


def load_base(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"split", "row_id", "pKd_true", "prediction_raw"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame.loc[:, ["split", "row_id", "pKd_true", "prediction_raw"]].rename(
        columns={"pKd_true": "pKd_true_base", "prediction_raw": "prediction_deep"}
    )


def load_prior(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"split", "row_id", "pKd_true", "prediction_raw", "max_train_drug_tanimoto"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame.loc[:, ["split", "row_id", "pKd_true", "prediction_raw", "max_train_drug_tanimoto"]].rename(
        columns={"pKd_true": "pKd_true_prior", "prediction_raw": "prediction_prior"}
    )


def fit_affine(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([y_pred, np.ones_like(y_pred)])
    slope, intercept = np.linalg.lstsq(design, y_true, rcond=None)[0]
    if slope <= 0:
        raise RuntimeError(f"Non-positive calibration slope ({slope:.4f}) would reverse rankings")
    return float(slope), float(intercept)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deep", required=True)
    parser.add_argument("--prior", required=True)
    parser.add_argument("--output-stem", default="dtiam_drugcold_f0_similarity_gated_ensemble")
    args = parser.parse_args()

    merged = load_base(Path(args.deep)).merge(load_prior(Path(args.prior)), on=["split", "row_id"], validate="one_to_one")
    if not np.allclose(merged.pKd_true_base, merged.pKd_true_prior, rtol=0.0, atol=1e-6):
        raise RuntimeError("Deep and prior files disagree on pKd labels")
    merged = merged.rename(columns={"pKd_true_base": "pKd_true"}).drop(columns=["pKd_true_prior"])
    val = merged[merged.split == "val"].reset_index(drop=True)
    test = merged[merged.split == "test"].reset_index(drop=True)
    if val.empty or test.empty:
        raise RuntimeError("Both validation and test rows are required")

    y_val = val.pKd_true.to_numpy()
    deep_val = val.prediction_deep.to_numpy()
    prior_val = val.prediction_prior.to_numpy()
    sim_mean = float(val.max_train_drug_tanimoto.mean())
    sim_std = float(val.max_train_drug_tanimoto.std())
    if sim_std < 1e-8:
        raise RuntimeError("Validation structural similarity has no variation")
    z_val = (val.max_train_drug_tanimoto.to_numpy() - sim_mean) / sim_std

    def mixed(params: np.ndarray, z: np.ndarray, deep: np.ndarray, prior: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        deep_weight = expit(params[0] + params[1] * z)
        return deep_weight * deep + (1.0 - deep_weight) * prior, deep_weight

    def objective(params: np.ndarray) -> float:
        prediction, _ = mixed(params, z_val, deep_val, prior_val)
        return float(np.mean((y_val - prediction) ** 2) + 1e-3 * params[1] ** 2)

    initial = np.array([logit(0.335), 0.0], dtype=np.float64)
    optimized = minimize(objective, initial, method="L-BFGS-B")
    if not optimized.success:
        raise RuntimeError(f"Gate optimization failed: {optimized.message}")
    raw_val, weight_val = mixed(optimized.x, z_val, deep_val, prior_val)
    slope, intercept = fit_affine(y_val, raw_val)

    y_test = test.pKd_true.to_numpy()
    z_test = (test.max_train_drug_tanimoto.to_numpy() - sim_mean) / sim_std
    raw_test, weight_test = mixed(
        optimized.x,
        z_test,
        test.prediction_deep.to_numpy(),
        test.prediction_prior.to_numpy(),
    )
    affine_val = slope * raw_val + intercept
    affine_test = slope * raw_test + intercept
    result = {
        "protocol": "fit similarity gate and affine calibration only on validation rows; evaluate once on untouched test rows",
        "rows": {"val": len(val), "test": len(test)},
        "gate": {
            "formula": "deep_weight = sigmoid(a + b * z_max_tanimoto)",
            "a": float(optimized.x[0]),
            "b": float(optimized.x[1]),
            "z_mean": sim_mean,
            "z_std": sim_std,
            "deep_weight_val_range": [float(weight_val.min()), float(weight_val.max())],
            "deep_weight_test_range": [float(weight_test.min()), float(weight_test.max())],
            "affine_slope": slope,
            "affine_intercept": intercept,
        },
        "validation": {"raw": regression_metrics(y_val, raw_val), "affine": regression_metrics(y_val, affine_val)},
        "test": {"raw": regression_metrics(y_test, raw_test), "affine": regression_metrics(y_test, affine_test)},
    }

    output = merged.copy()
    output["deep_weight"] = np.nan
    # Preserve original row order while assigning split-specific outputs.
    output.loc[output.split == "val", "deep_weight"] = weight_val
    output.loc[output.split == "test", "deep_weight"] = weight_test
    output.loc[output.split == "val", "prediction_gated_raw"] = raw_val
    output.loc[output.split == "test", "prediction_gated_raw"] = raw_test
    output["prediction_gated_affine"] = slope * output.prediction_gated_raw + intercept
    results_dir = PROJECT / "results"
    prediction_dir = results_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    out_json = results_dir / f"{args.output_stem}.json"
    out_csv = prediction_dir / f"{args.output_stem}_predictions.csv"
    out_json.write_text(json.dumps(result, indent=2))
    output.to_csv(out_csv, index=False)
    print(json.dumps(result, indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
