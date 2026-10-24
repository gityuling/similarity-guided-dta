#!/usr/bin/env python3
"""Validation-selected convex ensemble for two DTA prediction files.

The mixture weight and an affine calibrator are fitted only on validation rows.
The selected transform is frozen before the held-out test evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_nexusdti_regressor import PROJECT, regression_metrics


def load_predictions(path: Path, label: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"split", "row_id", "pKd_true", "prediction_raw"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return frame.loc[:, ["split", "row_id", "pKd_true", "prediction_raw"]].rename(
        columns={"prediction_raw": f"prediction_{label}"}
    )


def fit_affine(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[float, float]:
    design = np.column_stack([y_pred, np.ones_like(y_pred)])
    slope, intercept = np.linalg.lstsq(design, y_true, rcond=None)[0]
    if slope <= 0:
        raise RuntimeError(f"Non-positive calibration slope ({slope:.4f}) would reverse rankings")
    return float(slope), float(intercept)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True, help="CSV from the original model calibration evaluator")
    parser.add_argument("--censor", required=True, help="CSV from the censor-aware model calibration evaluator")
    parser.add_argument("--grid-size", type=int, default=1001)
    parser.add_argument("--output-stem", default="dtiam_drugcold_f0_base_censor_ensemble")
    args = parser.parse_args()

    base = load_predictions(Path(args.base), "base")
    censor = load_predictions(Path(args.censor), "censor")
    merged = base.merge(censor, on=["split", "row_id"], how="inner", validate="one_to_one", suffixes=("_base", "_censor"))
    if len(merged) != len(base) or len(merged) != len(censor):
        raise RuntimeError("Prediction files do not contain identical split/row IDs")
    if not np.allclose(merged.pKd_true_base, merged.pKd_true_censor, rtol=0.0, atol=1e-6):
        raise RuntimeError("Prediction files disagree on the ground-truth pKd values")
    merged = merged.rename(columns={"pKd_true_base": "pKd_true"}).drop(columns=["pKd_true_censor"])

    val = merged[merged.split == "val"].reset_index(drop=True)
    test = merged[merged.split == "test"].reset_index(drop=True)
    if len(val) == 0 or len(test) == 0:
        raise RuntimeError("Both validation and test rows are required")
    y_val = val.pKd_true.to_numpy()
    base_val = val.prediction_base.to_numpy()
    censor_val = val.prediction_censor.to_numpy()

    weights = np.linspace(0.0, 1.0, args.grid_size)
    blended_val = weights[:, None] * base_val[None, :] + (1.0 - weights[:, None]) * censor_val[None, :]
    mse_by_weight = np.mean((blended_val - y_val[None, :]) ** 2, axis=1)
    base_weight = float(weights[int(np.argmin(mse_by_weight))])
    raw_val = base_weight * base_val + (1.0 - base_weight) * censor_val
    slope, intercept = fit_affine(y_val, raw_val)

    y_test = test.pKd_true.to_numpy()
    raw_test = base_weight * test.prediction_base.to_numpy() + (1.0 - base_weight) * test.prediction_censor.to_numpy()
    calibrated_val = slope * raw_val + intercept
    calibrated_test = slope * raw_test + intercept

    result = {
        "protocol": "select convex base/censor mixture and affine calibration only on validation rows; evaluate once on untouched test rows",
        "rows": {"val": len(val), "test": len(test)},
        "selection": {
            "base_weight": base_weight,
            "censor_weight": 1.0 - base_weight,
            "affine_slope": slope,
            "affine_intercept": intercept,
        },
        "validation": {
            "base_raw": regression_metrics(y_val, base_val),
            "censor_raw": regression_metrics(y_val, censor_val),
            "ensemble_raw": regression_metrics(y_val, raw_val),
            "ensemble_affine": regression_metrics(y_val, calibrated_val),
        },
        "test": {
            "base_raw": regression_metrics(y_test, test.prediction_base.to_numpy()),
            "censor_raw": regression_metrics(y_test, test.prediction_censor.to_numpy()),
            "ensemble_raw": regression_metrics(y_test, raw_test),
            "ensemble_affine": regression_metrics(y_test, calibrated_test),
        },
    }

    results_dir = PROJECT / "results"
    prediction_dir = results_dir / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    out_json = results_dir / f"{args.output_stem}.json"
    out_csv = prediction_dir / f"{args.output_stem}_predictions.csv"
    result_frame = merged.copy()
    result_frame["prediction_ensemble_raw"] = base_weight * result_frame.prediction_base + (1.0 - base_weight) * result_frame.prediction_censor
    result_frame["prediction_ensemble_affine"] = slope * result_frame.prediction_ensemble_raw + intercept
    result_frame.to_csv(out_csv, index=False)
    out_json.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
