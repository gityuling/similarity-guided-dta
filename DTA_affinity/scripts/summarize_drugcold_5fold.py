#!/usr/bin/env python3
"""Summarize five DTIAM-protocol drug-cold outer folds reproducibly."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from train_nexusdti_regressor import PROJECT


METRICS = ("mse", "rmse", "ci", "pearson", "spearman")


def summarize(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std(ddof=1)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def main() -> None:
    results_dir = PROJECT / "results"
    rows = []
    for fold in range(5):
        deep_path = results_dir / f"nexusdti_regressor_dtiam_drugcold_f{fold}_hybrid_e12.json"
        gated_path = results_dir / f"dtiam_drugcold_f{fold}_similarity_gated_ensemble.json"
        deep = json.loads(deep_path.read_text())
        gated = json.loads(gated_path.read_text())
        row = {"fold": fold, "test_rows": deep["rows_test"]}
        for metric in METRICS:
            row[f"deep_{metric}"] = deep["metrics_test"][metric]
            row[f"gated_{metric}"] = gated["test"]["raw"][metric]
        rows.append(row)

    frame = pd.DataFrame(rows)
    aggregate = {
        "protocol": {
            "dataset": "Davis full 30,056-pair matrix with pKd=5.0 floor retained",
            "outer_evaluation": "official DTIAM 5-fold drug-cold splitter (KFold seed 0)",
            "deep_model": "NEXUS-DTA hybrid Uni-Mol2 + ESM-2 model, identical training configuration across folds",
            "gated_model": "deep score blended with training-only Morgan/Tanimoto same-target neighbor prior; blend gate selected on the corresponding validation split",
            "calibration": "not used in final gated aggregate because its held-out effect was inconsistent across the first three folds",
        },
        "reference": {
            "name": "DTIAM, Nature Communications 2025 Figure 3",
            "drug_cold_mse_approx": 0.55,
            "drug_cold_pearson_approx": 0.49,
            "note": "Approximate visual readings of published five-fold medians; not treated as exact tabulated values.",
        },
        "per_fold": rows,
        "deep": {metric: summarize(frame[f"deep_{metric}"].tolist()) for metric in METRICS},
        "gated": {metric: summarize(frame[f"gated_{metric}"].tolist()) for metric in METRICS},
    }
    aggregate["improvement"] = {
        "mse_mean_relative_reduction": float(1.0 - aggregate["gated"]["mse"]["mean"] / aggregate["deep"]["mse"]["mean"]),
        "mse_median_relative_reduction": float(1.0 - aggregate["gated"]["mse"]["median"] / aggregate["deep"]["mse"]["median"]),
        "ci_mean_gain": float(aggregate["gated"]["ci"]["mean"] - aggregate["deep"]["ci"]["mean"]),
        "pearson_mean_gain": float(aggregate["gated"]["pearson"]["mean"] - aggregate["deep"]["pearson"]["mean"]),
    }

    out_json = results_dir / "dtiam_drugcold_5fold_summary.json"
    out_csv = results_dir / "dtiam_drugcold_5fold_summary.csv"
    out_json.write_text(json.dumps(aggregate, indent=2))
    frame.to_csv(out_csv, index=False)
    print(json.dumps(aggregate, indent=2))
    print(f"Wrote {out_json}")
    print(f"Wrote {out_csv}")


if __name__ == "__main__":
    main()
