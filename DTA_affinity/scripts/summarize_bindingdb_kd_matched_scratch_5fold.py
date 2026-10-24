#!/usr/bin/env python3
"""Summarize the strict batch-matched Davis scratch-versus-transfer ablation."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from train_nexusdti_regressor import PROJECT


METRICS = ("mse", "rmse", "ci", "pearson", "spearman")
TRANSFER_TAG = "bindingdb_kd"
SCRATCH_TAG = "scratch_b64_lr1e4_e20"


def aggregate(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std(ddof=1)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def main() -> None:
    results = PROJECT / "results"
    rows = []
    for fold in range(5):
        transfer_path = results / f"dtiam_drugcold_f{fold}_{TRANSFER_TAG}_similarity_gated_ensemble.json"
        scratch_path = results / f"dtiam_drugcold_f{fold}_{SCRATCH_TAG}_similarity_gated_ensemble.json"
        transfer = json.loads(transfer_path.read_text())
        scratch = json.loads(scratch_path.read_text())
        row = {"fold": fold, "test_rows": int(transfer["rows"]["test"])}
        if row["test_rows"] != int(scratch["rows"]["test"]):
            raise RuntimeError(f"Fold {fold}: scratch and transfer test sizes differ")
        for metric in METRICS:
            transfer_value = float(transfer["test"]["raw"][metric])
            scratch_value = float(scratch["test"]["raw"][metric])
            row[f"transfer_{metric}"] = transfer_value
            row[f"scratch_{metric}"] = scratch_value
            row[f"delta_transfer_minus_scratch_{metric}"] = transfer_value - scratch_value
        rows.append(row)

    frame = pd.DataFrame(rows)
    result = {
        "protocol": {
            "dataset": "Davis full 30,056-pair matrix with pKd=5.0 floor retained",
            "outer_evaluation": "Official DTIAM five-fold drug-cold split (KFold shuffle=True, random_state=0)",
            "paired_control": "same manifest, NEXUS-DTA hybrid architecture, seed, batch_size=64, epochs=20, optimizer, early stopping, Morgan prior and validation-selected gate",
            "transfer_only_difference": "shared encoder initialized from the fold-specific strict BindingDB exact-Kd source checkpoint; scratch has no source initialization",
            "final_metric_variant": "raw validation-gated prediction; affine calibration is recorded but excluded from the primary aggregate",
        },
        "per_fold": rows,
        "bindingdb_kd_transfer_gated": {
            metric: aggregate(frame[f"transfer_{metric}"].tolist()) for metric in METRICS
        },
        "scratch_b64_gated": {
            metric: aggregate(frame[f"scratch_{metric}"].tolist()) for metric in METRICS
        },
        "paired_delta_transfer_minus_scratch": {
            metric: aggregate(frame[f"delta_transfer_minus_scratch_{metric}"].tolist()) for metric in METRICS
        },
    }
    result["comparison"] = {
        "mean_mse_relative_reduction_from_transfer": float(
            1.0
            - result["bindingdb_kd_transfer_gated"]["mse"]["mean"]
            / result["scratch_b64_gated"]["mse"]["mean"]
        ),
        "mean_ci_gain_from_transfer": float(
            result["bindingdb_kd_transfer_gated"]["ci"]["mean"]
            - result["scratch_b64_gated"]["ci"]["mean"]
        ),
        "mse_wins_transfer": int((frame["delta_transfer_minus_scratch_mse"] < 0).sum()),
        "mse_wins_scratch": int((frame["delta_transfer_minus_scratch_mse"] > 0).sum()),
    }
    json_path = results / "dtiam_drugcold_bindingdb_kd_matched_scratch_b64_5fold_summary.json"
    csv_path = results / "dtiam_drugcold_bindingdb_kd_matched_scratch_b64_5fold_summary.csv"
    json_path.write_text(json.dumps(result, indent=2))
    frame.to_csv(csv_path, index=False)
    print(json.dumps(result, indent=2))
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
