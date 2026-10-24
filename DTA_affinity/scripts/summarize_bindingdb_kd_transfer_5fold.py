#!/usr/bin/env python3
"""Summarize strict continuous-BindingDB-Kd transfer on DTIAM drug-cold folds.

The report deliberately keeps the earlier gated system beside the new transfer
system and records the batch-size caveat.  It is an audit artifact, not a
claim of statistical significance from five correlated outer-fold estimates.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from train_nexusdti_regressor import PROJECT


METRICS = ("mse", "rmse", "ci", "pearson", "spearman")


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
        legacy = json.loads((results / f"dtiam_drugcold_f{fold}_similarity_gated_ensemble.json").read_text())
        transfer = json.loads((results / f"dtiam_drugcold_f{fold}_bindingdb_kd_similarity_gated_ensemble.json").read_text())
        deep = json.loads(
            (results / f"nexusdti_regressor_dtiam_drugcold_f{fold}_bindingdb_kd_e10_transfer_b64_lr1e4_e20.json").read_text()
        )
        row = {
            "fold": fold,
            "test_rows": int(deep["rows_test"]),
            "source_encoder_best_epoch": int(
                min(
                    range(1, 11),
                    key=lambda epoch: json.loads(
                        (results / f"nexusdti_bindingdb_kd_strict_f{fold}_e10_b64.json").read_text()
                    )["history"][epoch - 1]["mse_normalized"],
                )
            ),
            "davis_transfer_best_val_rmse": float(deep["best_val_rmse"]),
        }
        for metric in METRICS:
            legacy_value = float(legacy["test"]["raw"][metric])
            transfer_value = float(transfer["test"]["raw"][metric])
            row[f"legacy_gated_{metric}"] = legacy_value
            row[f"bindingdb_kd_gated_{metric}"] = transfer_value
            row[f"delta_{metric}"] = transfer_value - legacy_value
        rows.append(row)

    frame = pd.DataFrame(rows)
    summary = {
        "protocol": {
            "dataset": "Davis full 30,056-pair matrix with pKd=5.0 floor retained",
            "outer_evaluation": "DTIAM official five-fold drug-cold splitter",
            "source": "Exact positive single-chain BindingDB Kd records, transformed to pKd; every outer-fold validation/test drug removed before source splitting",
            "source_model": "NEXUS-DTA shared Uni-Mol2/ESM-2 interaction encoder pretrained on source pKd, then transferred without source head",
            "final_prediction": "Transferred Davis hybrid model gated with a same-target, training-only Morgan/Tanimoto neighbor prior; gate fitted only on that fold's Davis validation split",
            "final_metric_variant": "raw gated prediction; affine calibration is not used because held-out effect is inconsistent",
        },
        "caveat": {
            "legacy_gated_models": "Historical five-fold hybrid runs used batch size 4.",
            "transfer_gated_models": "Continuous-Kd transfer runs use batch size 64 for feasible strict source-transfer training.",
            "matched_ablation": "F0 batch-64 scratch + gate MSE=0.689696 versus F0 batch-64 continuous-Kd transfer + gate MSE=0.675780; this isolates a positive transfer contribution on one fold only.",
            "interpretation": "Five folds are not enough for a significance claim; report per-fold outcomes and mean/median rather than calling this an unqualified SOTA result.",
        },
        "per_fold": rows,
        "legacy_gated": {metric: aggregate(frame[f"legacy_gated_{metric}"].tolist()) for metric in METRICS},
        "bindingdb_kd_gated": {metric: aggregate(frame[f"bindingdb_kd_gated_{metric}"].tolist()) for metric in METRICS},
        "paired_delta_new_minus_legacy": {metric: aggregate(frame[f"delta_{metric}"].tolist()) for metric in METRICS},
    }
    summary["comparison"] = {
        "mean_mse_relative_reduction": float(
            1.0 - summary["bindingdb_kd_gated"]["mse"]["mean"] / summary["legacy_gated"]["mse"]["mean"]
        ),
        "mean_ci_gain": float(
            summary["bindingdb_kd_gated"]["ci"]["mean"] - summary["legacy_gated"]["ci"]["mean"]
        ),
    }
    json_path = results / "dtiam_drugcold_bindingdb_kd_transfer_5fold_summary.json"
    csv_path = results / "dtiam_drugcold_bindingdb_kd_transfer_5fold_summary.csv"
    json_path.write_text(json.dumps(summary, indent=2))
    frame.to_csv(csv_path, index=False)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {json_path}")
    print(f"Wrote {csv_path}")


if __name__ == "__main__":
    main()
