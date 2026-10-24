#!/usr/bin/env python3
"""Evaluate four matched predictors using native affinity labels and one prior."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit

from train_nexusdti_regressor import PROJECT, regression_metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--deep", type=Path, required=True)
    parser.add_argument("--prior", type=Path, required=True)
    parser.add_argument("--data-table", type=Path, required=True)
    parser.add_argument("--manifest-file", type=Path, required=True)
    parser.add_argument("--target-column", default="kiba_score")
    parser.add_argument("--output-stem", required=True)
    args = parser.parse_args()
    deep, prior = pd.read_csv(args.deep), pd.read_csv(args.prior)
    for frame in (deep, prior):
        if "affinity_true" not in frame:
            frame.rename(columns={"pKd_true": "affinity_true"}, inplace=True)
        if frame.duplicated(["split", "row_id"]).any():
            raise ValueError("Duplicate prediction row IDs")
    frame = deep[["split", "row_id", "affinity_true", "prediction_raw"]].merge(
        prior[["split", "row_id", "affinity_true", "prediction_raw", "max_train_drug_tanimoto"]],
        on=["split", "row_id"], validate="one_to_one", suffixes=("_deep", "_prior"),
    )
    if len(frame) != len(deep) or len(frame) != len(prior):
        raise RuntimeError("Deep/prior predictions do not cover the same rows")
    if not np.allclose(frame.affinity_true_deep, frame.affinity_true_prior, rtol=0, atol=2e-6):
        raise RuntimeError("Deep/prior affinity labels disagree")
    frame.rename(columns={"affinity_true_deep": "affinity_true", "prediction_raw_deep": "prediction_deep",
                          "prediction_raw_prior": "prediction_neighbor_prior"}, inplace=True)
    frame.drop(columns="affinity_true_prior", inplace=True)
    table, manifest = pd.read_csv(args.data_table), pd.read_csv(args.manifest_file)
    expected = manifest.loc[manifest.split.isin(["val", "test"]), ["row_id", "split"]]
    aligned = expected.merge(frame, on=["split", "row_id"], validate="one_to_one")
    if len(aligned) != len(expected) or len(frame) != len(expected):
        raise RuntimeError("Predictions do not match validation/test manifest")
    truth = aligned.merge(table[["row_id", args.target_column]], on="row_id", validate="one_to_one")
    if not np.allclose(truth.affinity_true, truth[args.target_column], rtol=0, atol=2e-6):
        raise RuntimeError("Predictions disagree with native benchmark labels")
    frame = aligned
    val = frame.loc[frame.split.eq("val")]
    if val.empty or not frame.split.eq("test").any():
        raise ValueError("Both validation and test rows are required")
    difference = val.prediction_deep.to_numpy() - val.prediction_neighbor_prior.to_numpy()
    denominator = float(np.dot(difference, difference))
    fixed_weight = float(np.clip(np.dot(val.affinity_true - val.prediction_neighbor_prior, difference) / denominator, 0, 1)) if denominator > 0 else 0.5
    sim_mean = float(val.max_train_drug_tanimoto.mean())
    sim_std = float(val.max_train_drug_tanimoto.std())
    if sim_std < 1e-8:
        raise ValueError("Validation similarity has no variation")
    z_val = (val.max_train_drug_tanimoto.to_numpy() - sim_mean) / sim_std

    def objective(parameters):
        weight = expit(parameters[0] + parameters[1] * z_val)
        prediction = weight * val.prediction_deep + (1 - weight) * val.prediction_neighbor_prior
        return float(np.mean((val.affinity_true - prediction) ** 2) + 1e-3 * parameters[1] ** 2)

    optimized = minimize(objective, np.array([logit(0.335), 0.0]), method="L-BFGS-B")
    if not optimized.success:
        raise RuntimeError(f"Validation gate fit failed: {optimized.message}")
    frame["deep_weight"] = expit(optimized.x[0] + optimized.x[1] * (frame.max_train_drug_tanimoto - sim_mean) / sim_std)
    frame["prediction_fixed_blend"] = fixed_weight * frame.prediction_deep + (1 - fixed_weight) * frame.prediction_neighbor_prior
    frame["prediction_similarity_gate"] = frame.deep_weight * frame.prediction_deep + (1 - frame.deep_weight) * frame.prediction_neighbor_prior
    models = ("deep", "neighbor_prior", "fixed_blend", "similarity_gate")
    result = {
        "target_column": args.target_column,
        "protocol": "All four predictors share the same deep output and frozen training-only prior. Blend/gate fit on validation rows only. Raw outputs, no affine calibration.",
        "rows": {part: int(frame.split.eq(part).sum()) for part in ("val", "test")},
        "fixed_deep_weight": fixed_weight,
        "gate": {"a": float(optimized.x[0]), "b": float(optimized.x[1]), "sim_mean": sim_mean, "sim_std": sim_std},
        "input_sha256": {name: hashlib.sha256(path.read_bytes()).hexdigest()
                         for name, path in (("deep", args.deep), ("prior", args.prior), ("table", args.data_table), ("manifest", args.manifest_file))},
    }
    for part in ("val", "test"):
        selected = frame.loc[frame.split.eq(part)]
        result["validation" if part == "val" else "test"] = {
            model: regression_metrics(selected.affinity_true.to_numpy(), selected[f"prediction_{model}"].to_numpy())
            for model in models
        }
    output = PROJECT / "results" / f"{args.output_stem}_ablation.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    prediction_dir = output.parent / "predictions"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(prediction_dir / f"{args.output_stem}_ablation_predictions.csv", index=False)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"output": str(output), "test": result["test"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
