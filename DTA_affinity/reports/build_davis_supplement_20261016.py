#!/usr/bin/env python3
"""Consolidate existing Davis scratch predictions without training or GPU use.

The same frozen neighbor predictions feed all four ablations. Validation-only
gate fitting follows evaluate_similarity_gated_ensemble.py exactly. These are
supplemental exploratory analyses of previously inspected development folds.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from scipy.optimize import minimize
from scipy.special import expit, logit


PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT / "scripts"))
from evaluate_similarity_gated_ensemble import load_base, load_prior  # noqa: E402
from train_nexusdti_regressor import regression_metrics  # noqa: E402


MODELS = {
    "deep": "prediction_deep",
    "neighbor_prior": "prediction_prior",
    "fixed_blend": "prediction_fixed",
    "similarity_gate": "prediction_gated",
}
METRICS = ("mse", "ci", "pearson")
OUTPUT = Path(__file__).with_name("davis_supplement_20261016.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def canonical(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    require(mol is not None, "Invalid input SMILES")
    return Chem.MolToSmiles(mol)


def aggregate(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(array.mean()), "sd_across_folds_ddof1": float(array.std(ddof=1)), "per_fold": array.tolist()}


def maximum_difference(a, b) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))))


def validate_frame(frame: pd.DataFrame, manifest: pd.DataFrame, table: pd.DataFrame, label: str) -> float:
    require(not frame.duplicated(["split", "row_id"]).any(), f"{label}: duplicate prediction IDs")
    expected = manifest.loc[manifest.split.isin(["val", "test"]), ["split", "row_id"]]
    check = expected.merge(frame[["split", "row_id"]], on=["split", "row_id"], how="outer", validate="one_to_one", indicator=True)
    require(check._merge.eq("both").all(), f"{label}: prediction/manifest row mismatch")
    aligned = frame.merge(table[["row_id", "pKd_floor"]], on="row_id", validate="one_to_one")
    difference = maximum_difference(aligned.pKd_true, aligned.pKd_floor)
    require(difference <= 1e-6, f"{label}: labels disagree with data table")
    return difference


def fit_gate(val: pd.DataFrame, frame: pd.DataFrame) -> tuple[dict, np.ndarray]:
    y_val = val.pKd_true.to_numpy()
    deep_val = val.prediction_deep.to_numpy()
    prior_val = val.prediction_prior.to_numpy()
    sim_mean = float(val.max_train_drug_tanimoto.mean())
    sim_std = float(val.max_train_drug_tanimoto.std())
    require(sim_std >= 1e-8, "Validation structural similarity has no variation")
    z_val = (val.max_train_drug_tanimoto.to_numpy() - sim_mean) / sim_std

    def mixed(params, z, deep, prior):
        weight = expit(params[0] + params[1] * z)
        return weight * deep + (1.0 - weight) * prior, weight

    def objective(params):
        prediction, _ = mixed(params, z_val, deep_val, prior_val)
        return float(np.mean((y_val - prediction) ** 2) + 1e-3 * params[1] ** 2)

    initial = np.array([logit(0.335), 0.0], dtype=np.float64)
    optimized = minimize(objective, initial, method="L-BFGS-B")
    require(optimized.success, f"Gate optimization failed: {optimized.message}")
    prediction, weights = mixed(
        optimized.x,
        (frame.max_train_drug_tanimoto.to_numpy() - sim_mean) / sim_std,
        frame.prediction_deep.to_numpy(),
        frame.prediction_prior.to_numpy(),
    )
    return {
        "a": float(optimized.x[0]), "b": float(optimized.x[1]),
        "validation_similarity_mean": sim_mean, "validation_similarity_sd_ddof1": sim_std,
        "slope_l2_penalty": 1e-3, "optimizer": "L-BFGS-B",
        "initial_parameters": initial.tolist(), "objective": float(optimized.fun),
        "optimization_success": bool(optimized.success),
        "prediction_deep_weight_range": [float(weights.min()), float(weights.max())],
    }, prediction


def main() -> None:
    inputs = {Path(__file__).resolve()}
    inputs.update(PROJECT / "scripts" / name for name in (
        "evaluate_similarity_gated_ensemble.py", "evaluate_drug_similarity_prior.py", "train_nexusdti_regressor.py",
    ))
    table_path = PROJECT / "data/processed/davis_dtiam_protocol/table.csv"
    inputs.add(table_path)
    table = pd.read_csv(table_path)
    require(table.row_id.is_unique, "Data table row IDs are not unique")
    canonical_map = {s: canonical(s) for s in table.Canonical_SMILES.unique()}
    table["canonical_smiles"] = table.Canonical_SMILES.map(canonical_map)
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = {s: generator.GetFingerprint(Chem.MolFromSmiles(s)) for s in table.canonical_smiles.unique()}
    per_fold, oof, integrity = [], [], []
    minor_differences = []
    for fold in range(5):
        tag = f"dtiam_drugcold_f{fold}_scratch_b64_lr1e4_e20"
        gate_tag = tag + "_similarity_gated_ensemble" if fold else "dtiam_drugcold_f0_scratch_b64_similarity_gated_ensemble"
        prior_tag = tag + "_morgan_neighbor_prior" if fold else "dtiam_drugcold_f0_morgan_neighbor_prior"
        paths = {
            "manifest": PROJECT / f"data/processed/davis_dtiam_protocol/folds/drug_coldstart/fold_{fold}.csv",
            "deep_predictions": PROJECT / f"results/predictions/nexusdti_{tag}_affine_calibration_predictions.csv",
            "deep_result": PROJECT / f"results/nexusdti_regressor_{tag}.json",
            "prior_predictions": PROJECT / f"results/predictions/{prior_tag}_predictions.csv",
            "prior_result": PROJECT / f"results/{prior_tag}.json",
            "gate_predictions": PROJECT / f"results/predictions/{gate_tag}_predictions.csv",
            "gate_result": PROJECT / f"results/{gate_tag}.json",
        }
        inputs.update(paths.values())
        manifest = pd.read_csv(paths["manifest"])
        require(manifest.row_id.is_unique, f"Fold {fold}: duplicate manifest row IDs")
        require(set(manifest.row_id) == set(table.row_id), f"Fold {fold}: manifest does not cover the table")
        require(set(manifest.split) == {"train", "val", "test"}, f"Fold {fold}: invalid split labels")
        entities = manifest[["row_id", "split"]].merge(table[["row_id", "canonical_smiles", "Sequence"]], on="row_id", validate="one_to_one")
        drugs = {split: set(entities.loc[entities.split == split, "canonical_smiles"]) for split in ("train", "val", "test")}
        for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
            require(not drugs[a] & drugs[b], f"Fold {fold}: {a}/{b} drug overlap")
        base = load_base(paths["deep_predictions"])
        prior = load_prior(paths["prior_predictions"])
        merged = base.merge(prior, on=["split", "row_id"], validate="one_to_one")
        require(len(merged) == len(base) == len(prior), f"Fold {fold}: deep/prior IDs mismatch")
        require(np.allclose(merged.pKd_true_base, merged.pKd_true_prior, atol=1e-6, rtol=0), "Deep/prior labels differ")
        merged = merged.rename(columns={"pKd_true_base": "pKd_true"}).drop(columns="pKd_true_prior")
        label_error = validate_frame(merged, manifest, table, f"Fold {fold} deep/prior")
        require(np.isfinite(merged[["pKd_true", "prediction_deep", "prediction_prior", "max_train_drug_tanimoto"]]).all().all(), "Non-finite predictions")
        require(merged.max_train_drug_tanimoto.between(0, 1).all(), "Similarity outside [0,1]")
        expected_similarity = {s: max(DataStructs.BulkTanimotoSimilarity(fps[s], [fps[t] for t in sorted(drugs["train"])])) for s in drugs["val"] | drugs["test"]}
        checked = merged.merge(table[["row_id", "canonical_smiles"]], on="row_id", validate="one_to_one")
        similarity_error = maximum_difference(checked.max_train_drug_tanimoto, checked.canonical_smiles.map(expected_similarity))
        require(similarity_error < 1e-12, f"Fold {fold}: stored similarity not equal to training-only Morgan similarity")
        val = merged.loc[merged.split == "val"].reset_index(drop=True)
        direction = val.prediction_deep.to_numpy() - val.prediction_prior.to_numpy()
        denominator = float(np.dot(direction, direction))
        alpha = float(np.clip(np.dot(direction, val.pKd_true.to_numpy() - val.prediction_prior.to_numpy()) / denominator, 0, 1)) if denominator else 0.0
        merged["prediction_fixed"] = alpha * merged.prediction_deep + (1 - alpha) * merged.prediction_prior
        gate, merged["prediction_gated"] = fit_gate(val, merged)
        old_gate = pd.read_csv(paths["gate_predictions"])
        validate_frame(old_gate, manifest, table, f"Fold {fold} existing gate")
        comparison = merged.merge(old_gate, on=["split", "row_id"], validate="one_to_one", suffixes=("_refit", "_existing"))
        deep_error = maximum_difference(comparison.prediction_deep_refit, comparison.prediction_deep_existing)
        prior_error = maximum_difference(comparison.prediction_prior_refit, comparison.prediction_prior_existing)
        gate_error = maximum_difference(comparison.prediction_gated, comparison.prediction_gated_raw)
        require(deep_error < 1e-12 and prior_error < 1e-12, "Existing gate used different component predictions")
        require(gate_error < 1e-7, f"Fold {fold}: refitted gate does not reproduce existing predictions ({gate_error})")
        test = merged.loc[merged.split == "test"].copy()
        metrics = {model: {key: float(value) for key, value in regression_metrics(test.pKd_true.to_numpy(), test[column].to_numpy()).items() if key in METRICS} for model, column in MODELS.items()}
        deep_result = json.loads(paths["deep_result"].read_text())
        prior_result = json.loads(paths["prior_result"].read_text())
        gate_result = json.loads(paths["gate_result"].read_text())
        for model, stored in (("deep", deep_result["metrics_test"]), ("neighbor_prior", prior_result["test"]["raw"]), ("similarity_gate", gate_result["test"]["raw"])):
            delta = {metric: metrics[model][metric] - stored[metric] for metric in METRICS}
            require(max(abs(value) for value in delta.values()) < 5e-6, f"Fold {fold}: stored {model} metrics disagree materially")
            if max(abs(value) for value in delta.values()) > 1e-12:
                minor_differences.append({"fold": fold, "model": model, "recomputed_minus_stored": delta})
        test["fold"] = fold
        test = test.merge(table[["row_id", "drug_idx", "drug_id", "protein_id", "is_censored_floor", "pKd_floor"]], on="row_id", validate="one_to_one")
        oof.append(test)
        per_fold.append({
            "fold": fold, "training_seed": deep_result["seed"], "metrics_test": metrics,
            "fixed_blend_deep_weight": alpha, "gate": gate,
            "prior_selected_hyperparameters": prior_result["selected_hyperparameters"],
            "paths": {key: str(path.relative_to(PROJECT)) for key, path in paths.items()},
            "fold_0_alias_used": fold == 0,
        })
        integrity.append({
            "fold": fold, "rows_by_split": {split: int((manifest.split == split).sum()) for split in ("train", "val", "test")},
            "canonical_drugs_by_split": {split: len(values) for split, values in drugs.items()},
            "train_val_drug_overlap": 0, "train_test_drug_overlap": 0, "val_test_drug_overlap": 0,
            "max_abs_label_error_vs_table": label_error, "max_abs_train_only_similarity_error": similarity_error,
            "max_abs_deep_error_vs_existing_gate": deep_error, "max_abs_prior_error_vs_existing_gate": prior_error,
            "max_abs_refitted_gate_error_vs_existing_gate": gate_error,
        })
        print(f"Fold {fold}: " + ", ".join(f"{model} MSE={values['mse']:.8f}" for model, values in metrics.items()), flush=True)
    combined = pd.concat(oof, ignore_index=True).sort_values("row_id").reset_index(drop=True)
    require(combined.row_id.is_unique and set(combined.row_id) == set(table.row_id), "Five-fold test coverage is incomplete or duplicated")
    require(combined.groupby("drug_idx").fold.nunique().eq(1).all(), "A test drug appears in multiple outer folds")
    summaries = {model: {metric: aggregate([item["metrics_test"][model][metric] for item in per_fold]) for metric in METRICS} for model in MODELS}
    masks = [
        ("all", np.ones(len(combined), dtype=bool)),
        ("pKd_floor_5", combined.is_censored_floor), ("pKd_above_5", ~combined.is_censored_floor),
        ("max_Tanimoto_below_0.3", combined.max_train_drug_tanimoto < 0.3),
        ("max_Tanimoto_0.3_to_0.5", combined.max_train_drug_tanimoto.between(0.3, 0.5, inclusive="left")),
        ("max_Tanimoto_0.5_to_0.7", combined.max_train_drug_tanimoto.between(0.5, 0.7, inclusive="left")),
        ("max_Tanimoto_at_least_0.7", combined.max_train_drug_tanimoto >= 0.7),
    ]
    subgroups = []
    for name, mask in masks:
        subset = combined.loc[mask]
        subgroups.append({
            "subgroup": name, "pairs": len(subset), "drugs": int(subset.drug_idx.nunique()),
            "folds_represented": sorted(int(fold) for fold in subset.fold.unique()),
            "pair_pooled_errors": {model: {
                "mse": float(np.mean((subset.pKd_true - subset[column]) ** 2)),
                "mae": float(np.mean(np.abs(subset.pKd_true - subset[column]))),
            } for model, column in MODELS.items()},
        })
    csv_path = OUTPUT.with_suffix(".csv")
    for model, column in MODELS.items():
        combined[f"squared_error_{model}"] = (combined.pKd_true - combined[column]) ** 2
    combined.to_csv(csv_path, index=False)
    record = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "dataset": "Davis full 30056-pair pKd table", "setting": "DTIAM official drug-cold five-fold outer tests",
        "scope": "Exploratory supplemental reanalysis of existing scratch-b64 predictions; no retraining, no new seed, no independent confirmation dataset.",
        "aggregation": {
            "primary": "Unweighted mean of five test-fold metrics; sample SD across folds uses ddof=1. Folds are not random-seed replicates.",
            "subgroups": "Pair-weighted errors from pooled out-of-fold test predictions; not a mean across folds. Similarity bins were selected in a prior audit.",
            "labels": "Use the stored float32-derived deep prediction labels for all four ablations, matching the existing gate evaluator. They agree with the full-precision table within 1e-6.",
            "gate": "Same validation normalization, initialization, L2 penalty, optimizer and raw gate formula as evaluate_similarity_gated_ensemble.py; no affine calibration.",
            "fixed_blend": "One deep weight constrained to [0,1] fitted by least squares on validation rows only.",
            "prior": "Existing scratch priors frozen by input SHA-256; identical predictions feed prior-only, fixed blend and gate. No neighbor regeneration or test-based retuning.",
        },
        "five_fold_metrics": summaries, "folds": per_fold, "pooled_out_of_fold_subgroups": subgroups,
        "integrity_checks": {"passed": True, "per_fold": integrity, "unique_test_rows": len(combined), "unique_test_drugs": int(combined.drug_idx.nunique()), "every_table_row_tested_once": True},
        "minor_metric_differences": minor_differences,
        "limitations": [
            "Historical test results were inspected during development; these analyses are exploratory and do not create a fresh holdout.",
            "Reusing hashed prior predictions avoids known historical prior drift, but does not independently regenerate the prior or audit pretrained encoder corpora.",
            "The pKd=5 floor is censored; pKd>5 denotes nonfloor labels, not a predefined high-affinity class.",
            "The highest similarity bin has very few drugs; inspect subgroup counts before drawing conclusions.",
        ],
        "environment": {"python": sys.version, "executable": sys.executable, "platform": platform.platform(), "rdkit": rdBase.rdkitVersion, "packages": {package: importlib.metadata.version(package) for package in ("numpy", "pandas", "scipy", "torch")}, "device_used": "CPU; no model checkpoints loaded"},
        "inputs": [{"path": str(path.relative_to(PROJECT)), "bytes": path.stat().st_size, "sha256": sha256(path)} for path in sorted(inputs)],
        "out_of_fold_csv": {"path": str(csv_path.relative_to(PROJECT)), "rows": len(combined), "sha256": sha256(csv_path)},
    }
    OUTPUT.write_text(json.dumps(record, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps(summaries, indent=2))
    print(f"Wrote {OUTPUT}\nWrote {csv_path}")


if __name__ == "__main__":
    main()
