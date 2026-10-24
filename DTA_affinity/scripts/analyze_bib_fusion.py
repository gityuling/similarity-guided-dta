#!/usr/bin/env python3
"""Matched fusion controls, similarity strata and clustered fixed-prediction CIs."""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import pearsonr, spearmanr

from bib_common import (DATASETS, PROJECT, baseline_paths, canonical,
                        check_predictions, manifest_path, record, table_path,
                        verify_freeze)
from train_nexusdti_regressor import regression_metrics


MODELS = ("deep", "neighbor_prior", "half_blend", "fixed_blend", "linear_stacking", "similarity_gate")
CONFIG = PROJECT / "configs/bib_analysis_20261017.json"
OUTPUT = PROJECT / "reports/bib_fusion_20261017"


def metric(y, prediction) -> dict:
    y, prediction = np.asarray(y, dtype=float), np.asarray(prediction, dtype=float)
    return {"mse": float(np.mean((y - prediction) ** 2)),
            "mae": float(np.mean(np.abs(y - prediction))),
            "pearson": float(pearsonr(y, prediction).statistic) if len(y) > 1 and np.std(y) > 0 and np.std(prediction) > 0 else None}


def fit_controls(frame: pd.DataFrame) -> dict:
    val = frame.loc[frame.split.eq("val")]
    y, deep, prior = (val[c].to_numpy() for c in ("truth", "prediction_deep", "prediction_neighbor_prior"))
    difference = deep - prior
    denominator = difference @ difference
    alpha = float(np.clip((y - prior) @ difference / denominator, 0, 1)) if denominator else 0.5
    design = np.column_stack([np.ones(len(val)), deep, prior])
    beta, _, rank, singular = np.linalg.lstsq(design, y, rcond=None)
    mean, sd = float(val.max_train_drug_tanimoto.mean()), float(val.max_train_drug_tanimoto.std(ddof=1))
    if sd < 1e-8:
        raise RuntimeError("No validation similarity variation")
    z = (val.max_train_drug_tanimoto.to_numpy() - mean) / sd

    def objective(parameters):
        weights = expit(parameters[0] + parameters[1] * z)
        prediction = weights * deep + (1 - weights) * prior
        return float(np.mean((y - prediction) ** 2) + 1e-3 * parameters[1] ** 2)

    optimized = minimize(objective, np.array([logit(0.335), 0.0]), method="L-BFGS-B")
    if not optimized.success:
        raise RuntimeError(f"Gate fit failed: {optimized.message}")
    frame["deep_weight"] = expit(optimized.x[0] + optimized.x[1] * (frame.max_train_drug_tanimoto - mean) / sd)
    d, p = frame.prediction_deep, frame.prediction_neighbor_prior
    frame["prediction_half_blend"] = (d + p) / 2
    frame["prediction_fixed_blend"] = alpha * d + (1 - alpha) * p
    frame["prediction_linear_stacking"] = beta[0] + beta[1] * d + beta[2] * p
    frame["prediction_similarity_gate"] = frame.deep_weight * d + (1 - frame.deep_weight) * p
    return {"fixed_deep_weight": alpha,
            "linear_stacking": {"intercept": float(beta[0]), "deep_coefficient": float(beta[1]),
                                "prior_coefficient": float(beta[2]), "design_rank": int(rank),
                                "singular_values": singular.tolist()},
            "gate": {"a": float(optimized.x[0]), "b": float(optimized.x[1]), "sim_mean": mean, "sim_std": sd}}


def bootstrap(frame: pd.DataFrame, count: int, seed: int) -> dict:
    # An identity may span multiple folds; share its count across folds and models.
    errors = np.column_stack([(frame.truth - frame[f"prediction_{m}"]) ** 2 for m in MODELS])
    working = frame[["fold", "chemical_identity"]].copy()
    for i, model in enumerate(MODELS):
        working[model] = errors[:, i]
    grouped = working.groupby(["fold", "chemical_identity"])
    sums, sizes = grouped[list(MODELS)].sum(), grouped.size()
    identities = sorted(frame.chemical_identity.unique())
    generator = np.random.default_rng(seed)
    samples = np.zeros((count, len(MODELS)))
    cluster_counts = {}
    fold_arrays = []
    for fold in sorted(frame.fold.unique()):
        se = sums.loc[fold].reindex(identities, fill_value=0).to_numpy()
        n = sizes.loc[fold].reindex(identities, fill_value=0).to_numpy()
        cluster_counts[str(fold)] = int(np.count_nonzero(n))
        fold_arrays.append((se, n))
    probabilities = np.full(len(identities), 1 / len(identities))
    offset, discarded = 0, 0
    while offset < count:
        draws = generator.multinomial(len(identities), probabilities, size=min(250, count - offset))
        denominators = np.column_stack([draws @ n for _, n in fold_arrays])
        valid = np.all(denominators > 0, axis=1)
        discarded += int(np.sum(~valid))
        draws, denominators = draws[valid], denominators[valid]
        for index, (se, _) in enumerate(fold_arrays):
            samples[offset:offset + len(draws)] += (draws @ se) / denominators[:, index, None] / len(fold_arrays)
        offset += len(draws)
    points = np.array([np.mean([np.mean((f.truth - f[f"prediction_{m}"]) ** 2)
                                       for _, f in frame.groupby("fold")]) for m in MODELS])
    comparisons = []
    gate = MODELS.index("similarity_gate")
    for i, model in enumerate(MODELS):
        if i == gate:
            continue
        difference = samples[:, gate] - samples[:, i]
        comparisons.append({"contrast": f"similarity_gate_minus_{model}", "mse_difference": float(points[gate] - points[i]),
                            "percentile_95": np.quantile(difference, [0.025, 0.975]).tolist(),
                            "bootstrap_fraction_below_zero": float(np.mean(difference < 0)),
                            "fraction_note": "Descriptive bootstrap fraction, not a p-value"})
    return {"repeats": count, "seed": seed, "sampling_unit": "canonical chemical identity, sampled globally with shared counts across folds and models",
            "total_identities": len(identities), "discarded_draws_with_empty_fold": discarded,
            "identities_per_fold": cluster_counts, "estimand": "unweighted five-fold MSE difference",
            "scope": "Conditional on saved fixed predictions; excludes training uncertainty; drug-family/scaffold dependence not modeled",
            "comparisons": comparisons}


def main() -> None:
    verify_freeze(include_features=False)
    config = json.loads(CONFIG.read_text())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    result = {"analysis_config": record(CONFIG), "training_config": record(PROJECT / config["training_config"]),
              "code": record(PROJECT / "scripts/analyze_bib_fusion.py"),
              "prediction_seed": 20261012, "datasets": {}, "inputs": [],
              "scope": "Post-hoc reanalysis of existing test predictions; no test-based model or checkpoint selection",
              "aggregation": "Unweighted outer-fold means for benchmark comparisons; pooled pair errors for subgroups"}
    subgroup_rows, controls, curves, ranking = [], [], [], []
    for dataset, settings in DATASETS.items():
        table = pd.read_csv(table_path(dataset))
        table["chemical_identity"] = table.Canonical_SMILES.map({s: canonical(s) for s in table.Canonical_SMILES.unique()})
        folds, test_frames = [], []
        for fold in range(5):
            paths = baseline_paths(dataset, fold)
            manifest = pd.read_csv(manifest_path(dataset, fold))
            for kind in ("deep_predictions", "prior_predictions"):
                check_predictions(paths[kind], table, manifest, settings["label"])
                result["inputs"].append(record(paths[kind]))
            deep, prior = pd.read_csv(paths["deep_predictions"]), pd.read_csv(paths["prior_predictions"])
            truth_column = "affinity_true" if "affinity_true" in deep else "pKd_true"
            frame = deep[["split", "row_id", truth_column, "prediction_raw"]].rename(
                columns={truth_column: "truth", "prediction_raw": "prediction_deep"})
            frame = frame.merge(prior[["split", "row_id", "prediction_raw", "max_train_drug_tanimoto"]],
                                on=["split", "row_id"], validate="one_to_one").rename(columns={"prediction_raw": "prediction_neighbor_prior"})
            fit_start = time.perf_counter()
            fitted = fit_controls(frame)
            fitted["fit_controls_wall_seconds"] = time.perf_counter() - fit_start
            saved = pd.read_csv(paths["fusion_predictions"])
            saved_column = "prediction_gated_raw" if dataset == "davis" else "prediction_similarity_gate"
            checked = frame.merge(saved[["split", "row_id", saved_column]], on=["split", "row_id"],
                                  validate="one_to_one", suffixes=("", "_saved"))
            actual_saved = saved_column if dataset == "davis" else saved_column + "_saved"
            error = float(np.max(np.abs(checked.prediction_similarity_gate - checked[actual_saved])))
            if error > 1e-7:
                raise RuntimeError(f"Reanalysis changed saved gate predictions: {dataset}, fold {fold}, delta={error}")
            frame = frame.merge(table[["row_id", "drug_idx", "drug_id", "protein_id", "chemical_identity", settings["label"]]],
                                on="row_id", validate="one_to_one")
            frame["fold"] = fold
            test = frame.loc[frame.split.eq("test")].copy()
            fold_metrics = {m: regression_metrics(test.truth.to_numpy(), test[f"prediction_{m}"].to_numpy()) for m in MODELS}
            folds.append({"fold": fold, "fit": fitted, "metrics": fold_metrics, "saved_gate_max_abs_difference": error})
            controls.append({"dataset": dataset, "fold": fold, "fixed_deep_weight": fitted["fixed_deep_weight"],
                             **fitted["linear_stacking"], **{f"gate_{k}": v for k, v in fitted["gate"].items()}})
            for similarity in np.linspace(0, 1, 101):
                gate = fitted["gate"]
                curves.append({"dataset": dataset, "fold": fold, "similarity": similarity,
                               "deep_weight": float(expit(gate["a"] + gate["b"] * (similarity - gate["sim_mean"]) / gate["sim_std"]))})
            test_frames.append(test)
        oof = pd.concat(test_frames, ignore_index=True).sort_values("row_id")
        if not oof.row_id.is_unique or set(oof.row_id) != set(table.row_id):
            raise RuntimeError("OOF predictions do not cover every benchmark pair exactly once")
        summary = {m: {k: {"mean": float(np.mean(values)), "sd_across_folds": float(np.std(values, ddof=1)), "per_fold": values}
                       for k in ("mse", "ci", "pearson") for values in [[f["metrics"][m][k] for f in folds]]} for m in MODELS}
        masks = [("all", np.ones(len(oof), dtype=bool))]
        for lo, hi in ((0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)):
            mask = oof.max_train_drug_tanimoto.ge(lo) & (oof.max_train_drug_tanimoto.lt(hi) if hi < 1 else oof.max_train_drug_tanimoto.le(1))
            masks.append((f"tanimoto_{lo}_{hi}", mask))
        if dataset == "davis":
            masks.extend([("pKd_floor_5", oof.pKd_floor.le(5 + 1e-8)), ("pKd_above_5", oof.pKd_floor.gt(5 + 1e-8))])
        subgroups = []
        for label, mask in masks:
            subset = oof.loc[mask]
            entry = {"subgroup": label, "pairs": len(subset), "drug_ids": int(subset.drug_idx.nunique()),
                     "chemical_identities": int(subset.chemical_identity.nunique()),
                     "folds_represented": [int(x) for x in sorted(subset.fold.unique())],
                     "deep_weight_mean": float(subset.deep_weight.mean()) if len(subset) else None,
                     "deep_weight_range": [float(subset.deep_weight.min()), float(subset.deep_weight.max())] if len(subset) else None,
                     "pooled_metrics": {m: metric(subset.truth, subset[f"prediction_{m}"]) for m in MODELS} if len(subset) else {}}
            subgroups.append(entry)
            for m, values in entry["pooled_metrics"].items():
                subgroup_rows.append({"dataset": dataset, "subgroup": label, "model": m, "pairs": len(subset),
                                      "drug_ids": entry["drug_ids"], "chemical_identities": entry["chemical_identities"],
                                      "mean_deep_weight": entry["deep_weight_mean"], **values})
        # All targets with at least 20 unique test identities and nonconstant labels are eligible.
        for target, subset in oof.groupby("protein_id", sort=True):
            if subset.chemical_identity.nunique() < 20 or subset.truth.nunique() < 2:
                continue
            for m in MODELS:
                prediction = subset[f"prediction_{m}"]
                rho = float(spearmanr(subset.truth, prediction).statistic) if prediction.nunique() > 1 else None
                ranking.append({"dataset": dataset, "target": str(target), "model": m, "pairs": len(subset),
                                "chemical_identities": int(subset.chemical_identity.nunique()), "spearman": rho,
                                **metric(subset.truth, prediction)})
        oof_path = OUTPUT / f"{dataset}_oof.csv"
        oof.to_csv(oof_path, index=False)
        uncertainty = bootstrap(oof, config["statistics"]["bootstrap_repeats"], config["statistics"]["bootstrap_seed"])
        result["datasets"][dataset] = {"summary": summary, "folds": folds, "subgroups": subgroups,
                                        "cluster_bootstrap": uncertainty, "oof": record(oof_path)}
        print(json.dumps({"dataset": dataset, "mean_mse": {m: v["mse"]["mean"] for m, v in summary.items()},
                          "bootstrap": uncertainty["comparisons"]}, indent=2), flush=True)
    pd.DataFrame(subgroup_rows).to_csv(OUTPUT / "similarity_subgroups.csv", index=False)
    pd.DataFrame(controls).to_csv(OUTPUT / "fusion_parameters.csv", index=False)
    pd.DataFrame(curves).to_csv(OUTPUT / "gate_curves.csv", index=False)
    pd.DataFrame(ranking).to_csv(OUTPUT / "per_target_ranking.csv", index=False)
    result["ranking_rule"] = config["ranking_rule"]
    result["ranking_macro"] = []
    ranking_frame = pd.DataFrame(ranking)
    for (dataset, model), group in ranking_frame.groupby(["dataset", "model"]):
        result["ranking_macro"].append({"dataset": dataset, "model": model, "targets": len(group),
                                        "spearman_macro": float(group.spearman.mean()),
                                        "pearson_macro": float(group.pearson.mean()), "mse_macro": float(group.mse.mean())})
    (OUTPUT / "analysis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
