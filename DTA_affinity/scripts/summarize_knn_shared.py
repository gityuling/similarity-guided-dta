#!/usr/bin/env python3
"""Aggregate all planned runs, verify predictions, and export matched comparisons."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

PROJECT = Path(__file__).resolve().parents[1]
RUNS = PROJECT / "results/knn_shared_20261025"
REPORT = PROJECT / "reports/knn_shared_20261025"
METHODS = ["base", "knn_dta", "ada_knn_dta", "sgf", "neighbor"]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-repeats", type=int, default=10000)
    args = parser.parse_args()
    config_path = PROJECT / "configs/knn_shared_20261025.json"
    config = json.loads(config_path.read_text())
    REPORT.mkdir(parents=True, exist_ok=True)
    metrics, predictions, provenance, missing = [], [], [], []
    metric_roundtrip_max_error = 0.
    for dataset, fold, seed in itertools.product(config["datasets"], config["folds"], config["seeds"]):
        folder = RUNS / dataset / f"fold_{fold}" / f"seed_{seed}"
        path = folder / "result.json"
        if not path.exists():
            missing.append({"dataset": dataset, "fold": fold, "seed": seed})
            continue
        result = json.loads(path.read_text())
        identity = result["identity"]
        if identity["config"] != sha(config_path):
            raise ValueError("Results have inconsistent configurations")
        for key, code in (("runner", "run_knn_shared.py"), ("core", "knn_dta_reference.py")):
            if identity[key] != sha(PROJECT / "scripts" / code):
                raise ValueError(f"Code changed after run: {dataset}/{fold}/{seed}/{key}")
        prediction_path = folder / "test_predictions.csv"
        if sha(prediction_path) != result["prediction_sha256"]:
            raise ValueError("Prediction hash mismatch")
        frame = pd.read_csv(prediction_path, dtype={"drug_id": str, "protein_id": str})
        expected = pd.read_csv(PROJECT / f"reports/knn_comparison_20261025/inputs/{dataset}/fold_{fold}/test_labels.csv")
        checked = frame[["row_id", "affinity"]].merge(expected, on="row_id", how="outer", validate="one_to_one", indicator=True)
        if not checked._merge.eq("both").all() or not np.allclose(checked.affinity_x, checked.affinity_y, atol=1e-12, rtol=0):
            raise ValueError("Test rows/labels differ from the common protocol")
        for method in METHODS:
            mse = float(np.mean((frame.affinity - frame[method]) ** 2))
            pearson = float(pearsonr(frame.affinity, frame[method]).statistic)
            cached_metrics = [result["test"][method]["mse"], result["test"][method]["pearson"]]
            metric_roundtrip_max_error = max(metric_roundtrip_max_error, float(np.max(np.abs(np.array([mse, pearson]) - cached_metrics))))
            # CSV serialization of float32 predictions keeps about seven digits.
            if not np.allclose([mse, pearson], cached_metrics, atol=2e-7, rtol=0):
                raise ValueError("Recomputed metrics differ")
            metrics.append({"dataset": dataset, "fold": fold, "seed": seed, "method": method, "mse": mse, "pearson": pearson})
        predictions.append(frame.assign(dataset=dataset, fold=fold, seed=seed))
        provenance.append({"path": str(path.relative_to(PROJECT)), "sha256": sha(path), "wall_seconds": result["wall_seconds"]})
    (REPORT / "progress.json").write_text(json.dumps({"completed": len(provenance), "planned": 30, "missing": missing}, indent=2) + "\n")
    if missing:
        print(json.dumps({"completed": len(provenance), "planned": 30, "status": "waiting for all planned runs"}))
        return
    metric_frame = pd.DataFrame(metrics)
    metric_frame.to_csv(REPORT / "per_run_metrics.csv", index=False)
    oof = pd.concat(predictions, ignore_index=True)
    oof.to_csv(REPORT / "oof_predictions.csv", index=False)
    fold_metrics = metric_frame.groupby(["dataset", "fold", "method"], as_index=False)[["mse", "pearson"]].mean()
    fold_metrics.to_csv(REPORT / "seed_averaged_folds.csv", index=False)
    summary = []
    for (dataset, method), rows in fold_metrics.groupby(["dataset", "method"], sort=False):
        summary.append({"dataset": dataset, "method": method,
                        "mse_mean": float(rows.mse.mean()), "mse_fold_sd": float(rows.mse.std(ddof=1)),
                        "pearson_mean": float(rows.pearson.mean()), "pearson_fold_sd": float(rows.pearson.std(ddof=1))})
    summary_frame = pd.DataFrame(summary)
    summary_frame.to_csv(REPORT / "summary.csv", index=False)
    rankings = []
    for (dataset, seed, target), frame in oof.groupby(["dataset", "seed", "protein_id"], sort=False):
        if frame.chemical_identity.nunique() < 20 or frame.affinity.nunique() < 2:
            continue
        for method in METHODS:
            if frame[method].nunique() < 2:
                continue
            rankings.append({"dataset": dataset, "seed": seed, "protein_id": target, "method": method,
                             "spearman": float(spearmanr(frame.affinity, frame[method]).statistic)})
    ranking_frame = pd.DataFrame(rankings)
    ranking_frame.to_csv(REPORT / "target_rankings.csv", index=False)
    rank_summary = ranking_frame.groupby(["dataset", "seed", "method"], as_index=False).spearman.mean()
    rank_summary.to_csv(REPORT / "ranking_seed_means.csv", index=False)
    rng = np.random.default_rng(20261025)
    contrasts = []
    for dataset in config["datasets"]:
        rows = oof.loc[oof.dataset.eq(dataset)].copy()
        for method in METHODS:
            rows[method + "_se"] = (rows[method] - rows.affinity) ** 2
        # Average squared errors over seeds; this matches the reported mean of
        # seed-specific MSEs, rather than scoring an ensemble of predictions.
        errors = rows.groupby(["fold", "row_id", "chemical_identity"], as_index=False)[[m + "_se" for m in METHODS]].mean()
        draws = np.zeros((args.bootstrap_repeats, len(METHODS)))
        for fold, frame in errors.groupby("fold", sort=True):
            grouped = frame.groupby("chemical_identity", sort=True)
            sums = grouped[[m + "_se" for m in METHODS]].sum().to_numpy()
            counts = grouped.size().to_numpy()
            n = len(counts)
            # Chemical identities are sampled within each fixed outer fold.
            for start in range(0, args.bootstrap_repeats, 500):
                end = min(start + 500, args.bootstrap_repeats)
                weights = rng.multinomial(n, np.full(n, 1. / n), size=end - start)
                draws[start:end] += (weights @ sums) / (weights @ counts)[:, None] / 5.
        for other in ("base", "knn_dta", "ada_knn_dta", "neighbor"):
            a = METHODS.index("sgf")
            b = METHODS.index(other)
            boot = draws[:, a] - draws[:, b]
            selected = metric_frame.loc[metric_frame.dataset.eq(dataset)]
            pivot = selected.pivot(index=["fold", "seed"], columns="method", values="mse")
            differences = pivot.sgf - pivot[other]
            baseline_mean = float(pivot[other].mean())
            contrasts.append({"dataset": dataset, "comparison": "sgf_minus_" + other,
                              "mse_difference": float(differences.mean()),
                              "ci95_low": float(np.quantile(boot, .025)), "ci95_high": float(np.quantile(boot, .975)),
                              "sgf_lower_mse_runs": int((differences < 0).sum()), "runs": len(differences),
                              "relative_mse_reduction_percent": float(-100 * differences.mean() / baseline_mean)})
    pd.DataFrame(contrasts).to_csv(REPORT / "paired_comparisons.csv", index=False)
    report = {"complete": True, "runs": len(provenance), "config_sha256": sha(config_path),
              "metric_roundtrip_max_abs_error": metric_roundtrip_max_error,
              "methods": METHODS, "summary": summary, "paired_comparisons": contrasts,
              "ranking": rank_summary.groupby(["dataset", "method"]).spearman.mean().reset_index().to_dict("records"),
              "provenance": provenance,
              "bootstrap": {"repeats": args.bootstrap_repeats, "seed": 20261025,
                            "unit": "chemical identity within each outer test fold",
                            "estimand": "mean over folds of seed-averaged MSE differences",
                            "scope": "conditional on the trained models and their fixed predictions"}}
    (REPORT / "analysis.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"complete": True, "runs": len(provenance), "summary": summary, "paired_comparisons": contrasts}, indent=2))


if __name__ == "__main__":
    main()
