#!/usr/bin/env python3
"""Evaluate the fixed affinity/response ranking against separate CTRP endpoints."""

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from bib_common import ROOT, record, sha256, write_once
from predict_bib_cellular_affinity import CASE, MODELS


def metrics(y, scores):
    active = y < 0.5
    count = len(y)
    order = np.argsort(-scores, kind="stable")
    both = 0 < active.sum() < count
    result = {"spearman": float(spearmanr(scores, -y).statistic) if np.ptp(scores) and np.ptp(y) else None,
              "auroc": float(roc_auc_score(active, scores)) if both else None,
              "average_precision": float(average_precision_score(active, scores)) if active.any() else None}
    for k in (5, 10):
        hits = int(active[order[:k]].sum())
        result[f"hits_at_{k}"] = hits
        result[f"precision_at_{k}"] = hits / min(k, count)
        result[f"enrichment_at_{k}"] = result[f"precision_at_{k}"] / active.mean() if active.any() else None
    return result


def main():
    config = json.loads((CASE / "plan.json").read_text())
    coverage_audit = json.loads((CASE / "coverage_audit.json").read_text())
    for filename in ("affinity_audit.json", "functional_audit.json"):
        report = json.loads((CASE / filename).read_text())
        for item in report["outputs"]:
            if sha256(ROOT / item["path"]) != item["sha256"]:
                raise RuntimeError("Candidate score inputs changed")
    for item in coverage_audit["sources"]:
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError("Audited source changed before endpoint evaluation")
    affinity = pd.read_csv(CASE / "affinity_predictions.csv")
    functional = pd.read_csv(CASE / "functional_scores.csv")
    joined = affinity.merge(functional, on="chemical_identity", how="outer", validate="one_to_one", indicator=True)
    if not joined._merge.eq("both").all():
        raise RuntimeError("Affinity and functional candidate libraries differ")
    joined = joined.drop(columns="_merge").sort_values("chemical_identity").reset_index(drop=True)
    scores = [*MODELS, "functional_only", *(f"{m}_plus_functional" for m in MODELS)]
    joined["functional_only"] = joined.functional_concordance
    functional_rank = joined.functional_only.rank(method="average", pct=True)
    for model in MODELS:
        joined[f"{model}_plus_functional"] = (joined[model].rank(method="average", pct=True) + functional_rank) / 2
    # Endpoint values enter only after the complete candidate scores have been fixed.
    write_once(CASE / "candidate_scores_before_endpoints.csv", joined.to_csv(index=False))
    coverage = pd.read_csv(CASE / "candidate_coverage.csv").fillna("")
    mapping = coverage.loc[coverage.in_candidate_library, ["chemical_identity", "ctrp_pubchem_ids"]].copy()
    mapping["pubchem_id"] = mapping.ctrp_pubchem_ids.str.split("|")
    mapping = mapping.explode("pubchem_id").drop(columns="ctrp_pubchem_ids")
    source = ROOT / "UniPert_DrEval/data/raw/extracted/CTRPv2/CTRPv2.csv"
    responses = pd.read_csv(source, usecols=["cellosaurus_id", "pubchem_id", "AUC_curvecurator"], dtype={"pubchem_id": str})
    responses = responses.loc[responses.cellosaurus_id.eq(config["cellosaurus_id"])].copy()
    responses["AUC_curvecurator"] = pd.to_numeric(responses.AUC_curvecurator, errors="coerce")
    responses = responses.loc[np.isfinite(responses.AUC_curvecurator)].merge(mapping, on="pubchem_id", validate="many_to_one")
    endpoint = responses.groupby("chemical_identity", sort=True).AUC_curvecurator.agg(["median", "count"]).reset_index()
    endpoint.rename(columns={"median": "cellular_auc", "count": "endpoint_records"}, inplace=True)
    joined = joined.merge(endpoint, how="left", on="chemical_identity", validate="one_to_one")
    evaluated = joined.loc[joined.cellular_auc.notna()].copy()
    if len(evaluated) < config["minimum_measured_candidates"]:
        raise RuntimeError("Too few measured candidates after endpoint aggregation")
    y = evaluated.cellular_auc.to_numpy()
    point = {model: metrics(y, evaluated[model].to_numpy()) for model in scores}
    comparisons = [
        ("similarity_gate_plus_functional", "similarity_gate"),
        ("similarity_gate_plus_functional", "functional_only"),
        ("similarity_gate_plus_functional", "fixed_blend_plus_functional"),
        ("similarity_gate_plus_functional", "deep_plus_functional"),
        ("similarity_gate_plus_functional", "neighbor_prior_plus_functional"),
    ]
    rng = np.random.default_rng(config["bootstrap"]["seed"])
    samples = {(left, right, metric): [] for left, right in comparisons
               for metric in ("spearman", "auroc", "average_precision")}
    for iteration in range(config["bootstrap"]["repeats"]):
        idx = rng.integers(0, len(y), len(y))
        sampled = {m: metrics(y[idx], evaluated[m].to_numpy()[idx]) for m in scores}
        for key, values in samples.items():
            left, right, metric = key
            a, b = sampled[left][metric], sampled[right][metric]
            if a is not None and b is not None and np.isfinite(a) and np.isfinite(b):
                values.append(a - b)
        if (iteration + 1) % 500 == 0:
            print(f"Paired candidate bootstrap {iteration + 1}/{config['bootstrap']['repeats']}", flush=True)
    intervals = [{"left": left, "right": right, "metric": metric,
                  "difference": point[left][metric] - point[right][metric]
                  if point[left][metric] is not None and point[right][metric] is not None else None,
                  "percentile_95": np.quantile(values, [0.025, 0.975]).tolist() if values else None,
                  "valid_resamples": len(values)} for (left, right, metric), values in samples.items()]
    write_once(CASE / "candidate_scores_and_endpoints.csv", joined.to_csv(index=False))
    write_once(CASE / "ranking_metrics.csv", pd.DataFrame([{"model": m, **point[m]} for m in scores]).to_csv(index=False))
    top = evaluated.sort_values("similarity_gate_plus_functional", ascending=False, kind="stable").head(10)
    write_once(CASE / "top10_joint_candidates.csv", top.to_csv(index=False))
    result = {"plan": record(CASE / "plan.json"), "candidates_scored": len(joined),
              "candidates_with_measured_endpoint": len(evaluated), "unmeasured_not_treated_as_negative": int(joined.cellular_auc.isna().sum()),
              "active_auc_below_0_5": int((y < 0.5).sum()),
              "models": point, "paired_bootstrap": intervals, "bootstrap_config": config["bootstrap"],
              "bootstrap_scope": "Intervals for differences in Spearman, AUROC and average precision, conditional on saved predictions; no multiplicity correction. Top-K metrics are descriptive point estimates.",
              "interpretation": config["interpretation"], "limitations": config["boundaries"],
              "inputs": [record(CASE / n) for n in ("coverage_audit.json", "affinity_audit.json", "functional_audit.json",
                                                     "affinity_predictions.csv", "functional_scores.csv")],
              "endpoint_source": record(source)}
    write_once(CASE / "analysis.json", json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"measured_candidates": len(evaluated), "models": point}, indent=2), flush=True)


if __name__ == "__main__":
    main()
