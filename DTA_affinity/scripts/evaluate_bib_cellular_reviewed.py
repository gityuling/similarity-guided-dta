#!/usr/bin/env python3
"""Evaluate frozen reviewed scores, guide sensitivities and shuffled-affinity controls."""

import json

import numpy as np
import pandas as pd

from bib_common import PROJECT, ROOT, record, sha256, write_once
from evaluate_bib_cellular_case import metrics
from predict_bib_cellular_affinity import CASE as ORIGINAL, MODELS


OUT = PROJECT / "reports/bib_cellular_case_reviewed_20261018"
METRICS = ("spearman", "auroc", "average_precision")


def verify(items):
    for item in items:
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"Frozen source changed: {item['path']}")


def interval(values):
    finite = np.asarray([v for v in values if v is not None and np.isfinite(v)])
    return {"percentile_95": np.quantile(finite, [0.025, 0.975]).tolist() if len(finite) else None,
            "valid_resamples": len(finite)}


def main():
    config = json.loads((OUT / "plan.json").read_text())
    freeze = json.loads((OUT / "freeze.json").read_text())
    verify([*freeze["files"], freeze["code"]])
    coverage_audit = json.loads((ORIGINAL / "coverage_audit.json").read_text())
    verify(coverage_audit["sources"] + coverage_audit["outputs"])
    for directory, filename in ((OUT, "functional_audit.json"), (ORIGINAL, "affinity_audit.json")):
        report = json.loads((directory / filename).read_text())
        verify(report["outputs"])
    affinity = pd.read_csv(ORIGINAL / "affinity_predictions.csv")
    functional = pd.read_csv(OUT / "functional_scores.csv")
    joined = affinity.merge(functional, on="chemical_identity", how="outer", validate="one_to_one", indicator=True)
    if not joined._merge.eq("both").all() or len(joined) != 284:
        raise RuntimeError("Candidate libraries differ")
    joined = joined.drop(columns="_merge").sort_values("chemical_identity").reset_index(drop=True)
    joined["functional_only"] = joined.functional_concordance
    functional_rank = joined.functional_only.rank(method="average", pct=True).to_numpy()
    gate_rank = joined.similarity_gate.rank(method="average", pct=True).to_numpy()
    scores = [*MODELS, "functional_only", *(f"{m}_plus_functional" for m in MODELS)]
    for model in MODELS:
        joined[f"{model}_plus_functional"] = (joined[model].rank(method="average", pct=True) + functional_rank) / 2
    sensitivity = []
    for column in config["sensitivity_scores"]:
        name = f"gate_plus_{column}"
        joined[name] = (gate_rank + joined[column].rank(method="average", pct=True)) / 2
        sensitivity.extend([column, name])
    if not np.isfinite(joined[scores + sensitivity].to_numpy()).all():
        raise RuntimeError("Non-finite ranking scores")
    # The scores and all analysis variants are saved before the endpoint values enter.
    write_once(OUT / "candidate_scores_before_endpoints.csv", joined.to_csv(index=False))
    coverage = pd.read_csv(ORIGINAL / "candidate_coverage.csv").fillna("")
    mapping = coverage.loc[coverage.in_candidate_library, ["chemical_identity", "ctrp_pubchem_ids"]].copy()
    mapping["pubchem_id"] = mapping.ctrp_pubchem_ids.str.split("|")
    mapping = mapping.explode("pubchem_id").drop(columns="ctrp_pubchem_ids")
    source = ROOT / "UniPert_DrEval/data/raw/extracted/CTRPv2/CTRPv2.csv"
    response = pd.read_csv(source, usecols=["cellosaurus_id", "pubchem_id", "AUC_curvecurator"], dtype={"pubchem_id": str})
    response = response.loc[response.cellosaurus_id.eq(config["cellosaurus_id"])].copy()
    response["AUC_curvecurator"] = pd.to_numeric(response.AUC_curvecurator, errors="coerce")
    response = response.loc[np.isfinite(response.AUC_curvecurator)].merge(mapping, on="pubchem_id", validate="many_to_one")
    endpoint = response.groupby("chemical_identity").AUC_curvecurator.agg(["median", "count"]).reset_index()
    endpoint.rename(columns={"median": "cellular_auc", "count": "endpoint_records"}, inplace=True)
    joined = joined.merge(endpoint, on="chemical_identity", how="left", validate="one_to_one")
    measured_mask = joined.cellular_auc.notna().to_numpy()
    evaluated = joined.loc[measured_mask]
    if len(evaluated) != 256:
        raise RuntimeError("Endpoint coverage differs from the frozen audit")
    y = evaluated.cellular_auc.to_numpy()
    names = scores + sensitivity
    vectors = {m: evaluated[m].to_numpy() for m in names}
    point = {m: metrics(y, v) for m, v in vectors.items()}
    main_joint = "similarity_gate_plus_functional"
    comparisons = [(main_joint, m) for m in ("functional_only", "deep_plus_functional", "similarity_gate",
                                            "fixed_blend_plus_functional", "neighbor_prior_plus_functional")]
    comparisons += [(f"gate_plus_{column}", column) for column in config["sensitivity_scores"]]
    samples = {(left, right, metric): [] for left, right in comparisons for metric in METRICS}
    rng = np.random.default_rng(config["bootstrap"]["seed"])
    for iteration in range(config["bootstrap"]["repeats"]):
        idx = rng.integers(0, len(y), len(y))
        sampled = {m: metrics(y[idx], v[idx]) for m, v in vectors.items()}
        for (left, right, metric), values in samples.items():
            a, b = sampled[left][metric], sampled[right][metric]
            values.append(a - b if a is not None and b is not None else None)
        if (iteration + 1) % 500 == 0:
            print(f"Paired bootstrap {iteration + 1}/{config['bootstrap']['repeats']}", flush=True)
    intervals = []
    for (left, right, metric), values in samples.items():
        a, b = point[left][metric], point[right][metric]
        intervals.append({"left": left, "right": right, "metric": metric,
                          "difference": a - b if a is not None and b is not None else None, **interval(values)})
    negative = config["negative_control"]
    rng = np.random.default_rng(negative["seed"])
    null = {m: [] for m in METRICS}
    for iteration in range(negative["repeats"]):
        shuffled = (rng.permutation(gate_rank) + functional_rank) / 2
        result = metrics(y, shuffled[measured_mask])
        for metric in METRICS:
            if result[metric] is not None:
                null[metric].append(result[metric])
        if (iteration + 1) % 500 == 0:
            print(f"Shuffled-affinity control {iteration + 1}/{negative['repeats']}", flush=True)
    null_summary = {}
    for metric, values in null.items():
        observed = point[main_joint][metric]
        null_summary[metric] = {"observed_joint": observed, "functional_only": point["functional_only"][metric],
                                "null_mean": float(np.mean(values)) if values else None,
                                "upper_tail_probability": float((1 + np.sum(np.asarray(values) >= observed)) / (len(values) + 1))
                                if values and observed is not None else None, **interval(values)}
    write_once(OUT / "candidate_scores_and_endpoints.csv", joined.to_csv(index=False))
    write_once(OUT / "ranking_metrics.csv", pd.DataFrame([{"model": m, **point[m]} for m in scores]).to_csv(index=False))
    write_once(OUT / "sensitivity_metrics.csv", pd.DataFrame([{"model": m, **point[m]} for m in sensitivity]).to_csv(index=False))
    top = evaluated.sort_values(main_joint, ascending=False, kind="stable").head(10)
    write_once(OUT / "top10_joint_candidates.csv", top.to_csv(index=False))
    result = {"plan": record(OUT / "plan.json"), "candidates_scored": len(joined), "measured_candidates": len(y),
              "unmeasured_not_treated_as_negative": int((~measured_mask).sum()), "active_auc_below_0_5": int((y < 0.5).sum()),
              "endpoint_range": [float(y.min()), float(y.max())], "models": {m: point[m] for m in scores},
              "sensitivity_models": {m: point[m] for m in sensitivity}, "paired_bootstrap": intervals,
              "shuffled_affinity": null_summary, "bootstrap_config": config["bootstrap"], "negative_control_config": negative,
              "statistical_scope": "Identity paired intervals conditional on fixed predictions. No multiplicity correction; secondary metrics and null probabilities are exploratory. Top-K estimates are descriptive. No model or weight fitted on CTRP.",
              "interpretation": config["interpretation"], "limitations": config["boundaries"],
              "inputs": [record(p) for p in (OUT / "freeze.json", OUT / "functional_audit.json", ORIGINAL / "affinity_audit.json",
                                             ORIGINAL / "affinity_predictions.csv", OUT / "functional_scores.csv", source)],
              "outputs": [record(OUT / n) for n in ("candidate_scores_before_endpoints.csv", "candidate_scores_and_endpoints.csv",
                                                     "ranking_metrics.csv", "sensitivity_metrics.csv", "top10_joint_candidates.csv")]}
    write_once(OUT / "analysis.json", json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"measured_candidates": len(y), "active": result["active_auc_below_0_5"], "models": result["models"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
