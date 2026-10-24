#!/usr/bin/env python3
"""Evaluate saved predictions using the frozen cellular panel rules."""

import json

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

from bib_common import PROJECT, record, write_once


OUT = PROJECT / "reports/bib_nanobret_20261018"


def save_json(name, payload):
    write_once(OUT / name, json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


MODELS = ("deep", "neighbor_prior", "fixed_blend", "similarity_gate")


def metrics(group, model, minimum):
    values = group[model].to_numpy()
    endpoint = group.endpoint_score.to_numpy()
    active = group.active.to_numpy(dtype=bool)
    order = group.sort_values([model, "protein_id"], ascending=[False, True], kind="stable")
    top = order.head(10)
    valid_rho = len(group) >= minimum and np.unique(endpoint).size > 1 and np.unique(values).size > 1
    binary = len(group) >= minimum and 0 < active.sum() < len(active)
    return {
        "pairs": len(group), "active_pairs": int(active.sum()),
        "spearman": float(spearmanr(endpoint, values).statistic) if valid_rho else None,
        "auroc": float(roc_auc_score(active, values)) if binary else None,
        "ap": float(average_precision_score(active, values)) if binary else None,
        "top10_hits": int(top.active.sum()), "top10_n": len(top),
        "ef10": float(top.active.mean() / active.mean()) if active.any() else None,
    }


def analyze(scores, plan):
    rows, summaries, intervals = [], [], []
    variants = {"primary": scores.copy()}
    clipped = scores.loc[scores.endpoint.eq("occupancy_pct")].copy()
    clipped["endpoint_score"] = clipped.endpoint_score.clip(0, 100)
    variants["occupancy_clipped"] = clipped
    variants["ic50_interior_only"] = scores.loc[
        scores.endpoint.eq("ic50_nm") & ~scores.reported_censor & ~scores.possible_assay_boundary].copy()
    for variant, frame in variants.items():
        for (compound, endpoint), group in frame.groupby(["compound", "endpoint"], sort=True):
            for model in MODELS:
                rows.append({"variant": variant, "compound": compound, "endpoint": endpoint, "model": model,
                             "davis_external": bool(group.davis_external.iloc[0]),
                             "kiba_external": bool(group.kiba_external.iloc[0]),
                             **metrics(group, model, plan["minimum_pairs_per_compound"])})
    per_compound = pd.DataFrame(rows)
    strata = {"all": per_compound, "davis_disjoint": per_compound.loc[per_compound.davis_external]}
    if not per_compound.kiba_external.all():
        strata.update(kiba_external=per_compound.loc[per_compound.kiba_external],
                      kiba_heldout_known=per_compound.loc[~per_compound.kiba_external])
    for stratum, frame in strata.items():
        for (variant, endpoint), group in frame.groupby(["variant", "endpoint"], sort=True):
            for metric in ("spearman", "auroc", "ap", "ef10"):
                wide = group.pivot(index="compound", columns="model", values=metric).dropna()
                if wide.empty:
                    continue
                for model in MODELS:
                    summaries.append({"stratum": stratum, "variant": variant, "endpoint": endpoint,
                                      "metric": metric, "model": model, "compounds": len(wide),
                                      "macro_mean": float(wide[model].mean())})
                rng = np.random.default_rng(plan["bootstrap_seed"])
                draws = rng.integers(0, len(wide), size=(plan["bootstrap_n"], len(wide)))
                for comparator in ("deep", "neighbor_prior", "fixed_blend"):
                    delta = (wide.similarity_gate - wide[comparator]).to_numpy()
                    resampled = delta[draws].mean(axis=1)
                    intervals.append({"stratum": stratum, "variant": variant, "endpoint": endpoint,
                                      "metric": metric, "comparison": f"similarity_gate - {comparator}",
                                      "compounds": len(wide), "delta": float(delta.mean()),
                                      "ci95": np.quantile(resampled, [0.025, 0.975]).tolist(),
                                      "drug_wins": int((delta > 0).sum()), "drug_ties": int((delta == 0).sum()),
                                      "conditional_compound_bootstrap": True})
    return per_compound, summaries, intervals


def main():
    plan = json.loads((OUT / "plan.json").read_text())
    query = pd.read_csv(OUT / "query_table.csv", float_precision="round_trip")
    predictions = pd.read_csv(OUT / "predictions.csv", float_precision="round_trip")
    measurements = pd.read_csv(OUT / "measurements.csv", float_precision="round_trip")
    if set(predictions.row_id) != set(query.row_id) or predictions.row_id.duplicated().any():
        raise RuntimeError("Prediction/query coverage differs")
    scores = query.drop(columns="Sequence").merge(predictions, on="row_id", validate="one_to_one")
    columns = ["measurement_id", "target_name", "value", "raw_value", "qualifier", "endpoint_score", "active",
               "reported_censor", "possible_assay_boundary", "occupancy_outside_0_100", "source_sheet", "source_row"]
    scores = scores.merge(measurements[columns], on="measurement_id", validate="one_to_one")
    if scores[list(MODELS)].isna().any().any():
        raise RuntimeError("Missing score")
    per_compound, summary, intervals = analyze(scores, plan)
    write_once(OUT / "scores.csv", scores.to_csv(index=False))
    write_once(OUT / "per_compound_metrics.csv", per_compound.to_csv(index=False))
    write_once(OUT / "macro_metrics.csv", pd.DataFrame(summary).to_csv(index=False))
    top = []
    for (compound, endpoint), group in scores.groupby(["compound", "endpoint"]):
        for model in MODELS:
            chosen = group.sort_values([model, "protein_id"], ascending=[False, True], kind="stable").head(10).copy()
            chosen["model"] = model
            chosen["rank"] = np.arange(1, len(chosen) + 1)
            top.append(chosen)
    write_once(OUT / "top10.csv", pd.concat(top, ignore_index=True).to_csv(index=False))
    report = {"model_training_performed": False, "new_endpoint_weight_fitting": False,
              "pairs": len(scores), "compounds": scores.compound.nunique(),
              "macro_metrics": summary, "paired_differences": intervals,
              "uncertainty": plan["uncertainty"], "limitations": plan["limitations"],
              "files": [record(OUT / n) for n in ("plan.json", "freeze.json", "predictions.csv", "scores.csv", "per_compound_metrics.csv", "macro_metrics.csv", "top10.csv")]}
    save_json("analysis.json", report)
    print(pd.DataFrame(summary).query("stratum == 'all' and variant == 'primary' and metric == 'spearman'").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
