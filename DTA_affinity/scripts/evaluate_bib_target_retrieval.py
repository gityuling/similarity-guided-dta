#!/usr/bin/env python3
"""Development-only target retrieval from DTA and measured/predicted responses."""

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from bib_g2cp_common import DATA, OUT, PROJECT, check_records, now, record, save_json, write_once
from evaluate_bib_g2cp_dta import correlation

CONFIG = PROJECT / "configs/bib_target_retrieval_20261019.json"
REFERENCE = PROJECT / "reports/bib_g2cp_reference_diagnostic_20261019"
DEST = PROJECT / "reports/bib_target_retrieval_20261019"
JOB = PROJECT / "results/bib_target_retrieval_20261019"
DTA = ("deep", "neighbor_prior", "fixed_blend", "similarity_gate")
METRICS = ("mrr", "hit_at_1", "hit_at_5", "hit_at_10", "recall_at_5")


def percentile(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("At least two finite candidate scores are required")
    return (rankdata(values, method="average")-1)/(len(values)-1)


def retrieval_metrics(values, positive):
    values, positive = np.asarray(values, dtype=float), np.asarray(positive, dtype=bool)
    if not positive.any():
        raise ValueError("Unknown annotations must not be scored as negatives")
    if not np.isfinite(values).all():
        raise ValueError("Nonfinite retrieval scores")
    best = values[positive].max()
    preceding = int((values > best).sum())
    tied = values == best
    n, p = int(tied.sum()), int((tied & positive).sum())
    denominator = math.comb(n, p)
    positions = np.arange(1, n-p+2)
    probabilities = np.array([math.comb(n-int(j), p-1)/denominator for j in positions])
    result = {"mrr": float(np.sum(probabilities/(preceding+positions)))}
    for k in (1, 5, 10):
        result[f"hit_at_{k}"] = float(probabilities[preceding+positions <= k].sum())
    recall = []
    for value in values[positive]:
        ahead = int((values > value).sum())
        group = int((values == value).sum())
        recall.append(float(np.clip((5-ahead)/group, 0, 1)))
    result["recall_at_5"] = float(np.mean(recall))
    return result


def macro_metric(rows, column="mrr"):
    return float(rows.groupby("canonical_identity")[column].mean().mean())


def select_weight(grid, excluded_parent=None):
    train = grid.loc[grid.split.eq("train") & grid.davis_external].copy()
    if excluded_parent is not None:
        train = train.loc[~train.parent_key.eq(excluded_parent)]
    if train.empty:
        raise RuntimeError("No annotated strict training drugs for weight fitting")
    means = train.groupby(["weight", "canonical_identity"]).mrr.mean().groupby("weight").mean()
    best = means.max()
    chosen = min(float(weight) for weight, value in means.items() if value >= best-1e-12)
    return chosen, [{"weight": float(weight), "mrr": float(value)} for weight, value in means.items()]


def summarize(metrics, cfg):
    summaries, comparisons = [], []
    comparison_pairs = [(cfg["primary_method"], key) for key in cfg["primary_comparators"]]
    comparison_pairs += [(f"similarity_gate_plus_{mode}_equal", comparator)
                         for mode in ("observed_centered", "g2cp_predicted")
                         for comparator in (mode, "similarity_gate", f"deep_plus_{mode}_equal")]
    comparison_pairs += [("similarity_gate_plus_observed_train_selected", key)
                         for key in ("observed", "similarity_gate", "deep_plus_observed_train_selected")]
    for scope in ("primary_davis_disjoint", "all_kiba_external"):
        current = metrics.loc[metrics.davis_external] if scope.startswith("primary") else metrics
        for split in ("train", "val"):
            frame = current.loc[current.split.eq(split)]
            for cell in ["all", *cfg["cells"]]:
                selected = frame if cell == "all" else frame.loc[frame.cell.eq(cell)]
                drug = selected.groupby(["method", "canonical_identity"])[list(METRICS)].mean().reset_index()
                for method, group in drug.groupby("method"):
                    summaries.append({"scope": scope, "split": split, "cell": cell, "method": method,
                                      "drug_identities": len(group), **group[list(METRICS)].mean().to_dict()})
            if split != "val":
                continue
            drug = frame.groupby(["canonical_identity", "method"])[list(METRICS)].mean()
            for metric in METRICS:
                matrix = drug[metric].unstack()
                if matrix.isna().any().any():
                    raise RuntimeError("Retrieval methods have unequal evaluation coverage")
                rng = np.random.default_rng(cfg["bootstrap_seed"])
                selection = rng.integers(0, len(matrix), size=(cfg["bootstrap_repeats"], len(matrix)))
                for left, right in comparison_pairs:
                    delta = (matrix[left]-matrix[right]).to_numpy()
                    comparisons.append({"scope": scope, "metric": metric, "left": left, "right": right,
                                        "difference": float(delta.mean()), "ci95": np.quantile(delta[selection].mean(1), [.025, .975]).tolist(),
                                        "drug_identities": len(matrix), "positive_drugs": int((delta>1e-12).sum()),
                                        "negative_drugs": int((delta < -1e-12).sum()),
                                        "primary": metric == cfg["primary_metric"] and left == cfg["primary_method"] and right in cfg["primary_comparators"]})
    return pd.DataFrame(summaries), comparisons


def main():
    cfg = json.loads(CONFIG.read_text())
    if cfg["splits"] != ["train", "val"]:
        raise RuntimeError("Test scoring is outside this exploratory protocol")
    JOB.mkdir(parents=True, exist_ok=True)
    save_json(JOB / "status.json", {"state": "running", "stage": "input_integrity", "started_utc": now()})
    source = json.loads((REFERENCE / "analysis.json").read_text())
    check_records(source["outputs"])
    frozen = json.loads((REFERENCE / "freeze.json").read_text())
    for key in ("inputs", "original_data_files", "original_analysis_outputs"):
        check_records(frozen[key])
    inputs = [record(path) for path in (CONFIG, Path(__file__), PROJECT / "scripts/evaluate_bib_g2cp_dta.py",
                                        PROJECT / "scripts/bib_g2cp_common.py", REFERENCE / "analysis.json",
                                        REFERENCE / "verification.json", REFERENCE / "rank_verification.json")]
    save_json(DEST / "freeze.json", {"config": cfg, "inputs": inputs, "reference_outputs": source["outputs"],
              "original_data_files": frozen["original_data_files"],
              "original_analysis_outputs": frozen["original_analysis_outputs"]}, immutable=True)
    candidates = pd.read_csv(OUT / "candidates.csv", keep_default_na=False).set_index("canonical_identity")
    train_candidates = candidates.loc[candidates.split.eq("train") & candidates.davis_external]
    annotation_by_parent = {row.parent_key: set(row.mapped_kiba_targets.split("|")) for row in train_candidates.itertuples()}
    query = pd.read_csv(OUT / "query_table.csv", usecols=["row_id", "Canonical_SMILES", "protein_id"])
    development_ids = set(candidates.loc[candidates.split.isin(cfg["splits"])].index)
    query = query.loc[query.Canonical_SMILES.isin(development_ids)]
    affinity = query.merge(pd.read_csv(OUT / "affinity_predictions.csv"), on="row_id", validate="one_to_one")
    cases, score_rows, coverage_rows, grid_rows = [], [], [], []
    for cell in cfg["cells"]:
        meta = pd.read_csv(REFERENCE / f"{cell}_development_rows.csv")
        if not meta.split.isin(cfg["splits"]).all():
            raise RuntimeError("Unexpected test row in development archive")
        with np.load(DATA / f"{cell}.npz", allow_pickle=False) as archive:
            proteins, library = archive["proteins"].astype(str), archive["library"].astype(float)
        with np.load(REFERENCE / f"{cell}_development_vectors.npz", allow_pickle=False) as archive:
            truth, training_mean = archive["truth"], archive["training_mean"]
            predicted = np.mean([archive[f"g2cp_{seed}"] for seed in cfg["seeds"]], axis=0)
        for index, item in enumerate(meta.itertuples()):
            details = {key: getattr(item, key) for key in ("cell", "canonical_identity", "parent_key", "names", "split", "davis_external")}
            annotated = set(candidates.loc[item.canonical_identity, "mapped_kiba_targets"].split("|"))
            positive = np.array([protein in annotated for protein in proteins])
            details.update(library_targets=len(proteins), positive_targets=int(positive.sum()))
            scores = affinity.loc[affinity.Canonical_SMILES.eq(item.canonical_identity)].set_index("protein_id").loc[proteins]
            raw = {method: scores[method].to_numpy() for method in DTA}
            raw["observed"] = correlation(library, np.broadcast_to(truth[index], library.shape))
            raw["observed_centered"] = correlation(library-library.mean(0), np.broadcast_to(truth[index]-training_mean[index], library.shape))
            raw["g2cp_predicted"] = correlation(library, np.broadcast_to(predicted[index], library.shape))
            parents = [parent for parent in annotation_by_parent if item.split != "train" or parent != item.parent_key]
            raw["training_target_frequency"] = np.array([sum(protein in annotation_by_parent[parent] for parent in parents) for protein in proteins], dtype=float)
            ranked = {key: percentile(value) for key, value in raw.items()}
            case = {"details": details, "proteins": proteins, "positive": positive, "raw": raw, "ranked": ranked}
            cases.append(case)
            coverage_rows.append({**details, "available_positive_ids": "|".join(proteins[positive]),
                                  "unavailable_positive_ids": "|".join(sorted(annotated-set(proteins))),
                                  "evaluable": bool(positive.any()), "frequency_training_parents": len(parents)})
            for protein_index, protein in enumerate(proteins):
                score_rows.append({**details, "protein_id": protein,
                                   "annotation_status": "known_positive" if positive[protein_index] else "unknown",
                                   **{key: float(value[protein_index]) for key, value in raw.items()}})
            if not positive.any() or item.split != "train":
                continue
            for method in DTA:
                for weight in cfg["weight_grid"]:
                    values = (1-weight)*ranked["observed"]+weight*ranked[method]
                    grid_rows.append({**details, "method": method, "weight": weight, **retrieval_metrics(values, positive)})
    grid = pd.DataFrame(grid_rows)
    weights, loo_weights = {}, []
    for method in DTA:
        current = grid.loc[grid.method.eq(method)]
        weight, curve = select_weight(current)
        weights[method] = {"weight": weight, "training_curve": curve,
                           "training_drugs": int(current.loc[current.davis_external].canonical_identity.nunique())}
        for parent in sorted(current.loc[current.davis_external, "parent_key"].unique()):
            selected, _ = select_weight(current, excluded_parent=parent)
            loo_weights.append({"method": method, "excluded_training_parent": parent, "selected_weight": selected})
    save_json(DEST / "frozen_weights.json", {"weights": weights, "validation_scores_computed": False,
              "config": record(CONFIG), "selection": cfg["weight_selection"]}, immutable=True)
    metric_rows, null_rows = [], []
    for case in cases:
        if not case["positive"].any():
            continue
        details, ranked, positive = case["details"], case["ranked"], case["positive"]
        methods = dict(ranked)
        for method in DTA:
            for mode in cfg["response_modes"]:
                methods[f"{method}_plus_{mode}_equal"] = .5*ranked[method]+.5*ranked[mode]
            weight = weights[method]["weight"]
            methods[f"{method}_plus_observed_train_selected"] = weight*ranked[method]+(1-weight)*ranked["observed"]
        for method, values in methods.items():
            metric_rows.append({**details, "method": method, **retrieval_metrics(values, positive)})
        if details["split"] != "val":
            continue
        for repeat in range(cfg["shuffle_repeats"]):
            key = f"{cfg['shuffle_seed']}:{repeat}:{details['cell']}:{details['parent_key']}"
            seed = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
            shuffled = np.random.default_rng(seed).permutation(ranked["similarity_gate"])
            for mode in ("observed", "g2cp_predicted"):
                null_rows.append({**details, "repeat": repeat, "mode": mode,
                                  **retrieval_metrics(.5*shuffled+.5*ranked[mode], positive)})
    metrics, null = pd.DataFrame(metric_rows), pd.DataFrame(null_rows)
    summary, comparisons = summarize(metrics, cfg)
    null_summary = []
    for scope in ("primary_davis_disjoint", "all_kiba_external"):
        current = null.loc[null.davis_external] if scope.startswith("primary") else null
        per_drug = current.groupby(["repeat", "mode", "canonical_identity"])[list(METRICS)].mean()
        per_repeat = per_drug.groupby(["repeat", "mode"]).mean().reset_index()
        for mode, values in per_repeat.groupby("mode"):
            null_summary.append({"scope": scope, "mode": mode, "repeats": cfg["shuffle_repeats"],
                                 "drug_identities": current.canonical_identity.nunique(),
                                 "metrics": {key: {"mean": float(values[key].mean()),
                                                   "interval95_across_shuffles": np.quantile(values[key], [.025,.975]).tolist()} for key in METRICS}})
    files = {"coverage.csv": pd.DataFrame(coverage_rows), "target_scores.csv": pd.DataFrame(score_rows),
             "training_weight_grid.csv": grid, "leave_one_training_drug_weights.csv": pd.DataFrame(loo_weights),
             "per_drug_cell_metrics.csv": metrics, "summary.csv": summary, "shuffle_metrics.csv": null}
    for name, frame in files.items():
        write_once(DEST / name, frame.to_csv(index=False))
    check_records(inputs)
    save_json(DEST / "analysis.json", {"config": cfg, "comparisons": comparisons, "shuffle_summary": null_summary,
              "test_rows_scored": 0, "weights": weights,
              "outputs": [record(DEST / name) for name in [*files, "frozen_weights.json"]]}, immutable=True)
    save_json(JOB / "status.json", {"state": "computed", "finished_utc": now(), "test_rows_scored": 0,
              "metric_rows": len(metrics), "shuffle_rows": len(null), "gpu_used": False, "verification": "pending"})
    print(summary.loc[summary.scope.eq("primary_davis_disjoint") & summary.split.eq("val") & summary.cell.eq("all")].to_string(index=False), flush=True)
    print(json.dumps(weights, indent=2), flush=True)


if __name__ == "__main__":
    main()
