#!/usr/bin/env python3
"""Independently reconstruct retrieval scores, ties, weight choice and intervals."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from bib_g2cp_common import DATA, OUT, PROJECT, check_records, now, record, save_json

DEST = PROJECT / "reports/bib_target_retrieval_20261019"
REFERENCE = PROJECT / "reports/bib_g2cp_reference_diagnostic_20261019"
JOB = PROJECT / "results/bib_target_retrieval_20261019"
METRICS = ["mrr", "hit_at_1", "hit_at_5", "hit_at_10", "recall_at_5"]
DTA = ["deep", "neighbor_prior", "fixed_blend", "similarity_gate"]


def percentiles(values):
    values = np.asarray(values)
    less = (values[:, None] > values[None, :]).sum(1)
    equal = (values[:, None] == values[None, :]).sum(1)
    return (less+.5*(equal-1))/(len(values)-1)


def expected_metrics(values, positive):
    values, positive = np.asarray(values), np.asarray(positive)
    best = values[positive].max()
    before = int((values>best).sum())
    block = values == best
    size, positives = int(block.sum()), int((block & positive).sum())
    survival, mrr = 1., 0.
    hits = {k: 0. for k in (1, 5, 10)}
    # Sequential sampling without replacement independently handles best-positive ties.
    for offset in range(1, size-positives+2):
        probability = survival*positives/(size-offset+1)
        rank = before+offset
        mrr += probability/rank
        for k in hits:
            if rank <= k:
                hits[k] += probability
        survival *= (size-offset+1-positives)/(size-offset+1)
    recall = 0.
    for score in values[positive]:
        before_score = int((values>score).sum())
        tie_count = int((values==score).sum())
        recall += min(max(5-before_score, 0), tie_count)/tie_count
    return np.array([mrr, hits[1], hits[5], hits[10], recall/positive.sum()])


def main():
    frozen = json.loads((DEST / "freeze.json").read_text())
    analysis = json.loads((DEST / "analysis.json").read_text())
    cfg = frozen["config"]
    for key in ("inputs", "reference_outputs", "original_data_files", "original_analysis_outputs"):
        check_records(frozen[key])
    check_records(analysis["outputs"])
    target_scores = pd.read_csv(DEST / "target_scores.csv", float_precision="round_trip")
    coverage = pd.read_csv(DEST / "coverage.csv", keep_default_na=False).set_index(["cell", "canonical_identity"])
    metrics = pd.read_csv(DEST / "per_drug_cell_metrics.csv", float_precision="round_trip")
    grid = pd.read_csv(DEST / "training_weight_grid.csv", float_precision="round_trip")
    null = pd.read_csv(DEST / "shuffle_metrics.csv", float_precision="round_trip")
    weights = json.loads((DEST / "frozen_weights.json").read_text())["weights"]
    candidates = pd.read_csv(OUT / "candidates.csv", keep_default_na=False).set_index("canonical_identity")
    train = candidates.loc[candidates.split.eq("train") & candidates.davis_external]
    candidate_lookup = {row.parent_key: set(row.mapped_kiba_targets.split("|")) for row in train.itertuples()}
    query = pd.read_csv(OUT / "query_table.csv", usecols=["row_id", "Canonical_SMILES", "protein_id"])
    affinity = query.merge(pd.read_csv(OUT / "affinity_predictions.csv", float_precision="round_trip"), on="row_id", validate="one_to_one")
    for frame in (target_scores, metrics, grid, null):
        if not set(frame.split) <= {"train", "val"}:
            raise RuntimeError("Unexpected test row")
    if set(grid.split) != {"train"} or set(null.split) != {"val"}:
        raise RuntimeError("Grid or shuffled control has incorrect partition")
    if metrics.duplicated(["cell", "canonical_identity", "method"]).any():
        raise RuntimeError("Duplicate retrieval results")
    score_error, metric_error, grid_error, null_error = 0., 0., 0., 0.
    score_count, metric_count, grid_count, null_count = 0, 0, 0, 0
    checked_metrics, checked_grid, checked_null = [], [], []
    for cell in cfg["cells"]:
        meta = pd.read_csv(REFERENCE / f"{cell}_development_rows.csv")
        with np.load(DATA / f"{cell}.npz", allow_pickle=False) as archive:
            proteins, library = archive["proteins"].astype(str), archive["library"].astype(float)
        with np.load(REFERENCE / f"{cell}_development_vectors.npz", allow_pickle=False) as archive:
            observed, center = archive["truth"], archive["training_mean"]
            predicted = np.mean([archive[f"g2cp_{seed}"] for seed in cfg["seeds"]], axis=0)
        for index, item in enumerate(meta.itertuples()):
            identity = item.canonical_identity
            details = {key: getattr(item, key) for key in ("cell", "canonical_identity", "parent_key", "split", "davis_external")}
            cov = coverage.loc[(cell, identity)]
            annotation = set(candidates.loc[identity, "mapped_kiba_targets"].split("|"))
            positive = np.array([protein in annotation for protein in proteins])
            if cov.available_positive_ids != "|".join(proteins[positive]) or cov.evaluable != bool(positive.any()):
                raise RuntimeError("Annotation coverage differs")
            if cov.positive_targets != int(positive.sum()) or cov.library_targets != len(proteins):
                raise RuntimeError("Target count mismatch")
            parents = [parent for parent in candidate_lookup if item.split != "train" or parent != item.parent_key]
            if cov.frequency_training_parents != len(parents):
                raise RuntimeError("Frequency baseline includes held-out or own annotation")
            original = affinity.loc[affinity.Canonical_SMILES.eq(identity)].set_index("protein_id").loc[proteins]
            raw = {method: original[method].to_numpy() for method in DTA}
            def correlate(matrix, vector):
                return np.array([pearsonr(row, vector).statistic if np.std(row)>0 and np.std(vector)>0 else 0. for row in matrix])
            raw["observed"] = correlate(library, observed[index])
            raw["observed_centered"] = correlate(library-library.mean(0), observed[index]-center[index])
            raw["g2cp_predicted"] = correlate(library, predicted[index])
            raw["training_target_frequency"] = np.array([sum(protein in candidate_lookup[parent] for parent in parents) for protein in proteins])
            saved = target_scores.loc[target_scores.cell.eq(cell) & target_scores.canonical_identity.eq(identity)].set_index("protein_id")
            if set(saved.index) != set(proteins) or saved.index.duplicated().any():
                raise RuntimeError("Candidate library mismatch")
            saved = saved.loc[proteins]
            if not np.array_equal(saved.annotation_status.eq("known_positive").to_numpy(), positive):
                raise RuntimeError("Annotation label mismatch")
            for key, values in raw.items():
                score_error = max(score_error, float(np.max(np.abs(saved[key].to_numpy()-values))))
            score_count += len(saved)
            # Values are verified above. Retain exact saved floats for numerical tie boundaries.
            ranked = {key: percentiles(saved[key].to_numpy()) for key in raw}
            if not positive.any():
                if ((metrics.cell.eq(cell)) & metrics.canonical_identity.eq(identity)).any():
                    raise RuntimeError("Unavailable annotation treated as negative")
                continue
            methods = dict(ranked)
            for method in DTA:
                for mode in cfg["response_modes"]:
                    methods[f"{method}_plus_{mode}_equal"] = .5*ranked[method]+.5*ranked[mode]
                weight = weights[method]["weight"]
                methods[f"{method}_plus_observed_train_selected"] = weight*ranked[method]+(1-weight)*ranked["observed"]
            actual = metrics.loc[metrics.cell.eq(cell) & metrics.canonical_identity.eq(identity)].set_index("method")
            if set(actual.index) != set(methods):
                raise RuntimeError("Missing matched comparator")
            for method, values in methods.items():
                expected = expected_metrics(values, positive)
                metric_error = max(metric_error, float(np.max(np.abs(expected-actual.loc[method, METRICS].to_numpy(dtype=float)))))
                checked_metrics.append({**details, "method": method, **dict(zip(METRICS, expected))})
                metric_count += 1
            if item.split == "train":
                current = grid.loc[grid.cell.eq(cell) & grid.canonical_identity.eq(identity)].set_index(["method", "weight"])
                for method in DTA:
                    for weight in cfg["weight_grid"]:
                        expected = expected_metrics((1-weight)*ranked["observed"]+weight*ranked[method], positive)
                        grid_error = max(grid_error, float(np.max(np.abs(expected-current.loc[(method,weight), METRICS].to_numpy(dtype=float)))))
                        checked_grid.append({**details, "method": method, "weight": weight, "mrr": float(expected[0])})
                        grid_count += 1
            else:
                current = null.loc[null.cell.eq(cell) & null.canonical_identity.eq(identity)].set_index(["mode", "repeat"])
                for repeat in range(cfg["shuffle_repeats"]):
                    key = f"{cfg['shuffle_seed']}:{repeat}:{cell}:{item.parent_key}"
                    random_seed = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
                    permutation = np.random.default_rng(random_seed).permutation(ranked["similarity_gate"])
                    for mode in ("observed", "g2cp_predicted"):
                        expected = expected_metrics(.5*permutation+.5*ranked[mode], positive)
                        null_error = max(null_error, float(np.max(np.abs(expected-current.loc[(mode,repeat), METRICS].to_numpy(dtype=float)))))
                        checked_null.append({**details, "mode": mode, "repeat": repeat, **dict(zip(METRICS, expected))})
                        null_count += 1
        print(f"Verified {cell}: source scores, rankings and shuffled controls", flush=True)
    if max(score_error, metric_error, grid_error, null_error) > 1e-10:
        raise RuntimeError(f"Retrieval numeric mismatch: {score_error}, {metric_error}, {grid_error}, {null_error}")
    if (score_count, metric_count, grid_count, null_count) != (len(target_scores),len(metrics),len(grid),len(null)):
        raise RuntimeError("Unverified or duplicate metric rows")
    checked_grid = pd.DataFrame(checked_grid)
    loo = pd.read_csv(DEST / "leave_one_training_drug_weights.csv")
    for method in DTA:
        frame = checked_grid.loc[checked_grid.method.eq(method) & checked_grid.davis_external]
        curve = frame.groupby(["weight", "canonical_identity"]).mrr.mean().groupby("weight").mean()
        selected = float(curve.loc[curve >= curve.max()-1e-12].index.min())
        if selected != weights[method]["weight"] or frame.canonical_identity.nunique() != weights[method]["training_drugs"]:
            raise RuntimeError("Weight was not selected from strict training drugs")
        for point in weights[method]["training_curve"]:
            if abs(curve.loc[point["weight"]]-point["mrr"])>1e-12:
                raise RuntimeError("Training weight curve differs")
        for row in loo.loc[loo.method.eq(method)].itertuples():
            held = frame.loc[frame.parent_key.ne(row.excluded_training_parent)]
            values = held.groupby(["weight", "canonical_identity"]).mrr.mean().groupby("weight").mean()
            if float(values.loc[values >= values.max()-1e-12].index.min()) != row.selected_weight:
                raise RuntimeError("Leave-one-training-drug stability differs")
    checked_metrics, checked_null = pd.DataFrame(checked_metrics), pd.DataFrame(checked_null)
    summary_error, interval_error, shuffle_summary_error = 0., 0., 0.
    summary = pd.read_csv(DEST / "summary.csv", float_precision="round_trip")
    for row in summary.itertuples():
        current = checked_metrics.loc[checked_metrics.split.eq(row.split) & checked_metrics.method.eq(row.method)]
        if row.scope == "primary_davis_disjoint":
            current = current.loc[current.davis_external]
        if row.cell != "all":
            current = current.loc[current.cell.eq(row.cell)]
        values = current.groupby("canonical_identity")[METRICS].mean()
        if len(values) != row.drug_identities:
            raise RuntimeError("Macro sample count mismatch")
        summary_error = max(summary_error, float(np.max(np.abs(values.mean().to_numpy()-np.array([getattr(row,key) for key in METRICS])))))
    for item in analysis["comparisons"]:
        current = checked_metrics.loc[checked_metrics.split.eq("val")]
        if item["scope"] == "primary_davis_disjoint":
            current = current.loc[current.davis_external]
        values = current.groupby(["canonical_identity", "method"])[item["metric"]].mean().unstack()
        difference = (values[item["left"]]-values[item["right"]]).to_numpy()
        rng = np.random.default_rng(cfg["bootstrap_seed"])
        means = rng.choice(difference, size=(cfg["bootstrap_repeats"],len(difference)), replace=True).mean(1)
        expected = np.quantile(means, [.025,.975])
        interval_error = max(interval_error, float(np.max(np.abs(expected-item["ci95"]))), abs(difference.mean()-item["difference"]))
        if len(difference) != item["drug_identities"] or int((difference>1e-12).sum()) != item["positive_drugs"]:
            raise RuntimeError("Paired comparison counts differ")
    for item in analysis["shuffle_summary"]:
        current = checked_null.loc[checked_null["mode"].eq(item["mode"])]
        if item["scope"] == "primary_davis_disjoint":
            current = current.loc[current.davis_external]
        values = current.groupby(["repeat","canonical_identity"])[METRICS].mean().groupby("repeat").mean()
        for metric in METRICS:
            expected = item["metrics"][metric]
            shuffle_summary_error = max(shuffle_summary_error, abs(values[metric].mean()-expected["mean"]),
                float(np.max(np.abs(np.quantile(values[metric],[.025,.975])-expected["interval95_across_shuffles"]))))
    if max(summary_error, interval_error, shuffle_summary_error)>1e-10:
        raise RuntimeError("Summary or interval mismatch")
    result = {"state":"passed", "target_score_rows": score_count, "metric_rows": metric_count,
              "weight_grid_rows": grid_count, "shuffle_rows": null_count, "test_rows_scored":0,
              "max_score_error":score_error, "max_metric_error":metric_error, "max_grid_error":grid_error,
              "max_shuffle_error":null_error, "max_summary_error":summary_error, "max_interval_error":interval_error,
              "max_shuffle_summary_error":shuffle_summary_error, "training_only_weight_selection_verified":True,
              "exact_ties_independently_verified":True, "source": record(DEST / "analysis.json"), "verifier":record(Path(__file__))}
    save_json(DEST / "verification.json", result, immutable=True)
    save_json(JOB / "status.json", {"state":"completed", "finished_utc":now(), "verification":"passed",
              "metric_rows":metric_count, "test_rows_scored":0, "gpu_used":False})
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
