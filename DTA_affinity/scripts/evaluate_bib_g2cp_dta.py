#!/usr/bin/env python3
"""Fit auxiliary weights on validation drugs, then score all fixed comparisons."""

import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from bib_g2cp_common import DATA, OUT, RUN, check_records, plan, record, save_json, verify_data, write_once

DTA_MODELS = ("deep", "neighbor_prior", "fixed_blend", "similarity_gate")
AUXILIARIES = (*DTA_MODELS, "generic", "shuffled_gate")


def correlation(prediction, truth):
    p = np.asarray(prediction, dtype=np.float64)
    t = np.asarray(truth, dtype=np.float64)
    p = p - p.mean(axis=1, keepdims=True)
    t = t - t.mean(axis=1, keepdims=True)
    denominator = np.linalg.norm(p, axis=1) * np.linalg.norm(t, axis=1)
    return np.divide(np.sum(p * t, axis=1), denominator, out=np.zeros(len(p)), where=denominator > 0)


def scores(prediction, truth):
    return {"pearson": correlation(prediction, truth),
            "spearman": correlation(rankdata(prediction, axis=1), rankdata(truth, axis=1)),
            "mse": np.mean(np.square(prediction.astype(np.float64) - truth), axis=1)}


def build_auxiliary(cell, metadata, arrays, predictions, cfg):
    protein_ids, library = arrays["proteins"].astype(str), arrays["library"]
    weights = 1 / np.arange(1, cfg["top_k"] + 1, dtype=np.float64)
    weights /= weights.sum()
    result = {m: [] for m in AUXILIARIES}
    details = []
    for row in metadata.itertuples():
        frame = predictions.loc[predictions.Canonical_SMILES.eq(row.canonical_identity)].set_index("protein_id")
        matched = frame.loc[protein_ids]
        for method in AUXILIARIES:
            if method == "generic":
                result[method].append(library.mean(0))
                continue
            raw = matched["similarity_gate" if method == "shuffled_gate" else method].to_numpy()
            if method == "shuffled_gate":
                key = f"{cfg['shuffle_seed']}:{cell}:{row.canonical_identity}"
                seed = int(hashlib.sha256(key.encode()).hexdigest()[:16], 16)
                raw = np.random.default_rng(seed).permutation(raw)
            order = np.lexsort((protein_ids, -raw))[:cfg["top_k"]]
            result[method].append(weights @ library[order])
            for rank, index in enumerate(order):
                details.append({"cell": cell, "canonical_identity": row.canonical_identity, "method": method,
                                "rank": rank + 1, "protein_id": protein_ids[index], "weight": float(weights[rank]),
                                "dta_score": float(raw[index])})
    return {key: np.asarray(value, dtype=np.float64) for key, value in result.items()}, details


def macro(values, metadata):
    frame = metadata[["canonical_identity"]].copy()
    frame["value"] = values
    return float(frame.groupby("canonical_identity").value.mean().mean())


def choose_alpha(base, auxiliary, truth, metadata, cfg):
    if not len(metadata):
        raise RuntimeError("No primary validation compounds for weight fitting")
    best_alpha, best_score = 0.0, -float("inf")
    grid = []
    for alpha in cfg["alpha_grid"]:
        value = macro(correlation((1-alpha)*base + alpha*auxiliary, truth), metadata)
        grid.append({"alpha": alpha, "pearson": value})
        if value > best_score + 1e-12:
            best_alpha, best_score = alpha, value
    return best_alpha, grid


def main():
    cfg = plan()
    verify_data()
    query = pd.read_csv(OUT / "query_table.csv")
    pred = pd.read_csv(OUT / "affinity_predictions.csv", float_precision="round_trip")
    prediction = query.merge(pred, on="row_id", validate="one_to_one")
    cells, detail_rows = {}, []
    inputs = [record(OUT / "data_audit.json"), record(OUT / "affinity_predictions.csv")]
    for cell in cfg["cells"]:
        arrays = np.load(DATA / f"{cell}.npz", allow_pickle=False)
        metadata = pd.read_csv(OUT / f"{cell}_evaluation.csv")
        auxiliary, details = build_auxiliary(cell, metadata, arrays, prediction, cfg)
        detail_rows.extend(details)
        cells[cell] = {"metadata": metadata, "truth": arrays["truth"], "auxiliary": auxiliary, "base": {}}
        for seed in cfg["training_seeds"]:
            directory = RUN / "models" / f"{cell}_seed{seed}"
            completed = json.loads((directory / "completed.json").read_text())
            check_records(completed["files"])
            cells[cell]["base"][seed] = np.load(directory / "evaluation_predictions.npy", allow_pickle=False)
            inputs.append(record(directory / "completed.json"))
    write_once(OUT / "selected_targets.csv", pd.DataFrame(detail_rows).to_csv(index=False))
    weights = {}
    for seed in cfg["training_seeds"]:
        base, truth, metadata = [], [], []
        auxiliary = {m: [] for m in AUXILIARIES}
        for value in cells.values():
            mask = (value["metadata"].split.eq("val") & value["metadata"].davis_external).to_numpy()
            base.append(value["base"][seed][mask])
            truth.append(value["truth"][mask])
            metadata.append(value["metadata"].loc[mask])
            for method in AUXILIARIES:
                auxiliary[method].append(value["auxiliary"][method][mask])
        b, t, meta = np.concatenate(base), np.concatenate(truth), pd.concat(metadata, ignore_index=True)
        weights[str(seed)] = {}
        for method in AUXILIARIES:
            alpha, grid = choose_alpha(b, np.concatenate(auxiliary[method]), t, meta, cfg)
            weights[str(seed)][method] = {"alpha": alpha, "validation_grid": grid,
                                         "validation_identities": meta.canonical_identity.nunique()}
    save_json(OUT / "frozen_weights.json", {"inputs": inputs, "weights": weights,
              "test_scores_computed": False, "rule": cfg["alpha_selection"]}, immutable=True)
    rows = []
    for cell, value in cells.items():
        metadata, truth = value["metadata"], value["truth"]
        for seed in cfg["training_seeds"]:
            base = value["base"][seed]
            predictions = {"g2cp": base}
            for method in AUXILIARIES:
                alpha = weights[str(seed)][method]["alpha"]
                predictions[f"aux_{method}"] = value["auxiliary"][method]
                predictions[f"g2cp_plus_{method}"] = (1-alpha)*base + alpha*value["auxiliary"][method]
            for method, matrix in predictions.items():
                metrics = scores(matrix, truth)
                frame = metadata.copy()
                frame["seed"], frame["method"] = seed, method
                for key, values in metrics.items():
                    frame[key] = values
                frame["prediction_constant"] = np.std(matrix, axis=1) == 0
                frame["truth_constant"] = np.std(truth, axis=1) == 0
                rows.append(frame)
    per_cell = pd.concat(rows, ignore_index=True)
    write_once(OUT / "per_drug_cell_metrics.csv", per_cell.to_csv(index=False))
    summaries, comparisons = [], []
    test = per_cell.loc[per_cell.split.eq("test")]
    for scope in ("primary_davis_disjoint", "all_kiba_external"):
        subset = test.loc[test.davis_external] if scope.startswith("primary") else test
        for cell in ["all", *cfg["cells"]]:
            current = subset if cell == "all" else subset.loc[subset.cell.eq(cell)]
            identity = current.groupby(["seed", "method", "canonical_identity"])[["pearson", "spearman", "mse"]].mean().reset_index()
            for (seed, method), group in identity.groupby(["seed", "method"]):
                summaries.append({"scope": scope, "cell": cell, "seed": seed, "method": method,
                                  "drug_identities": len(group), **group[["pearson", "spearman", "mse"]].mean().to_dict()})
            if cell != "all":
                continue
            # Average seeds within each drug; bootstrap drugs jointly across every method.
            averaged = identity.groupby(["method", "canonical_identity"])[["pearson", "spearman", "mse"]].mean().reset_index()
            for metric in ("pearson", "spearman", "mse"):
                matrix = averaged.pivot(index="canonical_identity", columns="method", values=metric)
                if matrix.isna().any().any() or len(matrix) < 2:
                    raise RuntimeError("Insufficient or unmatched primary evaluation coverage")
                rng = np.random.default_rng(cfg["bootstrap_seed"])
                selections = rng.integers(0, len(matrix), (cfg["bootstrap_repeats"], len(matrix)))
                for comparator in ("g2cp", "g2cp_plus_deep", "g2cp_plus_neighbor_prior", "g2cp_plus_fixed_blend", "g2cp_plus_generic", "g2cp_plus_shuffled_gate"):
                    delta = (matrix.g2cp_plus_similarity_gate - matrix[comparator]).to_numpy()
                    samples = delta[selections].mean(axis=1)
                    comparisons.append({"scope": scope, "metric": metric, "comparison": f"g2cp_plus_similarity_gate-minus-{comparator}",
                                        "difference": float(delta.mean()), "ci95": np.quantile(samples, [0.025, 0.975]).tolist(),
                                        "drug_identities": len(matrix), "condition": "fixed split, seed-averaged saved predictions; drug-cluster interval"})
    write_once(OUT / "test_summary.csv", pd.DataFrame(summaries).to_csv(index=False))
    save_json(OUT / "analysis.json", {"comparisons": comparisons, "alpha": weights,
              "primary_comparisons": ["g2cp_plus_similarity_gate vs g2cp", "g2cp_plus_similarity_gate vs g2cp_plus_deep"],
              "limitations": ["Exploratory kinase subset, not article-wide G2CP reproduction.",
                              "Same chemical split across seeds; seeds do not create independent test drugs.",
                              "Intervals conditional on saved predictions; no multiple comparison adjustment.",
                              "Near-zero/constant vectors are flagged; exact constant correlation defined as zero.",
                              "Inhibition need not equal knockout; 24 h chemical vs 96 h genetic exposure."],
              "outputs": [record(OUT / name) for name in ("frozen_weights.json", "per_drug_cell_metrics.csv", "test_summary.csv", "selected_targets.csv")]}, immutable=True)
    print(json.dumps(comparisons[:6], indent=2), flush=True)


if __name__ == "__main__":
    main()
