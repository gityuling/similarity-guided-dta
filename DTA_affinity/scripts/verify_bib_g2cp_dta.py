#!/usr/bin/env python3
"""Independently check data boundaries and final Results 5 arithmetic."""

import argparse
import json

import anndata as ad
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from bib_g2cp_common import DATA, KIBA, OUT, RUN, SOURCE, check_records, plan, record, save_json, verify_data


def verify_prepared():
    audit, cfg = verify_data(), plan()
    split = pd.read_csv(OUT / "chemical_split.csv")
    if split.groupby("parent_key").split.nunique().max() != 1:
        raise RuntimeError("Chemical parent crosses partitions")
    candidates = pd.read_csv(OUT / "candidates.csv")
    if candidates.pert_id.isin(cfg["annotation_exclusions"]).any():
        raise RuntimeError("Excluded mechanism in candidate set")
    parent_split = split.drop_duplicates("parent_key").set_index("parent_key").split
    mapping = split.set_index("pert_id")
    library = pd.read_csv(OUT / "genetic_library_audit.csv")
    records = []
    for cell in cfg["cells"]:
        with np.load(DATA / f"{cell}.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        rows = pd.read_csv(DATA / f"{cell}_rows.csv", keep_default_na=False)
        evaluation = pd.read_csv(OUT / f"{cell}_evaluation.csv")
        chem = rows.loc[rows.pert_type.eq("trt_cp")]
        if not np.array_equal(chem.split, chem.parent_key.map(parent_split)):
            raise RuntimeError("Cell split differs from global parent assignment")
        if len(rows) != len(arrays["y"]) or len(evaluation) != len(arrays["truth"]):
            raise RuntimeError("Matrix alignment mismatch")
        source = ad.read_h5ad(SOURCE / f"data/LINCS/{cell}/{cell}_gene_chem_ctrl_adata.h5ad", backed="r")
        source_obs = source.obs.iloc[rows.source_row.to_numpy()]
        if source_obs.sig_id.tolist() != rows.sig_id.tolist() or source_obs.condition.astype(str).tolist() != rows.condition.tolist():
            raise RuntimeError("Source signature alignment differs")
        if source.var_names.astype(str).tolist() != arrays["gene_names"].tolist():
            raise RuntimeError("Landmark gene order differs")
        source.file.close()
        selected = library.loc[library.cell.eq(cell) & library.included]
        if set(selected.protein_id) != set(arrays["proteins"]):
            raise RuntimeError("Genetic library target coverage differs")
        for protein, expected in zip(arrays["proteins"], arrays["library"]):
            info = selected.loc[selected.protein_id.eq(protein)].iloc[0]
            gene_names = set(info.genes.split("|"))
            group = rows.loc[rows.condition.isin(gene_names) & rows.split.eq("pretrain_train")]
            if group.pert_id.nunique() < cfg["minimum_guides"]:
                raise RuntimeError("Genetic library contains validation-only or insufficient-guide target")
            means = np.stack([arrays["y"][part.index].mean(0) for _, part in group.groupby("pert_id")])
            if not np.allclose(means.mean(0), expected, atol=1e-6, rtol=0):
                raise RuntimeError("Genetic response aggregation differs")
        for row, expected in zip(evaluation.itertuples(), arrays["truth"]):
            group = rows.loc[rows.canonical_identity.eq(row.canonical_identity) & rows.pert_type.eq("trt_cp")]
            if set(group.split) != {row.split} or row.split != parent_split[row.parent_key]:
                raise RuntimeError("Evaluation chemical crosses train/val/test")
            means = [arrays["y"][part.index].mean(0) for _, part in group.groupby("pert_id")]
            if not np.allclose(np.mean(means, axis=0), expected, atol=1e-6, rtol=0):
                raise RuntimeError("Drug response aggregation differs")
        records.append({"cell": cell, "rows": len(rows), "library_targets": len(selected),
                        "test_drugs": int(evaluation.split.eq("test").sum()),
                        "primary_test_drugs": int((evaluation.split.eq("test") & evaluation.davis_external).sum())})
    result = {"state": "passed", "global_parent_disjoint": True, "cells": records,
              "data_audit": record(OUT / "data_audit.json"), "model_effects_scored": False}
    save_json(OUT / "prepared_verification.json", result, immutable=True)
    print(json.dumps(result, indent=2), flush=True)


def verify_final():
    verify_prepared()
    cfg = plan()
    analysis = json.loads((OUT / "analysis.json").read_text())
    check_records(analysis["outputs"])
    weight_record = json.loads((OUT / "frozen_weights.json").read_text())
    check_records(weight_record["inputs"])
    metrics = pd.read_csv(OUT / "per_drug_cell_metrics.csv", float_precision="round_trip")
    targets = pd.read_csv(OUT / "selected_targets.csv", float_precision="round_trip")
    expected_methods = {"g2cp"} | {f"{prefix}{method}" for prefix in ("aux_", "g2cp_plus_")
          for method in ("deep", "neighbor_prior", "fixed_blend", "similarity_gate", "generic", "shuffled_gate")}
    if set(metrics.method) != expected_methods or metrics.duplicated(["cell", "canonical_identity", "seed", "method"]).any():
        raise RuntimeError("Missing comparator or duplicate result")
    max_error, count = 0.0, 0
    validation_values = []
    for cell in cfg["cells"]:
        with np.load(DATA / f"{cell}.npz", allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        meta = pd.read_csv(OUT / f"{cell}_evaluation.csv")
        lookup = dict(zip(arrays["proteins"], arrays["library"]))
        aux = {}
        for method in ("deep", "neighbor_prior", "fixed_blend", "similarity_gate", "generic", "shuffled_gate"):
            vectors = []
            for item in meta.itertuples():
                if method == "generic":
                    vector = arrays["library"].mean(0)
                else:
                    selected = targets.loc[targets.cell.eq(cell) & targets.canonical_identity.eq(item.canonical_identity) & targets.method.eq(method)].sort_values("rank")
                    if len(selected) != cfg["top_k"] or abs(selected.weight.sum()-1) > 1e-12:
                        raise RuntimeError("Incorrect top-k auxiliary weights")
                    vector = sum(float(r.weight)*lookup[r.protein_id].astype(np.float64) for r in selected.itertuples())
                vectors.append(vector)
            aux[method] = np.asarray(vectors)
        for seed in cfg["training_seeds"]:
            path = RUN / "models" / f"{cell}_seed{seed}"
            completed = json.loads((path / "completed.json").read_text())
            check_records(completed["files"])
            base = np.load(path / "evaluation_predictions.npy", allow_pickle=False)
            predictions = {"g2cp": base}
            for method, vector in aux.items():
                alpha = weight_record["weights"][str(seed)][method]["alpha"]
                predictions[f"aux_{method}"] = vector
                predictions[f"g2cp_plus_{method}"] = (1-alpha)*base+alpha*vector
                for i, item in enumerate(meta.itertuples()):
                    if item.split != "val" or not item.davis_external:
                        continue
                    for trial in cfg["alpha_grid"]:
                        p = (1-trial)*base[i] + trial*vector[i]
                        truth = arrays["truth"][i]
                        value = pearsonr(p.astype(float), truth.astype(float)).statistic if np.std(p)>0 and np.std(truth)>0 else 0.0
                        validation_values.append({"seed": seed, "method": method, "alpha": trial, "identity": item.canonical_identity, "pearson": value})
            for method, values in predictions.items():
                saved = metrics.loc[metrics.cell.eq(cell) & metrics.seed.eq(seed) & metrics.method.eq(method)].set_index("canonical_identity")
                for i, item in enumerate(meta.itertuples()):
                    p, truth = values[i].astype(float), arrays["truth"][i].astype(float)
                    r = pearsonr(p, truth).statistic if np.std(p)>0 and np.std(truth)>0 else 0.0
                    s = spearmanr(p, truth).statistic if np.std(p)>0 and np.std(truth)>0 else 0.0
                    expected = np.array([r, s, np.mean((p-truth)**2)])
                    actual = saved.loc[item.canonical_identity, ["pearson", "spearman", "mse"]].to_numpy(dtype=float)
                    max_error = max(max_error, float(np.max(np.abs(expected-actual))))
                    count += 1
    if max_error > 1e-5:
        raise RuntimeError(f"Independent metric mismatch: {max_error}")
    val = pd.DataFrame(validation_values)
    aggregate = val.groupby(["seed", "method", "alpha", "identity"]).pearson.mean().groupby(["seed", "method", "alpha"]).mean()
    for (seed, method), group in aggregate.groupby(level=["seed", "method"]):
        best = group.max()
        alpha = min(index[2] for index, value in group.items() if value >= best-1e-12)
        if alpha != weight_record["weights"][str(seed)][method]["alpha"]:
            raise RuntimeError("Selected alpha disagrees with independent validation recomputation")
    save_json(OUT / "verification.json", {"state": "passed", "metric_rows_checked": count,
              "maximum_absolute_metric_error": max_error, "validation_weights_recomputed": True,
              "analysis": record(OUT / "analysis.json")}, immutable=True)
    print(f"Independent verification passed: {count} rows; max error {max_error}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--final", action="store_true")
    args = parser.parse_args()
    verify_final() if args.final else verify_prepared()
