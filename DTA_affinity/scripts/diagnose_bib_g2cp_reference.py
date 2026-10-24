#!/usr/bin/env python3
"""Diagnose target selection and genetic-response transfer on development drugs."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from bib_g2cp_common import DATA, OUT, PROJECT, RUN, check_records, now, record, save_json, verify_data, write_once
from evaluate_bib_g2cp_dta import build_auxiliary, correlation, scores

CONFIG = PROJECT / "configs/bib_g2cp_reference_diagnostic_20261019.json"
DEST = PROJECT / "reports/bib_g2cp_reference_diagnostic_20261019"
JOB = PROJECT / "results/bib_g2cp_reference_diagnostic_20261019"
METRICS = ["pearson", "spearman", "mse", "centered_pearson"]


def development_rows(metadata):
    selected = metadata.loc[metadata.split.isin(["train", "val"])].copy()
    if selected.duplicated(["cell", "canonical_identity"]).any():
        raise RuntimeError("Duplicate development identity")
    return selected.reset_index(drop=True)


def training_parent_means(y, rows):
    chemical = rows.loc[rows.split.eq("train") & rows.pert_type.eq("trt_cp")]
    result = {}
    for parent, group in chemical.groupby("parent_key", sort=True):
        means = [np.asarray(y[part.index], dtype=np.float64).mean(0)
                 for _, part in group.groupby("pert_id", sort=True)]
        result[parent] = np.mean(means, axis=0)
    return result


def query_training_mean(parent_means, parent, split):
    included = [key for key in sorted(parent_means) if split != "train" or key != parent]
    if not included:
        raise RuntimeError("No independent training parents for chemical mean")
    return np.mean([parent_means[key] for key in included], axis=0), len(included)


def reference_vectors(proteins, library, annotated, affinity):
    available = sorted(set(annotated) & set(proteins))
    if not available:
        return {}, available
    lookup = {protein: i for i, protein in enumerate(proteins)}
    vectors = {"annotated_reference": np.mean([library[lookup[p]] for p in available], axis=0)}
    for method in ("similarity_gate", "deep"):
        order = np.lexsort((proteins, -affinity[method].to_numpy()))[:len(available)]
        vectors[f"{method}_matched_count"] = library[order].mean(0)
    return vectors, available


def add_summaries(metrics):
    summaries, comparisons = [], []
    for scope in ("primary_davis_disjoint", "all_kiba_external"):
        scoped = metrics.loc[metrics.davis_external] if scope.startswith("primary") else metrics
        for subset in ("all_available", "annotation_available"):
            subset_rows = scoped.loc[scoped.reference_available] if subset.startswith("annotation") else scoped
            for (split, method), group in subset_rows.groupby(["split", "method"], sort=True):
                for cell in ["all", *sorted(group.cell.unique())]:
                    current = group if cell == "all" else group.loc[group.cell.eq(cell)]
                    per_drug = current.groupby(["seed", "canonical_identity"])[METRICS].mean()
                    per_drug = per_drug.groupby("canonical_identity").mean()
                    summaries.append({"scope": scope, "subset": subset, "split": split,
                                      "cell": cell, "method": method, "drug_identities": len(per_drug),
                                      "drug_cell_pairs": len(current[["cell", "canonical_identity"]].drop_duplicates()),
                                      **per_drug.mean().to_dict()})
            if subset != "annotation_available":
                continue
            validation = subset_rows.loc[subset_rows.split.eq("val")]
            aggregate = validation.groupby(["method", "seed", "canonical_identity"])[METRICS].mean()
            aggregate = aggregate.groupby(["method", "canonical_identity"]).mean().reset_index()
            pairs = [("aux_annotated_reference", "aux_similarity_gate"),
                     ("aux_annotated_reference", "aux_similarity_gate_matched_count"),
                     ("aux_annotated_reference", "aux_generic"),
                     ("aux_annotated_reference", "aux_chemical_training_mean"),
                     ("g2cp_plus_annotated_reference", "g2cp"),
                     ("g2cp_plus_annotated_reference", "g2cp_plus_similarity_gate")]
            cfg = json.loads(CONFIG.read_text())
            for metric in ("pearson", "centered_pearson"):
                matrix = aggregate.pivot(index="canonical_identity", columns="method", values=metric)
                for left, right in pairs:
                    if left not in matrix or right not in matrix:
                        continue
                    values = matrix[[left, right]].dropna()
                    if len(values) < 2:
                        continue
                    delta = (values[left] - values[right]).to_numpy()
                    rng = np.random.default_rng(cfg["bootstrap_seed"])
                    draws = rng.integers(0, len(delta), (cfg["bootstrap_repeats"], len(delta)))
                    comparisons.append({"scope": scope, "subset": subset, "split": "val", "metric": metric,
                                        "left": left, "right": right, "difference": float(delta.mean()),
                                        "ci95": np.quantile(delta[draws].mean(1), [.025, .975]).tolist(),
                                        "drug_identities": len(delta), "positive_drugs": int((delta > 0).sum()),
                                        "inference": "descriptive, conditional on development data; not confirmatory"})
    return pd.DataFrame(summaries), comparisons


def main():
    cfg = json.loads(CONFIG.read_text())
    if cfg["splits"] != ["train", "val"]:
        raise RuntimeError("This diagnostic must not score test drugs")
    JOB.mkdir(parents=True, exist_ok=True)
    save_json(JOB / "status.json", {"state": "running", "started_utc": now(), "stage": "verify_inputs"})
    audit = verify_data()
    prior = json.loads((OUT / "analysis.json").read_text())
    check_records(prior["outputs"])
    paths = [CONFIG, Path(__file__), PROJECT / "scripts/bib_g2cp_common.py",
             PROJECT / "scripts/evaluate_bib_g2cp_dta.py", OUT / "data_audit.json", OUT / "analysis.json"]
    for cell in cfg["cells"]:
        for seed in cfg["seeds"]:
            completed_path = RUN / "models" / f"{cell}_seed{seed}" / "completed.json"
            completed = json.loads(completed_path.read_text())
            check_records(completed["files"])
            paths.append(completed_path)
    sources = [record(path) for path in paths]
    save_json(DEST / "freeze.json", {"config": cfg, "inputs": sources,
              "original_data_files": audit["files"], "original_analysis_outputs": prior["outputs"]}, immutable=True)
    candidates = pd.read_csv(OUT / "candidates.csv", keep_default_na=False).set_index("canonical_identity")
    query = pd.read_csv(OUT / "query_table.csv", usecols=["row_id", "Canonical_SMILES", "protein_id"])
    query = query.loc[query.Canonical_SMILES.isin(candidates.loc[candidates.split.isin(cfg["splits"])].index)]
    affinity = query.merge(pd.read_csv(OUT / "affinity_predictions.csv"), on="row_id", validate="one_to_one")
    records, coverage, target_details, cell_audits = [], [], [], []
    for cell in cfg["cells"]:
        save_json(JOB / "status.json", {"state": "running", "stage": cell, "updated_utc": now()})
        full_meta = pd.read_csv(OUT / f"{cell}_evaluation.csv")
        meta = development_rows(full_meta)
        rows = pd.read_csv(DATA / f"{cell}_rows.csv", keep_default_na=False)
        with np.load(DATA / f"{cell}.npz", allow_pickle=False) as archive:
            truth = archive["truth"][meta.matrix_row.to_numpy()].astype(np.float64)
            proteins = archive["proteins"].astype(str)
            library = archive["library"].astype(np.float64)
            parent_means = training_parent_means(archive["y"], rows)
        arrays = {"proteins": proteins, "library": library}
        aux, _ = build_auxiliary(cell, meta, arrays, affinity, audit["plan"])
        aux.pop("neighbor_prior")
        aux.update({key: np.full_like(truth, np.nan) for key in
                    ("annotated_reference", "similarity_gate_matched_count", "deep_matched_count")})
        train_means, counts = [], []
        for i, row in enumerate(meta.itertuples()):
            info = candidates.loc[row.canonical_identity]
            annotated = set(filter(None, info.mapped_kiba_targets.split("|")))
            scores_frame = affinity.loc[affinity.Canonical_SMILES.eq(row.canonical_identity)].set_index("protein_id").loc[proteins]
            refs, available = reference_vectors(proteins, library, annotated, scores_frame)
            for key, vector in refs.items():
                aux[key][i] = vector
            mean, count = query_training_mean(parent_means, row.parent_key, row.split)
            train_means.append(mean)
            counts.append(count)
            coverage.append({"cell": cell, "canonical_identity": row.canonical_identity, "parent_key": row.parent_key,
                             "names": row.names, "split": row.split, "davis_external": row.davis_external,
                             "source_annotated_genes": info.gene_targets,
                             "mapped_annotated_targets": "|".join(sorted(annotated)),
                             "available_annotated_targets": "|".join(available),
                             "unavailable_annotated_targets": "|".join(sorted(annotated-set(available))),
                             "mapped_target_count": len(annotated), "available_target_count": len(available),
                             "reference_available": bool(available), "training_mean_parents": count})
            for method in ("deep", "fixed_blend", "similarity_gate"):
                order = np.lexsort((proteins, -scores_frame[method].to_numpy()))
                ranked = proteins[order]
                ranks = [int(np.flatnonzero(ranked == p)[0])+1 for p in available]
                target_details.append({"cell": cell, "canonical_identity": row.canonical_identity,
                                       "names": row.names, "split": row.split, "davis_external": row.davis_external,
                                       "method": method, "library_targets": len(proteins),
                                       "available_annotation_count": len(available), "top5_targets": "|".join(ranked[:5]),
                                       "annotated_target_ranks": "|".join(map(str, sorted(ranks))),
                                       "hit_at_1": int(1 in ranks) if ranks else np.nan,
                                       "hit_at_5": int(any(r <= 5 for r in ranks)) if ranks else np.nan,
                                       "recall_at_5": sum(r <= 5 for r in ranks)/len(ranks) if ranks else np.nan,
                                       "reciprocal_rank": 1/min(ranks) if ranks else np.nan})
        train_means = np.asarray(train_means)
        aux["chemical_training_mean"] = train_means
        meta["reference_available"] = np.isfinite(aux["annotated_reference"]).all(1)
        payload = {"truth": truth, "training_mean": train_means, "matrix_rows": meta.matrix_row.to_numpy()}
        payload.update({f"aux_{key}": value for key, value in aux.items()})
        for seed in cfg["seeds"]:
            base = np.load(RUN / "models" / f"{cell}_seed{seed}" / "evaluation_predictions.npy",
                           mmap_mode="r")[meta.matrix_row.to_numpy()].astype(np.float64)
            payload[f"g2cp_{seed}"] = base
            predictions = {"g2cp": base}
            for method, vector in aux.items():
                predictions[f"aux_{method}"] = vector
                predictions[f"g2cp_plus_{method}"] = (1-cfg["mixture_alpha"])*base + cfg["mixture_alpha"]*vector
            for method, values in predictions.items():
                mask = np.isfinite(values).all(1)
                result = meta.loc[mask].copy()
                result["method"], result["seed"] = method, seed
                for key, value in scores(values[mask], truth[mask]).items():
                    result[key] = value
                result["centered_pearson"] = correlation(values[mask]-train_means[mask], truth[mask]-train_means[mask])
                result["prediction_constant"] = np.std(values[mask], axis=1) == 0
                result["centered_prediction_constant"] = np.std(values[mask]-train_means[mask], axis=1) == 0
                records.append(result)
        archive_path = DEST / f"{cell}_development_vectors.npz"
        if archive_path.exists():
            with np.load(archive_path, allow_pickle=False) as previous:
                if set(previous.files) != set(payload) or any(not np.array_equal(previous[k], v, equal_nan=True) for k,v in payload.items()):
                    raise RuntimeError("Existing diagnostic vectors differ")
        else:
            np.savez_compressed(archive_path, **payload)
        write_once(DEST / f"{cell}_development_rows.csv", meta.to_csv(index=False))
        cell_audits.append({"cell": cell, "development_pairs": len(meta),
                           "excluded_test_pairs": int(full_meta.split.eq("test").sum()),
                           "reference_available_pairs": int(meta.reference_available.sum()),
                           "training_chemical_parents": len(parent_means)})
        print(json.dumps(cell_audits[-1]), flush=True)
    metrics = pd.concat(records, ignore_index=True)
    if not set(metrics.split) <= {"train", "val"}:
        raise RuntimeError("Test metric escaped development boundary")
    summary, comparisons = add_summaries(metrics)
    files = {"coverage.csv": pd.DataFrame(coverage), "target_ranks.csv": pd.DataFrame(target_details),
             "per_drug_cell_metrics.csv": metrics, "summary.csv": summary}
    for filename, frame in files.items():
        write_once(DEST / filename, frame.to_csv(index=False))
    check_records(sources)
    output_paths = [DEST / name for name in files]
    output_paths += [DEST / f"{cell}_{suffix}" for cell in cfg["cells"]
                     for suffix in ("development_vectors.npz", "development_rows.csv")]
    save_json(DEST / "analysis.json", {"config": cfg, "cells": cell_audits, "test_rows_scored": 0,
              "comparisons": comparisons, "outputs": [record(path) for path in output_paths]}, immutable=True)
    save_json(JOB / "status.json", {"state": "computed", "finished_utc": now(), "metric_rows": len(metrics),
              "test_rows_scored": 0, "verification": "pending", "gpu_used": False})
    primary = summary.loc[summary.scope.eq("primary_davis_disjoint") & summary.subset.eq("annotation_available")
                          & summary.split.eq("val") & summary.cell.eq("all")]
    print(primary[["method", "drug_identities", "pearson", "centered_pearson"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
