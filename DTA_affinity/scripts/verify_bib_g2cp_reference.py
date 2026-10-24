#!/usr/bin/env python3
"""Reconstruct reference diagnostics independently from their frozen inputs."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from bib_g2cp_common import DATA, OUT, PROJECT, RUN, check_records, now, record, save_json

DEST = PROJECT / "reports/bib_g2cp_reference_diagnostic_20261019"
JOB = PROJECT / "results/bib_g2cp_reference_diagnostic_20261019"
METRICS = ["pearson", "spearman", "mse", "centered_pearson"]


def corr(a, b):
    return float(pearsonr(a, b).statistic) if np.std(a) > 0 and np.std(b) > 0 else 0.0


def main():
    frozen = json.loads((DEST / "freeze.json").read_text())
    analysis = json.loads((DEST / "analysis.json").read_text())
    cfg = frozen["config"]
    for key in ("inputs", "original_data_files", "original_analysis_outputs"):
        check_records(frozen[key])
    check_records(analysis["outputs"])
    candidates = pd.read_csv(OUT / "candidates.csv", keep_default_na=False).set_index("canonical_identity")
    coverage = pd.read_csv(DEST / "coverage.csv", keep_default_na=False).set_index(["cell", "canonical_identity"])
    metrics = pd.read_csv(DEST / "per_drug_cell_metrics.csv", float_precision="round_trip")
    if not set(metrics.split) <= {"train", "val"} or metrics.duplicated(["cell", "canonical_identity", "seed", "method"]).any():
        raise RuntimeError("Test rows or duplicates in diagnostic")
    query = pd.read_csv(OUT / "query_table.csv", usecols=["row_id", "Canonical_SMILES", "protein_id"])
    predictions = query.merge(pd.read_csv(OUT / "affinity_predictions.csv"), on="row_id", validate="one_to_one")
    max_error, vector_error, checked, checked_vectors = 0., 0., 0, 0
    independently_scored = []
    for cell in cfg["cells"]:
        meta = pd.read_csv(DEST / f"{cell}_development_rows.csv")
        original = pd.read_csv(OUT / f"{cell}_evaluation.csv")
        expected_rows = original.loc[original.split.isin(["train", "val"])].matrix_row.to_numpy()
        if not np.array_equal(meta.matrix_row, expected_rows):
            raise RuntimeError("Development source rows changed")
        rows = pd.read_csv(DATA / f"{cell}_rows.csv", keep_default_na=False)
        with np.load(DATA / f"{cell}.npz", allow_pickle=False) as archive:
            source_truth = archive["truth"][expected_rows].astype(float)
            proteins, library = archive["proteins"], archive["library"].astype(float)
            train = rows.loc[rows.split.eq("train") & rows.pert_type.eq("trt_cp")]
            matrix = pd.DataFrame(archive["y"][train.index].astype(float))
        matrix["parent_key"] = train.parent_key.to_numpy()
        matrix["pert_id"] = train.pert_id.to_numpy()
        parent_means = matrix.groupby(["parent_key", "pert_id"]).mean().groupby("parent_key").mean()
        with np.load(DEST / f"{cell}_development_vectors.npz", allow_pickle=False) as archive:
            saved = {key: archive[key] for key in archive.files}
        if not np.array_equal(saved["truth"], source_truth) or not np.array_equal(saved["matrix_rows"], expected_rows):
            raise RuntimeError("Truth or matrix alignment differs")
        expected_aux = {key: [] for key in saved if key.startswith("aux_")}
        for i, row in enumerate(meta.itertuples()):
            frame = predictions.loc[predictions.Canonical_SMILES.eq(row.canonical_identity)].set_index("protein_id").loc[proteins]
            annotated = set(filter(None, candidates.loc[row.canonical_identity, "mapped_kiba_targets"].split("|")))
            available = sorted(annotated.intersection(proteins))
            cov = coverage.loc[(cell, row.canonical_identity)]
            if cov.available_annotated_targets != "|".join(available) or bool(cov.reference_available) != bool(available):
                raise RuntimeError("Annotation coverage mismatch")
            if bool(meta.reference_available.iloc[i]) != bool(available):
                raise RuntimeError("Comparison coverage mask mismatch")
            means = parent_means.drop(index=row.parent_key) if row.split == "train" else parent_means
            chemical_mean = means.mean().to_numpy()
            if len(means) != cov.training_mean_parents or not np.allclose(chemical_mean, saved["training_mean"][i], atol=1e-10):
                raise RuntimeError("Training chemical baseline includes incorrect rows or weights")
            for key in expected_aux:
                method = key.removeprefix("aux_")
                if method == "chemical_training_mean":
                    vector = chemical_mean
                elif method == "generic":
                    vector = library.mean(0)
                elif method == "annotated_reference":
                    vector = np.mean([library[np.flatnonzero(proteins == p)[0]] for p in available], axis=0) if available else np.full(978, np.nan)
                else:
                    source_method = method.removesuffix("_matched_count")
                    raw = frame["similarity_gate" if method == "shuffled_gate" else source_method].to_numpy()
                    if method == "shuffled_gate":
                        original_cfg = json.loads((PROJECT / "configs/bib_g2cp_dta_20261019.json").read_text())
                        key_string = f"{original_cfg['shuffle_seed']}:{cell}:{row.canonical_identity}"
                        seed = int(hashlib.sha256(key_string.encode()).hexdigest()[:16], 16)
                        raw = np.random.default_rng(seed).permutation(raw)
                    order = sorted(range(len(proteins)), key=lambda index: (-raw[index], proteins[index]))
                    if method.endswith("matched_count"):
                        vector = np.mean(library[order[:len(available)]], axis=0) if available else np.full(978, np.nan)
                    else:
                        weight = np.array([1., .5, 1/3, .25, .2])
                        vector = sum(w*library[index] for w, index in zip(weight/weight.sum(), order[:5]))
                expected_aux[key].append(vector)
                actual = saved[key][i]
                if not np.array_equal(np.isnan(actual), np.isnan(vector)):
                    raise RuntimeError("Unavailable target reference incorrectly imputed")
                if np.isfinite(vector).all():
                    error = float(np.max(np.abs(actual-vector)))
                    vector_error = max(vector_error, error)
                    if error > 1e-9:
                        raise RuntimeError(f"Auxiliary mismatch: {cell} {method} {error}")
                checked_vectors += 1
        expected_aux = {key: np.asarray(value) for key, value in expected_aux.items()}
        for seed in cfg["seeds"]:
            base = np.load(RUN / "models" / f"{cell}_seed{seed}" / "evaluation_predictions.npy", mmap_mode="r")[expected_rows].astype(float)
            if not np.array_equal(saved[f"g2cp_{seed}"], base):
                raise RuntimeError("Saved G2CP prediction mismatch")
            methods = {"g2cp": base, **expected_aux}
            for key, vector in expected_aux.items():
                methods["g2cp_plus_"+key.removeprefix("aux_")] = .75*base+.25*vector
            for method, values in methods.items():
                current = metrics.loc[metrics.cell.eq(cell) & metrics.seed.eq(seed) & metrics.method.eq(method)].set_index("canonical_identity")
                expected_ids = set(meta.loc[np.isfinite(values).all(1), "canonical_identity"])
                if set(current.index) != expected_ids:
                    raise RuntimeError("Comparator evaluates a different set of drug/cell pairs")
                for i, row in enumerate(meta.itertuples()):
                    if row.canonical_identity not in expected_ids:
                        continue
                    value, truth = values[i], source_truth[i]
                    center = expected_aux["aux_chemical_training_mean"][i]
                    rank = float(spearmanr(value, truth).statistic) if np.std(value)>0 and np.std(truth)>0 else 0.
                    expected = np.array([corr(value, truth), rank, np.mean((value-truth)**2), corr(value-center, truth-center)])
                    actual = current.loc[row.canonical_identity, METRICS].to_numpy(dtype=float)
                    max_error = max(max_error, float(np.max(np.abs(expected-actual))))
                    independently_scored.append({"cell": cell, "canonical_identity": row.canonical_identity,
                        "split": row.split, "davis_external": row.davis_external, "reference_available": bool(meta.reference_available.iloc[i]),
                        "seed": seed, "method": method, **dict(zip(METRICS, expected))})
                    checked += 1
        print(f"Verified {cell}", flush=True)
    if max_error > 1e-9:
        raise RuntimeError(f"Diagnostic metric mismatch: {max_error}")
    independently_scored = pd.DataFrame(independently_scored)
    summary = pd.read_csv(DEST / "summary.csv", float_precision="round_trip")
    summary_error = 0.
    for item in summary.itertuples():
        subset = independently_scored.loc[independently_scored.method.eq(item.method) & independently_scored.split.eq(item.split)]
        if item.scope == "primary_davis_disjoint":
            subset = subset.loc[subset.davis_external]
        if item.subset == "annotation_available":
            subset = subset.loc[subset.reference_available]
        if item.cell != "all":
            subset = subset.loc[subset.cell.eq(item.cell)]
        means = subset.groupby(["canonical_identity", "seed"])[METRICS].mean().groupby("canonical_identity").mean()
        if len(means) != item.drug_identities or len(subset[["cell", "canonical_identity"]].drop_duplicates()) != item.drug_cell_pairs:
            raise RuntimeError("Summary sample size mismatch")
        summary_error = max(summary_error, float(np.max(np.abs(means.mean().to_numpy()-np.array([getattr(item, k) for k in METRICS])))))
    interval_error = 0.
    for item in analysis["comparisons"]:
        subset = independently_scored.loc[independently_scored.split.eq("val") & independently_scored.reference_available]
        if item["scope"] == "primary_davis_disjoint":
            subset = subset.loc[subset.davis_external]
        metric = item["metric"]
        means = subset.groupby(["canonical_identity", "seed", "method"])[metric].mean().groupby(["canonical_identity", "method"]).mean().unstack()
        differences = (means[item["left"]] - means[item["right"]]).dropna().to_numpy()
        rng = np.random.default_rng(cfg["bootstrap_seed"])
        resamples = rng.choice(differences, size=(cfg["bootstrap_repeats"], len(differences)), replace=True).mean(1)
        interval = np.percentile(resamples, [2.5, 97.5])
        if len(differences) != item["drug_identities"]:
            raise RuntimeError("Interval drug count mismatch")
        interval_error = max(interval_error, abs(differences.mean()-item["difference"]), float(np.max(np.abs(interval-item["ci95"]))))
    if max(summary_error, interval_error) > 1e-9:
        raise RuntimeError("Summary or paired interval verification failed")
    result = {"state": "passed", "metric_rows": checked, "auxiliary_vectors": checked_vectors,
              "test_rows_scored": 0, "maximum_metric_error": max_error, "maximum_vector_error": vector_error,
              "maximum_summary_error": summary_error, "maximum_interval_error": interval_error,
              "reference_mapping_verified": True, "training_mean_excludes_heldout_and_self_parent": True,
              "verifier": record(Path(__file__)), "analysis": record(DEST / "analysis.json")}
    save_json(DEST / "verification.json", result, immutable=True)
    save_json(JOB / "status.json", {"state": "completed", "finished_utc": now(), "verification": "passed",
              "metric_rows": checked, "test_rows_scored": 0, "gpu_used": False})
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
