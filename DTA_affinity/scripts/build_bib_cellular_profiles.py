#!/usr/bin/env python3
"""Construct endpoint-blind measured perturbation scores with explicit row alignment."""

import json

import anndata as ad
import numpy as np
import pandas as pd

from audit_bib_cellular_case import G2CP, treated_masks
from bib_common import PROJECT, ROOT, record, sha256, write_once


CASE = PROJECT / "reports/bib_cellular_case_external_20261018"


def grouped_profiles(matrix, obs, mask):
    positions = np.flatnonzero(np.asarray(mask))
    selected = obs.iloc[positions].copy()
    selected["matrix_row"] = positions
    # Group indices must refer to the original matrix, not a filtered frame.
    selected = selected.drop_duplicates(["condition", "sig_id"])
    return {str(key): matrix[group.matrix_row.to_numpy()].mean(axis=0)
            for key, group in selected.groupby("condition", observed=True, sort=True)}


def correlation(x, y):
    x, y = x - x.mean(), y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom == 0:
        raise ValueError("A perturbation profile has no variation")
    return float(np.dot(x, y) / denom)


def main():
    config = json.loads((CASE / "plan.json").read_text())
    audit = json.loads((CASE / "coverage_audit.json").read_text())
    if not audit["primary_coverage_sufficient"]:
        raise RuntimeError("Case coverage is below the fixed minimum")
    for source in [audit["plan"], *audit["sources"], *audit["outputs"]]:
        if sha256(ROOT / source["path"]) != source["sha256"]:
            raise RuntimeError(f"Audited case input changed: {source['path']}")
    coverage = pd.read_csv(CASE / "candidate_coverage.csv").fillna("")
    candidates = coverage.loc[coverage.in_candidate_library].sort_values("chemical_identity")
    path = G2CP / f"data/LINCS/{config['primary_cell']}/{config['primary_cell']}_gene_chem_ctrl_adata.h5ad"
    data = ad.read_h5ad(path, backed="r")
    obs = data.obs.copy()
    matrix = np.asarray(data.X[:], dtype=np.float64)
    genes = data.var_names.astype(str).to_numpy()
    data.file.close()
    if len(genes) != 978 or len(set(genes)) != 978 or not np.isfinite(matrix).all():
        raise RuntimeError("Expected 978 unique aligned finite landmark features")
    chemical_mask, genetic_mask = treated_masks(obs, config)
    time = pd.to_numeric(obs.pert_time, errors="coerce")
    cp_ctrl = obs.pert_type.eq("ctl_vehicle.cns") & time.eq(config["chemical_time_hours"])
    gen_ctrl = obs.pert_type.eq("ctl_vector.cns") & time.eq(config["genetic_time_hours"])
    if not cp_ctrl.any() or not gen_ctrl.any():
        raise RuntimeError("Missing perturbation-type/time-matched controls")
    cp_base = matrix[np.flatnonzero(cp_ctrl.to_numpy())].mean(axis=0)
    gen_base = matrix[np.flatnonzero(gen_ctrl.to_numpy())].mean(axis=0)
    chemical = grouped_profiles(matrix, obs, chemical_mask)
    genetic = grouped_profiles(matrix, obs, genetic_mask & obs.condition.eq(config["target_gene"]))
    target = genetic[config["target_gene"]] - gen_base
    rows, profile_rows = [], []
    for row in candidates.itertuples(index=False):
        pert_ids = row.lincs_pert_ids.split("|")
        profile = np.mean([chemical[p] - cp_base for p in pert_ids], axis=0)
        count = int((chemical_mask & obs.condition.isin(pert_ids)).sum())
        rows.append({"chemical_identity": row.chemical_identity, "drug_names": row.drug_names,
                     "lincs_pert_ids": row.lincs_pert_ids, "chemical_profiles": count,
                     "functional_concordance": correlation(profile, target)})
        profile_rows.append(profile)
    result = pd.DataFrame(rows)
    write_once(CASE / "functional_scores.csv", result.to_csv(index=False))
    arrays = CASE / "measured_profiles.npz"
    if not arrays.exists():
        np.savez_compressed(arrays, chemical_identity=result.chemical_identity.to_numpy(dtype=str),
                            genes=genes, chemical_effect=np.asarray(profile_rows),
                            target_effect=target, vehicle_mean=cp_base, vector_mean=gen_base)
    with np.load(arrays, allow_pickle=False) as saved:
        if not np.array_equal(saved["chemical_effect"], np.asarray(profile_rows)):
            raise RuntimeError("Archived profiles disagree on repeat")
    genetic_indices = np.flatnonzero((genetic_mask & obs.condition.eq(config["target_gene"])).to_numpy())
    report = {"plan": record(CASE / "plan.json"), "sources": [record(path), record(CASE / "candidate_coverage.csv")],
              "candidates": len(result), "landmark_genes": len(genes),
              "target_genetic_matrix_rows": genetic_indices.tolist(),
              "vehicle_matrix_rows": np.flatnonzero(cp_ctrl.to_numpy()).tolist(),
              "vector_matrix_rows": np.flatnonzero(gen_ctrl.to_numpy()).tolist(),
              "target_guide_pair_correlation": correlation(matrix[genetic_indices[0]] - gen_base,
                                                            matrix[genetic_indices[1]] - gen_base)
              if len(genetic_indices) == 2 else None,
              "perturbation_type_source": "https://clue.io/connectopedia/perturbagen_types_and_controls",
              "interpretation": "trt_xpr is CRISPR loss of function; positive response concordance, not gene-overexpression reversal",
              "endpoint_values_read": False,
              "outputs": [record(CASE / "functional_scores.csv"), record(arrays)]}
    write_once(CASE / "functional_audit.json", json.dumps(report, indent=2) + "\n")
    print(json.dumps({"candidates": len(result), "functional_profiles": "complete", "target_guide_pair_correlation": report["target_guide_pair_correlation"]}))


if __name__ == "__main__":
    main()
