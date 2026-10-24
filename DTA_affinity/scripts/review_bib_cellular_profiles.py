#!/usr/bin/env python3
"""Freeze an endpoint-blind signature review without replacing the original case."""

import json
from datetime import datetime, timezone

import anndata as ad
import numpy as np
import pandas as pd
import requests

from audit_bib_cellular_case import G2CP, treated_masks
from bib_common import PROJECT, ROOT, record, sha256, write_once
from build_bib_cellular_profiles import correlation, grouped_profiles
from fetch_bib_case_sources import TextParser


ORIGINAL = PROJECT / "reports/bib_cellular_case_external_20261018"
OUT = PROJECT / "reports/bib_cellular_case_reviewed_20261018"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    audit = json.loads((ORIGINAL / "coverage_audit.json").read_text())
    for item in [audit["plan"], *audit["sources"], *audit["outputs"]]:
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"Original coverage source changed: {item['path']}")
    source = OUT / "sources/clue_data_levels.html"
    url = "https://clue.io/connectopedia/data_levels"
    if not source.exists():
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        write_once(source, response.text)
    parser = TextParser()
    parser.feed(source.read_text())
    text = " ".join(parser.parts)
    if "replicate-collapsed z-score vectors" not in text or "entire plate population" not in text:
        raise RuntimeError("Cannot verify the official signature-level documentation")
    start = text.index("Level 1")
    write_once(OUT / "sources/data_levels_excerpt.txt", text[start:start + 2500] + "\n")

    config = json.loads((ORIGINAL / "plan.json").read_text())
    config.update({
        "revision": "signature_review_before_cellular_endpoint_evaluation",
        "original_plan": record(ORIGINAL / "plan.json"),
        "original_candidate_manifest": record(ORIGINAL / "candidate_coverage.csv"),
        "functional_score": "Pearson correlation of released, processed chemical and BRAF-CRISPR signatures across all 978 landmark genes. Do not subtract consensus controls again. Average unique signatures within perturbagen, then perturbagens within chemical identity; average the two BRAF guide signatures.",
        "preprocessing_decision": "Released signed, replicate-collapsed profiles are treated as processed response signatures. Their metadata and scale are consistent with L1000 Level 5; an exact upstream transformation chain was not supplied and is not claimed to have been reconstructed. Consensus controls are not matched treatment-plate baselines. Preserve the previous subtraction as a sensitivity analysis, never select preprocessing by CTRP performance.",
        "quality": "Original QC/dose/time/candidate inclusion unchanged; controls are audited but not subtracted in the primary signature analysis.",
        "sensitivity_scores": ["guide_1_concordance", "guide_2_concordance", "legacy_control_subtracted"],
        "negative_control": {"type": "permute similarity-gate affinity across all 284 identities before rank averaging with fixed functional scores", "repeats": 2000, "seed": 20261019, "evaluation": "Spearman, AUROC and AP null distributions; Monte Carlo upper-tail probabilities are exploratory, not proof of target specificity"},
        "primary_comparison": "similarity_gate_plus_functional versus functional_only, Spearman with negative cellular AUC",
        "secondary_comparison": "similarity_gate_plus_functional versus deep_plus_functional on the identical candidate subset",
    })
    config["boundaries"].append("Original raw L1000 plate matrices and a complete author preprocessing log are not locally available; published processed signatures are used without claiming raw-data reconstruction.")
    write_once(OUT / "plan.json", json.dumps(config, indent=2) + "\n")
    coverage = pd.read_csv(ORIGINAL / "candidate_coverage.csv").fillna("")
    candidates = coverage.loc[coverage.in_candidate_library].sort_values("chemical_identity")
    query = pd.read_csv(ORIGINAL / "query_table.csv")
    if set(query.Canonical_SMILES) != set(candidates.chemical_identity) or len(candidates) != 284:
        raise RuntimeError("Original candidate library changed")

    path = G2CP / "data/LINCS/A375/A375_gene_chem_ctrl_adata.h5ad"
    data = ad.read_h5ad(path, backed="r")
    matrix = np.asarray(data.X[:], dtype=np.float64)
    obs, genes = data.obs.copy(), data.var_names.to_numpy(dtype=str)
    data.file.close()
    if len(genes) != 978 or len(set(genes)) != 978 or not np.isfinite(matrix).all():
        raise RuntimeError("Unexpected landmark matrix")
    if obs.sig_id.duplicated().any():
        raise RuntimeError("Duplicate signature IDs require an explicit replicate review")
    chemical_mask, genetic_mask = treated_masks(obs, config)
    target_mask = genetic_mask & obs.condition.eq("BRAF")
    positions = np.flatnonzero(target_mask.to_numpy())
    if len(positions) != 2 or obs.iloc[positions].pert_id.nunique() != 2:
        raise RuntimeError("Expected the two frozen, distinct BRAF guides")
    chemical = grouped_profiles(matrix, obs, chemical_mask)
    guides = matrix[positions]
    target = guides.mean(axis=0)
    times = pd.to_numeric(obs.pert_time, errors="coerce")
    vehicle = np.flatnonzero((obs.pert_type.eq("ctl_vehicle.cns") & times.eq(24)).to_numpy())
    vector = np.flatnonzero((obs.pert_type.eq("ctl_vector.cns") & times.eq(96)).to_numpy())
    if not len(vehicle) or not len(vector):
        raise RuntimeError("Original sensitivity controls are missing")
    cp_base, gen_base = matrix[vehicle].mean(axis=0), matrix[vector].mean(axis=0)
    rows, profiles = [], []
    for row in candidates.itertuples(index=False):
        ids = row.lincs_pert_ids.split("|")
        profile = np.mean([chemical[p] for p in ids], axis=0)
        rows.append({"chemical_identity": row.chemical_identity, "drug_names": row.drug_names,
                     "lincs_pert_ids": row.lincs_pert_ids,
                     "chemical_profiles": int((chemical_mask & obs.condition.isin(ids)).sum()),
                     "functional_concordance": correlation(profile, target),
                     "guide_1_concordance": correlation(profile, guides[0]),
                     "guide_2_concordance": correlation(profile, guides[1]),
                     "legacy_control_subtracted": correlation(profile - cp_base, target - gen_base)})
        profiles.append(profile)
    scores = pd.DataFrame(rows)
    old = pd.read_csv(ORIGINAL / "functional_scores.csv")
    matched = scores.merge(old[["chemical_identity", "functional_concordance"]], on="chemical_identity",
                           suffixes=("", "_old"), validate="one_to_one")
    error = float(np.max(np.abs(matched.legacy_control_subtracted - matched.functional_concordance_old)))
    if error > 1e-12:
        raise RuntimeError("Sensitivity calculation does not reproduce the archived original score")
    write_once(OUT / "functional_scores.csv", scores.to_csv(index=False))
    archive = OUT / "measured_profiles.npz"
    arrays = dict(chemical_identity=scores.chemical_identity.to_numpy(dtype=str), genes=genes,
                  chemical_signatures=np.asarray(profiles), target_signature=target,
                  guide_signatures=guides, vehicle_mean=cp_base, vector_mean=gen_base)
    if not archive.exists():
        np.savez_compressed(archive, **arrays)
    with np.load(archive, allow_pickle=False) as saved:
        if set(saved.files) != set(arrays) or any(not np.array_equal(saved[k], v) for k, v in arrays.items()):
            raise RuntimeError("Reviewed signature archive changed")
    selected_mask = chemical_mask & obs.condition.isin(set("|".join(candidates.lincs_pert_ids).split("|")))
    selected_mask |= target_mask | obs.pert_type.isin(["ctl_vehicle.cns", "ctl_vector.cns"])
    selected = obs.loc[selected_mask].copy()
    selected["matrix_row"] = np.flatnonzero(selected_mask.to_numpy())
    fields = ["matrix_row", "sig_id", "condition", "pert_id", "pert_type", "pert_time", "pert_dose",
              "qc_pass", "nsample", "distil_ids", "det_plates", "bead_batch", "batch"]
    write_once(OUT / "selected_profile_metadata.csv", selected[fields].to_csv(index=False))
    code_path = ROOT / "UniPert-G2CP_reproduce/analysis_assets/fig6_cause_effect_joint_analysis_ER_case.ipynb"
    report = {
        "plan": record(OUT / "plan.json"), "endpoint_values_read": False,
        "sources": [record(path), record(ORIGINAL / "coverage_audit.json"), record(code_path)],
        "documentation": {"url": url, **record(source)},
        "matrix_shape": list(matrix.shape), "negative_fraction": float((matrix < 0).mean()),
        "matrix_minimum": float(matrix.min()), "matrix_maximum": float(matrix.max()),
        "duplicate_sig_ids": int(obs.sig_id.duplicated().sum()),
        "target_guides": obs.iloc[positions][fields[1:]].astype(object).fillna("").to_dict("records"),
        "guide_signature_correlation": correlation(guides[0], guides[1]),
        "vehicle_consensus_rows": vehicle.tolist(), "vector_consensus_rows": vector.tolist(),
        "batch_field_crosstab": pd.crosstab(obs.batch, obs.pert_type).to_dict(),
        "batch_interpretation": "This file's batch field encodes chemical/genetic/control groups, not matched experimental plates.",
        "legacy_score_maximum_absolute_error": error,
        "primary_vs_legacy_score_pearson": correlation(scores.functional_concordance.to_numpy(), scores.legacy_control_subtracted.to_numpy()),
        "data_level_evidence": "Unique sig_id rows, nsample/distil_ids describing replicate collapse, signed standardized values, and the author's direct signature-connectivity analysis. Consistent with Level 5; full raw-to-export pipeline not reconstructed.",
        "decision": config["preprocessing_decision"],
        "outputs": [record(OUT / n) for n in ("functional_scores.csv", "measured_profiles.npz", "selected_profile_metadata.csv")],
    }
    write_once(OUT / "functional_audit.json", json.dumps(report, indent=2) + "\n")
    frozen = {"frozen_utc": datetime.now(timezone.utc).isoformat(), "endpoint_evaluation_started": False,
              "files": [record(OUT / n) for n in ("plan.json", "functional_scores.csv", "functional_audit.json")],
              "code": record(PROJECT / "scripts/review_bib_cellular_profiles.py")}
    if not (OUT / "freeze.json").exists():
        write_once(OUT / "freeze.json", json.dumps(frozen, indent=2) + "\n")
    print(json.dumps({"reviewed_candidates": len(scores), "guide_correlation": report["guide_signature_correlation"],
                      "primary_vs_legacy_correlation": report["primary_vs_legacy_score_pearson"],
                      "endpoint_values_read": False}), flush=True)


if __name__ == "__main__":
    main()
