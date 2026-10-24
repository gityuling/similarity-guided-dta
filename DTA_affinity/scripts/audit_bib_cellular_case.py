#!/usr/bin/env python3
"""Audit local DTA, LINCS and CTRP coverage before candidate scoring."""

from __future__ import annotations

import json
import argparse
import zipfile
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from bib_common import PROJECT, ROOT, record, write_once


CONFIG = PROJECT / "configs/bib_cellular_case_20261018.json"
OUT = PROJECT / "reports/bib_cellular_case_20261018"
G2CP = ROOT / "UniPert-G2CP_reproduce/G2CP"
CTRP = ROOT / "UniPert_DrEval/data/raw/extracted/CTRPv2"
CELLS = ("A375", "A549", "MCF7", "PC3", "HT29")


def canonical(value):
    if pd.isna(value):
        return None
    mol = Chem.MolFromSmiles(str(value))
    return Chem.MolToSmiles(mol, isomericSmiles=True) if mol is not None else None


def treated_masks(obs, config):
    quality = obs.qc_pass.astype(str).str.lower().isin({"1", "true"})
    time = pd.to_numeric(obs.pert_time, errors="coerce")
    dose = pd.to_numeric(obs.pert_dose, errors="coerce")
    chemical = (quality & obs.pert_type.eq(config["chemical_pert_type"])
                & time.eq(config["chemical_time_hours"]) & obs.pert_time_unit.eq("h")
                & dose.eq(config["chemical_dose_um"]) & obs.pert_dose_unit.eq("uM"))
    genetic = (quality & obs.pert_type.eq(config["genetic_pert_type"])
               & time.eq(config["genetic_time_hours"]) & obs.pert_time_unit.eq("h"))
    return chemical, genetic


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG)
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    config_path, out = args.config, args.out_dir
    RDLogger.DisableLog("rdApp.*")
    config = json.loads(config_path.read_text())
    out.mkdir(parents=True, exist_ok=True)
    write_once(out / "plan.json", json.dumps(config, indent=2) + "\n")
    table_path = PROJECT / "data/processed/kiba_dtiam_protocol/table.csv"
    table = pd.read_csv(table_path, usecols=["row_id", "drug_id", "protein_id", "Canonical_SMILES", "Sequence"])
    table["chemical_identity"] = table.Canonical_SMILES.map(canonical)
    kiba_ids = set(table.chemical_identity.dropna())
    target_table = table.loc[table.protein_id.eq(config["target_uniprot"])]
    target_ids = set(target_table.chemical_identity)
    target_sequences = target_table.Sequence.unique()
    if len(target_sequences) != 1:
        raise RuntimeError("Expected a single known-target sequence")
    compounds_path = G2CP / "data/processed_CMAP_compound_info.csv"
    compounds = pd.read_csv(compounds_path, usecols=["pert_id", "canonical_smiles"])
    compounds["chemical_identity"] = compounds.canonical_smiles.map(canonical)
    ambiguity = compounds.dropna().groupby("pert_id").chemical_identity.nunique()
    if (ambiguity > 1).any():
        raise RuntimeError("Ambiguous LINCS perturbagen-to-structure mapping")
    compounds = compounds.dropna().drop_duplicates(["pert_id", "chemical_identity"])
    ctrp_smiles_path = CTRP / "drug_smiles.csv"
    drugs = pd.read_csv(ctrp_smiles_path, usecols=["pubchem_id", "drug_name", "canonical_smiles"], dtype={"pubchem_id": str})
    drugs["chemical_identity"] = drugs.canonical_smiles.map(canonical)
    ctrp_ids = set(drugs.chemical_identity.dropna())
    if (drugs.dropna().groupby("pubchem_id").chemical_identity.nunique() > 1).any():
        raise RuntimeError("Ambiguous CTRP PubChem-to-structure mapping")
    response_path = CTRP / "CTRPv2.csv"
    response = pd.read_csv(response_path, usecols=["cellosaurus_id", "cell_line_name", "pubchem_id", "AUC_curvecurator"],
                           dtype={"pubchem_id": str}, low_memory=False)
    response["label_available"] = np.isfinite(pd.to_numeric(response.pop("AUC_curvecurator"), errors="coerce"))
    response = response.merge(drugs[["pubchem_id", "chemical_identity"]].drop_duplicates(),
                              on="pubchem_id", how="left", validate="many_to_one")
    atom_counts = {s: Chem.AddHs(Chem.MolFromSmiles(s)).GetNumAtoms() for s in ctrp_ids}
    supported_ids = {s for s, n in atom_counts.items() if n <= 512}
    drugs_by_identity = {k: g for k, g in drugs.dropna(subset=["chemical_identity"]).groupby("chemical_identity")}
    kiba_by_identity = table.groupby("chemical_identity").drug_id.agg(lambda x: "|".join(sorted(x.unique()))).to_dict()
    sources = [record(p) for p in (config_path, table_path, compounds_path, ctrp_smiles_path, response_path)]
    summaries, primary = [], None
    for cell in CELLS:
        path = G2CP / f"data/LINCS/{cell}/{cell}_gene_chem_ctrl_adata.h5ad"
        data = ad.read_h5ad(path, backed="r")
        obs = data.obs.copy()
        chemical, genetic = treated_masks(obs, config)
        cell_ids = set(obs.cellosaurus_id.dropna().astype(str))
        if len(cell_ids) != 1:
            raise RuntimeError(f"Ambiguous cell identity in {cell}")
        cell_id = next(iter(cell_ids))
        measured = response.loc[response.cellosaurus_id.eq(cell_id) & response.label_available]
        measured_ids = set(measured.chemical_identity.dropna())
        eligible_pert = set(obs.loc[chemical, "condition"].astype(str))
        cell_compounds = compounds.loc[compounds.pert_id.isin(eligible_pert)]
        lincs_ids = set(cell_compounds.chemical_identity)
        candidates = ((ctrp_ids & lincs_ids) - kiba_ids if config.get("novel_only")
                      else kiba_ids & ctrp_ids & lincs_ids)
        candidates &= supported_ids
        scored = candidates & target_ids if config.get("requires_saved_target_prediction", True) else candidates
        target_obs = obs.loc[genetic & obs.condition.eq(config["target_gene"])]
        item = {"cell": cell, "cellosaurus_id": cell_id, "landmark_genes": data.n_vars,
                "qc_chemical_profiles": int(chemical.sum()), "qc_chemical_identities": len(lincs_ids),
                "target_genetic_profiles": len(target_obs),
                "target_genetic_identifiers": sorted(target_obs.pert_id.astype(str).unique().tolist()),
                "target_uniprot_annotations": sorted(target_obs.perturbagen.astype(str).unique().tolist()),
                "candidate_library": len(candidates), "scoreable_candidates": len(scored),
                "saved_target_prediction_candidates": len(candidates & target_ids),
                "kiba_overlap_candidates": len(candidates & kiba_ids),
                "measured_endpoint_candidates": len(scored & measured_ids),
                "vehicle_controls": int(obs.pert_type.eq("ctl_vehicle.cns").sum()),
                "vector_controls": int(obs.pert_type.eq("ctl_vector.cns").sum())}
        summaries.append(item)
        if cell == config["primary_cell"]:
            if cell_id != config["cellosaurus_id"]:
                raise RuntimeError("Primary cell identifier disagrees")
            primary = item
            rows = []
            lincs_by_identity = {k: g for k, g in cell_compounds.groupby("chemical_identity")}
            for identity in sorted(kiba_ids | ctrp_ids):
                match_drugs = drugs_by_identity.get(identity, drugs.iloc[:0])
                match_lincs = lincs_by_identity.get(identity, cell_compounds.iloc[:0])
                reasons = []
                membership = (identity not in kiba_ids, "present_in_KIBA") if config.get("novel_only") else (identity in kiba_ids, "outside_KIBA")
                target_requirement = (not config.get("requires_saved_target_prediction", True) or identity in target_ids)
                for flag, reason in (membership,
                                     (identity in ctrp_ids, "no_CTRP_structure"),
                                     (identity in lincs_ids, "no_QC_LINCS_profile"),
                                     (identity in supported_ids, "outside_encoder_atom_capacity"),
                                     (target_requirement, "no_saved_target_OOF_prediction"),
                                     (identity in measured_ids, "no_finite_cellular_endpoint")):
                    if not flag:
                        reasons.append(reason)
                rows.append({"chemical_identity": identity,
                             "kiba_drug_ids": kiba_by_identity.get(identity, ""),
                             "ctrp_pubchem_ids": "|".join(sorted(match_drugs.pubchem_id.unique())),
                             "drug_names": "|".join(sorted(match_drugs.drug_name.dropna().unique())),
                             "lincs_pert_ids": "|".join(sorted(match_lincs.pert_id.unique())),
                             "in_candidate_library": identity in candidates,
                             "has_saved_target_prediction": identity in target_ids,
                             "has_endpoint": identity in measured_ids,
                             "included": not reasons, "exclusion_reasons": "|".join(reasons)})
            manifest = pd.DataFrame(rows)
            write_once(out / "candidate_coverage.csv", manifest.to_csv(index=False))
            query = pd.DataFrame({"row_id": range(len(scored)), "Canonical_SMILES": sorted(scored),
                                  "Sequence": target_sequences[0], "protein_id": config["target_uniprot"]})
            write_once(out / "query_table.csv", query.to_csv(index=False))
            selected = obs.loc[chemical & obs.condition.isin(set(cell_compounds.loc[cell_compounds.chemical_identity.isin(scored), "pert_id"]))
                               | (genetic & obs.condition.eq(config["target_gene"]))
                               | obs.pert_type.isin(["ctl_vehicle.cns", "ctl_vector.cns"])].copy()
            selected["matrix_row"] = obs.index.get_indexer(selected.index)
            fields = ["matrix_row", "sig_id", "condition", "pert_id", "pert_type", "perturbagen",
                      "qc_pass", "pert_dose", "pert_dose_unit", "pert_time", "pert_time_unit", "cellosaurus_id", "batch"]
            write_once(out / "selected_profile_metadata.csv", selected[fields].to_csv(index=False))
        data.file.close()
        sources.append(record(path))
        print(json.dumps(item), flush=True)
    archive = G2CP / "_zenodo/G2CP_saved_models.zip"
    report = {"status": "coverage_audit_before_scoring", "plan": record(out / "plan.json"),
              "sources": sources, "kiba_identities": len(kiba_ids), "ctrp_identities": len(ctrp_ids),
              "strict_kiba_ctrp_overlap": len(kiba_ids & ctrp_ids), "cells": summaries,
              "primary_coverage_sufficient": bool(primary["measured_endpoint_candidates"] >= config["minimum_measured_candidates"]
                                                  and primary["target_genetic_profiles"] >= 1),
              "g2cp_weights": {"local_bytes": archive.stat().st_size, "valid_zip": zipfile.is_zipfile(archive),
                               "inference_checkpoint_directory_exists": (G2CP / "saved_models").exists()},
              "endpoint_values_used_for_selection": False,
              "training_overlap": config.get("dta_prediction", "Use saved strict-identity test predictions only; verify candidate identities absent from each fold training/validation before scoring."),
              "existing_audit_reuse": "Do not reuse older cross-project concordance scores: filtered groupby positions must not index the unfiltered expression matrix. Resolve explicit absolute matrix rows for this analysis.",
              "outputs": [record(out / n) for n in ("candidate_coverage.csv", "selected_profile_metadata.csv", "query_table.csv")]}
    write_once(out / "coverage_audit.json", json.dumps(report, indent=2) + "\n")
    print(json.dumps({"primary_coverage_sufficient": report["primary_coverage_sufficient"],
                      "report": str(out / "coverage_audit.json")}), flush=True)


if __name__ == "__main__":
    main()
