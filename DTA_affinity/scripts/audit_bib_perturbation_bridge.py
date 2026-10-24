#!/usr/bin/env python3
"""Inventory Results 5 coverage without loading expression or model scores."""

import json
from collections import defaultdict
from functools import lru_cache

import anndata as ad
import numpy as np
import pandas as pd
from rdkit import RDLogger

from audit_bib_cellular_case import treated_masks
from bib_common import PROJECT, ROOT, canonical, record, write_once
from prepare_bib_nanobret import parent_key


OUT = PROJECT / "reports/bib_perturbation_bridge_20261019"
NB = PROJECT / "reports/bib_nanobret_20261018"
DATA = ROOT / "UniPert-G2CP_reproduce/G2CP/data"
CELLS = ("A375", "A549", "MCF7", "PC3", "HT29")
CONDITIONS = {
    "chemical_pert_type": "trt_cp", "chemical_time_hours": 24,
    "chemical_dose_um": 10, "genetic_pert_type": "trt_xpr",
    "genetic_time_hours": 96,
}


@lru_cache(maxsize=None)
def structure(smiles):
    try:
        identity = canonical(smiles)
        return identity, parent_key(identity)
    except (ValueError, RuntimeError):
        return "", ""


def main():
    RDLogger.DisableLog("rdApp.*")
    paths = {
        "compounds": DATA / "processed_CMAP_compound_info.csv",
        "annotations": DATA / "new_Touchstone.csv",
        "proteins": NB / "protein_annotation_audit.csv",
        "nanobret_compounds": NB / "compound_audit.csv",
        "nanobret_queries": NB / "query_table.csv",
        "kiba": PROJECT / "data/processed/kiba_dtiam_protocol/table.csv",
        "davis": PROJECT / "data/processed/davis_dtiam_protocol/table.csv",
    }
    snapshots = {name: record(path) for name, path in paths.items()}
    drug_sets = {}
    for name in ("kiba", "davis"):
        smiles = pd.read_csv(paths[name], usecols=["Canonical_SMILES"]).Canonical_SMILES.unique()
        values = [structure(s) for s in smiles]
        if any(not identity or not parent for identity, parent in values):
            raise RuntimeError(f"Invalid {name} reference identity")
        drug_sets[name] = ({v[0] for v in values}, {v[1] for v in values})

    compounds = pd.read_csv(paths["compounds"], usecols=["pert_id", "canonical_smiles"])
    compounds["canonical_identity"], compounds["parent_key"] = zip(*[
        structure(s) if isinstance(s, str) else ("", "") for s in compounds.canonical_smiles
    ])
    invalid = int(compounds.canonical_identity.eq("").sum())
    compounds = compounds.loc[compounds.canonical_identity.ne("")].copy()
    ambiguous = compounds.groupby("pert_id").canonical_identity.nunique()
    ambiguous_ids = set(ambiguous[ambiguous.gt(1)].index)
    compounds = compounds.loc[~compounds.pert_id.isin(ambiguous_ids)].drop_duplicates("pert_id")
    for name, (identities, parents) in drug_sets.items():
        compounds[f"{name}_external"] = ~(compounds.canonical_identity.isin(identities)
                                                      | compounds.parent_key.isin(parents))

    proteins = pd.read_csv(paths["proteins"])
    aliases = defaultdict(set)
    sequence_ok = set(proteins.loc[proteins.canonical_sequence_matches, "protein_id"])
    for row in proteins.to_dict("records"):
        if row["Organism (ID)"] == 9606:
            for alias in str(row["Gene Names"]).split():
                aliases[alias.upper()].add(row["protein_id"])

    def map_gene(gene):
        matches = aliases.get(str(gene).upper(), set())
        return next(iter(matches)) if len(matches) == 1 and matches.issubset(sequence_ok) else ""

    annotations = pd.read_csv(paths["annotations"]).fillna("")
    ann = defaultdict(list)
    for row in annotations.loc[annotations.Type.eq("trt_cp")].to_dict("records"):
        ann[row["Query ID"]].append(row)
    annotation_rows = []
    for row in compounds.to_dict("records"):
        records = ann.get(row["pert_id"], [])
        # Curated inhibition and structure agreement are coverage flags, not validated mechanisms.
        supported = []
        for item in records:
            identity, parent = structure(item["SMILES"]) if item["SMILES"] else ("", "")
            if identity == row["canonical_identity"]:
                supported.append(item)
        moa = "|".join(sorted({x["Mechanism of action"] for x in supported}))
        genes = sorted({g for x in supported for g in x["gene_targets"].split("|") if g})
        targets = sorted({map_gene(g) for g in genes} - {""})
        lower = moa.lower()
        inhibitor = "inhibitor" in lower and not any(x in lower for x in ("agonist", "activator", "degrader"))
        annotation_rows.append({
            **row, "names": "|".join(sorted({x["Name"] for x in supported})),
            "moa": moa, "annotation_structure_exact": bool(supported),
            "annotation_rows": len(records), "gene_targets": "|".join(genes),
            "mapped_kiba_targets": "|".join(targets),
            "inhibitor_with_kiba_annotation": bool(inhibitor and targets),
        })
    annotated = pd.DataFrame(annotation_rows)
    nb_drugs = pd.read_csv(paths["nanobret_compounds"]).fillna("")
    nb_queries = pd.read_csv(paths["nanobret_queries"], usecols=["compound", "protein_id"])
    genetic_rows, chemical_rows, nb_rows, cells = [], [], [], []
    reference_genes = None
    for cell in CELLS:
        path = DATA / "LINCS" / cell / f"{cell}_gene_chem_ctrl_adata.h5ad"
        before = record(path)
        data = ad.read_h5ad(path, backed="r")
        obs = data.obs.copy()
        genes = data.var_names.astype(str).tolist()
        data.file.close()
        if len(genes) != 978 or len(set(genes)) != 978:
            raise RuntimeError(f"Unexpected landmark gene set: {cell}")
        if reference_genes is None:
            reference_genes = genes
        if genes != reference_genes or obs.sig_id.duplicated().any():
            raise RuntimeError(f"Gene order or signature identity mismatch: {cell}")
        chemical, genetic = treated_masks(obs, CONDITIONS)
        gen = obs.loc[genetic].copy()
        gen["protein_id"] = gen.condition.map(map_gene)
        mapped = gen.loc[gen.protein_id.ne("")]
        usable_targets = set()
        for protein, group in mapped.groupby("protein_id", observed=True):
            n_guides = group.pert_id.nunique()
            if n_guides >= 2:
                usable_targets.add(protein)
            genetic_rows.append({"cell": cell, "protein_id": protein,
                                 "gene_names": "|".join(sorted(group.condition.astype(str).unique())),
                                 "signature_rows": len(group), "distinct_guides": n_guides,
                                 "at_least_two_guides": n_guides >= 2})
        chem_obs = obs.loc[chemical]
        candidates = annotated.loc[annotated.pert_id.isin(chem_obs.pert_id)].copy()
        candidates["cell"] = cell
        candidates["annotated_targets_with_two_guides"] = candidates.mapped_kiba_targets.map(
            lambda value: "|".join(sorted(set(value.split("|")) & usable_targets)))
        candidates["preliminary_eligible"] = (candidates.kiba_external
            & candidates.inhibitor_with_kiba_annotation
            & candidates.annotated_targets_with_two_guides.ne(""))
        chemical_rows.extend(candidates.to_dict("records"))
        for row in nb_drugs.to_dict("records"):
            exact = compounds.loc[compounds.canonical_identity.eq(row["Canonical_SMILES"])]
            parent = compounds.loc[compounds.parent_key.eq(row["parent_connectivity_key"])]
            current_ids = set(exact.pert_id) & set(chem_obs.pert_id)
            available = set(nb_queries.loc[nb_queries.compound.eq(row["compound"]), "protein_id"])
            nb_rows.append({"cell": cell, "compound": row["compound"],
                            "exact_pert_ids": "|".join(sorted(exact.pert_id)),
                            "parent_pert_ids": "|".join(sorted(parent.pert_id)),
                            "available_exact_pert_ids": "|".join(sorted(current_ids)),
                            "chemical_signature_rows": int(chem_obs.pert_id.isin(current_ids).sum()),
                            "existing_predicted_targets_with_two_guides": len(available & usable_targets),
                            "davis_external": row["davis_external"]})
        eligible = candidates.loc[candidates.preliminary_eligible]
        cells.append({"cell": cell, "chemical_signatures": int(chemical.sum()),
                      "genetic_signatures": int(genetic.sum()), "kiba_targets_any_guide": mapped.protein_id.nunique(),
                      "kiba_targets_two_guides": len(usable_targets),
                      "preliminary_eligible_identities": eligible.canonical_identity.nunique(),
                      "preliminary_eligible_davis_external": eligible.loc[eligible.davis_external].canonical_identity.nunique(),
                      "nanobret_drugs_with_exact_chemical_profiles": sum(
                          bool(x["available_exact_pert_ids"]) for x in nb_rows if x["cell"] == cell)})
        if record(path) != before:
            raise RuntimeError(f"Source changed during audit: {cell}")
        snapshots[cell] = before
        print(json.dumps(cells[-1]), flush=True)

    chemical_frame = pd.DataFrame(chemical_rows)
    for name, path in paths.items():
        if record(path) != snapshots[name]:
            raise RuntimeError(f"Source changed during audit: {name}")
    tables = {"cell_coverage.csv": pd.DataFrame(cells),
              "genetic_target_coverage.csv": pd.DataFrame(genetic_rows),
              "chemical_annotation_coverage.csv": chemical_frame,
              "nanobret_overlap.csv": pd.DataFrame(nb_rows)}
    for name, table in tables.items():
        write_once(OUT / name, table.to_csv(index=False))
    report = {
        "stage": "metadata_only_feasibility", "expression_values_read": False,
        "model_scores_read": False, "gpu_used": False, "conditions": CONDITIONS,
        "landmark_genes": 978, "cells": cells,
        "invalid_compound_structures": invalid, "ambiguous_pert_ids": sorted(ambiguous_ids),
        "preliminary_eligible_unique_identities": int(chemical_frame.loc[
            chemical_frame.preliminary_eligible, "canonical_identity"].nunique()),
        "sources": snapshots, "code": record(PROJECT / "scripts/audit_bib_perturbation_bridge.py"),
        "outputs": [record(OUT / name) for name in tables],
        "limitations": [
            "Annotation flags are preliminary; compound-specific inhibition and target coverage need review.",
            "Two distinct guide IDs do not imply agreement, knockdown efficacy or independent biological replicates.",
            "Only metadata were inspected; no expression or prediction performance selected this inventory.",
            "Chemical/genetic exposures are 24/96 h; genetic loss and pharmacological inhibition differ.",
            "LINCS availability does not establish independence from earlier workspace analyses.",
            "Parent-only matches are reported but not merged into NanoBRET exact-identity cases.",
        ],
    }
    write_once(OUT / "coverage_audit.json", json.dumps(report, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
