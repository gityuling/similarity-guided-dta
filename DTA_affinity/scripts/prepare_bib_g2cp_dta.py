#!/usr/bin/env python3
"""Freeze a global parent-disjoint G2CP evaluation before model scoring."""

import hashlib
import json

import anndata as ad
import numpy as np
import pandas as pd
from rdkit import Chem

from audit_bib_cellular_case import treated_masks
from bib_g2cp_common import CONFIG, COVERAGE, DATA, KIBA, OUT, PROJECT, SOURCE, plan, record, save_json, write_once


def ordered(values, salt):
    return sorted(set(values), key=lambda value: hashlib.sha256(f"{salt}:{value}".encode()).hexdigest())


def assign_groups(values, fractions, salt):
    values = ordered(values, salt)
    n_train, n_val = int(len(values) * fractions[0]), int(len(values) * fractions[1])
    return {value: ("train" if i < n_train else "val" if i < n_train + n_val else "test")
            for i, value in enumerate(values)}


def dense(value):
    return np.asarray(value.toarray() if hasattr(value, "toarray") else value, dtype=np.float32)


def save_npz(path, **arrays):
    if path.exists():
        with np.load(path, allow_pickle=False) as old:
            if set(old.files) != set(arrays) or any(not np.array_equal(old[k], v) for k, v in arrays.items()):
                raise RuntimeError(f"Existing arrays differ: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **arrays)


def main():
    cfg = plan()
    OUT.mkdir(parents=True, exist_ok=True)
    sources = [CONFIG, COVERAGE / "coverage_audit.json", COVERAGE / "chemical_annotation_coverage.csv",
               COVERAGE / "genetic_target_coverage.csv", KIBA / "table.csv",
               PROJECT / "scripts/prepare_bib_g2cp_dta.py"]
    chemical = pd.read_csv(COVERAGE / "chemical_annotation_coverage.csv").fillna("")
    targets = pd.read_csv(COVERAGE / "genetic_target_coverage.csv")
    compounds = chemical.drop_duplicates("pert_id").copy()
    candidates = chemical.loc[chemical.preliminary_eligible].drop_duplicates("pert_id").copy()
    candidates["review_exclusion"] = candidates.pert_id.map(cfg["annotation_exclusions"]).fillna("")
    candidates["explicit_H_atoms"] = candidates.canonical_identity.map(lambda s: Chem.AddHs(Chem.MolFromSmiles(s)).GetNumAtoms())
    candidates.loc[candidates.explicit_H_atoms.gt(512), "review_exclusion"] = "outside encoder capacity"
    candidates["mechanism_consistent"] = candidates.review_exclusion.eq("")
    write_once(OUT / "annotation_review.csv", candidates.to_csv(index=False))
    accepted = candidates.loc[candidates.mechanism_consistent].copy()
    if accepted.parent_key.duplicated().any():
        raise RuntimeError("Resolve multiple accepted forms of a parent before splitting")
    split = {}
    for external in (True, False):
        values = accepted.loc[accepted.davis_external.eq(external), "parent_key"]
        split.update(assign_groups(values, cfg["candidate_split_fractions"], f"{cfg['split_seed']}:candidate:{external}"))
    split.update(assign_groups(set(compounds.parent_key) - set(split), cfg["other_chemical_split_fractions"],
                               f"{cfg['split_seed']}:other"))
    compounds["split"] = compounds.parent_key.map(split)
    accepted["split"] = accepted.parent_key.map(split)
    write_once(OUT / "chemical_split.csv", compounds[["pert_id", "canonical_identity", "parent_key", "split"]].to_csv(index=False))
    write_once(OUT / "candidates.csv", accepted.to_csv(index=False))
    genes_all = set()
    source_records = [record(p) for p in sources]
    for cell in cfg["cells"]:
        path = SOURCE / f"data/LINCS/{cell}/{cell}_gene_chem_ctrl_adata.h5ad"
        source_records.append(record(path))
        a = ad.read_h5ad(path, backed="r")
        _, mask = treated_masks(a.obs, cfg["conditions"])
        genes_all.update(a.obs.loc[mask, "condition"].astype(str))
        a.file.close()
    genes_order = ordered(genes_all, f"{cfg['split_seed']}:genes")
    gene_val = set(genes_order[:max(1, int(len(genes_order) * cfg["genetic_validation_fraction"]))])
    gene_split = {g: "pretrain_val" if g in gene_val else "pretrain_train" for g in genes_order}
    write_once(OUT / "genetic_split.csv", pd.DataFrame(sorted(gene_split.items()), columns=["gene", "split"]).to_csv(index=False))
    outputs, summaries, library_audits, exclusions = [], [], [], []
    query_targets = set()
    reference_genes = None
    for cell in cfg["cells"]:
        a = ad.read_h5ad(SOURCE / f"data/LINCS/{cell}/{cell}_gene_chem_ctrl_adata.h5ad")
        obs = a.obs.copy().reset_index(drop=True)
        obs["source_row"] = np.arange(len(obs))
        names = a.var_names.astype(str).to_numpy()
        if reference_genes is None:
            reference_genes = names
        if not np.array_equal(reference_genes, names) or len(names) != 978 or obs.sig_id.duplicated().any():
            raise RuntimeError("Unexpected genes or duplicate signatures")
        cp, ge = treated_masks(obs, cfg["conditions"])
        mapping = compounds.set_index("pert_id")
        cp_obs = obs.loc[cp & obs.pert_id.isin(mapping.index)].copy()
        for key in ("canonical_identity", "parent_key", "split"):
            cp_obs[key] = cp_obs.pert_id.map(mapping[key])
        ge_obs = obs.loc[ge].copy()
        ge_obs["split"] = ge_obs.condition.astype(str).map(gene_split)
        reps = a.uns["UniPert_reps"]
        keep = ["source_row", "sig_id", "condition", "pert_id", "pert_type", "split", "canonical_identity", "parent_key"]
        selected = pd.concat([cp_obs, ge_obs], ignore_index=True)[keep].copy()
        for key in keep[1:]:
            selected[key] = selected[key].astype(object).fillna("").astype(str)
        missing = ~selected.condition.astype(str).isin(reps)
        for row in selected.loc[missing].itertuples():
            exclusions.append({"cell": cell, "sig_id": row.sig_id, "reason": "missing_UniPert_representation"})
        selected = selected.loc[~missing].reset_index(drop=True)
        y = dense(a.X[selected.source_row.to_numpy()])
        representations = np.asarray([reps[str(v)] for v in selected.condition], dtype=np.float32)
        if representations.shape != (len(selected), 256) or not np.isfinite(y).all() or not np.isfinite(representations).all():
            raise RuntimeError("Invalid response or representation")
        # Only genetic training data supply the auxiliary library for all comparators.
        available = targets.loc[targets.cell.eq(cell)]
        library, proteins = [], []
        for row in available.itertuples():
            gene_names = set(row.gene_names.split("|"))
            current = selected.loc[selected.split.eq("pretrain_train") & selected.condition.isin(gene_names)]
            guides = [y[group.index].mean(0) for _, group in current.groupby("pert_id", observed=True)]
            median = None
            if len(guides) >= cfg["minimum_guides"]:
                g = np.asarray(guides, dtype=np.float64)
                cor = np.corrcoef(g)
                upper = cor[np.triu_indices(len(g), 1)]
                if np.isfinite(upper).all():
                    median = float(np.median(upper))
            include = median is not None and median >= cfg["minimum_median_guide_pearson"]
            library_audits.append({"cell": cell, "protein_id": row.protein_id, "genes": row.gene_names,
                                   "training_guides": len(guides), "median_guide_pearson": median, "included": include})
            if include:
                proteins.append(row.protein_id)
                library.append(np.mean(guides, axis=0))
        if len(proteins) < cfg["top_k"]:
            raise RuntimeError("Insufficient genetic training library")
        query_targets.update(proteins)
        eligible_ids = set(accepted.canonical_identity)
        eval_rows, truth = [], []
        eligible_obs = selected.loc[selected.pert_type.eq("trt_cp") & selected.canonical_identity.isin(eligible_ids)]
        for identity, group in eligible_obs.groupby("canonical_identity", sort=True):
            info = accepted.loc[accepted.canonical_identity.eq(identity)].iloc[0]
            # Equal weight per BRD ID, then per unique released signature.
            means = [y[part.index].mean(0) for _, part in group.groupby("pert_id", observed=True)]
            eval_rows.append({"cell": cell, "canonical_identity": identity, "parent_key": info.parent_key,
                              "names": info.names, "split": info.split, "davis_external": bool(info.davis_external),
                              "pert_ids": "|".join(sorted(group.pert_id.unique())), "signature_count": len(group),
                              "matrix_row": len(truth)})
            truth.append(np.mean(means, axis=0))
        controls = dense(a.X[obs.condition.eq("ctrl").to_numpy()])
        if len(controls) == 0 or not np.isfinite(controls).all():
            raise RuntimeError("Invalid released control inputs")
        arrays = DATA / f"{cell}.npz"
        save_npz(arrays, y=y, representations=representations, controls=controls,
                 gene_names=names.astype(str), library=np.asarray(library, dtype=np.float32),
                 proteins=np.asarray(proteins, dtype=str), truth=np.asarray(truth, dtype=np.float32))
        meta = DATA / f"{cell}_rows.csv"
        write_once(meta, selected[keep].to_csv(index=False))
        ev = OUT / f"{cell}_evaluation.csv"
        frame = pd.DataFrame(eval_rows)
        write_once(ev, frame.to_csv(index=False))
        outputs.extend([arrays, meta, ev])
        entry = {"cell": cell, "rows_by_split": selected.groupby("split").size().to_dict(),
                 "library_targets": len(proteins), "evaluation_drugs_by_split": frame.groupby("split").size().to_dict(),
                 "primary_drugs_by_split": frame.loc[frame.davis_external].groupby("split").size().to_dict()}
        summaries.append(entry)
        print(json.dumps(entry), flush=True)
    write_once(OUT / "genetic_library_audit.csv", pd.DataFrame(library_audits).to_csv(index=False))
    write_once(OUT / "representation_exclusions.csv", pd.DataFrame(exclusions, columns=["cell", "sig_id", "reason"]).to_csv(index=False))
    kiba = pd.read_csv(KIBA / "table.csv", usecols=["protein_id", "Sequence"]).drop_duplicates()
    proteins = kiba.loc[kiba.protein_id.isin(query_targets)].sort_values("protein_id")
    if proteins.protein_id.duplicated().any() or set(proteins.protein_id) != query_targets:
        raise RuntimeError("Target sequence mapping is not one-to-one")
    query = accepted[["canonical_identity", "parent_key", "names"]].rename(columns={"canonical_identity": "Canonical_SMILES"}).merge(proteins, how="cross")
    query.insert(0, "row_id", np.arange(len(query)))
    write_once(OUT / "query_table.csv", query.to_csv(index=False))
    outputs.extend(OUT / name for name in ("annotation_review.csv", "candidates.csv", "chemical_split.csv", "genetic_split.csv",
                                           "genetic_library_audit.csv", "representation_exclusions.csv", "query_table.csv"))
    for source in source_records:
        from bib_g2cp_common import ROOT
        if record(ROOT / source["path"]) != source:
            raise RuntimeError("Source changed during preparation")
    save_json(OUT / "data_audit.json", {"plan": cfg, "sources": source_records, "files": [record(p) for p in outputs],
              "accepted_candidates": len(accepted), "primary_candidates": int(accepted.davis_external.sum()),
              "rejected_annotations": int((~candidates.mechanism_consistent).sum()), "cells": summaries,
              "models_scored": False, "chemical_parent_split_global": True,
              "limitations": ["Annotations reviewed for consistency, not experimentally revalidated.",
                              "UniPert representation pretraining exposure is not fully reconstructable.",
                              "Processed signatures reused without claiming a complete upstream normalization reconstruction."]}, immutable=True)


if __name__ == "__main__":
    main()
