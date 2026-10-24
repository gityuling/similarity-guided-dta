#!/usr/bin/env python3
"""Assess external exact-Kd coverage without loading any model predictions."""

import json

import numpy as np
import pandas as pd

from bib_common import PROJECT, canonical, record, table_path, write_once
from prepare_bindingdb_articles_kd import RAW_ZIP, SMILES, KD, CHAINS, SEQUENCE

OUTPUT = PROJECT / "reports/bib_external_feasibility_20261017"


def main():
    OUTPUT.mkdir(exist_ok=True)
    tables = {d: pd.read_csv(table_path(d)) for d in ("davis", "kiba")}
    drugs = {d: set(t.Canonical_SMILES.map({s: canonical(s) for s in t.Canonical_SMILES.unique()}))
             for d, t in tables.items()}
    sequences = {d: set(t.Sequence) for d, t in tables.items()}
    plan = {
        "scope": "Feasibility and overlap only; no external prediction scores inspected",
        "measurements": "Exact positive numeric Kd in nM; single protein chain with a sequence; valid canonical isomeric SMILES",
        "identity": "Same canonicalization as the benchmark freeze; retain salt, charge and tautomer states",
        "external_drugs": "Exclude every canonical identity in either full Davis or full KIBA benchmark",
        "known_target": "Require exact sequence membership in full Davis; KIBA-only targets are not eligible for pKd evaluation",
        "replicates": "Median pKd for each chemical identity and protein sequence, without label clipping",
        "feasibility_rule": "At least 100 eligible pairs, 50 chemical identities and 5 protein sequences to prioritize this panel; otherwise prepare scaffold stress test",
        "future_evaluation": "If feasible, freeze eligible table before scoring; equally average five original-seed Davis fold models and their validation-fitted component/fusion predictions. No external fitting or model selection. Explicitly report below-floor pKd behavior.",
        "independence_limit": "This BindingDB snapshot was used in earlier weak transfer experiments, although current scratch models have no BindingDB supervised initialization. It is not a prospectively untouched cohort; encoder pretraining overlap is unknown.",
        "inputs": [record(RAW_ZIP), *[record(table_path(d)) for d in tables]],
    }
    write_once(OUTPUT / "plan.json", json.dumps(plan, indent=2) + "\n")
    columns = [SMILES, KD, CHAINS, SEQUENCE, "Article DOI", "PMID", "Target Name",
               "BindingDB Reactant_set_id", "UniProt (SwissProt) Primary ID of Target Chain 1"]
    frames, counts = [], {"rows_read": 0, "exact_positive_kd": 0, "single_chain_sequence": 0, "valid_structure": 0}
    for chunk in pd.read_csv(RAW_ZIP, sep="\t", compression="zip", usecols=columns,
                             chunksize=100000, low_memory=False):
        counts["rows_read"] += len(chunk)
        kd = pd.to_numeric(chunk[KD].astype(str).str.strip(), errors="coerce")
        frame = chunk.loc[kd.gt(0)].copy()
        frame["kd_nM"] = kd.loc[frame.index]
        counts["exact_positive_kd"] += len(frame)
        chain = pd.to_numeric(frame[CHAINS], errors="coerce")
        seq = frame[SEQUENCE].fillna("").astype(str).str.strip()
        frame = frame.loc[chain.eq(1) & seq.ne("")].copy()
        frame["sequence"] = seq.loc[frame.index]
        counts["single_chain_sequence"] += len(frame)
        cache = {}
        for smiles in frame[SMILES].unique():
            try:
                cache[smiles] = canonical(str(smiles))
            except ValueError:
                cache[smiles] = None
        frame["chemical_identity"] = frame[SMILES].map(cache)
        frame = frame.dropna(subset=["chemical_identity"])
        counts["valid_structure"] += len(frame)
        frame["pKd"] = -np.log10(frame.kd_nM * 1e-9)
        frames.append(frame)
    source = pd.concat(frames, ignore_index=True)
    for d in tables:
        counts[f"rows_with_{d}_drug"] = int(source.chemical_identity.isin(drugs[d]).sum())
        counts[f"rows_with_exact_{d}_sequence"] = int(source.sequence.isin(sequences[d]).sum())
    source["novel_drug"] = ~source.chemical_identity.isin(drugs["davis"] | drugs["kiba"])
    eligible = source.loc[source.novel_drug & source.sequence.isin(sequences["davis"])].copy()
    eligible.to_csv(OUTPUT / "eligible_measurements.csv", index=False)
    aggregated = eligible.groupby(["chemical_identity", "sequence"], as_index=False).agg(
        pKd=("pKd", "median"), measurements=("pKd", "size"),
        dois=("Article DOI", lambda x: ";".join(sorted(set(x.dropna().astype(str))))),
        pmids=("PMID", lambda x: ";".join(sorted(set(x.dropna().astype(str))))),
        target_names=("Target Name", lambda x: ";".join(sorted(set(x.dropna().astype(str))))))
    aggregated.to_csv(OUTPUT / "eligible_pairs.csv", index=False)
    counts.update({"eligible_raw_measurements": len(eligible), "eligible_pairs": len(aggregated),
                   "eligible_identities": int(aggregated.chemical_identity.nunique()),
                   "eligible_exact_sequences": int(aggregated.sequence.nunique())})
    feasible = len(aggregated) >= 100 and aggregated.chemical_identity.nunique() >= 50 and aggregated.sequence.nunique() >= 5
    target_counts = aggregated.groupby("sequence").size().sort_values(ascending=False)
    target_counts.rename("pairs").to_csv(OUTPUT / "target_coverage.csv")
    audit = {"plan": record(OUTPUT / "plan.json"), "code": record(PROJECT / "scripts/audit_bib_external.py"),
             "counts": counts, "feasible_under_fixed_rule": bool(feasible),
             "next": "Freeze panel provenance and inference plan before any scoring" if feasible else "Prepare scaffold-grouped validation; external exact-sequence support is insufficient under fixed rule",
             "prediction_scores_loaded": False, "eligible_pairs": record(OUTPUT / "eligible_pairs.csv")}
    (OUTPUT / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"counts": counts, "feasible": bool(feasible), "next": audit["next"]}, indent=2))


if __name__ == "__main__":
    main()
