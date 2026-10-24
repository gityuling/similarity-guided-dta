#!/usr/bin/env python3
"""Freeze a scaffold-disjoint KIBA stress-test plan before model evaluation."""

import json
from itertools import combinations

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import KFold

from bib_common import PROJECT, canonical, record, table_path, write_once

OUTPUT = PROJECT / "data/processed/kiba_scaffold_20261017"


def main():
    table = pd.read_csv(table_path("kiba"))
    structures = {s: canonical(s) for s in table.Canonical_SMILES.unique()}
    scaffolds = {s: MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(s), includeChirality=False) or "__ACYCLIC__"
                 for s in set(structures.values())}
    table["chemical_identity"] = table.Canonical_SMILES.map(structures)
    table["scaffold"] = table.chemical_identity.map(scaffolds)
    groups = np.array(sorted(table.scaffold.unique()))
    OUTPUT.mkdir(exist_ok=True)
    config = json.loads((PROJECT / "configs/bib_validation_20261017.json").read_text())
    plan = {"dataset": "kiba", "group": "RDKit non-generic Bemis-Murcko scaffold, includeChirality=False; all acyclic compounds share one group; no salt/tautomer/charge normalization",
            "outer": "KFold(5, shuffle=True, random_state=0) over sorted unique scaffolds",
            "inner": "10% of outer-training scaffolds (rounded up), shuffled with seed 20261012+fold",
            "training_seed": 20261012, "training": config["training"], "table": record(table_path("kiba")),
            "models": ["deep", "neighbor_prior", "fixed_blend", "similarity_gate"],
            "selection": "Same frozen training and validation-only fitting rules; retain all planned folds; no result-based split selection",
            "comparison": "Matched local controls only; published drug-ID scores are not a same-protocol comparator",
            "status": "Prepared before predictions; execute after the repeated-seed and identity queue"}
    write_once(OUTPUT / "plan.json", json.dumps(plan, indent=2) + "\n")
    folds, tested = [], []
    for fold, (train_indices, test_indices) in enumerate(KFold(5, shuffle=True, random_state=0).split(groups)):
        remaining = groups[train_indices].copy()
        np.random.default_rng(20261012 + fold).shuffle(remaining)
        count = max(1, int(np.ceil(0.1 * len(remaining))))
        partitions = {"test": set(groups[test_indices]), "val": set(remaining[:count]), "train": set(remaining[count:])}
        for a, b in combinations(partitions, 2):
            if partitions[a] & partitions[b]:
                raise RuntimeError("Cross-partition scaffold overlap")
        manifest = table[["row_id"]].copy()
        mapping = {g: split for split, values in partitions.items() for g in values}
        manifest["split"] = table.scaffold.map(mapping)
        frames = {s: table.loc[manifest.split.eq(s)] for s in partitions}
        same_target = set(frames["train"].Sequence)
        unsupported = {s: int((~frames[s].Sequence.isin(same_target)).sum()) for s in ("val", "test")}
        if any(unsupported.values()):
            raise RuntimeError(f"Scaffold protocol has unsupported target queries: {unsupported}")
        identity_overlaps = {f"{a}/{b}": len(set(frames[a].chemical_identity) & set(frames[b].chemical_identity))
                             for a, b in combinations(partitions, 2)}
        if any(identity_overlaps.values()):
            raise RuntimeError("Identity overlap despite scaffold grouping")
        path = OUTPUT / "folds/drug_coldstart" / f"fold_{fold}.csv"
        write_once(path, manifest.to_csv(index=False))
        folds.append({"fold": fold, "manifest": record(path), "rows": {s: len(f) for s, f in frames.items()},
                      "scaffolds": {s: int(f.scaffold.nunique()) for s, f in frames.items()},
                      "chemical_identity_overlap": identity_overlaps,
                      "scaffold_overlap": {f"{a}/{b}": 0 for a, b in combinations(partitions, 2)},
                      "unsupported_same_target_rows": unsupported})
        tested.extend(frames["test"].row_id.tolist())
    if sorted(tested) != sorted(table.row_id):
        raise RuntimeError("Scaffold tests do not cover all pairs exactly once")
    keys = table[["drug_idx", "drug_id", "chemical_identity", "scaffold"]].drop_duplicates().sort_values("drug_idx")
    write_once(OUTPUT / "drug_scaffolds.csv", keys.to_csv(index=False))
    summary = {"plan": record(OUTPUT / "plan.json"), "pairs": len(table), "identities": len(scaffolds),
               "scaffolds": len(groups), "acyclic_identities": sum(s == "__ACYCLIC__" for s in scaffolds.values()),
               "pairs_removed": 0, "every_test_row_once": True, "folds": folds}
    write_once(OUTPUT / "summary.json", json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"pairs": len(table), "scaffolds": len(groups), "fold_test_pairs": [f["rows"]["test"] for f in folds],
                      "all_scaffold_intersections_zero": True, "training": "pending"}, indent=2))


if __name__ == "__main__":
    main()
