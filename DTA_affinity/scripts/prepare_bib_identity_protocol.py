#!/usr/bin/env python3
"""Create canonical-identity-grouped KIBA splits without removing pairs."""

import json
from itertools import combinations

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from bib_common import PROJECT, canonical, record, table_path, write_once


def main() -> None:
    table_file = table_path("kiba")
    table = pd.read_csv(table_file)
    table["chemical_identity"] = table.Canonical_SMILES.map(
        {s: canonical(s) for s in table.Canonical_SMILES.unique()})
    groups = np.array(sorted(table.chemical_identity.unique()))
    output = PROJECT / "data/processed/kiba_identity_20261017"
    summary = {"group_definition": "RDKit canonical isomeric SMILES; retain salts, charge and tautomer state",
               "table": record(table_file), "pairs": len(table), "drug_ids": int(table.drug_idx.nunique()),
               "chemical_identities": len(groups), "outer_seed": 0, "inner_seed": 20261012,
               "pairs_removed": 0, "folds": []}
    tests = []
    for fold, (outer_train, outer_test) in enumerate(KFold(5, shuffle=True, random_state=0).split(groups)):
        remaining = groups[outer_train].copy()
        np.random.default_rng(20261012 + fold).shuffle(remaining)
        n_val = max(1, int(round(0.10 * len(remaining))))
        sets = {"train": set(remaining[n_val:]), "val": set(remaining[:n_val]), "test": set(groups[outer_test])}
        manifest = table[["row_id", "drug_idx", "protein_idx"]].copy()
        manifest["split"] = ""
        for part, identities in sets.items():
            manifest.loc[table.chemical_identity.isin(identities), "split"] = part
        if set(manifest.split) != set(sets):
            raise RuntimeError("Incomplete identity-group split")
        overlap = {f"{a}/{b}": len(sets[a] & sets[b]) for a, b in combinations(sets, 2)}
        if any(overlap.values()):
            raise RuntimeError("Canonical identity overlaps partitions")
        support = set(table.loc[manifest.split.eq("train"), "Sequence"])
        unsupported = {p: int((~table.loc[manifest.split.eq(p), "Sequence"].isin(support)).sum())
                       for p in ("val", "test")}
        if any(unsupported.values()):
            raise RuntimeError(f"Same-target support missing: {unsupported}")
        path = output / f"folds/drug_coldstart/fold_{fold}.csv"
        write_once(path, manifest.to_csv(index=False))
        tests.extend(manifest.loc[manifest.split.eq("test"), "row_id"])
        summary["folds"].append({"fold": fold, "manifest": record(path), "canonical_overlap": overlap,
                                 "rows": {p: int(manifest.split.eq(p).sum()) for p in sets},
                                 "identities": {p: len(v) for p, v in sets.items()},
                                 "unsupported_same_target_rows": unsupported})
    if sorted(tests) != sorted(table.row_id):
        raise RuntimeError("Outer test coverage is not exactly once")
    summary["every_test_row_once"] = True
    write_once(output / "summary.json", json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"prepared": str(output), "pairs": len(table), "identities": len(groups),
                      "canonical_overlap": [f["canonical_overlap"] for f in summary["folds"]]}, indent=2))


if __name__ == "__main__":
    main()
