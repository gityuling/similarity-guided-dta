#!/usr/bin/env python3
"""Freeze the complete KIBA benchmark and DTIAM's official drug-ID outer folds."""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from prepare_dtiam_davis_protocol import inner_validation, outer_folds


PROJECT = Path(__file__).resolve().parents[1]


def write_frozen(path: Path, content: str) -> None:
    encoded = content.encode()
    if path.exists():
        if path.read_bytes() != encoded:
            raise RuntimeError(f"Frozen output differs; choose a new output directory: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(encoded)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dir", type=Path, default=PROJECT / "baselines/DTIAM/data/dta/kiba")
    parser.add_argument("--out-dir", type=Path, default=PROJECT / "data/processed/kiba_dtiam_protocol")
    parser.add_argument("--inner-seed", type=int, default=20261012)
    args = parser.parse_args()
    source = args.reference_dir
    ligands = json.loads((source / "ligands_can.txt").read_text(), object_pairs_hook=OrderedDict)
    proteins = json.loads((source / "proteins.txt").read_text(), object_pairs_hook=OrderedDict)
    with (source / "Y").open("rb") as handle:
        affinity = np.asarray(pickle.load(handle, encoding="latin1"), dtype=np.float64)
    if affinity.shape != (2111, 229) or (len(ligands), len(proteins)) != affinity.shape:
        raise ValueError(f"Unexpected KIBA matrix shape: {affinity.shape}")
    rows, cols = np.where(np.isfinite(affinity))
    if len(rows) != 118254:
        raise ValueError(f"Expected all 118254 KIBA pairs, found {len(rows)}")
    drug_ids, protein_ids = list(ligands), list(proteins)
    smiles = []
    for value in ligands.values():
        molecule = Chem.MolFromSmiles(value)
        if molecule is None:
            raise ValueError(f"Invalid benchmark SMILES: {value}")
        smiles.append(Chem.MolToSmiles(molecule, isomericSmiles=True))
    sequences = list(proteins.values())
    table = pd.DataFrame({
        "row_id": np.arange(len(rows)), "drug_idx": rows, "protein_idx": cols,
        "drug_id": [drug_ids[i] for i in rows], "protein_id": [protein_ids[i] for i in cols],
        "Canonical_SMILES": [smiles[i] for i in rows], "Sequence": [sequences[i] for i in cols],
        "kiba_score": affinity[rows, cols],
    })
    # The existing long table is an independent alignment check, never a subset filter.
    existing = pd.read_csv(PROJECT / "data/processed/kiba/kiba_deepdta_affinity_long.csv")
    if not (np.array_equal(existing.drug_index.to_numpy(), rows)
            and np.array_equal(existing.target_index.to_numpy(), cols)
            and np.allclose(existing.kiba_score, table.kiba_score, rtol=0, atol=1e-12)):
        raise RuntimeError("Existing KIBA table does not match the official pair order/labels")
    table_content = table.to_csv(index=False)
    summary = {
        "dataset": "KIBA complete DeepDTA/DTIAM benchmark", "label": "native kiba_score; no log transform or pKd floor",
        "pairs": len(table), "drug_ids": len(ligands), "canonical_drugs": len(set(smiles)),
        "protein_ids": len(proteins), "unique_sequences": len(set(sequences)),
        "outer_splitter": "KFold(n_splits=5, shuffle=True, random_state=0) over original drug IDs",
        "inner_validation": "10% of outer-train drug IDs, numpy.default_rng(inner_seed + fold)",
        "inner_seed": args.inner_seed,
        "source_sha256": {name: hashlib.sha256((source / name).read_bytes()).hexdigest()
                          for name in ("ligands_can.txt", "proteins.txt", "Y")},
        "table_sha256": hashlib.sha256(table_content.encode()).hexdigest(),
        "caveat": "Official drug-ID cold folds contain repeated canonical molecules across partitions; do not describe as strictly chemical-identity-disjoint.",
        "folds": [],
    }
    test_rows = []
    manifests = []
    for fold, (outer_train, outer_test) in enumerate(outer_folds(table, "drug_coldstart", 5, 0)):
        train_ids, val_ids = inner_validation(table.loc[outer_train], "drug_coldstart", fold, args.inner_seed)
        manifest = table[["row_id", "drug_idx", "protein_idx"]].copy()
        manifest["split"] = "test"
        manifest.loc[manifest.row_id.isin(train_ids), "split"] = "train"
        manifest.loc[manifest.row_id.isin(val_ids), "split"] = "val"
        if not np.array_equal(manifest.split.eq("test"), outer_test):
            raise RuntimeError("Official outer test rows changed")
        frames = {part: table.loc[manifest.split.eq(part)] for part in ("train", "val", "test")}
        sets = {part: set(frame.Canonical_SMILES) for part, frame in frames.items()}
        support = set(frames["train"].Sequence)
        unsupported = {part: int((~frames[part].Sequence.isin(support)).sum()) for part in ("val", "test")}
        if any(unsupported.values()):
            raise RuntimeError(f"Missing same-target prior support in fold {fold}: {unsupported}")
        content = manifest.to_csv(index=False)
        manifests.append((args.out_dir / "folds/drug_coldstart" / f"fold_{fold}.csv", content))
        summary["folds"].append({
            "fold": fold, "rows": {part: len(frame) for part, frame in frames.items()},
            "drug_ids": {part: int(frame.drug_idx.nunique()) for part, frame in frames.items()},
            "canonical_train_test_overlap": len(sets["train"] & sets["test"]),
            "canonical_train_val_overlap": len(sets["train"] & sets["val"]),
            "test_rows_with_train_canonical_drug": int(frames["test"].Canonical_SMILES.isin(sets["train"]).sum()),
            "unsupported_same_target_rows": unsupported,
            "manifest_sha256": hashlib.sha256(content.encode()).hexdigest(),
        })
        test_rows.extend(manifest.loc[manifest.split.eq("test"), "row_id"].tolist())
    if sorted(test_rows) != list(range(len(table))):
        raise RuntimeError("Outer tests do not cover every benchmark pair exactly once")
    write_frozen(args.out_dir / "table.csv", table_content)
    for path, content in manifests:
        write_frozen(path, content)
    write_frozen(args.out_dir / "summary.json", json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
