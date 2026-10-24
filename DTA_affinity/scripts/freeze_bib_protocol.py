#!/usr/bin/env python3
"""Freeze data, splits, cached representations and existing prediction evidence."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from itertools import combinations
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
from rdkit import rdBase

from bib_common import (DATASETS, FREEZE, PROJECT, ROOT, baseline_paths, canonical,
                        check_predictions, manifest_path, record, relative,
                        sha256, table_path, verify_freeze, write_once)
from prepare_dtiam_davis_protocol import outer_folds, inner_validation


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify or (FREEZE / "protocol.json").exists():
        result = verify_freeze()
        print(json.dumps({"verified": True, "datasets": list(result["datasets"])}))
        return
    files, features, datasets = set(), [], {}
    for dataset, config in DATASETS.items():
        path = table_path(dataset)
        table = pd.read_csv(path)
        if (len(table) != config["pairs"] or not table.row_id.is_unique
                or table.drug_idx.nunique() != config["drug_ids"]):
            raise RuntimeError(f"Unexpected benchmark: {dataset}")
        table["chemical_identity"] = table.Canonical_SMILES.map(
            {s: canonical(s) for s in table.Canonical_SMILES.unique()})
        source = PROJECT / f"baselines/DTIAM/data/dta/{dataset}"
        files.update(source / name for name in ("Y", "ligands_can.txt", "proteins.txt"))
        files.add(path)
        fold_records, test_ids = [], []
        expected_outer = list(outer_folds(table, "drug_coldstart", 5, 0))
        for fold in range(5):
            manifest_file = manifest_path(dataset, fold)
            manifest = pd.read_csv(manifest_file)
            if not manifest.row_id.is_unique or set(manifest.row_id) != set(table.row_id):
                raise RuntimeError("Manifest does not cover table exactly once")
            joined = table.merge(manifest[["row_id", "split"]], on="row_id", validate="one_to_one")
            if set(joined.split) != {"train", "val", "test"}:
                raise RuntimeError("Invalid manifest split names")
            expected_train, expected_test = expected_outer[fold]
            if not np.array_equal(joined.split.eq("test"), expected_test):
                raise RuntimeError("Outer drug-ID split differs from official algorithm")
            train_rows, val_rows = inner_validation(table.loc[expected_train], "drug_coldstart", fold, 20261012)
            if (set(joined.loc[joined.split.eq("train"), "row_id"]) != set(train_rows)
                    or set(joined.loc[joined.split.eq("val"), "row_id"]) != set(val_rows)):
                raise RuntimeError("Inner validation changed")
            frames = {p: joined.loc[joined.split.eq(p)] for p in ("train", "val", "test")}
            ids = {p: set(frame.drug_idx) for p, frame in frames.items()}
            structures = {p: set(frame.chemical_identity) for p, frame in frames.items()}
            for a, b in combinations(frames, 2):
                if ids[a] & ids[b]:
                    raise RuntimeError("Drug-ID split has cross-partition drug IDs")
            supported = set(frames["train"].Sequence)
            unsupported = {p: int((~frames[p].Sequence.isin(supported)).sum()) for p in ("val", "test")}
            if any(unsupported.values()):
                raise RuntimeError("Missing same-target neighbor support")
            paths = baseline_paths(dataset, fold)
            for key in ("deep_predictions", "prior_predictions", "fusion_predictions"):
                check_predictions(paths[key], table, manifest, config["label"])
            checkpoint = torch.load(paths["deep_checkpoint"], map_location="cpu", weights_only=False)
            options = checkpoint["args"]
            if options["seed"] != 20261012 or options["init_checkpoint"] or options["encoder_init_checkpoint"]:
                raise RuntimeError("Existing baseline is not the expected scratch run")
            prior = json.loads(paths["prior_result"].read_text())
            fold_records.append({
                "fold": fold, "manifest": relative(manifest_file),
                "rows": {p: len(f) for p, f in frames.items()},
                "drug_ids": {p: int(f.drug_idx.nunique()) for p, f in frames.items()},
                "canonical_overlap": {f"{a}/{b}": len(structures[a] & structures[b]) for a, b in combinations(frames, 2)},
                "unsupported_same_target_rows": unsupported,
                "checkpoint_args": options,
                "prior_hyperparameters": prior["selected_hyperparameters"],
                "artifacts": {k: relative(v) for k, v in paths.items()},
            })
            files.add(manifest_file)
            files.update(paths.values())
            test_ids.extend(frames["test"].row_id.tolist())
        if sorted(test_ids) != sorted(table.row_id):
            raise RuntimeError("Outer test coverage is not exactly once")
        for kind, values, width in (("drugs", table.chemical_identity.unique(), 1536),
                                    ("proteins", table.Sequence.unique(), 1280)):
            for value in sorted(values):
                token_file = config["tokens"] / kind / f"{hashlib.md5(value.encode()).hexdigest()}.npy"
                array = np.load(token_file, mmap_mode="r")
                if array.ndim != 2 or array.shape[1] != width or not np.isfinite(array).all():
                    raise RuntimeError(f"Invalid cached representation: {token_file}")
                features.append({"dataset": dataset, "kind": kind, **record(token_file),
                                 "tokens": array.shape[0], "width": width})
        datasets[dataset] = {
            "table": relative(path), "label": config["label"], "pairs": len(table),
            "drug_ids": int(table.drug_idx.nunique()),
            "canonical_structures": int(table.chemical_identity.nunique()),
            "protein_ids": int(table.protein_idx.nunique()),
            "unique_sequences": int(table.Sequence.nunique()),
            "label_min": float(table[config["label"]].min()),
            "label_max": float(table[config["label"]].max()),
            "label_scale": "-log10(Kd_nM / 1e9); retain pKd=5 floor" if dataset == "davis" else "native KIBA score; no log transform",
            "features": relative(config["tokens"]),
            "feature_provenance": "Legacy cache; per-file hash/shape verified, original checkpoint lineage not independently recoverable" if dataset == "davis" else "Fresh local-weight inference; build_audit.json records model and extractor hashes",
            "folds": fold_records, "every_test_row_once": True,
        }
        print(f"Verified {dataset}: {len(table)} pairs; {len(features)} cumulative feature files", flush=True)
    files.update(PROJECT / p for p in (
        "reports/davis_supplement_20261016.json", "reports/davis_supplement_20261016.csv",
        "reports/dtiam_published_comparison_20261016.json", "data/raw/dtiam_source_data_2025.zip",
        "results/kiba_drugcold_supplement_5fold.json", "configs/kiba_drugcold_20261016.json",
        "data/processed/kiba_dtiam_protocol/tokens/build_audit.json"))
    code = sorted((PROJECT / "scripts").glob("*.py")) + sorted((PROJECT / "reports").glob("*.py"))
    files.update(code)
    split_code = PROJECT / "baselines/DTIAM/code/data_process/data_split_dta.py"
    files.add(split_code)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT / "baselines/DTIAM", text=True).strip()
    FREEZE.mkdir(parents=True, exist_ok=True)
    inventory = pd.DataFrame(features).to_csv(index=False)
    write_once(FREEZE / "feature_inventory.csv", inventory)
    archive = FREEZE / "source_snapshot.zip"
    with ZipFile(archive, "x") as bundle:
        for path in [*code, split_code]:
            bundle.write(path, relative(path))
    result = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Frozen evidence from completed drug-ID cold runs; no historical source revision is reconstructed",
        "outer_seed": 0, "inner_seed": 20261012, "original_training_seed": 20261012,
        "planned_additional_training_seeds": [20261017, 20261018],
        "identity_definition": "RDKit MolToSmiles(MolFromSmiles(s), isomericSmiles=True); no salt stripping, tautomer or charge normalization",
        "prior": "Same protein sequence; training labels only; Morgan radius=2, fpSize=2048; validation-selected k/gamma; raw outputs",
        "fusion": "Validation-only clipped constant blend and L2 slope-regularized sigmoid gate; no test fitting or affine calibration",
        "datasets": datasets, "files": [record(p) for p in sorted(files)],
        "feature_inventory": record(FREEZE / "feature_inventory.csv"),
        "current_source_snapshot": record(archive), "dtiam_git_revision": revision,
        "environment": {"python": sys.version, "executable": sys.executable,
                        "platform": platform.platform(), "rdkit": rdBase.rdkitVersion,
                        "packages": {p: importlib.metadata.version(p) for p in ("numpy", "pandas", "scipy", "torch", "scikit-learn")}},
        "limitations": ["Project has no Git history; source snapshot records current code, not proof of historical training source.",
                        "Davis legacy token cache has incomplete historical model provenance.",
                        "Davis tests were inspected during development; repeats do not constitute a new holdout.",
                        "DTIAM-study baseline values are reported, not local reproductions.",
                        "KIBA original drug-ID folds contain canonical identity overlap; strict grouped experiments use separate outputs."],
    }
    write_once(FREEZE / "protocol.json", json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    print(json.dumps({"frozen": str(FREEZE), "files": len(files), "features": len(features)}))


if __name__ == "__main__":
    main()
