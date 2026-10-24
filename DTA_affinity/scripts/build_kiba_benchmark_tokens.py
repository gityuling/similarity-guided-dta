#!/usr/bin/env python3
"""Build a complete, resumable KIBA cache without removing benchmark pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from build_bindingdb_articles_kd_tokens import (
    ESM2_DIR, UNIMOL2_CHECKPOINT, atom_count_with_hydrogens, md5,
    save_drug_tokens, save_protein_tokens,
)


PROJECT = Path(__file__).resolve().parents[1]


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_cache(path: Path, width: int, length: int) -> bool:
    if not path.exists():
        return False
    array = np.load(path, mmap_mode="r", allow_pickle=False)
    if array.shape != (length, width) or not np.isfinite(array).all():
        raise ValueError(f"Unexpected or non-finite token matrix at {path}: {array.shape}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-table", type=Path, default=PROJECT / "data/processed/kiba_dtiam_protocol/table.csv")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT / "data/processed/kiba_dtiam_protocol/tokens")
    parser.add_argument("--reuse-cache", type=Path, action="append", default=[], help="Explicit legacy cache root to reuse; model provenance is recorded as legacy.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reuse-only", action="store_true", help="Copy compatible existing matrices without model inference.")
    parser.add_argument("--protein-batch-size", type=int, default=2)
    parser.add_argument("--drug-batch-size", type=int, default=2)
    args = parser.parse_args()
    if min(args.protein_batch_size, args.drug_batch_size) < 1:
        raise ValueError("Batch sizes must be positive")
    table = pd.read_csv(args.data_table)
    smiles = sorted(table.Canonical_SMILES.unique())
    sequences = sorted(table.Sequence.unique())
    sizes = {value: atom_count_with_hydrogens(value) for value in smiles}
    if any(size is None or size > 512 for size in sizes.values()):
        raise ValueError("A benchmark molecule exceeds the 512-token extractor capacity; no rows were filtered")
    entities = {
        "drugs": [(value, int(sizes[value]), 1536) for value in smiles],
        "proteins": [(value, min(len(value) + 2, 1024), 1280) for value in sequences],
    }
    available = {kind: 0 for kind in entities}
    reusable = {kind: 0 for kind in entities}
    copied = []
    for kind, items in entities.items():
        destination = args.cache_dir / kind
        for value, length, width in items:
            name = f"{md5(value)}.npy"
            path = destination / name
            if valid_cache(path, width, length):
                available[kind] += 1
                continue
            for source in args.reuse_cache:
                candidate = source / kind / name
                if not valid_cache(candidate, width, length):
                    continue
                reusable[kind] += 1
                if not args.dry_run:
                    destination.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(candidate, path)
                    copied.append({"source": str(candidate), "destination": str(path),
                                   "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
                break
    audit = {
        "data_table": str(args.data_table), "table_sha256": hashlib.sha256(args.data_table.read_bytes()).hexdigest(),
        "cache_dir": str(args.cache_dir), "benchmark_pairs": len(table),
        "unique_drugs": len(smiles), "unique_proteins": len(sequences),
        "max_explicit_H_atoms": max(sizes.values()), "molecules_over_256_atoms_retained": sum(n > 256 for n in sizes.values()),
        "available_at_start": available, "reusable_from_explicit_legacy_roots": reusable,
        "remaining_after_reuse": {kind: len(items) - available[kind] - reusable[kind] for kind, items in entities.items()},
        "legacy_provenance": "Explicitly selected existing caches are shape/length/finite checked; their original checkpoint hashes are not independently recoverable. Omit --reuse-cache to regenerate every entity.",
        "requested_reuse_roots": [str(path) for path in args.reuse_cache], "copied": copied,
        "extractor": "build_bindingdb_articles_kd_tokens.py UniMol2Extractor / save_protein_tokens",
        "unimol2_weights": str(UNIMOL2_CHECKPOINT), "esm2_model": str(ESM2_DIR),
        "protein_max_tokens": 1024, "drug_max_batched_atoms": 96,
        "dry_run": args.dry_run, "reuse_only": args.reuse_only,
    }
    print(json.dumps(audit, indent=2), flush=True)
    if args.dry_run:
        return
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    audit_path = args.cache_dir / "build_audit.json"
    previous = {}
    if audit_path.exists():
        previous = json.loads(audit_path.read_text())
        if previous["table_sha256"] != audit["table_sha256"]:
            raise RuntimeError("Cache audit belongs to a different benchmark table")
        audit["copied"] = previous.get("copied", []) + copied
        for key in ("model_sha256", "extractor_sha256"):
            if key in previous:
                audit[key] = previous[key]
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    if args.reuse_only:
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for KIBA token inference; existing compatible features were retained for resumption")
    device = torch.device("cuda")
    weight_paths = [UNIMOL2_CHECKPOINT, ESM2_DIR / "config.json", ESM2_DIR / "model.safetensors"]
    weight_hashes = {str(path): file_sha256(path) for path in weight_paths}
    previous_hashes = previous.get("model_sha256")
    if previous_hashes is not None and previous_hashes != weight_hashes:
        raise RuntimeError("Local encoder weights changed; choose a new cache directory")
    extractor_hash = file_sha256(Path(__file__).with_name("build_bindingdb_articles_kd_tokens.py"))
    if previous.get("extractor_sha256", extractor_hash) != extractor_hash:
        raise RuntimeError("Token extractor changed; choose a new cache directory")
    audit["model_sha256"] = weight_hashes
    audit["extractor_sha256"] = extractor_hash
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}", flush=True)
    for kind in entities:
        (args.cache_dir / kind).mkdir(parents=True, exist_ok=True)
    protein_done, protein_failed = save_protein_tokens(sequences, args.cache_dir / "proteins", device, args.protein_batch_size)
    drug_done, drug_failed = save_drug_tokens(
        [(value, int(sizes[value])) for value in smiles], args.cache_dir / "drugs", device, args.drug_batch_size, 96,
    )
    remaining = {
        kind: [md5(value) for value, length, width in items
               if not valid_cache(args.cache_dir / kind / f"{md5(value)}.npy", width, length)]
        for kind, items in entities.items()
    }
    audit.update({"device": torch.cuda.get_device_name(0), "generated_drugs": drug_done,
                  "generated_proteins": protein_done, "failed_drugs": drug_failed,
                  "failed_proteins": protein_failed, "remaining": remaining,
                  "complete": not any(remaining.values())})
    audit_path.write_text(json.dumps(audit, indent=2) + "\n")
    if not audit["complete"]:
        raise RuntimeError("Feature generation is incomplete; training must not filter missing benchmark rows")
    print("Complete KIBA feature cache is ready.", flush=True)


if __name__ == "__main__":
    main()
