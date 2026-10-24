#!/usr/bin/env python3
"""Align Davis entity tables to the existing Uni-Mol2/ESM-2 token cache.

By default this is a cache-only operation. ``--compute-missing`` uses the
local ESM-2 checkpoint to fill protein sequences absent from the old Davis
classification cache; no model or data is downloaded by this script.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
TOKEN_DIR = ROOT / "dataset/processed_DTI_Token/token_features_davis"


def md5(value: str) -> str:
    return hashlib.md5(value.encode()).hexdigest()


def load_mean(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.ndim != 2:
        raise ValueError(f"Expected token matrix, got {arr.shape}: {path}")
    return arr.astype(np.float32).mean(axis=0)


def drug_key(smiles: str) -> str:
    from rdkit import Chem

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return md5(Chem.MolToSmiles(mol))


def cache_features(proteins: pd.DataFrame, drugs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, list[int]]:
    drug_files = {p.stem: p for p in (TOKEN_DIR / "drugs").glob("*.npy")}
    protein_files = {p.stem: p for p in (TOKEN_DIR / "proteins").glob("*.npy")}

    drug_features = []
    for smiles in drugs["Canonical_SMILES"]:
        key = drug_key(str(smiles))
        if key not in drug_files:
            raise FileNotFoundError(f"No cached Uni-Mol2 token file for {smiles} ({key})")
        drug_features.append(load_mean(drug_files[key]))

    protein_features = []
    missing = []
    for index, sequence in enumerate(proteins["Sequence"]):
        key = md5(str(sequence))
        if key not in protein_files:
            protein_features.append(np.zeros(1280, dtype=np.float32))
            missing.append(index)
        else:
            protein_features.append(load_mean(protein_files[key]))
    return np.stack(drug_features), np.stack(protein_features), missing


def fill_missing_with_esm(proteins: pd.DataFrame, features: np.ndarray, missing: list[int], batch_size: int) -> None:
    import torch
    from transformers import EsmModel, EsmTokenizer

    partial_dir = PROJECT / "data/processed/davis/esm2_pooled_partial"
    partial_dir.mkdir(parents=True, exist_ok=True)
    pending = []
    for index in missing:
        partial = partial_dir / f"{index}.npy"
        if partial.exists():
            cached = np.load(partial)
            if cached.ndim == 2 and cached.shape[1] == 1280:
                features[index] = cached.mean(axis=0).astype(np.float32)
                sequence_key = md5(str(proteins.iloc[index]["Sequence"]))
                np.save(TOKEN_DIR / "proteins" / f"{sequence_key}.npy", cached.astype(np.float32))
                continue
        pending.append(index)
    if not pending:
        print("All missing proteins are already present in the partial ESM-2 cache")
        return

    model_dir = ROOT / "ESM2/esm2_model"
    tokenizer = EsmTokenizer.from_pretrained(model_dir, local_files_only=True)
    model = EsmModel.from_pretrained(model_dir, local_files_only=True)
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    sequences = [str(proteins.iloc[i]["Sequence"]) for i in pending]
    for start in range(0, len(sequences), batch_size):
        batch = sequences[start : start + batch_size]
        encoded = tokenizer(
            batch,
            padding=True,
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            hidden = model(**encoded).last_hidden_state.detach().cpu().numpy()
        mask = encoded["attention_mask"].detach().cpu().numpy().astype(bool)
        for offset, valid in enumerate(mask):
            index = pending[start + offset]
            token_matrix = hidden[offset, valid].astype(np.float32)
            features[index] = token_matrix.mean(axis=0)
            # Keep the complete token matrix for the final cross-modal model;
            # the pooled vector is only a convenience for the smoke baseline.
            sequence_key = md5(str(proteins.iloc[index]["Sequence"]))
            np.save(partial_dir / f"{index}.npy", token_matrix)
            np.save(TOKEN_DIR / "proteins" / f"{sequence_key}.npy", token_matrix)
        print(f"ESM-2: {min(start + len(batch), len(sequences))}/{len(sequences)} missing proteins (checkpointed)")
    del model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compute-missing", action="store_true")
    parser.add_argument("--batch-size", type=int, default=1)
    args = parser.parse_args()

    proteins = pd.read_csv(ROOT / "DTA_affinity/data/raw/davis/dta_entities_proteins.csv")
    drugs = pd.read_csv(ROOT / "DTA_affinity/data/raw/davis/dta_entities_drugs.csv")
    drug_features, protein_features, missing = cache_features(proteins, drugs)
    if args.compute_missing and missing:
        fill_missing_with_esm(proteins, protein_features, missing, args.batch_size)

    out = PROJECT / "data/processed/davis/davis_pooled_features.npz"
    np.savez_compressed(
        out,
        drug_features=drug_features,
        protein_features=protein_features,
        protein_available=np.array([i not in missing for i in range(len(proteins))]),
        drug_indices=drugs["Drug_Index"].to_numpy(),
        protein_indices=proteins["Protein_Index"].to_numpy(),
    )
    print(f"Wrote {out}")
    print(f"Drug features: {drug_features.shape}; protein features: {protein_features.shape}")
    print(f"Missing cached proteins: {len(missing)}; computed={args.compute_missing and bool(missing)}")


if __name__ == "__main__":
    main()
