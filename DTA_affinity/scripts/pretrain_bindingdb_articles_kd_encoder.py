#!/usr/bin/env python3
"""Strict continuous-affinity pretraining on exact BindingDB Kd measurements.

This is intentionally a *source-only* stage for drug-cold Davis DTA.  Every
downstream validation/test drug is removed before source splitting; the
source's pKd labels therefore provide no direct supervision for a held-out
Davis drug.  Only shared token/interactions layers are transferred later, not
the source regression head or its label scale.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem
from scipy.stats import pearsonr
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from train_nexusdti_regressor import (
    NexusDTIAffinityRegressor,
    PROJECT,
    ROOT,
    sequence_phys,
    standardize_smiles,
)


DEFAULT_SOURCE = PROJECT / "data/processed/bindingdb_articles_kd/bindingdb_articles_kd_f0.csv"
DEFAULT_TOKENS = ROOT / "dataset/processed_DTI_Token/token_features_bindingdb_articles_kd_f0"


def md5(value: str) -> str:
    return hashlib.md5(value.encode()).hexdigest()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def canonical_or_none(smiles: str) -> str | None:
    try:
        return standardize_smiles(str(smiles))
    except (TypeError, ValueError):
        return None


def explicit_atom_count(smiles: str) -> int | None:
    try:
        molecule = Chem.MolFromSmiles(smiles)
        return Chem.AddHs(molecule).GetNumAtoms() if molecule is not None else None
    except (TypeError, ValueError):
        return None


class BindingDBKdDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, token_dir: Path, target_mean: float, target_std: float):
        protein_dir = token_dir / "proteins"
        drug_dir = token_dir / "drugs"
        self.records = [
            {
                "protein_path": protein_dir / f"{row.protein_hash}.npy",
                "drug_path": drug_dir / f"{row.drug_hash}.npy",
                "sequence": row.sequence,
                "target": (float(row.pKd) - target_mean) / target_std,
            }
            for row in frame.itertuples(index=False)
        ]
        if not self.records:
            raise RuntimeError("No BindingDB Kd rows with complete token features")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        item = self.records[index]
        protein = np.load(item["protein_path"]).astype(np.float32)
        drug = np.load(item["drug_path"]).astype(np.float32)
        phys = sequence_phys(item["sequence"], protein.shape[0])
        return (
            torch.from_numpy(protein),
            torch.from_numpy(drug),
            torch.from_numpy(phys),
            torch.tensor(item["target"], dtype=torch.float32),
        )


def collate_fn(batch):
    proteins, drugs, phys, targets = zip(*batch)
    protein_lengths = torch.tensor([item.shape[0] for item in proteins], dtype=torch.long)
    drug_lengths = torch.tensor([item.shape[0] for item in drugs], dtype=torch.long)
    proteins = pad_sequence(proteins, batch_first=True)
    drugs = pad_sequence(drugs, batch_first=True)
    phys = pad_sequence(phys, batch_first=True)
    protein_mask = torch.arange(proteins.shape[1])[None, :] >= protein_lengths[:, None]
    drug_mask = torch.arange(drugs.shape[1])[None, :] >= drug_lengths[:, None]
    return proteins, drugs, phys, protein_mask, drug_mask, torch.stack(targets)


def quantile_split(frame: pd.DataFrame, validation_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    bins = pd.qcut(frame.pKd, q=10, labels=False, duplicates="drop")
    rng = np.random.default_rng(seed)
    validation_indices: list[int] = []
    for _, group in frame.groupby(bins, observed=True):
        indices = group.index.to_numpy()
        rng.shuffle(indices)
        validation_indices.extend(indices[: max(1, int(round(len(indices) * validation_fraction)))].tolist())
    validation_set = set(validation_indices)
    train = frame.loc[~frame.index.isin(validation_set)].reset_index(drop=True)
    validation = frame.loc[frame.index.isin(validation_set)].reset_index(drop=True)
    if len(train) < 2 or len(validation) < 2:
        raise RuntimeError("Source quantile split is too small")
    return train, validation


@torch.no_grad()
def evaluate(
    model: NexusDTIAffinityRegressor,
    loader: DataLoader,
    device: torch.device,
    target_mean: float,
    target_std: float,
) -> dict[str, float]:
    model.eval()
    actual, predicted = [], []
    amp = device.type == "cuda"
    for protein, drug, phys, protein_mask, drug_mask, target in loader:
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            output = model(
                protein.to(device, non_blocking=amp),
                drug.to(device, non_blocking=amp),
                phys.to(device, non_blocking=amp),
                protein_mask.to(device, non_blocking=amp),
                drug_mask.to(device, non_blocking=amp),
            )
        actual.append(target.numpy())
        predicted.append(output.float().cpu().numpy())
    y_true = np.concatenate(actual)
    y_pred = np.concatenate(predicted)
    pKd_true = y_true * target_std + target_mean
    pKd_pred = y_pred * target_std + target_mean
    return {
        "mse_normalized": float(np.mean((y_true - y_pred) ** 2)),
        "rmse_pKd": float(np.sqrt(np.mean((pKd_true - pKd_pred) ** 2))),
        "pearson": float(pearsonr(pKd_true, pKd_pred).statistic),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--davis-manifest", required=True, help="Davis outer-fold manifest relative to DTA_affinity.")
    parser.add_argument("--source-table", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--token-dir", type=Path, default=DEFAULT_TOKENS)
    parser.add_argument("--max-drug-atoms", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--run-tag", required=True)
    args = parser.parse_args()
    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("--validation-fraction must be between 0 and 0.5")
    if args.max_drug_atoms < 1:
        raise ValueError("--max-drug-atoms must be positive")
    set_seed(args.seed)
    resolve = lambda value: value if value.is_absolute() else ROOT / value
    source_table = resolve(args.source_table)
    token_dir = resolve(args.token_dir)
    manifest_path = Path(args.davis_manifest)
    if not manifest_path.is_absolute():
        manifest_path = PROJECT / manifest_path
    manifest = pd.read_csv(manifest_path)
    davis = manifest.merge(
        pd.read_csv(PROJECT / "data/processed/davis_dtiam_protocol/table.csv"),
        on="row_id",
        how="left",
        validate="one_to_one",
    )
    if davis.Canonical_SMILES.isna().any():
        raise RuntimeError("Davis manifest does not align to the affinity table")
    held_drugs = {
        canonical
        for canonical in davis.loc[davis.split != "train", "Canonical_SMILES"].map(canonical_or_none)
        if canonical is not None
    }

    source = pd.read_csv(source_table)
    required = {"canonical_smiles", "sequence", "pKd"}
    missing_columns = required - set(source.columns)
    if missing_columns:
        raise ValueError(f"Source table lacks {sorted(missing_columns)}")
    source["canonical_smiles"] = source.canonical_smiles.map(canonical_or_none)
    source["sequence"] = source.sequence.fillna("").astype(str).str.strip()
    source["pKd"] = pd.to_numeric(source.pKd, errors="coerce")
    source = source.dropna(subset=["canonical_smiles", "pKd"]).loc[lambda frame: frame.sequence.ne("")].copy()
    rows_before_held_drug_check = len(source)
    source = source.loc[~source.canonical_smiles.isin(held_drugs)].reset_index(drop=True)
    held_drug_rows_removed = rows_before_held_drug_check - len(source)
    source["atom_count"] = source.canonical_smiles.map(explicit_atom_count)
    invalid_atom_count_rows = int(source.atom_count.isna().sum())
    source = source.dropna(subset=["atom_count"]).copy()
    oversized_rows_removed = int((source.atom_count > args.max_drug_atoms).sum())
    source = source.loc[source.atom_count <= args.max_drug_atoms].copy()
    source["protein_hash"] = source.sequence.map(md5)
    source["drug_hash"] = source.canonical_smiles.map(md5)
    cached_proteins = {path.stem for path in (token_dir / "proteins").glob("*.npy")}
    cached_drugs = {path.stem for path in (token_dir / "drugs").glob("*.npy")}
    before_cache_filter = len(source)
    source = source.loc[
        source.protein_hash.isin(cached_proteins) & source.drug_hash.isin(cached_drugs)
    ].reset_index(drop=True)
    missing_cache_rows = before_cache_filter - len(source)
    if source.empty:
        raise RuntimeError("No strict BindingDB Kd rows have complete token features")
    train_frame, val_frame = quantile_split(source, args.validation_fraction, args.seed)
    target_mean = float(train_frame.pKd.mean())
    target_std = float(train_frame.pKd.std())
    if target_std < 1e-6:
        raise RuntimeError("BindingDB pKd target has near-zero standard deviation")
    train_ds = BindingDBKdDataset(train_frame, token_dir, target_mean, target_std)
    val_ds = BindingDBKdDataset(val_frame, token_dir, target_mean, target_std)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required for the intended source-pretraining protocol")
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
        persistent_workers=args.workers > 0,
        collate_fn=collate_fn,
    )
    model = NexusDTIAffinityRegressor(pool_fusion=True, hybrid=False, dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    audit = {
        "source_table": str(source_table),
        "token_dir": str(token_dir),
        "davis_manifest": str(manifest_path),
        "downstream_held_drugs": len(held_drugs),
        "source_rows_before_held_drug_check": rows_before_held_drug_check,
        "source_rows_removed_for_downstream_val_or_test_drugs": held_drug_rows_removed,
        "source_rows_removed_for_invalid_atom_count": invalid_atom_count_rows,
        "source_rows_removed_for_max_drug_atoms": oversized_rows_removed,
        "max_drug_atoms": args.max_drug_atoms,
        "source_rows_before_cache_filter": before_cache_filter,
        "source_rows_removed_for_missing_cache": missing_cache_rows,
        "source_rows_after_all_filters": len(source),
        "source_unique_drugs_after_all_filters": int(source.canonical_smiles.nunique()),
        "source_unique_proteins_after_all_filters": int(source.sequence.nunique()),
        "cached_rows_train": len(train_ds),
        "cached_rows_val": len(val_ds),
        "target_mean_train_pKd": target_mean,
        "target_std_train_pKd": target_std,
    }
    print(json.dumps({"device": str(device), "audit": audit, "epochs": args.epochs, "batch": args.batch_size}), flush=True)
    best_mse = float("inf")
    best_state = None
    history = []
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for protein, drug, phys, protein_mask, drug_mask, target in train_loader:
            protein = protein.to(device, non_blocking=True)
            drug = drug.to(device, non_blocking=True)
            phys = phys.to(device, non_blocking=True)
            protein_mask = protein_mask.to(device, non_blocking=True)
            drug_mask = drug_mask.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16, enabled=True):
                prediction = model(protein, drug, phys, protein_mask, drug_mask)
                loss = F.smooth_l1_loss(prediction, target, beta=0.25)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().float().cpu()))
        scheduler.step()
        metrics = evaluate(model, val_loader, device, target_mean, target_std)
        record = {"epoch": epoch + 1, "train_smooth_l1": float(np.mean(losses)), **metrics}
        history.append(record)
        print(json.dumps(record), flush=True)
        if metrics["mse_normalized"] < best_mse:
            best_mse = metrics["mse_normalized"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("No source checkpoint was selected")
    model.load_state_dict(best_state)
    results_dir = PROJECT / "results"
    checkpoint_dir = results_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result = {
        "args": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "audit": audit,
        "history": history,
        "best_val_mse_normalized": best_mse,
    }
    checkpoint_path = checkpoint_dir / f"nexusdti_bindingdb_kd_strict_{args.run_tag}.pt"
    result_path = results_dir / f"nexusdti_bindingdb_kd_strict_{args.run_tag}.json"
    torch.save({"model": model.state_dict(), "result": result}, checkpoint_path)
    result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps({"checkpoint": str(checkpoint_path), "result": str(result_path), "best_val_mse_normalized": best_mse}), flush=True)


if __name__ == "__main__":
    main()
