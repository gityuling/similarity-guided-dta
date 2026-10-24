#!/usr/bin/env python3
"""Strict drug-cold KIBA affinity pretraining for a Davis outer fold.

KIBA is used only to learn transferable token-interaction representations.
The KIBA regression head is not transferred to Davis because KIBA scores and
Davis pKd values have different scales.  All downstream validation/test drugs
are removed from KIBA before its source split is formed.
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


KIBA_TABLE = PROJECT / "data/processed/kiba/kiba_deepdta_affinity_long.csv"
KIBA_TOKENS = ROOT / "dataset/processed_DTI_Token/token_features"


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


class KibaDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, target_mean: float, target_std: float):
        protein_dir = KIBA_TOKENS / "proteins"
        drug_dir = KIBA_TOKENS / "drugs"
        records = []
        for row in frame.itertuples(index=False):
            sequence = str(row.protein_sequence)
            smiles = str(row.canonical_smiles)
            protein_path = protein_dir / f"{md5(sequence)}.npy"
            drug_path = drug_dir / f"{md5(smiles)}.npy"
            if protein_path.exists() and drug_path.exists():
                records.append(
                    {
                        "protein_path": protein_path,
                        "drug_path": drug_path,
                        "sequence": sequence,
                        "target": (float(row.kiba_score) - target_mean) / target_std,
                    }
                )
        if not records:
            raise RuntimeError("No KIBA rows have both cached token features")
        self.records = records

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
    bins = pd.qcut(frame.kiba_score, q=10, labels=False, duplicates="drop")
    rng = np.random.default_rng(seed)
    val_indices = []
    for _, group in frame.groupby(bins, observed=True):
        indices = group.index.to_numpy()
        rng.shuffle(indices)
        val_indices.extend(indices[: max(1, int(round(len(indices) * validation_fraction)))].tolist())
    val_set = set(val_indices)
    return (
        frame.loc[~frame.index.isin(val_set)].reset_index(drop=True),
        frame.loc[frame.index.isin(val_set)].reset_index(drop=True),
    )


@torch.no_grad()
def evaluate(model: NexusDTIAffinityRegressor, loader: DataLoader, device: torch.device) -> dict[str, float]:
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
    return {
        "mse_normalized": float(np.mean((y_true - y_pred) ** 2)),
        "pearson": float(pearsonr(y_true, y_pred).statistic),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--davis-manifest", required=True, help="Davis outer-fold manifest relative to DTA_affinity.")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=20261012)
    parser.add_argument("--run-tag", required=True)
    args = parser.parse_args()
    if not 0.0 < args.validation_fraction < 0.5:
        raise ValueError("--validation-fraction must be between 0 and 0.5")
    set_seed(args.seed)

    manifest_path = Path(args.davis_manifest)
    if not manifest_path.is_absolute():
        manifest_path = PROJECT / manifest_path
    davis = pd.read_csv(manifest_path).merge(
        pd.read_csv(PROJECT / "data/processed/davis_dtiam_protocol/table.csv"),
        on="row_id",
        how="left",
        validate="one_to_one",
    )
    held_drugs = {
        canonical
        for canonical in davis.loc[davis.split != "train", "Canonical_SMILES"].map(canonical_or_none)
        if canonical is not None
    }
    source = pd.read_csv(KIBA_TABLE)
    source["canonical_smiles"] = source.smiles.map(canonical_or_none)
    source = source.dropna(subset=["canonical_smiles", "protein_sequence", "kiba_score"]).copy()
    rows_before_exclusion = len(source)
    source = source.loc[~source.canonical_smiles.isin(held_drugs)].reset_index(drop=True)
    rows_after_drug_exclusion = len(source)
    cached_drugs = {path.stem for path in (KIBA_TOKENS / "drugs").glob("*.npy")}
    cached_proteins = {path.stem for path in (KIBA_TOKENS / "proteins").glob("*.npy")}
    source["drug_hash"] = source.canonical_smiles.map(md5)
    source["protein_hash"] = source.protein_sequence.map(md5)
    source = source.loc[
        source.drug_hash.isin(cached_drugs) & source.protein_hash.isin(cached_proteins)
    ].reset_index(drop=True)
    train_frame, val_frame = quantile_split(source, args.validation_fraction, args.seed)
    target_mean = float(train_frame.kiba_score.mean())
    target_std = float(train_frame.kiba_score.std())
    if target_std < 1e-6:
        raise RuntimeError("KIBA source target has near-zero standard deviation")
    train_ds = KibaDataset(train_frame, target_mean, target_std)
    val_ds = KibaDataset(val_frame, target_mean, target_std)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=args.workers > 0,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=args.workers > 0,
    )
    model = NexusDTIAffinityRegressor(pool_fusion=True, hybrid=False, dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    scaler = torch.amp.GradScaler("cuda", enabled=pin_memory)
    audit = {
        "kiba_table": str(KIBA_TABLE),
        "davis_manifest": str(manifest_path),
        "downstream_held_drugs": len(held_drugs),
        "source_rows_before_downstream_drug_exclusion": rows_before_exclusion,
        "source_rows_excluded_for_downstream_val_or_test_drug": rows_before_exclusion - rows_after_drug_exclusion,
        "source_rows_after_drug_exclusion_before_cache_filter": rows_after_drug_exclusion,
        "source_rows_after_cache_filter": len(source),
        "cached_rows_train": len(train_ds),
        "cached_rows_val": len(val_ds),
        "target_mean": target_mean,
        "target_std": target_std,
    }
    print(json.dumps({"device": str(device), "audit": audit, "epochs": args.epochs, "batch": args.batch_size}), flush=True)
    best_mse = float("inf")
    best_state = None
    history = []
    amp = device.type == "cuda"
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for protein, drug, phys, protein_mask, drug_mask, target in train_loader:
            protein = protein.to(device, non_blocking=amp)
            drug = drug.to(device, non_blocking=amp)
            phys = phys.to(device, non_blocking=amp)
            protein_mask = protein_mask.to(device, non_blocking=amp)
            drug_mask = drug_mask.to(device, non_blocking=amp)
            target = target.to(device, non_blocking=amp)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                prediction = model(protein, drug, phys, protein_mask, drug_mask)
                loss = F.smooth_l1_loss(prediction, target, beta=0.25)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().float().cpu()))
        scheduler.step()
        metrics = evaluate(model, val_loader, device)
        row = {"epoch": epoch + 1, "train_smooth_l1": float(np.mean(losses)), **metrics}
        history.append(row)
        print(json.dumps(row), flush=True)
        if metrics["mse_normalized"] < best_mse:
            best_mse = metrics["mse_normalized"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    if best_state is None:
        raise RuntimeError("No KIBA checkpoint was selected")
    model.load_state_dict(best_state)
    results_dir = PROJECT / "results"
    checkpoint_dir = results_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result = {"args": vars(args), "audit": audit, "history": history, "best_val_mse_normalized": best_mse}
    checkpoint_path = checkpoint_dir / f"nexusdti_kiba_strict_{args.run_tag}.pt"
    result_path = results_dir / f"nexusdti_kiba_strict_{args.run_tag}.json"
    torch.save({"model": model.state_dict(), "result": result}, checkpoint_path)
    result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps({"checkpoint": str(checkpoint_path), "result": str(result_path), "best_val_mse_normalized": best_mse}), flush=True)


if __name__ == "__main__":
    main()
