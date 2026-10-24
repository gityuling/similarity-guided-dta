#!/usr/bin/env python3
"""Strict drug-cold BindingDB interaction pretraining for Davis DTA.

This script pretrains the NEXUS-DTA shared Uni-Mol2/ESM-2 interaction
encoder on the local BindingDB binary interaction data.  For a Davis outer
fold, every Davis validation/test drug is removed from BindingDB *before*
source train/validation splitting.  Thus the source task supplies no
interaction label for any downstream held-out drug.

The exported checkpoint is deliberately encoder-compatible with
``train_nexusdti_regressor.py``.  The binary source prediction head should
not be interpreted as an affinity head and is not selected by Davis labels.
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
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from train_nexusdti_regressor import (
    NexusDTIAffinityRegressor,
    PROJECT,
    ROOT,
    sequence_phys,
    standardize_smiles,
)


BINDINGDB_TABLE = ROOT / "DrugBAN/datasets/bindingdb/full.csv"
BINDINGDB_TOKENS = ROOT / "dataset/processed_DTI_Token/token_features_bindingdb"


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


class BindingDBDataset(Dataset):
    def __init__(self, frame: pd.DataFrame):
        prot_dir = BINDINGDB_TOKENS / "proteins"
        drug_dir = BINDINGDB_TOKENS / "drugs"
        records = []
        for row in frame.itertuples(index=False):
            sequence = str(row.Protein)
            smiles = str(row.canonical_smiles)
            protein_path = prot_dir / f"{md5(sequence)}.npy"
            drug_path = drug_dir / f"{md5(smiles)}.npy"
            if protein_path.exists() and drug_path.exists():
                records.append(
                    {
                        "protein_path": protein_path,
                        "drug_path": drug_path,
                        "sequence": sequence,
                        "label": float(row.Y),
                    }
                )
        if not records:
            raise RuntimeError("No BindingDB rows have both cached token features")
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
            torch.tensor(item["label"], dtype=torch.float32),
        )


def collate_fn(batch):
    proteins, drugs, phys, labels = zip(*batch)
    protein_lengths = torch.tensor([item.shape[0] for item in proteins], dtype=torch.long)
    drug_lengths = torch.tensor([item.shape[0] for item in drugs], dtype=torch.long)
    proteins = pad_sequence(proteins, batch_first=True)
    drugs = pad_sequence(drugs, batch_first=True)
    phys = pad_sequence(phys, batch_first=True)
    protein_mask = torch.arange(proteins.shape[1])[None, :] >= protein_lengths[:, None]
    drug_mask = torch.arange(drugs.shape[1])[None, :] >= drug_lengths[:, None]
    return proteins, drugs, phys, protein_mask, drug_mask, torch.stack(labels)


def stratified_split(frame: pd.DataFrame, validation_fraction: float, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    validation_indices = []
    for label, group in frame.groupby("Y", sort=True):
        indices = group.index.to_numpy()
        rng.shuffle(indices)
        count = max(1, int(round(len(indices) * validation_fraction)))
        validation_indices.extend(indices[:count].tolist())
    validation_set = set(validation_indices)
    val = frame.loc[frame.index.isin(validation_set)].reset_index(drop=True)
    train = frame.loc[~frame.index.isin(validation_set)].reset_index(drop=True)
    if train.Y.nunique() != 2 or val.Y.nunique() != 2:
        raise RuntimeError("Source stratification did not preserve both BindingDB classes")
    return train, val


@torch.no_grad()
def evaluate(model: NexusDTIAffinityRegressor, loader: DataLoader, device: torch.device) -> dict[str, float]:
    model.eval()
    labels, logits = [], []
    amp = device.type == "cuda"
    for protein, drug, phys, protein_mask, drug_mask, label in loader:
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            output = model(
                protein.to(device, non_blocking=amp),
                drug.to(device, non_blocking=amp),
                phys.to(device, non_blocking=amp),
                protein_mask.to(device, non_blocking=amp),
                drug_mask.to(device, non_blocking=amp),
            )
        labels.append(label.numpy())
        logits.append(output.float().cpu().numpy())
    y_true = np.concatenate(labels)
    y_logit = np.concatenate(logits)
    prob = 1.0 / (1.0 + np.exp(-y_logit))
    return {
        "bce": float(F.binary_cross_entropy_with_logits(torch.from_numpy(y_logit), torch.from_numpy(y_true))),
        "auroc": float(roc_auc_score(y_true, prob)),
        "aupr": float(average_precision_score(y_true, prob)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--davis-table",
        default="data/processed/davis_dtiam_protocol/table.csv",
        help="Davis affinity table relative to DTA_affinity.",
    )
    parser.add_argument(
        "--davis-manifest",
        required=True,
        help="Outer-fold manifest; its val/test drugs are excluded from BindingDB.",
    )
    parser.add_argument("--epochs", type=int, default=5)
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
    resolve = lambda value: Path(value) if Path(value).is_absolute() else PROJECT / value
    davis_table = pd.read_csv(resolve(args.davis_table))
    manifest_path = resolve(args.davis_manifest)
    manifest = pd.read_csv(manifest_path)
    davis = manifest.merge(davis_table, on="row_id", how="left", validate="one_to_one")
    if davis.Canonical_SMILES.isna().any():
        raise RuntimeError("Davis manifest does not align to the affinity table")
    held_drugs = {
        canonical
        for canonical in davis.loc[davis.split != "train", "Canonical_SMILES"].map(canonical_or_none)
        if canonical is not None
    }

    source = pd.read_csv(BINDINGDB_TABLE)
    source["canonical_smiles"] = source.SMILES.map(canonical_or_none)
    invalid_smiles = int(source.canonical_smiles.isna().sum())
    source = source.dropna(subset=["canonical_smiles", "Protein", "Y"]).copy()
    source.Y = source.Y.astype(np.float32)
    if not set(source.Y.unique()).issubset({0.0, 1.0}):
        raise RuntimeError("Expected binary BindingDB DTI labels in Y")
    before_exclusion = len(source)
    source = source.loc[~source.canonical_smiles.isin(held_drugs)].reset_index(drop=True)
    excluded_rows = before_exclusion - len(source)
    train_frame, val_frame = stratified_split(source, args.validation_fraction, args.seed)
    train_ds, val_ds = BindingDBDataset(train_frame), BindingDBDataset(val_frame)
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
    best_bce = float("inf")
    best_state = None
    history = []
    audit = {
        "bindingdb_table": str(BINDINGDB_TABLE),
        "davis_manifest": str(manifest_path),
        "downstream_held_drugs": len(held_drugs),
        "source_rows_before_drug_exclusion": before_exclusion,
        "source_rows_excluded_for_downstream_val_or_test_drug": excluded_rows,
        "source_rows_after_drug_exclusion_before_cache_filter": len(source),
        "invalid_source_smiles": invalid_smiles,
        "source_class_counts_before_cache_filter": {str(key): int(value) for key, value in source.Y.value_counts().items()},
        "cached_rows_train": len(train_ds),
        "cached_rows_val": len(val_ds),
    }
    print(json.dumps({"device": str(device), "audit": audit, "epochs": args.epochs, "batch": args.batch_size}), flush=True)
    amp = device.type == "cuda"
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for protein, drug, phys, protein_mask, drug_mask, label in train_loader:
            protein = protein.to(device, non_blocking=amp)
            drug = drug.to(device, non_blocking=amp)
            phys = phys.to(device, non_blocking=amp)
            protein_mask = protein_mask.to(device, non_blocking=amp)
            drug_mask = drug_mask.to(device, non_blocking=amp)
            label = label.to(device, non_blocking=amp)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                logits = model(protein, drug, phys, protein_mask, drug_mask)
                loss = F.binary_cross_entropy_with_logits(logits, label)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().float().cpu()))
        scheduler.step()
        metrics = evaluate(model, val_loader, device)
        row = {"epoch": epoch + 1, "train_bce": float(np.mean(losses)), **metrics}
        history.append(row)
        print(json.dumps(row), flush=True)
        if metrics["bce"] < best_bce:
            best_bce = metrics["bce"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    if best_state is None:
        raise RuntimeError("No BindingDB checkpoint was selected")
    model.load_state_dict(best_state)
    output_dir = PROJECT / "results"
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    result = {"args": vars(args), "audit": audit, "history": history, "best_val_bce": best_bce}
    checkpoint_path = checkpoint_dir / f"nexusdti_bindingdb_strict_{args.run_tag}.pt"
    result_path = output_dir / f"nexusdti_bindingdb_strict_{args.run_tag}.json"
    torch.save({"model": model.state_dict(), "result": result}, checkpoint_path)
    result_path.write_text(json.dumps(result, indent=2))
    print(json.dumps({"checkpoint": str(checkpoint_path), "result": str(result_path), "best_val_bce": best_bce}), flush=True)


if __name__ == "__main__":
    main()
