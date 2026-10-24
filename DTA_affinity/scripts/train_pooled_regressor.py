#!/usr/bin/env python3
"""Train a small regression head on pooled Uni-Mol2/ESM-2 features.

This is a smoke baseline for the new DTA project, not the final NEXUSDTI
cross-attention model. It deliberately runs on the cached subset when some
revised Davis protein sequences still need ESM-2 extraction.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


ROOT = Path(__file__).resolve().parents[1]


class RegressionHead(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(512, 128),
            nn.GELU(),
            nn.Dropout(0.10),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def cindex(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    concordant = 0.0
    comparable = 0.0
    for i in range(len(y_true)):
        delta = y_true[i] - y_true[i + 1 :]
        valid = delta != 0
        if not valid.any():
            continue
        pred_delta = y_pred[i] - y_pred[i + 1 :]
        concordant += np.sum((pred_delta[valid] * delta[valid]) > 0)
        concordant += 0.5 * np.sum(pred_delta[valid] == 0)
        comparable += np.sum(valid)
    return float(concordant / comparable) if comparable else float("nan")


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mse": float(np.mean((y_true - y_pred) ** 2)),
        "ci": cindex(y_true, y_pred),
        "pearson": float(pearsonr(y_true, y_pred).statistic),
        "spearman": float(spearmanr(y_true, y_pred).statistic),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["random", "cold_drug", "cold_target", "double_cold"], default="random")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261012)
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    table = pd.read_csv(ROOT / "data/processed/davis/davis_entity_affinity_clean.csv")
    split = pd.read_csv(ROOT / "data/processed/davis_clean/splits" / args.split / "manifest.csv")
    features = np.load(ROOT / "data/processed/davis/davis_pooled_features.npz")
    drug_position = dict(zip(features["drug_indices"].tolist(), range(len(features["drug_indices"]))))
    target_position = dict(zip(features["protein_indices"].tolist(), range(len(features["protein_indices"]))))

    target_available = dict(zip(features["protein_indices"].tolist(), features["protein_available"].tolist()))
    available = table["Protein_Index"].map(target_available).fillna(False).to_numpy(dtype=bool)
    if not available.any():
        raise RuntimeError("No Davis rows have cached protein features")
    split = split.loc[available].copy()
    table = table.loc[available].reset_index(drop=True)

    drug_idx = table["Drug_Index"].map(drug_position).to_numpy()
    target_idx = table["Protein_Index"].map(target_position).to_numpy()
    x = np.concatenate(
        [features["drug_features"][drug_idx], features["protein_features"][target_idx]], axis=1
    ).astype(np.float32)
    y = table["pKd_floor"].to_numpy(dtype=np.float32)
    split_labels = split["split"].to_numpy()
    train_mask = split_labels == "train"
    val_mask = split_labels == "val"
    test_mask = split_labels == "test"
    if not train_mask.any() or not test_mask.any():
        raise RuntimeError(f"Invalid split after cache filtering: train={train_mask.sum()}, test={test_mask.sum()}")

    mean = x[train_mask].mean(axis=0, keepdims=True)
    std = x[train_mask].std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    x = (x - mean) / std
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RegressionHead(x.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss(beta=0.25)
    train_ds = TensorDataset(torch.from_numpy(x[train_mask]), torch.from_numpy(y[train_mask]))
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    best_state = None
    best_val = float("inf")
    for epoch in range(args.epochs):
        model.train()
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            pred_val = model(torch.from_numpy(x[val_mask]).to(device)).cpu().numpy() if val_mask.any() else np.array([])
        val_rmse = float(np.sqrt(np.mean((pred_val - y[val_mask]) ** 2))) if val_mask.any() else float("inf")
        if val_rmse < best_val:
            best_val = val_rmse
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pred_test = model(torch.from_numpy(x[test_mask]).to(device)).cpu().numpy()
        pred_train = model(torch.from_numpy(x[train_mask]).to(device)).cpu().numpy()
    result = {
        "dataset": "davis_clean_cached_features",
        "split": args.split,
        "seed": args.seed,
        "device": str(device),
        "epochs": args.epochs,
        "rows_total_after_feature_filter": int(len(table)),
        "rows_train": int(train_mask.sum()),
        "rows_val": int(val_mask.sum()),
        "rows_test": int(test_mask.sum()),
        "proteins_with_cached_features": int(features["protein_available"].sum()),
        "metrics_train": metrics(y[train_mask], pred_train),
        "metrics_test": metrics(y[test_mask], pred_test),
    }
    out = ROOT / "results" / f"pooled_regressor_{args.split}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
