#!/usr/bin/env python3
"""Train a NEXUSDTI-style token-interaction regressor on Davis DTA data.

This is intentionally an independent experiment for the quantitative DTA
extension. It reuses the existing Uni-Mol2 drug tokens and ESM-2 protein
tokens, adds protein self-attention plus bilinear Top-K drug--protein token
interaction, and replaces the original binary head with a continuous
affinity regression head.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from rdkit import Chem
from scipy.stats import pearsonr, spearmanr
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
TOKEN_DIR = ROOT / "dataset/processed_DTI_Token/token_features_davis"
PROT_DIR = TOKEN_DIR / "proteins"
DRUG_DIR = TOKEN_DIR / "drugs"


AA_PHYS_MAP = {
    "A": [1.8, 6.00, 0.1, 0.0, 0, 0, 0], "R": [-4.5, 10.76, 0.9, 1.0, 4, 1, 1],
    "N": [-3.5, 5.41, 0.5, 1.0, 2, 2, 0], "D": [-3.5, 2.77, 0.5, 1.0, 1, 4, -1],
    "C": [2.5, 5.07, 0.4, 0.0, 1, 1, 0], "Q": [-3.5, 5.65, 0.6, 1.0, 2, 2, 0],
    "E": [-3.5, 3.22, 0.6, 1.0, 1, 4, -1], "G": [-0.4, 5.97, 0.0, 0.0, 0, 0, 0],
    "H": [-3.2, 7.59, 0.7, 1.0, 2, 2, 1], "I": [4.5, 6.02, 0.6, 0.0, 0, 0, 0],
    "L": [3.8, 5.98, 0.6, 0.0, 0, 0, 0], "K": [-3.9, 9.74, 0.6, 1.0, 3, 1, 1],
    "M": [1.9, 5.74, 0.7, 0.0, 0, 1, 0], "F": [2.8, 5.48, 0.8, 0.0, 0, 0, 0],
    "P": [-1.6, 6.30, 0.4, 0.0, 0, 0, 0], "S": [-0.8, 5.68, 0.3, 1.0, 1, 2, 0],
    "T": [-0.7, 5.60, 0.4, 1.0, 1, 2, 0], "W": [-0.9, 5.89, 1.0, 0.0, 1, 0, 0],
    "Y": [-1.3, 5.66, 0.9, 1.0, 1, 1, 0], "V": [4.2, 5.96, 0.4, 0.0, 0, 0, 0],
    "X": [0.0, 6.00, 0.5, 0.5, 1, 1, 0],
}


def md5(value: str) -> str:
    return hashlib.md5(value.encode()).hexdigest()


def standardize_smiles(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol)


def sequence_phys(sequence: str, token_length: int) -> np.ndarray:
    """Align residue descriptors with the existing token convention.

    The original NEXUSDTI loader places residue descriptors at positions
    1..L, leaving position 0 and any special/truncated positions as zeros.
    We preserve that convention for both the old Uni-Mol2-aligned cache and
    the newly generated ESM-2 matrices.
    """
    out = np.zeros((token_length, 7), dtype=np.float32)
    usable = min(len(sequence), max(0, token_length - 2))
    if usable:
        out[1 : 1 + usable] = np.asarray(
            [AA_PHYS_MAP.get(char, AA_PHYS_MAP["X"]) for char in sequence[:usable]],
            dtype=np.float32,
        )
    return out


def kinase_pocket_selector(sequence: str, token_length: int, radius: int = 48) -> np.ndarray:
    """Select a sequence-defined kinase ATP-pocket neighborhood.

    Davis is a kinase panel, including resistant variants.  Rather than using
    an assay label or a ligand-bound structure, this selector finds the
    conserved beta3 lysine / HRD / DFG catalytic motif triplet in the supplied
    sequence.  The selected span covers the glycine-rich loop, beta3 lysine,
    catalytic loop and activation-loop entrance.  If a non-canonical kinase
    has no complete triplet, DFG-centered and finally full-sequence fallbacks
    keep inference well-defined without using affinity information.

    Positions are returned in the cached ESM convention: token 0 is BOS and
    residue i is at token i + 1.  Long proteins are clipped to their cached
    ESM token length before motif matching, exactly as their embeddings are.
    """
    selector = np.zeros(token_length, dtype=bool)
    usable = min(len(sequence), max(0, token_length - 2))
    residues = sequence[:usable]
    if not residues:
        return selector

    beta3 = [match.start() for match in re.finditer(r"[VILM][A-Z][VILM]K", residues)]
    hrd = [match.start() for match in re.finditer(r"H[RKQN]D", residues)]
    dfg = [match.start() for match in re.finditer(r"DFG", residues)]
    candidates: list[tuple[int, int, int, int]] = []
    for lys_pos in beta3:
        for hrd_pos in hrd:
            if not 25 <= hrd_pos - lys_pos <= 180:
                continue
            for dfg_pos in dfg:
                if not 0 <= dfg_pos - hrd_pos <= 80:
                    continue
                # Canonical kinase-domain spacing is approximately 70 and 25
                # residues respectively; use it only to break motif ties.
                score = abs((hrd_pos - lys_pos) - 70) + abs((dfg_pos - hrd_pos) - 25)
                candidates.append((score, lys_pos, hrd_pos, dfg_pos))

    if candidates:
        _, lys_pos, _, dfg_pos = min(candidates)
        start = max(0, lys_pos - radius)
        end = min(usable, dfg_pos + radius + 3)
    elif dfg:
        dfg_pos = dfg[0]
        start = max(0, dfg_pos - (2 * radius + 24))
        end = min(usable, dfg_pos + radius + 3)
    else:
        # A documented fallback for atypical/poorly annotated entries.  It is
        # intentionally label-free and simply reduces to the original full
        # protein branch rather than fabricating a pocket location.
        start, end = 0, usable

    selector[1 + start : 1 + end] = True
    return selector


class AffinityDataset(Dataset):
    def __init__(
        self, subset: pd.DataFrame, kinase_pocket: bool = False, pocket_radius: int = 48,
        token_dir: Path | None = None, target_column: str = "pKd_floor",
    ):
        protein_dir = PROT_DIR if token_dir is None else Path(token_dir) / "proteins"
        drug_dir = DRUG_DIR if token_dir is None else Path(token_dir) / "drugs"
        records = []
        pocket_cache: dict[str, np.ndarray] = {}
        for row in subset.itertuples(index=False):
            sequence = str(row.Sequence)
            smiles = str(row.Canonical_SMILES)
            protein_path = protein_dir / f"{md5(sequence)}.npy"
            drug_path = drug_dir / f"{md5(standardize_smiles(smiles))}.npy"
            if not protein_path.exists() or not drug_path.exists():
                raise FileNotFoundError(f"Missing token cache for row {row.row_id}")
            if kinase_pocket and sequence not in pocket_cache:
                token_length = int(np.load(protein_path, mmap_mode="r").shape[0])
                pocket_cache[sequence] = kinase_pocket_selector(sequence, token_length, pocket_radius)
            records.append(
                {
                    "row_id": int(row.row_id),
                    "protein_path": protein_path,
                    "drug_path": drug_path,
                    "sequence": sequence,
                    "pocket_selector": pocket_cache.get(sequence),
                    "y": float(getattr(row, target_column)),
                }
            )
        self.records = records

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        item = self.records[index]
        protein = np.load(item["protein_path"]).astype(np.float32)
        drug = np.load(item["drug_path"]).astype(np.float32)
        phys = sequence_phys(item["sequence"], protein.shape[0])
        pocket_selector = item["pocket_selector"]
        if pocket_selector is None:
            pocket_selector = np.zeros(protein.shape[0], dtype=bool)
        return (
            torch.from_numpy(protein),
            torch.from_numpy(drug),
            torch.from_numpy(phys),
            torch.from_numpy(pocket_selector),
            torch.tensor(item["y"], dtype=torch.float32),
        )


def build_pooled_stats(train_df: pd.DataFrame, token_dir: Path | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Build train-only statistics for the explicit pooled baseline branch."""
    protein_dir = PROT_DIR if token_dir is None else Path(token_dir) / "proteins"
    drug_dir = DRUG_DIR if token_dir is None else Path(token_dir) / "drugs"
    drug_cache: dict[str, np.ndarray] = {}
    protein_cache: dict[str, np.ndarray] = {}
    rows = []
    for row in train_df.itertuples(index=False):
        sequence = str(row.Sequence)
        smiles = standardize_smiles(str(row.Canonical_SMILES))
        if smiles not in drug_cache:
            drug_cache[smiles] = np.load(drug_dir / f"{md5(smiles)}.npy").astype(np.float32).mean(axis=0)
        if sequence not in protein_cache:
            protein_cache[sequence] = np.load(protein_dir / f"{md5(sequence)}.npy").astype(np.float32).mean(axis=0)
        rows.append(np.concatenate([drug_cache[smiles], protein_cache[sequence]]))
    features = np.stack(rows).astype(np.float32)
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


def collate_fn(batch):
    proteins, drugs, phys, pocket_selectors, labels = zip(*batch)
    protein_lengths = torch.tensor([item.shape[0] for item in proteins], dtype=torch.long)
    drug_lengths = torch.tensor([item.shape[0] for item in drugs], dtype=torch.long)
    proteins = pad_sequence(proteins, batch_first=True)
    drugs = pad_sequence(drugs, batch_first=True)
    phys = pad_sequence(phys, batch_first=True)
    pocket_selectors = pad_sequence(pocket_selectors, batch_first=True)
    protein_mask = torch.arange(proteins.shape[1])[None, :] >= protein_lengths[:, None]
    drug_mask = torch.arange(drugs.shape[1])[None, :] >= drug_lengths[:, None]
    return proteins, drugs, phys, pocket_selectors, protein_mask, drug_mask, torch.stack(labels)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class MaskedBilinearTopK(nn.Module):
    def __init__(self, latent_dim: int, num_heads: int = 8):
        super().__init__()
        if latent_dim % num_heads:
            raise ValueError("latent_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = latent_dim // num_heads
        self.bilinear_w = nn.Parameter(torch.empty(num_heads, self.head_dim, self.head_dim))
        nn.init.xavier_uniform_(self.bilinear_w)
        self.q_proj = nn.Linear(latent_dim, latent_dim)
        self.k_proj = nn.Linear(latent_dim, latent_dim)
        self.v_proj = nn.Linear(latent_dim, latent_dim)
        self.out_proj = nn.Linear(latent_dim, latent_dim)

    def forward(
        self,
        drug_query: torch.Tensor,
        protein_tokens: torch.Tensor,
        protein_mask: torch.Tensor,
        k_ratio: float,
    ) -> torch.Tensor:
        batch_size, token_count, _ = protein_tokens.shape
        q = self.q_proj(drug_query).view(batch_size, self.num_heads, 1, self.head_dim)
        k = self.k_proj(protein_tokens).view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(protein_tokens).view(batch_size, token_count, self.num_heads, self.head_dim).transpose(1, 2)
        q_weighted = torch.matmul(q, self.bilinear_w.unsqueeze(0))
        scores = torch.matmul(q_weighted, k.transpose(-1, -2)).squeeze(2)
        scores = scores.masked_fill(protein_mask[:, None, :], -1e4)

        valid_counts = (~protein_mask).sum(dim=1)
        k_value = max(1, int(token_count * k_ratio))
        k_value = min(k_value, int(valid_counts.min().item()))
        _, topk = torch.topk(scores, k_value, dim=-1)
        topk_mask = torch.full_like(scores, -1e4)
        topk_mask.scatter_(2, topk, 0.0)
        attention = F.softmax(scores + topk_mask, dim=-1)
        context = torch.matmul(attention.unsqueeze(2), v).squeeze(2).reshape(batch_size, -1)
        return self.out_proj(context)


class NexusDTIAffinityRegressor(nn.Module):
    def __init__(
        self,
        drug_dim: int = 1536,
        protein_dim: int = 1280,
        latent_dim: int = 512,
        pool_fusion: bool = False,
        drug_attention: bool = False,
        pocket_fusion: bool = False,
        pocket_residual: bool = False,
        hybrid: bool = False,
        censor_aware: bool = False,
        hard_censor_output: bool = True,
        prediction_floor: float = 5.0,
        pooled_mean: np.ndarray | None = None,
        pooled_std: np.ndarray | None = None,
        dropout: float = 0.10,
    ):
        super().__init__()
        self.pool_fusion = pool_fusion
        self.drug_attention = drug_attention
        self.pocket_fusion = pocket_fusion
        self.pocket_residual = pocket_residual
        if pocket_residual and not pocket_fusion:
            raise ValueError("Pocket residual mode requires --pocket-fusion")
        self.hybrid = hybrid
        self.censor_aware = censor_aware
        self.hard_censor_output = hard_censor_output
        drug_encoder_input_dim = latent_dim if drug_attention else drug_dim
        self.drug_encoder = nn.Sequential(
            nn.Linear(drug_encoder_input_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(),
            ResidualBlock(latent_dim),
            nn.Linear(latent_dim, latent_dim),
        )
        if drug_attention:
            self.drug_token_proj = nn.Linear(drug_dim, latent_dim)
            self.drug_self_attention = nn.MultiheadAttention(latent_dim, 8, batch_first=True)
            self.drug_norm = nn.LayerNorm(latent_dim)
        self.protein_esm_proj = nn.Linear(protein_dim, latent_dim)
        self.protein_phys_proj = nn.Sequential(nn.Linear(7, 64), nn.SiLU(), nn.Linear(64, 32))
        self.protein_token_proj = nn.Linear(latent_dim + 32, latent_dim)
        self.protein_attention = nn.MultiheadAttention(latent_dim, 8, batch_first=True)
        self.protein_norm = nn.LayerNorm(latent_dim)
        self.iab1 = MaskedBilinearTopK(latent_dim)
        self.iab2 = MaskedBilinearTopK(latent_dim)
        self.iab_norm = nn.LayerNorm(latent_dim)
        if pocket_fusion:
            # Retain atom-level Uni-Mol2 tokens for direct ligand--pocket
            # message passing rather than forcing the ligand through a mean
            # pool before it can interact with the protein.
            self.drug_pocket_proj = nn.Sequential(nn.Linear(drug_dim, latent_dim), nn.LayerNorm(latent_dim))
            self.drug_to_pocket_attention = nn.MultiheadAttention(latent_dim, 8, batch_first=True)
            self.pocket_to_drug_attention = nn.MultiheadAttention(latent_dim, 8, batch_first=True)
            self.drug_to_pocket_norm = nn.LayerNorm(latent_dim)
            self.pocket_to_drug_norm = nn.LayerNorm(latent_dim)
            if pocket_residual:
                self.pocket_residual_head = nn.Sequential(
                    nn.Linear(latent_dim * 2, 256),
                    nn.SiLU(),
                    nn.Dropout(dropout),
                    nn.Linear(256, 1),
                )
                nn.init.zeros_(self.pocket_residual_head[-1].weight)
                nn.init.zeros_(self.pocket_residual_head[-1].bias)
                # Near-zero initialization lets a loaded base model remain
                # predictive while the pocket expert earns a correction.
                self.pocket_residual_gate = nn.Parameter(torch.tensor(-3.0))
        head_input_dim = latent_dim * (
            (3 if pool_fusion else 2) + (2 if pocket_fusion and not pocket_residual else 0)
        )
        self.regressor = nn.Sequential(
            nn.Linear(head_input_dim, 1024),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, 1),
        )
        if hybrid:
            if pooled_mean is None or pooled_std is None:
                raise ValueError("Hybrid mode requires pooled train statistics")
            self.register_buffer("pooled_mean", torch.from_numpy(pooled_mean))
            self.register_buffer("pooled_std", torch.from_numpy(pooled_std))
            self.pooled_head = nn.Sequential(
                nn.Linear(drug_dim + protein_dim, 512),
                nn.LayerNorm(512),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(512, 128),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(128, 1),
            )
            self.residual_gate = nn.Parameter(torch.tensor(-2.0))
        if censor_aware:
            self.activity_head = nn.Sequential(
                nn.Linear(head_input_dim, 256),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(256, 1),
            )
            self.register_buffer("prediction_floor", torch.tensor(float(prediction_floor)))

    def forward(
        self,
        protein_tokens: torch.Tensor,
        drug_tokens: torch.Tensor,
        protein_phys: torch.Tensor,
        protein_mask: torch.Tensor,
        drug_mask: torch.Tensor,
        pocket_selector: torch.Tensor | None = None,
        k_ratio: float = 0.15,
        return_activity: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        drug_valid = (~drug_mask).unsqueeze(-1).to(drug_tokens.dtype)
        raw_drug_mean = (drug_tokens * drug_valid).sum(dim=1) / drug_valid.sum(dim=1).clamp_min(1.0)
        if self.drug_attention:
            drug_repr_tokens = self.drug_token_proj(drug_tokens)
            drug_attended, _ = self.drug_self_attention(
                drug_repr_tokens,
                drug_repr_tokens,
                drug_repr_tokens,
                key_padding_mask=drug_mask,
                need_weights=False,
            )
            drug_repr_tokens = self.drug_norm(drug_repr_tokens + drug_attended)
            drug_mean = (drug_repr_tokens * drug_valid).sum(dim=1) / drug_valid.sum(dim=1).clamp_min(1.0)
        else:
            drug_mean = raw_drug_mean
        drug_repr = self.drug_encoder(drug_mean)

        protein_raw = torch.cat(
            [self.protein_esm_proj(protein_tokens), self.protein_phys_proj(protein_phys)], dim=-1
        )
        protein_repr = self.protein_token_proj(protein_raw)
        attended, _ = self.protein_attention(
            protein_repr,
            protein_repr,
            protein_repr,
            key_padding_mask=protein_mask,
            need_weights=False,
        )
        protein_repr = self.protein_norm(protein_repr + attended)

        interaction_a = self.iab1(drug_repr, protein_repr, protein_mask, k_ratio=min(0.25, k_ratio * 2.0))
        interaction_b = self.iab2(drug_repr, protein_repr, protein_mask, k_ratio=k_ratio)
        interaction = self.iab_norm(interaction_a + interaction_b)
        pocket_features = []
        if self.pocket_fusion:
            if pocket_selector is None:
                pocket_mask = protein_mask
            else:
                pocket_mask = protein_mask | ~pocket_selector.bool()
                # The dataset has an explicit full-protein fallback, but keep
                # this guard so an external caller cannot pass an empty pocket.
                empty_pockets = pocket_mask.all(dim=1)
                if empty_pockets.any():
                    pocket_mask = pocket_mask.clone()
                    pocket_mask[empty_pockets] = protein_mask[empty_pockets]
            # Attention masks do not reduce the quadratic kernel's sequence
            # length.  Materialize the selected residues into compact padded
            # tensors so this branch actually computes on the ATP pocket,
            # rather than on all 1,024 cached ESM positions.
            pocket_lengths = (~pocket_mask).sum(dim=1)
            pocket_tokens = pad_sequence(
                [protein_repr[index, ~pocket_mask[index]] for index in range(protein_repr.shape[0])],
                batch_first=True,
            )
            pocket_padding_mask = (
                torch.arange(pocket_tokens.shape[1], device=protein_tokens.device)[None, :]
                >= pocket_lengths[:, None]
            )
            drug_pocket_tokens = self.drug_pocket_proj(drug_tokens)
            drug_context, _ = self.drug_to_pocket_attention(
                drug_pocket_tokens,
                pocket_tokens,
                pocket_tokens,
                key_padding_mask=pocket_padding_mask,
                need_weights=False,
            )
            drug_context = self.drug_to_pocket_norm(drug_pocket_tokens + drug_context)
            drug_context = (drug_context * drug_valid).sum(dim=1) / drug_valid.sum(dim=1).clamp_min(1.0)
            pocket_context, _ = self.pocket_to_drug_attention(
                pocket_tokens,
                drug_pocket_tokens,
                drug_pocket_tokens,
                key_padding_mask=drug_mask,
                need_weights=False,
            )
            pocket_context = self.pocket_to_drug_norm(pocket_tokens + pocket_context)
            pocket_valid = (~pocket_padding_mask).unsqueeze(-1).to(pocket_context.dtype)
            pocket_context = (pocket_context * pocket_valid).sum(dim=1) / pocket_valid.sum(dim=1).clamp_min(1.0)
            pocket_features = [drug_context, pocket_context]
        if self.pool_fusion:
            protein_valid = (~protein_mask).unsqueeze(-1).to(protein_repr.dtype)
            protein_global = (protein_repr * protein_valid).sum(dim=1) / protein_valid.sum(dim=1).clamp_min(1.0)
            base_head_input = torch.cat([drug_repr, interaction, protein_global], dim=-1)
        else:
            base_head_input = torch.cat([drug_repr, interaction], dim=-1)
        head_input = (
            base_head_input
            if self.pocket_residual
            else torch.cat([base_head_input, *pocket_features], dim=-1)
        )
        interaction_pred = self.regressor(head_input).squeeze(-1)
        if self.hybrid:
            raw_protein_valid = (~protein_mask).unsqueeze(-1).to(protein_tokens.dtype)
            raw_protein_mean = (protein_tokens * raw_protein_valid).sum(dim=1) / raw_protein_valid.sum(dim=1).clamp_min(1.0)
            pooled = torch.cat([raw_drug_mean, raw_protein_mean], dim=-1)
            pooled = (pooled - self.pooled_mean.to(pooled.dtype)) / self.pooled_std.to(pooled.dtype)
            pooled_pred = self.pooled_head(pooled).squeeze(-1)
            raw_prediction = pooled_pred + torch.sigmoid(self.residual_gate) * interaction_pred
        else:
            raw_prediction = interaction_pred
        if self.pocket_residual:
            pocket_prediction = self.pocket_residual_head(torch.cat(pocket_features, dim=-1)).squeeze(-1)
            raw_prediction = raw_prediction + torch.sigmoid(self.pocket_residual_gate) * pocket_prediction
        if not self.censor_aware:
            return raw_prediction
        activity_logit = self.activity_head(head_input).squeeze(-1)
        if self.hard_censor_output:
            active_probability = torch.sigmoid(activity_logit)
            prediction = self.prediction_floor + active_probability * F.softplus(raw_prediction - self.prediction_floor)
        else:
            prediction = raw_prediction
        return (prediction, activity_logit) if return_activity else prediction


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cindex(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    concordant = 0.0
    comparable = 0.0
    for index in range(len(y_true)):
        delta = y_true[index] - y_true[index + 1 :]
        valid = delta != 0
        if not valid.any():
            continue
        pred_delta = y_pred[index] - y_pred[index + 1 :]
        concordant += np.sum((pred_delta[valid] * delta[valid]) > 0)
        concordant += 0.5 * np.sum(pred_delta[valid] == 0)
        comparable += np.sum(valid)
    return float(concordant / comparable) if comparable else float("nan")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "mse": float(np.mean((y_true - y_pred) ** 2)),
        "ci": cindex(y_true, y_pred),
        "pearson": float(pearsonr(y_true, y_pred).statistic),
        "spearman": float(spearmanr(y_true, y_pred).statistic),
    }


def initialize_from_checkpoint(model: nn.Module, checkpoint_path: Path) -> dict[str, int | str]:
    """Load shape-compatible baseline weights and preserve its old head path.

    Pocket fusion appends two latent vectors to the original regression input.
    The old columns of the first regression layer are copied exactly and the
    new columns are initialized to zero.  Therefore the first forward pass is
    the initialized NEXUS-DTA model, while gradients can progressively turn on
    the new pocket branch.  This avoids conflating architectural value with a
    fresh random initialization on the small Davis panel.
    """
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = payload.get("model", payload)
    target = model.state_dict()
    compatible = {
        key: value
        for key, value in source.items()
        if key in target and target[key].shape == value.shape
    }
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    copied_head_columns = 0
    source_head = source.get("regressor.0.weight")
    if source_head is not None and source_head.ndim == 2:
        target_head = model.regressor[0].weight
        overlap = min(source_head.shape[1], target_head.shape[1])
        if source_head.shape[0] == target_head.shape[0]:
            with torch.no_grad():
                target_head.zero_()
                target_head[:, :overlap].copy_(source_head[:, :overlap])
            copied_head_columns = int(overlap)
    source_bias = source.get("regressor.0.bias")
    if source_bias is not None and source_bias.shape == model.regressor[0].bias.shape:
        with torch.no_grad():
            model.regressor[0].bias.copy_(source_bias)
    return {
        "checkpoint": str(checkpoint_path),
        "compatible_tensors": len(compatible),
        "missing_tensors": len(missing),
        "unexpected_tensors": len(unexpected),
        "copied_regressor_input_columns": copied_head_columns,
    }


def initialize_shared_encoder_from_checkpoint(model: nn.Module, checkpoint_path: Path) -> dict[str, int | str]:
    """Overlay only transferable token-interaction parameters from a source task.

    In particular, a BindingDB DTI logit head and a Davis pKd head have
    incompatible semantics.  This loader transfers the shared drug/protein
    encoders, protein attention and bilinear interaction blocks while keeping
    the current affinity heads intact.
    """
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    source = payload.get("model", payload)
    transferable_prefixes = (
        "drug_encoder.",
        "protein_esm_proj.",
        "protein_phys_proj.",
        "protein_token_proj.",
        "protein_attention.",
        "protein_norm.",
        "iab1.",
        "iab2.",
        "iab_norm.",
    )
    target = model.state_dict()
    compatible = {
        key: value
        for key, value in source.items()
        if key.startswith(transferable_prefixes) and key in target and target[key].shape == value.shape
    }
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return {
        "checkpoint": str(checkpoint_path),
        "transferred_tensors": len(compatible),
        "missing_tensors": len(missing),
        "unexpected_tensors": len(unexpected),
    }


@torch.no_grad()
def predict(model, loader, device: torch.device, k_ratio: float) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    targets, predictions = [], []
    amp = device.type == "cuda"
    for protein, drug, phys, pocket_selector, protein_mask, drug_mask, labels in loader:
        protein = protein.to(device, non_blocking=amp)
        drug = drug.to(device, non_blocking=amp)
        phys = phys.to(device, non_blocking=amp)
        pocket_selector = pocket_selector.to(device, non_blocking=amp)
        protein_mask = protein_mask.to(device, non_blocking=amp)
        drug_mask = drug_mask.to(device, non_blocking=amp)
        with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
            pred = model(
                protein,
                drug,
                phys,
                protein_mask,
                drug_mask,
                pocket_selector=pocket_selector,
                k_ratio=k_ratio,
            )
        targets.append(labels.numpy())
        predictions.append(pred.float().cpu().numpy())
    return np.concatenate(targets), np.concatenate(predictions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        default="random",
        help="Result/split label. Without --manifest-file it must name a native Davis split.",
    )
    parser.add_argument(
        "--data-table",
        default="data/processed/davis/davis_entity_affinity_clean.csv",
        help="Affinity table relative to the DTA project (or an absolute path).",
    )
    parser.add_argument(
        "--manifest-file",
        default=None,
        help="Optional train/val/test manifest relative to the DTA project (or absolute).",
    )
    parser.add_argument(
        "--dataset-label",
        default="davis_clean",
        help="Descriptive dataset label recorded in the result JSON.",
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--token-dir", default=None, help="Token cache root; defaults to the existing Davis cache.")
    parser.add_argument("--target-column", default="pKd_floor", help="Native affinity label column, e.g. kiba_score.")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--save-predictions", action="store_true", help="Save raw validation/test predictions for ablation.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--latent-dim", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--k-ratio", type=float, default=0.15)
    parser.add_argument("--pool-fusion", action="store_true", help="Add masked global protein representation to the regression head")
    parser.add_argument("--drug-attention", action="store_true", help="Use self-attention over drug atom tokens before pooling")
    parser.add_argument(
        "--pocket-fusion",
        action="store_true",
        help="Fuse atom-level drug tokens with a sequence-defined kinase ATP-pocket neighborhood.",
    )
    parser.add_argument(
        "--pocket-radius",
        type=int,
        default=48,
        help="Residues retained around VAIK--HRD--DFG kinase-pocket anchors.",
    )
    parser.add_argument(
        "--pocket-residual",
        action="store_true",
        help="Add a gated pocket-expert correction while preserving the original NEXUS-DTA prediction path.",
    )
    parser.add_argument(
        "--freeze-base",
        action="store_true",
        help="Freeze inherited NEXUS-DTA parameters; train only the added pocket expert and its gate.",
    )
    parser.add_argument("--hybrid", action="store_true", help="Use pooled baseline prediction plus gated token-interaction residual")
    parser.add_argument(
        "--censor-aware",
        action="store_true",
        help="Add a pKd-floor activity head; outputs are constrained only unless --activity-aux-only is set.",
    )
    parser.add_argument("--censor-floor", type=float, default=5.0, help="Observed pKd floor used to identify censored samples.")
    parser.add_argument(
        "--activity-aux-only",
        action="store_true",
        help="Train the floor-activity head as an auxiliary task without forcing final predictions above --censor-floor.",
    )
    parser.add_argument(
        "--activity-loss-weight",
        type=float,
        default=0.25,
        help="Weight of BCE loss for pKd > --censor-floor activity classification in censor-aware mode.",
    )
    parser.add_argument("--target-normalize", action="store_true", help="Train in standardized pKd space and invert predictions for metrics")
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument(
        "--init-checkpoint",
        default=None,
        help="Optional compatible baseline checkpoint used to initialize shared layers and the old regression-head columns.",
    )
    parser.add_argument(
        "--encoder-init-checkpoint",
        default=None,
        help="Optional source-task checkpoint whose shared encoders replace the current initialization; affinity heads are retained.",
    )
    parser.add_argument("--run-tag", default="", help="Suffix for result/checkpoint filenames")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261012)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was explicitly requested but is unavailable in this execution environment.")
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_name)
    set_seed(args.seed)

    def resolve_project_path(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else PROJECT / path

    table_path = resolve_project_path(args.data_table)
    token_dir = resolve_project_path(args.token_dir) if args.token_dir else TOKEN_DIR
    if args.manifest_file is None:
        native_splits = {"random", "cold_drug", "cold_target", "double_cold"}
        if args.split not in native_splits:
            raise ValueError(
                "A custom --split label requires --manifest-file; expected one of "
                f"{sorted(native_splits)} for the native Davis manifests."
            )
        manifest_path = PROJECT / "data/processed/davis_clean/splits" / args.split / "manifest.csv"
    else:
        manifest_path = resolve_project_path(args.manifest_file)

    table = pd.read_csv(table_path)
    if args.target_column not in table or not np.isfinite(table[args.target_column].to_numpy(dtype=float)).all():
        raise ValueError(f"Missing or non-finite affinity column: {args.target_column}")
    if args.censor_aware and args.target_column != "pKd_floor":
        raise ValueError("The pKd censoring head is not applicable to a different affinity scale.")
    if "row_id" not in table.columns:
        table = table.copy()
        table.insert(0, "row_id", np.arange(len(table), dtype=np.int64))
    manifest = pd.read_csv(manifest_path)
    joined = manifest.merge(table, on="row_id", how="left", validate="one_to_one")
    if joined[["Sequence", "Canonical_SMILES"]].isna().any().any():
        raise RuntimeError("Split manifest and Davis affinity table are not aligned")

    train_df = joined[joined.split == "train"].reset_index(drop=True)
    val_df = joined[joined.split == "val"].reset_index(drop=True)
    test_df = joined[joined.split == "test"].reset_index(drop=True)
    dataset_options = dict(kinase_pocket=args.pocket_fusion, pocket_radius=args.pocket_radius,
                           token_dir=token_dir, target_column=args.target_column)
    train_ds = AffinityDataset(train_df, **dataset_options)
    val_ds = AffinityDataset(val_df, **dataset_options)
    test_ds = AffinityDataset(test_df, **dataset_options)
    pin_memory = device.type == "cuda"
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=True,
        persistent_workers=args.workers > 0,
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
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        persistent_workers=args.workers > 0,
    )

    target_mean = float(train_df[args.target_column].mean())
    target_std = float(train_df[args.target_column].std())
    if target_std < 1e-6:
        target_std = 1.0
    prediction_floor = (
        (args.censor_floor - target_mean) / target_std if args.target_normalize else args.censor_floor
    )
    pooled_mean, pooled_std = (build_pooled_stats(train_df, token_dir=token_dir) if args.hybrid else (None, None))
    model = NexusDTIAffinityRegressor(
        latent_dim=args.latent_dim,
        pool_fusion=args.pool_fusion,
        drug_attention=args.drug_attention,
        pocket_fusion=args.pocket_fusion,
        pocket_residual=args.pocket_residual,
        hybrid=args.hybrid,
        censor_aware=args.censor_aware,
        hard_censor_output=not args.activity_aux_only,
        prediction_floor=prediction_floor,
        pooled_mean=pooled_mean,
        pooled_std=pooled_std,
        dropout=args.dropout,
    ).to(device)
    init_report = None
    if args.init_checkpoint:
        init_path = Path(args.init_checkpoint)
        if not init_path.is_absolute():
            init_path = PROJECT / init_path
        if not init_path.exists():
            raise FileNotFoundError(f"Initialization checkpoint does not exist: {init_path}")
        init_report = initialize_from_checkpoint(model, init_path)
        print(f"initialized_from={json.dumps(init_report)}", flush=True)
    encoder_init_report = None
    if args.encoder_init_checkpoint:
        encoder_init_path = Path(args.encoder_init_checkpoint)
        if not encoder_init_path.is_absolute():
            encoder_init_path = PROJECT / encoder_init_path
        if not encoder_init_path.exists():
            raise FileNotFoundError(f"Encoder initialization checkpoint does not exist: {encoder_init_path}")
        encoder_init_report = initialize_shared_encoder_from_checkpoint(model, encoder_init_path)
        print(f"encoder_initialized_from={json.dumps(encoder_init_report)}", flush=True)
    frozen_base_parameters = 0
    if args.freeze_base:
        if not args.pocket_residual:
            raise ValueError("--freeze-base is only valid with --pocket-residual")
        trainable_prefixes = (
            "drug_pocket_proj",
            "drug_to_pocket_attention",
            "pocket_to_drug_attention",
            "drug_to_pocket_norm",
            "pocket_to_drug_norm",
            "pocket_residual_head",
            "pocket_residual_gate",
        )
        for name, parameter in model.named_parameters():
            if not name.startswith(trainable_prefixes):
                parameter.requires_grad_(False)
                frozen_base_parameters += parameter.numel()
        if init_report is None:
            raise ValueError("--freeze-base requires --init-checkpoint so the frozen path is a trained base model")
        print(f"frozen_base_parameters={frozen_base_parameters}", flush=True)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad], lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loss_fn = nn.SmoothL1Loss(beta=0.25)
    best_val = float("inf")
    best_state = None
    stale_epochs = 0
    history = []
    initial_validation = None
    # A source-/base-initialized model is itself a valid validation candidate.
    # This is essential for residual and transfer experiments: an update that
    # does not beat the inherited model must not overwrite it merely because
    # epoch 1 is the first evaluated state.
    if args.init_checkpoint or args.encoder_init_checkpoint:
        y_initial, pred_initial = predict(model, val_loader, device, args.k_ratio)
        if args.target_normalize:
            pred_initial = pred_initial * target_std + target_mean
        initial_validation = regression_metrics(y_initial, pred_initial)
        best_val = initial_validation["rmse"]
        best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}

    print(
        f"device={device} dataset={args.dataset_label} split={args.split} rows="
        f"{len(train_ds)}/{len(val_ds)}/{len(test_ds)} "
        f"batch={args.batch_size} latent={args.latent_dim} "
        f"pool_fusion={args.pool_fusion} pocket_fusion={args.pocket_fusion} "
        f"pocket_residual={args.pocket_residual} freeze_base={args.freeze_base} hybrid={args.hybrid} "
        f"censor_aware={args.censor_aware} "
        f"target_normalize={args.target_normalize}"
    )
    if initial_validation is not None:
        print(f"initial_val_rmse={initial_validation['rmse']:.5f}", flush=True)
    for epoch in range(args.epochs):
        model.train()
        losses = []
        for protein, drug, phys, pocket_selector, protein_mask, drug_mask, labels in train_loader:
            protein = protein.to(device, non_blocking=amp)
            drug = drug.to(device, non_blocking=amp)
            phys = phys.to(device, non_blocking=amp)
            pocket_selector = pocket_selector.to(device, non_blocking=amp)
            protein_mask = protein_mask.to(device, non_blocking=amp)
            drug_mask = drug_mask.to(device, non_blocking=amp)
            labels = labels.to(device, non_blocking=amp)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                model_output = model(
                    protein,
                    drug,
                    phys,
                    protein_mask,
                    drug_mask,
                    pocket_selector=pocket_selector,
                    k_ratio=args.k_ratio,
                    return_activity=args.censor_aware,
                )
                if args.censor_aware:
                    pred, activity_logit = model_output
                else:
                    pred = model_output
                train_targets = (labels - target_mean) / target_std if args.target_normalize else labels
                loss = loss_fn(pred, train_targets)
                if args.censor_aware:
                    activity_targets = (labels > args.censor_floor).to(labels.dtype)
                    activity_loss = F.binary_cross_entropy_with_logits(activity_logit, activity_targets)
                    loss = loss + args.activity_loss_weight * activity_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.detach().float().cpu()))
        scheduler.step()

        y_val, pred_val = predict(model, val_loader, device, args.k_ratio)
        if args.target_normalize:
            pred_val = pred_val * target_std + target_mean
        val_rmse = float(np.sqrt(np.mean((y_val - pred_val) ** 2)))
        mean_loss = float(np.mean(losses))
        history.append({"epoch": epoch + 1, "train_loss": mean_loss, "val_rmse": val_rmse})
        print(f"epoch={epoch + 1:02d} train_loss={mean_loss:.5f} val_rmse={val_rmse:.5f}", flush=True)
        if val_rmse < best_val:
            best_val = val_rmse
            stale_epochs = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"early_stop at epoch={epoch + 1}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    y_test, pred_test = predict(model, test_loader, device, args.k_ratio)
    y_train, pred_train = predict(model, train_loader, device, args.k_ratio)
    if args.target_normalize:
        pred_test = pred_test * target_std + target_mean
        pred_train = pred_train * target_std + target_mean
    tag = f"_{args.run_tag}" if args.run_tag else ""
    result = {
        "dataset": f"{args.dataset_label}_nexusdti_style_regressor",
        "data_table": str(table_path),
        "manifest_file": str(manifest_path),
        "token_dir": str(token_dir),
        "target_column": args.target_column,
        "input_sha256": {
            "data_table": hashlib.sha256(table_path.read_bytes()).hexdigest(),
            "manifest": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        },
        "split": args.split,
        "seed": args.seed,
        "device": str(device),
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "batch_size": args.batch_size,
        "lr": args.lr,
        "patience": args.patience,
        "workers": args.workers,
        "latent_dim": args.latent_dim,
        "k_ratio": args.k_ratio,
        "pool_fusion": args.pool_fusion,
        "drug_attention": args.drug_attention,
        "pocket_fusion": args.pocket_fusion,
        "pocket_radius": args.pocket_radius if args.pocket_fusion else None,
        "pocket_residual": args.pocket_residual,
        "freeze_base": args.freeze_base,
        "frozen_base_parameters": frozen_base_parameters,
        "hybrid": args.hybrid,
        "censor_aware": args.censor_aware,
        "censor_floor": args.censor_floor,
        "activity_aux_only": args.activity_aux_only if args.censor_aware else None,
        "activity_loss_weight": args.activity_loss_weight if args.censor_aware else None,
        "target_normalize": args.target_normalize,
        "target_mean": target_mean,
        "target_std": target_std,
        "dropout": args.dropout,
        "initialization": init_report,
        "encoder_initialization": encoder_init_report,
        "run_tag": args.run_tag,
        "rows_train": len(train_ds),
        "rows_val": len(val_ds),
        "rows_test": len(test_ds),
        "best_val_rmse": best_val,
        "initial_validation": initial_validation,
        "metrics_train": regression_metrics(y_train, pred_train),
        "metrics_test": regression_metrics(y_test, pred_test),
        "history": history,
    }
    if args.hybrid:
        result["residual_gate"] = float(torch.sigmoid(model.residual_gate).detach().cpu())
    if args.pocket_residual:
        result["pocket_residual_gate"] = float(torch.sigmoid(model.pocket_residual_gate).detach().cpu())
    result_dir = PROJECT / "results"
    checkpoint_dir = result_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args), "result": result}, checkpoint_dir / f"nexusdti_{args.split}{tag}.pt")
    out = result_dir / f"nexusdti_regressor_{args.split}{tag}.json"
    out.write_text(json.dumps(result, indent=2))
    if args.save_predictions:
        y_val, pred_val = predict(model, val_loader, device, args.k_ratio)
        if args.target_normalize:
            pred_val = pred_val * target_std + target_mean
        frames = []
        for split_name, frame, labels, predictions in (
            ("val", val_df, y_val, pred_val), ("test", test_df, y_test, pred_test),
        ):
            frames.append(pd.DataFrame({
                "split": split_name, "row_id": frame.row_id.to_numpy(),
                "affinity_true": labels, "prediction_raw": predictions,
            }))
        prediction_dir = result_dir / "predictions"
        prediction_dir.mkdir(parents=True, exist_ok=True)
        pd.concat(frames, ignore_index=True).to_csv(
            prediction_dir / f"nexusdti_{args.split}{tag}_predictions.csv", index=False,
        )
    print(json.dumps(result, indent=2))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
