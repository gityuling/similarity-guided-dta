#!/usr/bin/env python3
"""Build an isolated Uni-Mol2/ESM2 token cache for strict BindingDB pKd data.

The cache is deliberately separate from the legacy DTI caches.  It is keyed by
canonical SMILES and raw amino-acid sequence, so it is safe to resume and can
be reused by every fold-specific BindingDB pKd table that contains the same
entity.  The input table is expected to come from
``prepare_bindingdb_articles_kd.py`` and therefore has already applied the
outer-fold downstream drug exclusion.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import sys
from argparse import Namespace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm
from transformers import EsmModel, EsmTokenizer


ROOT = Path(__file__).resolve().parents[2]
PROJECT = ROOT / "DTA_affinity"
UNIMOL2_DIR = ROOT / "unimol2"
UNIMOL2_CHECKPOINT = UNIMOL2_DIR / "checkpoint.pt"
ESM2_DIR = ROOT / "ESM2" / "esm2_model"
DEFAULT_TABLE = PROJECT / "data/processed/bindingdb_articles_kd/bindingdb_articles_kd_f0.csv"
ATOM_TYPES = [
    "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne", "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
    "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn", "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y",
    "Zr", "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce",
    "Pr", "Nd", "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir",
    "Pt", "Au", "Hg", "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th", "Pa", "U", "Np", "Pu", "Am", "Cm",
    "Bk", "Cf", "Es", "Fm", "Md", "No", "Lr", "Rf", "Db", "Sg", "Bh", "Hs", "Mt", "Ds", "Rg", "Cn", "Nh", "Fl", "Mc",
    "Lv", "Ts", "Og",
]


def md5(value: str) -> str:
    return hashlib.md5(value.encode()).hexdigest()


def canonical_smiles(value: str) -> str | None:
    try:
        molecule = Chem.MolFromSmiles(str(value))
        return Chem.MolToSmiles(molecule) if molecule is not None else None
    except (TypeError, ValueError):
        return None


def atom_count_with_hydrogens(smiles: str) -> int | None:
    """Return the actual token count used by Uni-Mol2 before padding.

    Large macrocycles/peptides are rare in this source but dominate the
    quadratic pair representation.  Counting explicit hydrogens here lets us
    keep normal drug-like molecules batched while treating long outliers
    safely and reproducibly.
    """
    try:
        molecule = Chem.MolFromSmiles(smiles)
        return Chem.AddHs(molecule).GetNumAtoms() if molecule is not None else None
    except (TypeError, ValueError):
        return None


class UniMol2Extractor:
    """Minimal inference wrapper matched to the original NexusDTI token cache."""

    def __init__(self, device: torch.device) -> None:
        if not UNIMOL2_CHECKPOINT.exists():
            raise FileNotFoundError(f"Missing Uni-Mol2 checkpoint: {UNIMOL2_CHECKPOINT}")
        if str(UNIMOL2_DIR) not in sys.path:
            sys.path.insert(0, str(UNIMOL2_DIR))
        from unimol2.data.molecule_dataset import get_graph
        from unimol2.data.unimol2_dataset import (
            get_graph_features,
            pad_1d,
            pad_1d_feat,
            pad_2d,
            pad_2d_feat,
            pad_attn_bias,
        )
        from unimol2.models.unimol2 import UniMol2Model

        args = Namespace(
            encoder_layers=64,
            encoder_embed_dim=1536,
            pair_embed_dim=512,
            pair_hidden_dim=64,
            encoder_ffn_embed_dim=1536,
            encoder_attention_heads=96,
            dropout=0.1,
            emb_dropout=0.1,
            attention_dropout=0.1,
            activation_dropout=0.0,
            pooler_dropout=0.0,
            max_seq_len=512,
            activation_fn="gelu",
            pooler_activation_fn="tanh",
            post_ln=False,
            masked_token_loss=-1.0,
            masked_coord_loss=-1.0,
            masked_dist_loss=-1.0,
            x_norm_loss=-1.0,
            delta_pair_repr_norm_loss=-1.0,
            notri=False,
            gaussian_std_width=1.0,
            gaussian_mean_start=0.0,
            gaussian_mean_stop=9.0,
            mode="infer",
            droppath_prob=0.0,
        )
        self.device = device
        self.atom_dict = {atom: index + 3 for index, atom in enumerate(ATOM_TYPES)}
        self.get_graph = get_graph
        self.get_graph_features = get_graph_features
        self.pad_1d = pad_1d
        self.pad_1d_feat = pad_1d_feat
        self.pad_2d = pad_2d
        self.pad_2d_feat = pad_2d_feat
        self.pad_attn_bias = pad_attn_bias
        self.model = UniMol2Model(args)
        state = torch.load(UNIMOL2_CHECKPOINT, map_location="cpu", weights_only=False)
        self.model.load_state_dict(state["model"], strict=False)
        self.model.to(device).eval()

    def _molecule_features(self, smiles: str) -> dict | None:
        molecule = Chem.MolFromSmiles(smiles)
        if molecule is None:
            return None
        molecule = AllChem.AddHs(molecule, addCoords=True)
        params = AllChem.ETKDG()
        params.randomSeed = 20261016
        if AllChem.EmbedMolecule(molecule, params) == -1:
            if AllChem.EmbedMolecule(molecule, randomSeed=20261016, useRandomCoords=True) == -1:
                return None
        try:
            positions = molecule.GetConformer().GetPositions()
        except ValueError:
            return None
        atoms = [atom.GetSymbol() for atom in molecule.GetAtoms()]
        x_h, edge_index_h, edge_attr_h = self.get_graph(molecule)
        feature = self.get_graph_features(edge_attr_h, edge_index_h, x_h, drop_feat=False)
        feature["src_token"] = torch.tensor([self.atom_dict.get(atom, 3) for atom in atoms], dtype=torch.long)
        feature["src_pos"] = torch.from_numpy(positions).float()
        return feature

    def encode(self, smiles_batch: list[str]) -> dict[str, np.ndarray]:
        features: list[dict] = []
        valid_smiles: list[str] = []
        for smiles in smiles_batch:
            feature = self._molecule_features(smiles)
            if feature is not None:
                features.append(feature)
                valid_smiles.append(smiles)
        if not features:
            return {}
        max_nodes = max(item["atom_mask"].shape[0] for item in features)
        max_nodes = (max_nodes + 1 + 3) // 4 * 4 - 1
        collated = {
            "src_token": self.pad_1d([item["src_token"] for item in features], max_nodes).to(self.device),
            "src_pos": self.pad_1d_feat([item["src_pos"] for item in features], max_nodes).to(self.device),
            "atom_feat": self.pad_1d_feat([item["atom_feat"] for item in features], max_nodes).to(self.device),
            "atom_mask": self.pad_1d([item["atom_mask"] for item in features], max_nodes).to(self.device),
            "edge_feat": self.pad_2d_feat([item["edge_feat"] for item in features], max_nodes).to(self.device),
            "pair_type": self.pad_2d_feat([item["pair_type"] for item in features], max_nodes).to(self.device),
            "attn_bias": self.pad_attn_bias([item["attn_bias"] for item in features], max_nodes).to(self.device),
            "degree": self.pad_1d([item["degree"] for item in features], max_nodes).to(self.device),
            "shortest_path": self.pad_2d([item["shortest_path"] for item in features], max_nodes).to(self.device),
        }
        with torch.inference_mode():
            representations, _ = self.model(collated, features_only=True)
        result = {}
        for index, smiles in enumerate(valid_smiles):
            atom_count = features[index]["atom_mask"].shape[0]
            result[smiles] = representations[index, 1 : atom_count + 1].float().cpu().numpy()
        return result


def chunks(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


def save_protein_tokens(
    sequences: list[str], protein_dir: Path, device: torch.device, batch_size: int
) -> tuple[int, int]:
    pending = [sequence for sequence in sequences if not (protein_dir / f"{md5(sequence)}.npy").exists()]
    if not pending:
        return 0, 0
    if not ESM2_DIR.exists():
        raise FileNotFoundError(f"Missing local ESM2 model: {ESM2_DIR}")
    tokenizer = EsmTokenizer.from_pretrained(ESM2_DIR, local_files_only=True)
    model = EsmModel.from_pretrained(ESM2_DIR, local_files_only=True).to(device).eval()
    completed = failures = 0
    for batch in tqdm(list(chunks(pending, batch_size)), desc="ESM2 protein tokens"):
        try:
            encoded = tokenizer(batch, padding=True, truncation=True, max_length=1024, return_tensors="pt").to(device)
            with torch.inference_mode():
                hidden = model(**encoded).last_hidden_state.float().cpu().numpy()
            masks = encoded.attention_mask.cpu().numpy().astype(bool)
            for index, sequence in enumerate(batch):
                np.save(protein_dir / f"{md5(sequence)}.npy", hidden[index][masks[index]].astype(np.float32))
                completed += 1
        except RuntimeError as error:
            print(f"Protein batch failed ({error}); retrying items singly.", flush=True)
            for sequence in batch:
                try:
                    encoded = tokenizer([sequence], truncation=True, max_length=1024, return_tensors="pt").to(device)
                    with torch.inference_mode():
                        hidden = model(**encoded).last_hidden_state[0].float().cpu().numpy()
                    mask = encoded.attention_mask[0].cpu().numpy().astype(bool)
                    np.save(protein_dir / f"{md5(sequence)}.npy", hidden[mask].astype(np.float32))
                    completed += 1
                except RuntimeError as item_error:
                    print(f"Protein failed: {item_error}", flush=True)
                    failures += 1
                    torch.cuda.empty_cache()
        torch.cuda.empty_cache()
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    return completed, failures


def save_drug_tokens(
    smiles_and_sizes: list[tuple[str, int]],
    drug_dir: Path,
    device: torch.device,
    batch_size: int,
    max_batched_atoms: int,
) -> tuple[int, int]:
    pending = [
        (smiles, atoms)
        for smiles, atoms in smiles_and_sizes
        if not (drug_dir / f"{md5(smiles)}.npy").exists()
    ]
    if not pending:
        return 0, 0
    # Padding is to the largest molecule in a batch and Uni-Mol2 has several
    # quadratic pair tensors.  Homogeneous small batches avoid one macrocycle
    # multiplying the memory/runtime of three ordinary drug-like molecules.
    small = sorted((item for item in pending if item[1] <= max_batched_atoms), key=lambda item: (item[1], item[0]))
    large = sorted((item for item in pending if item[1] > max_batched_atoms), key=lambda item: (item[1], item[0]))
    batches = [batch for batch in chunks([smiles for smiles, _ in small], batch_size)]
    batches.extend([[smiles] for smiles, _ in large])
    extractor = UniMol2Extractor(device)
    completed = failures = 0
    for batch in tqdm(batches, desc="Uni-Mol2 drug tokens"):
        try:
            result = extractor.encode(batch)
            for smiles, token in result.items():
                np.save(drug_dir / f"{md5(smiles)}.npy", token.astype(np.float32))
                completed += 1
            failures += len(batch) - len(result)
        except RuntimeError as error:
            print(f"Drug batch failed ({error}); retrying items singly.", flush=True)
            for smiles in batch:
                try:
                    result = extractor.encode([smiles])
                    if smiles in result:
                        np.save(drug_dir / f"{md5(smiles)}.npy", result[smiles].astype(np.float32))
                        completed += 1
                    else:
                        failures += 1
                except RuntimeError as item_error:
                    print(f"Drug failed: {item_error}", flush=True)
                    failures += 1
                    torch.cuda.empty_cache()
        torch.cuda.empty_cache()
    del extractor
    gc.collect()
    torch.cuda.empty_cache()
    return completed, failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-table", type=Path, default=DEFAULT_TABLE)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--audit-filename",
        default="build_audit.json",
        help="Filename for this source-table build audit inside --cache-dir.",
    )
    parser.add_argument("--protein-batch-size", type=int, default=4)
    parser.add_argument("--drug-batch-size", type=int, default=4)
    parser.add_argument(
        "--max-batched-drug-atoms",
        type=int,
        default=96,
        help="Maximum explicit-atom count allowed in a multi-drug Uni-Mol2 batch.",
    )
    parser.add_argument(
        "--max-drug-atoms",
        type=int,
        default=256,
        help="Exclude larger molecules: they exceed practical full-Uni-Mol2 inference memory.",
    )
    parser.add_argument("--limit-proteins", type=int, default=None, help="Smoke-test cap; applied after deterministic sorting.")
    parser.add_argument("--limit-drugs", type=int, default=None, help="Smoke-test cap; applied after deterministic sorting.")
    args = parser.parse_args()
    source_table = args.source_table if args.source_table.is_absolute() else ROOT / args.source_table
    cache_dir = args.cache_dir if args.cache_dir.is_absolute() else ROOT / args.cache_dir
    source = pd.read_csv(source_table)
    required = {"canonical_smiles", "sequence", "pKd"}
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"Source table lacks required columns: {sorted(missing)}")
    source["canonical_smiles"] = source.canonical_smiles.map(canonical_smiles)
    source = source.dropna(subset=["canonical_smiles", "sequence"]).copy()
    source["sequence"] = source.sequence.astype(str).str.strip()
    source = source.loc[source.sequence.ne("")].copy()
    sequences = sorted(source.sequence.unique().tolist())
    drug_sizes = {
        smiles: atom_count_with_hydrogens(smiles)
        for smiles in sorted(source.canonical_smiles.unique().tolist())
    }
    invalid_smiles = [smiles for smiles, atoms in drug_sizes.items() if atoms is None]
    oversized_smiles = [
        smiles
        for smiles, atoms in drug_sizes.items()
        if atoms is not None and atoms > args.max_drug_atoms
    ]
    smiles_and_sizes = sorted(
        [
            (smiles, int(atoms))
            for smiles, atoms in drug_sizes.items()
            if atoms is not None and atoms <= args.max_drug_atoms
        ],
        key=lambda item: item[0],
    )
    if args.limit_proteins is not None:
        sequences = sequences[: args.limit_proteins]
    if args.limit_drugs is not None:
        smiles_and_sizes = smiles_and_sizes[: args.limit_drugs]
    if args.protein_batch_size < 1 or args.drug_batch_size < 1:
        raise ValueError("Batch sizes must be positive")
    if args.max_batched_drug_atoms < 1 or args.max_drug_atoms < args.max_batched_drug_atoms:
        raise ValueError("Require 1 <= --max-batched-drug-atoms <= --max-drug-atoms")
    if Path(args.audit_filename).name != args.audit_filename:
        raise ValueError("--audit-filename must be a plain filename")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("CUDA is required: these Uni-Mol2 embeddings must match the GPU inference cache.")
    protein_dir = cache_dir / "proteins"
    drug_dir = cache_dir / "drugs"
    protein_dir.mkdir(parents=True, exist_ok=True)
    drug_dir.mkdir(parents=True, exist_ok=True)
    print(json.dumps({
        "source_table": str(source_table),
        "cache_dir": str(cache_dir),
        "device": str(device),
        "source_pairs": int(len(source)),
        "selected_unique_proteins": len(sequences),
        "source_unique_drugs": len(drug_sizes),
        "selected_unique_drugs": len(smiles_and_sizes),
        "oversized_drugs_excluded": len(oversized_smiles),
        "invalid_drugs_excluded": len(invalid_smiles),
        "source_pairs_after_drug_size_filter": int(source.canonical_smiles.isin({smiles for smiles, _ in smiles_and_sizes}).sum()),
        "max_batched_drug_atoms": args.max_batched_drug_atoms,
        "max_drug_atoms": args.max_drug_atoms,
    }, indent=2), flush=True)
    proteins_done, proteins_failed = save_protein_tokens(sequences, protein_dir, device, args.protein_batch_size)
    drugs_done, drugs_failed = save_drug_tokens(
        smiles_and_sizes,
        drug_dir,
        device,
        args.drug_batch_size,
        args.max_batched_drug_atoms,
    )
    audit = {
        "source_table": str(source_table),
        "device": str(device),
        "selected_unique_proteins": len(sequences),
        "source_unique_drugs": len(drug_sizes),
        "selected_unique_drugs": len(smiles_and_sizes),
        "oversized_drugs_excluded": len(oversized_smiles),
        "invalid_drugs_excluded": len(invalid_smiles),
        "source_pairs_after_drug_size_filter": int(source.canonical_smiles.isin({smiles for smiles, _ in smiles_and_sizes}).sum()),
        "max_batched_drug_atoms": args.max_batched_drug_atoms,
        "max_drug_atoms": args.max_drug_atoms,
        "protein_tokens_completed_this_run": proteins_done,
        "protein_tokens_failed_this_run": proteins_failed,
        "drug_tokens_completed_this_run": drugs_done,
        "drug_tokens_failed_this_run": drugs_failed,
        "protein_tokens_total": len(list(protein_dir.glob("*.npy"))),
        "drug_tokens_total": len(list(drug_dir.glob("*.npy"))),
    }
    (cache_dir / args.audit_filename).write_text(json.dumps(audit, indent=2))
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
