#!/usr/bin/env python3
"""Create deterministic pairwise and entity-cold split manifests.

Split files contain row ids and labels only; the full feature strings remain
in the processed table so protein sequences and SMILES are not duplicated
across many CSVs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


DATASETS = {
    "davis_clean": {
        "table": ROOT / "data/processed/davis/davis_entity_affinity_clean.csv",
        "drug": "Drug_Index",
        "target": "Protein_Index",
        "label": "pKd_floor",
    },
    "davis_floor": {
        "table": ROOT / "data/processed/davis/davis_entity_affinity_floor.csv",
        "drug": "Drug_Index",
        "target": "Protein_Index",
        "label": "pKd_floor",
    },
    "kiba_standard": {
        "table": ROOT / "data/processed/kiba/kiba_deepdta_affinity_long.csv",
        "drug": "drug_id",
        "target": "protein_id",
        "label": "kiba_score",
    },
}


def group_assignment(values: pd.Series, rng: np.random.Generator) -> dict[object, str]:
    groups = np.asarray(pd.unique(values), dtype=object)
    groups = rng.permutation(groups)
    n = len(groups)
    n_train = max(1, int(round(0.8 * n)))
    n_val = max(1, int(round(0.1 * n)))
    if n_train + n_val >= n:
        n_train = max(1, n - 2)
        n_val = 1
    labels = np.full(n, "test", dtype=object)
    labels[:n_train] = "train"
    labels[n_train : n_train + n_val] = "val"
    return dict(zip(groups.tolist(), labels.tolist()))


def make_manifest(df: pd.DataFrame, drug_col: str, target_col: str, mode: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = len(df)
    out = pd.DataFrame(
        {
            "row_id": np.arange(n, dtype=np.int64),
            "drug_id": df[drug_col].astype(str).to_numpy(),
            "target_id": df[target_col].astype(str).to_numpy(),
        }
    )

    if mode == "random":
        order = rng.permutation(n)
        labels = np.full(n, "test", dtype=object)
        n_train = int(round(0.8 * n))
        n_val = int(round(0.1 * n))
        labels[order[:n_train]] = "train"
        labels[order[n_train : n_train + n_val]] = "val"
        out["split"] = labels
        return out

    drug_map = group_assignment(df[drug_col], rng)
    target_map = group_assignment(df[target_col], rng)
    drug_split = df[drug_col].map(drug_map).to_numpy()
    target_split = df[target_col].map(target_map).to_numpy()

    if mode == "cold_drug":
        out["split"] = drug_split
    elif mode == "cold_target":
        out["split"] = target_split
    elif mode == "double_cold":
        # Strict double-cold evaluation: each retained split uses disjoint
        # drug and target blocks. Cross-block pairs are intentionally omitted
        # from this manifest rather than leaking a held-out entity into train.
        same_block = drug_split == target_split
        out.loc[~same_block, "split"] = "unused"
        out.loc[same_block, "split"] = drug_split[same_block]
    else:  # pragma: no cover
        raise ValueError(mode)
    return out


def write_dataset(name: str, seed: int) -> None:
    spec = DATASETS[name]
    table = pd.read_csv(spec["table"])
    required = {spec["drug"], spec["target"], spec["label"]}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(f"{name}: missing columns {sorted(missing)}")

    base = ROOT / "data/processed" / name / "splits"
    base.mkdir(parents=True, exist_ok=True)
    summary: dict[str, object] = {"dataset": name, "table": str(spec["table"]), "seed": seed}
    for mode in ("random", "cold_drug", "cold_target", "double_cold"):
        manifest = make_manifest(table, spec["drug"], spec["target"], mode, seed)
        out_dir = base / mode
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(out_dir / "manifest.csv", index=False)
        counts = manifest["split"].value_counts().to_dict()
        summary[mode] = {
            "rows": {str(k): int(v) for k, v in counts.items()},
            "unique_drugs": {
                str(k): int(manifest.loc[manifest["split"] == k, "drug_id"].nunique())
                for k in counts
            },
            "unique_targets": {
                str(k): int(manifest.loc[manifest["split"] == k, "target_id"].nunique())
                for k in counts
            },
        }
    (base / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"{name}: {len(table)} rows; split manifests written to {base}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=[*DATASETS, "all"], default="all")
    parser.add_argument("--seed", type=int, default=20261012)
    args = parser.parse_args()
    names = list(DATASETS) if args.dataset == "all" else [args.dataset]
    for name in names:
        write_dataset(name, args.seed)


if __name__ == "__main__":
    main()
