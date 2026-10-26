#!/usr/bin/env python3
"""Prepare exact existing folds for the two-stage kNN-DTA comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/knn_comparison_20261025.json")
    parser.add_argument("--output", type=Path, default=PROJECT / "reports/knn_comparison_20261025")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    args.output.mkdir(parents=True, exist_ok=True)
    original = json.loads((PROJECT / "reports/bib_protocol_20261017/protocol.json").read_text())
    frozen_hashes = {x["path"]: x["sha256"] for x in original["files"]}

    def verify_frozen(path):
        key = str(path.relative_to(PROJECT.parent))
        if frozen_hashes.get(key) != digest(path):
            raise ValueError(f"Input differs from the frozen protocol: {key}")

    records, jobs = [], []
    for dataset in config["datasets"]:
        source = PROJECT / f"data/processed/{dataset}_dtiam_protocol"
        table_file = source / "table.csv"
        verify_frozen(table_file)
        table = pd.read_csv(table_file, dtype={"drug_id": str, "protein_id": str})
        label = "pKd_floor" if dataset == "davis" else "kiba_score"
        if table.row_id.duplicated().any() or not np.isfinite(table[label]).all():
            raise ValueError(f"Invalid table: {dataset}")
        entity_folder = args.output / "inputs" / dataset
        entity_folder.mkdir(parents=True, exist_ok=True)
        entities = {}
        for name, columns in (("drugs.csv", ["drug_id", "Canonical_SMILES"]),
                              ("targets.csv", ["protein_id", "Sequence"])):
            values = table[columns].drop_duplicates()
            if values[columns[0]].duplicated().any():
                raise ValueError(f"One entity ID has different inputs: {dataset}/{name}")
            path = entity_folder / name
            values.to_csv(path, index=False)
            entities[name] = {"path": str(path.relative_to(PROJECT)), "sha256": digest(path)}
        test_row_ids = []
        for fold in config["folds"]:
            manifest_file = source / f"folds/drug_coldstart/fold_{fold}.csv"
            verify_frozen(manifest_file)
            manifest = pd.read_csv(manifest_file)
            joined = table.merge(manifest[["row_id", "split"]], on="row_id", how="outer",
                                 validate="one_to_one", indicator=True)
            if not joined._merge.eq("both").all() or set(joined.split) != {"train", "val", "test"}:
                raise ValueError(f"Manifest coverage mismatch: {dataset}/{fold}")
            groups = {s: joined.loc[joined.split.eq(s)].sort_values("row_id") for s in ("train", "val", "test")}
            test_row_ids.extend(groups["test"].row_id.tolist())
            drug_sets = {s: set(g.drug_id) for s, g in groups.items()}
            for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
                if drug_sets[a] & drug_sets[b]:
                    raise ValueError(f"Drug-ID overlap: {dataset}/{fold}/{a}/{b}")
            folder = args.output / "inputs" / dataset / f"fold_{fold}"
            folder.mkdir(parents=True, exist_ok=True)
            files = {}
            for split, frame in groups.items():
                inputs = frame[["row_id", "drug_id", "protein_id"]]
                targets = frame[["row_id", label]].rename(columns={label: "affinity"})
                for name, content in ((f"{split}_inputs.csv", inputs), (f"{split}_labels.csv", targets)):
                    path = folder / name
                    content.to_csv(path, index=False)
                    files[name] = {"path": str(path.relative_to(PROJECT)), "sha256": digest(path)}
            overlap = {}
            for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
                overlap[f"{a}/{b}"] = len(set(groups[a].Canonical_SMILES) & set(groups[b].Canonical_SMILES))
            records.append({"dataset": dataset, "fold": fold, "native_label": label,
                            "rows": {s: len(g) for s, g in groups.items()},
                            "drug_ids": {s: g.drug_id.nunique() for s, g in groups.items()},
                            "target_ids": int(table.protein_id.nunique()),
                            "input_sequences": int(table.Sequence.nunique()),
                            "drug_id_overlap": 0, "canonical_structure_overlap": overlap,
                            "source_table": {"path": str(table_file.relative_to(PROJECT)), "sha256": digest(table_file)},
                            "source_manifest": {"path": str(manifest_file.relative_to(PROJECT)), "sha256": digest(manifest_file)},
                            "entities": entities, "files": files})
            for seed in config["training_seeds"]:
                jobs.append({"dataset": dataset, "fold": fold, "seed": seed,
                             "external_backbone": "awaiting_verified_assets",
                             "complete_method_comparison": "pending",
                             "matched_backbone_comparison": "pending"})
        if sorted(test_row_ids) != sorted(table.row_id.tolist()):
            raise ValueError(f"The five test folds do not cover every pair exactly once: {dataset}")
    report = {"config": {"path": str(args.config.relative_to(PROJECT)), "sha256": digest(args.config)},
              "sgf_amendment_sha256": digest(PROJECT / config["sgf_amendment"]),
              "folds": records, "jobs": jobs,
              "result_status": "inputs_prepared; no new benchmark predictions generated"}
    (args.output / "prepared_inputs.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"folds": len(records), "planned_backbone_runs": len(jobs),
                      "output": str(args.output / "prepared_inputs.json")}, indent=2))


if __name__ == "__main__":
    main()
