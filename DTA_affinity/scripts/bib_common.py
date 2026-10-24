"""Shared paths and integrity checks for the BIB validation experiments."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem


PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
FREEZE = PROJECT / "reports/bib_protocol_20261017"
DATASETS = {
    "davis": {
        "label": "pKd_floor", "pairs": 30056, "drug_ids": 68,
        "tokens": ROOT / "dataset/processed_DTI_Token/token_features_davis",
    },
    "kiba": {
        "label": "kiba_score", "pairs": 118254, "drug_ids": 2111,
        "tokens": PROJECT / "data/processed/kiba_dtiam_protocol/tokens",
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def record(path: Path) -> dict:
    return {"path": relative(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def canonical(smiles: str) -> str:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles}")
    return Chem.MolToSmiles(mol, isomericSmiles=True)


def table_path(dataset: str) -> Path:
    return PROJECT / f"data/processed/{dataset}_dtiam_protocol/table.csv"


def manifest_path(dataset: str, fold: int) -> Path:
    return PROJECT / f"data/processed/{dataset}_dtiam_protocol/folds/drug_coldstart/fold_{fold}.csv"


def baseline_paths(dataset: str, fold: int) -> dict[str, Path]:
    if dataset == "davis":
        deep = f"dtiam_drugcold_f{fold}_scratch_b64_lr1e4_e20"
        prior = (f"{deep}_morgan_neighbor_prior" if fold
                 else "dtiam_drugcold_f0_morgan_neighbor_prior")
        pred = f"nexusdti_{deep}_affine_calibration_predictions.csv"
        gate = (f"{deep}_similarity_gated_ensemble" if fold
                else "dtiam_drugcold_f0_scratch_b64_similarity_gated_ensemble")
        gate_predictions = f"{gate}_predictions.csv"
    else:
        deep = f"kiba_dtiam_drugcold_f{fold}_scratch_b64_lr1e4_e20_seed20261012"
        prior = f"kiba_dtiam_drugcold_f{fold}_frozen_morgan_prior"
        pred = f"nexusdti_{deep}_predictions.csv"
        gate = f"{deep}_ablation"
        gate_predictions = f"{gate}_predictions.csv"
    return {
        "deep_result": PROJECT / f"results/nexusdti_regressor_{deep}.json",
        "deep_checkpoint": PROJECT / f"results/checkpoints/nexusdti_{deep}.pt",
        "deep_predictions": PROJECT / f"results/predictions/{pred}",
        "prior_result": PROJECT / f"results/{prior}.json",
        "prior_predictions": PROJECT / f"results/predictions/{prior}_predictions.csv",
        "fusion_result": PROJECT / f"results/{gate}.json",
        "fusion_predictions": PROJECT / f"results/predictions/{gate_predictions}",
    }


def check_predictions(path: Path, table: pd.DataFrame, manifest: pd.DataFrame,
                      target: str) -> None:
    prediction = pd.read_csv(path)
    truth = "affinity_true" if "affinity_true" in prediction else "pKd_true"
    if prediction.duplicated(["split", "row_id"]).any():
        raise RuntimeError(f"Duplicate prediction IDs: {path}")
    expected = manifest.loc[manifest.split.isin(["val", "test"]), ["row_id", "split"]]
    joined = expected.merge(prediction, on=["row_id", "split"], how="outer",
                            validate="one_to_one", indicator=True)
    if not joined._merge.eq("both").all():
        raise RuntimeError(f"Prediction coverage mismatch: {path}")
    joined = joined.merge(table[["row_id", target]], on="row_id", validate="one_to_one")
    if not np.allclose(joined[truth], joined[target], atol=2e-6, rtol=0):
        raise RuntimeError(f"Prediction label mismatch: {path}")


def verify_freeze(include_features: bool = True) -> dict:
    frozen = json.loads((FREEZE / "protocol.json").read_text())
    records = [*frozen["files"], frozen["feature_inventory"], frozen["current_source_snapshot"]]
    if include_features:
        records.extend(pd.read_csv(FREEZE / "feature_inventory.csv").to_dict("records"))
    for item in records:
        path = ROOT / item["path"]
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise RuntimeError(f"Frozen input changed: {path}")
    return frozen


def write_once(path: Path, text: str) -> None:
    if path.exists():
        if path.read_text() != text:
            raise RuntimeError(f"Refusing to replace frozen output: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
