#!/usr/bin/env python3
"""Compare retrieval mechanisms on one trained pooled-feature backbone per fold.

All methods receive the same frozen affinity head. kNN-DTA and Ada-kNN-DTA
include both feature aggregation and label aggregation. Experimental choices
and their provenance are recorded in the configuration and each result file.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import itertools
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.optimize import minimize
from scipy.special import expit, logit
from scipy.stats import pearsonr
import torch
from torch import nn

from knn_dta_reference import (AdaptiveAggregation, FixedParameters, Mix, Neighbors,
                               Query, TrainingStore, _average)

PROJECT = Path(__file__).resolve().parents[1]
PREPARED = PROJECT / "reports/knn_comparison_20261025"
OUTPUT = PROJECT / "results/knn_shared_20261025"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def emit(stage, **values):
    print(json.dumps({"stage": stage, **values}), flush=True)


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def features(dataset):
    folder = OUTPUT / "features"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{dataset}.npz"
    meta_path = folder / f"{dataset}.json"
    entities = PREPARED / "inputs" / dataset
    entity_hashes = {n: sha(entities / n) for n in ("drugs.csv", "targets.csv")}
    if path.exists():
        metadata = json.loads(meta_path.read_text())
        if metadata["entity_hashes"] != entity_hashes or sha(path) != metadata["cache_sha256"]:
            raise ValueError("Feature cache provenance mismatch")
        with np.load(path) as content:
            return {name: content[name] for name in content.files}, metadata
    token_root = (PROJECT.parent / "dataset/processed_DTI_Token/token_features_davis"
                  if dataset == "davis" else PROJECT / "data/processed/kiba_dtiam_protocol/tokens")
    drugs = pd.read_csv(entities / "drugs.csv", dtype={"drug_id": str})
    drugs["Canonical_SMILES"] = drugs.Canonical_SMILES.map(lambda value: Chem.MolToSmiles(Chem.MolFromSmiles(value)))
    proteins = pd.read_csv(entities / "targets.csv", dtype={"protein_id": str})
    values, source_hashes = {}, []
    for kind, frame, column, id_column in (("drug", drugs, "Canonical_SMILES", "drug_id"),
                                            ("protein", proteins, "Sequence", "protein_id")):
        keys = sorted(frame[column].unique())
        lookup = {key: i for i, key in enumerate(keys)}
        pooled = []
        for key in keys:
            filename = hashlib.md5(key.encode()).hexdigest() + ".npy"
            source = token_root / ("drugs" if kind == "drug" else "proteins") / filename
            token = np.load(source).astype(np.float32)
            if token.ndim != 2 or not np.isfinite(token).all():
                raise ValueError(f"Invalid features: {source}")
            pooled.append(token.mean(axis=0))
            source_hashes.append({"path": str(source.relative_to(PROJECT.parent)), "sha256": sha(source)})
        values[kind] = np.stack(pooled)
        values[kind + "_keys"] = np.array(keys)
        values[kind + "_ids"] = frame[id_column].to_numpy(dtype=str)
        values[kind + "_indices"] = frame[column].map(lookup).to_numpy(dtype=np.int64)
    np.savez_compressed(path, **values)
    metadata = {"cache_sha256": sha(path), "entity_hashes": entity_hashes, "sources": source_hashes,
                "pooling": "arithmetic mean of all cached token rows, matching the existing pooled branch"}
    save_json(meta_path, metadata)
    emit("features_ready", dataset=dataset, drugs=len(values["drug"]), proteins=len(values["protein"]))
    return values, metadata


class AffinityHead(nn.Module):
    def __init__(self, latent, hidden, dropout, y_mean, y_std):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(2 * latent, hidden), nn.Tanh(), nn.Dropout(dropout), nn.Linear(hidden, 1))
        self.register_buffer("y_mean", torch.tensor(float(y_mean)))
        self.register_buffer("y_std", torch.tensor(float(y_std)))

    def forward(self, drug, protein):
        return self.layers(torch.cat((drug, protein), -1)).squeeze(-1) * self.y_std + self.y_mean


class Backbone(nn.Module):
    def __init__(self, drug_dim, protein_dim, config, stats, y_mean, y_std):
        super().__init__()
        self.register_buffer("drug_mean", torch.tensor(stats[0]))
        self.register_buffer("drug_std", torch.tensor(stats[1]))
        self.register_buffer("protein_mean", torch.tensor(stats[2]))
        self.register_buffer("protein_std", torch.tensor(stats[3]))

        def encoder(dim):
            return nn.Sequential(nn.Linear(dim, config["entity_hidden"]), nn.LayerNorm(config["entity_hidden"]),
                                 nn.GELU(), nn.Dropout(config["dropout"]),
                                 nn.Linear(config["entity_hidden"], config["latent_dim"]))

        self.drug_encoder = encoder(drug_dim)
        self.protein_encoder = encoder(protein_dim)
        self.head = AffinityHead(config["latent_dim"], config["head_hidden"], config["dropout"], y_mean, y_std)

    def encode_drug(self, drug):
        return self.drug_encoder((drug - self.drug_mean) / self.drug_std)

    def encode_protein(self, protein):
        return self.protein_encoder((protein - self.protein_mean) / self.protein_std)

    def forward(self, drug, protein):
        return self.head(self.encode_drug(drug), self.encode_protein(protein))


def weighted_stats(features, indices):
    weights = np.bincount(indices, minlength=len(features)).astype(np.float64)
    weights /= weights.sum()
    mean = weights @ features
    var = weights @ (features.astype(np.float64) - mean) ** 2
    std = np.sqrt(var)
    std[std < 1e-6] = 1.
    return mean.astype(np.float32), std.astype(np.float32)


@torch.no_grad()
def head_predict(head, drug, protein, batch=8192):
    return torch.cat([head(drug[i:i + batch], protein[i:i + batch]) for i in range(0, len(drug), batch)])


def train_backbone(values, frames, labels, config, seed, folder, identity, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    stats = (*weighted_stats(values["drug"], frames["train"].di.to_numpy()),
             *weighted_stats(values["protein"], frames["train"].pi.to_numpy()))
    y_mean, y_std = labels["train"].mean(), labels["train"].std()
    if y_std <= 0:
        raise ValueError("Constant training labels")
    model = Backbone(values["drug"].shape[1], values["protein"].shape[1], config, stats, y_mean, y_std).to(device)
    path = folder / "backbone.pt"
    if path.exists():
        saved = torch.load(path, map_location=device, weights_only=True)
        if saved["identity"] != identity:
            raise ValueError(f"Existing checkpoint uses a different protocol: {path}")
        model.load_state_dict(saved["state"])
        model.eval()
        emit("backbone_reused", epoch=saved["best_epoch"], validation_mse=saved["validation_mse"])
        return model, saved["history"]
    d, p = torch.tensor(values["drug"], device=device), torch.tensor(values["protein"], device=device)
    tr = frames["train"]
    di, pi = torch.tensor(tr.di.to_numpy(), device=device), torch.tensor(tr.pi.to_numpy(), device=device)
    y = torch.tensor(labels["train"], device=device, dtype=torch.float32)
    v = frames["val"]
    vdi, vpi = torch.tensor(v.di.to_numpy(), device=device), torch.tensor(v.pi.to_numpy(), device=device)
    vy = torch.tensor(labels["val"], device=device, dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    best, stale, history = float("inf"), 0, []
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        order = torch.randperm(len(tr), device=device)
        total = 0.
        for start in range(0, len(tr), config["batch_size"]):
            index = order[start:start + config["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(d[di[index]], p[pi[index]])
            loss = (((prediction - y[index]) / y_std) ** 2).mean()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"])
            optimizer.step()
            total += loss.item() * len(index)
        model.eval()
        with torch.no_grad():
            de, pe = model.encode_drug(d), model.encode_protein(p)
            value = float(((head_predict(model.head, de[vdi], pe[vpi]) - vy) ** 2).mean())
        if not np.isfinite(value):
            raise RuntimeError("Nonfinite validation loss")
        history.append({"epoch": epoch, "training_normalized_mse": total / len(tr), "validation_mse": value})
        if value < best:
            best, stale = value, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            emit("backbone_training", epoch=epoch, validation_mse=value, best_epoch=best_epoch)
        if stale >= config["patience"]:
            break
    model.load_state_dict(best_state)
    torch.save({"identity": identity, "state": best_state, "best_epoch": best_epoch,
                "validation_mse": best, "history": history}, path)
    emit("backbone_ready", best_epoch=best_epoch, validation_mse=best)
    return model.eval(), history


def subset(query, index):
    def neighbors(n):
        return Neighbors(n.distances[index], n.values[index], n.indices[index])
    return Query(query.drug[index], query.protein[index], neighbors(query.labels),
                 neighbors(query.drug_neighbors), neighbors(query.protein_neighbors))


def trimmed(query, ks):
    return Query(query.drug, query.protein, query.labels.first(ks[0]),
                 query.drug_neighbors.first(ks[1]), query.protein_neighbors.first(ks[2]))


@torch.no_grad()
def tune_fixed(query, head, truth, config):
    names = ("drug", "protein", "label")
    caches, grid = {}, {}
    for name, n in zip(("label", "drug", "protein"),
                       (query.labels, query.drug_neighbors, query.protein_neighbors)):
        ks = sorted({min(k, n.distances.shape[1]) for k in config["k"]})
        caches[name] = {(k, tau): _average(n, Mix(k, tau, 1.))
                        for k, tau in itertools.product(ks, config["temperature"])}
        grid[name] = [Mix(k, tau, w) for k, tau, w in itertools.product(ks, config["temperature"], config["base_weight"])]
    base = {name: Mix(grid[name][0].k, grid[name][0].temperature, 1.) for name in names}
    trace = []

    def score(current):
        d, p, l = current["drug"], current["protein"], current["label"]
        dr = d.base_weight * query.drug + (1 - d.base_weight) * caches["drug"][d.k, d.temperature]
        pr = p.base_weight * query.protein + (1 - p.base_weight) * caches["protein"][p.k, p.temperature]
        prediction = head_predict(head, dr, pr, config["score_batch_size"])
        prediction = l.base_weight * prediction + (1 - l.base_weight) * caches["label"][l.k, l.temperature]
        return float(((prediction - truth) ** 2).mean())

    def search(name, current, phase):
        best, selected = score(current), current[name]
        for candidate in grid[name]:
            trial = {**current, name: candidate}
            value = score(trial)
            trace.append({"phase": phase, "component": name, **asdict(candidate), "validation_mse": value})
            if value < best:
                best, selected = value, candidate
        emit("fixed_search", phase=phase, component=name, validation_mse=best)
        return selected

    current = {name: search(name, base.copy(), "independent") for name in names}
    for sweep in range(config["coordinate_passes"]):
        for name in names:
            current[name] = search(name, current, f"joint_{sweep + 1}")
    parameters = FixedParameters(current["label"], current["drug"], current["protein"])
    return parameters, trace


@torch.no_grad()
def adaptive_predict(model, query, batch=512):
    return torch.cat([model(subset(query, slice(i, i + batch))) for i in range(0, len(query.drug), batch)])


def tune_adaptive(query, head, truth, parameters, config, seed, folder):
    torch.manual_seed(seed)
    ks = (parameters.label.k, parameters.drug.k, parameters.protein.k)
    query = trimmed(query, ks)
    model = AdaptiveAggregation(copy.deepcopy(head), *ks, hidden_dim=config["hidden_dim"]).to(query.drug.device)
    with torch.no_grad():
        for i, (n, network) in enumerate(zip((query.labels, query.drug_neighbors, query.protein_neighbors),
                                             (model.label_network, model.drug_network, model.protein_network))):
            model.distance_scale[i] = n.distances.square().mean().sqrt().clamp_min(1e-6)
            nn.init.zeros_(network[-1].weight)
            network[-1].bias.fill_(float(np.log(.5 / ks[i])))
            network[-1].bias[0] = float(np.log(.5))
    optimizer = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=config["lr"])
    best = float(((adaptive_predict(model, query) - truth) ** 2).mean())
    best_state = copy.deepcopy(model.state_dict())
    best_step, history = 0, [{"step": 0, "validation_mse": best}]
    for step in range(1, config["steps"] + 1):
        model.train()
        indices = torch.randint(len(truth), (config["batch_size"],), device=truth.device)
        optimizer.zero_grad(set_to_none=True)
        pred = model(subset(query, indices))
        loss = (((pred - truth[indices]) / head.y_std) ** 2).mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite adaptive fitting loss")
        loss.backward()
        nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], config["gradient_clip"])
        optimizer.step()
        if step % config["evaluate_every"] == 0 or step == config["steps"]:
            model.eval()
            value = float(((adaptive_predict(model, query) - truth) ** 2).mean())
            history.append({"step": step, "validation_mse": value})
            if value < best:
                best, best_step, best_state = value, step, copy.deepcopy(model.state_dict())
            if step % 1000 == 0:
                emit("adaptive_training", step=step, validation_mse=value, best_step=best_step)
    model.load_state_dict(best_state)
    torch.save({"state": {k: v.detach().cpu() for k, v in best_state.items()}, "ks": ks,
                "best_step": best_step, "history": history}, folder / "adaptive.pt")
    return model.eval(), {"ks": ks, "best_step": best_step, "validation_mse": best,
                          "distance_scale": model.distance_scale.cpu().tolist(), "history": history}


def prior_file(dataset, fold):
    stem = (f"bib_davis_target_id_f{fold}_frozen_prior" if dataset == "davis"
            else f"kiba_dtiam_drugcold_f{fold}_frozen_morgan_prior")
    return PROJECT / "results/predictions" / f"{stem}_predictions.csv"


def fit_gate(deep, prior, similarity, truth):
    mean, std = float(np.mean(similarity)), float(np.std(similarity, ddof=1))
    if std < 1e-8:
        raise ValueError("Constant validation similarities")
    z = (similarity - mean) / std

    def objective(ab):
        w = expit(ab[0] + ab[1] * z)
        return np.mean((w * deep + (1 - w) * prior - truth) ** 2) + .001 * ab[1] ** 2

    opt = minimize(objective, np.array([logit(.335), 0.]), method="L-BFGS-B")
    if not opt.success:
        raise RuntimeError(f"Gate optimization failed: {opt.message}")
    return {"a": float(opt.x[0]), "b": float(opt.x[1]), "mean": mean, "std": std}


def run(dataset, fold, seed, config, config_path, device):
    start = time.monotonic()
    folder = OUTPUT / dataset / f"fold_{fold}" / f"seed_{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    identity = {"config": sha(config_path), "runner": sha(__file__), "core": sha(Path(__file__).with_name("knn_dta_reference.py")),
                "prepared": sha(PREPARED / "prepared_inputs.json"), "dataset": dataset, "fold": fold, "seed": seed}
    result_path = folder / "result.json"
    if result_path.exists():
        previous = json.loads(result_path.read_text())
        if previous["identity"] != identity:
            raise ValueError("Completed result has different implementation/configuration")
        emit("already_complete", dataset=dataset, fold=fold, seed=seed)
        return
    emit("run_started", dataset=dataset, fold=fold, seed=seed)
    values, feature_metadata = features(dataset)
    drug_lookup = dict(zip(values["drug_ids"], values["drug_indices"]))
    protein_lookup = dict(zip(values["protein_ids"], values["protein_indices"]))
    input_folder = PREPARED / "inputs" / dataset / f"fold_{fold}"
    frames, labels = {}, {}
    for split in ("train", "val", "test"):
        frame = pd.read_csv(input_folder / f"{split}_inputs.csv", dtype={"drug_id": str, "protein_id": str})
        frame["di"], frame["pi"] = frame.drug_id.map(drug_lookup), frame.protein_id.map(protein_lookup)
        if frame[["di", "pi"]].isna().any().any():
            raise ValueError("Missing entity features")
        frames[split] = frame
        if split != "test":
            label = pd.read_csv(input_folder / f"{split}_labels.csv")
            labels[split] = frame[["row_id"]].merge(label, on="row_id", validate="one_to_one").affinity.to_numpy()
    model, training_history = train_backbone(values, frames, labels, config["backbone"], seed, folder, identity, device)
    model.requires_grad_(False)
    with torch.no_grad():
        drug = model.encode_drug(torch.tensor(values["drug"], device=device))
        protein = model.encode_protein(torch.tensor(values["protein"], device=device))
    tr = frames["train"]
    store = TrainingStore(drug[tr.di.to_numpy()], protein[tr.pi.to_numpy()],
                          torch.tensor(labels["train"], device=device, dtype=torch.float32),
                          values["drug_keys"][tr.di.to_numpy()].tolist(),
                          values["protein_keys"][tr.pi.to_numpy()].tolist(),
                          tr.row_id.tolist(), ["train"] * len(tr))
    va = frames["val"]
    query = store.retrieve(drug[va.di.to_numpy()], protein[va.pi.to_numpy()], 128, 128, 128)
    truth = torch.tensor(labels["val"], dtype=torch.float32, device=device)
    emit("validation_neighbors_ready", rows=len(va))
    fixed, search_trace = tune_fixed(query, model.head, truth, config["fixed"])
    save_json(folder / "fixed_search.json", {"parameters": asdict(fixed), "trace": search_trace})
    adaptive, adaptive_info = tune_adaptive(query, model.head, truth, fixed, config["adaptive"], seed, folder)
    prior_path = prior_file(dataset, fold)
    prior = pd.read_csv(prior_path, usecols=["row_id", "split", "prediction_raw", "max_train_drug_tanimoto"])
    vp = va[["row_id"]].merge(prior.loc[prior.split.eq("val")], on="row_id", validate="one_to_one")
    if len(vp) != len(va):
        raise ValueError("Prior coverage mismatch")
    base_val = head_predict(model.head, query.drug, query.protein).cpu().numpy()
    gate = fit_gate(base_val, vp.prediction_raw.to_numpy(), vp.max_train_drug_tanimoto.to_numpy(), labels["val"])
    # Freeze every selection before evaluating test predictions or reading test labels.
    save_json(folder / "selected.json", {"identity": identity, "fixed": asdict(fixed),
                                         "adaptive": adaptive_info, "gate": gate, "prior_sha256": sha(prior_path)})
    del query
    torch.cuda.empty_cache() if device == "cuda" else None
    te = frames["test"]
    query = store.retrieve(drug[te.di.to_numpy()], protein[te.pi.to_numpy()], fixed.label.k, fixed.drug.k, fixed.protein.k)
    from knn_dta_reference import fixed_prediction
    with torch.no_grad():
        base_test = head_predict(model.head, query.drug, query.protein).cpu().numpy()
        knn_test = fixed_prediction(query, model.head, fixed).cpu().numpy()
        ada_test = adaptive_predict(adaptive, query).cpu().numpy()
    tp = te[["row_id"]].merge(prior.loc[prior.split.eq("test")], on="row_id", validate="one_to_one")
    if len(tp) != len(te):
        raise ValueError("Test prior coverage mismatch")
    weights = expit(gate["a"] + gate["b"] * (tp.max_train_drug_tanimoto.to_numpy() - gate["mean"]) / gate["std"])
    predictions = te[["row_id", "drug_id", "protein_id"]].copy()
    predictions["base"] = base_test
    predictions["knn_dta"] = knn_test
    predictions["ada_knn_dta"] = ada_test
    predictions["sgf"] = weights * base_test + (1 - weights) * tp.prediction_raw.to_numpy()
    predictions["neighbor"] = tp.prediction_raw.to_numpy()
    predictions["chemical_identity"] = values["drug_keys"][te.di.to_numpy()]
    test_truth = pd.read_csv(input_folder / "test_labels.csv")
    predictions = predictions.merge(test_truth, on="row_id", validate="one_to_one")
    metrics = {}
    for name in ("base", "knn_dta", "ada_knn_dta", "sgf", "neighbor"):
        pred, actual = predictions[name].to_numpy(), predictions.affinity.to_numpy()
        if not np.isfinite(pred).all():
            raise ValueError("Nonfinite test predictions")
        metrics[name] = {"mse": float(np.mean((pred - actual) ** 2)),
                         "pearson": float(pearsonr(pred, actual).statistic)}
    predictions.to_csv(folder / "test_predictions.csv", index=False)
    save_json(result_path, {"identity": identity, "feature_cache_sha256": feature_metadata["cache_sha256"],
                           "scope": config["comparison"], "configuration": config,
                           "training_history": training_history, "fixed": asdict(fixed),
                           "adaptive": adaptive_info, "gate": gate, "test": metrics,
                           "prediction_sha256": sha(folder / "test_predictions.csv"),
                           "wall_seconds": time.monotonic() - start})
    emit("run_complete", dataset=dataset, fold=fold, seed=seed, test=metrics,
         seconds=time.monotonic() - start)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/knn_shared_20261025.json")
    parser.add_argument("--dataset", choices=["davis", "kiba"])
    parser.add_argument("--fold", type=int, choices=range(5))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    config = json.loads(args.config.read_text())
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("GPU is not visible; run in the host GPU environment")
    for dataset, fold, seed in itertools.product([args.dataset] if args.dataset else config["datasets"],
                                                [args.fold] if args.fold is not None else config["folds"],
                                                [args.seed] if args.seed else config["seeds"]):
        run(dataset, fold, seed, config, args.config, args.device)


if __name__ == "__main__":
    main()
