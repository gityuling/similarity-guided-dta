#!/usr/bin/env python3
"""Train the released G2CP architecture with audited global chemical holdouts."""

import argparse
import json
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch

from bib_g2cp_common import DATA, OUT, RUN, SOURCE, now, plan, record, save_json, verify_data

sys.path.insert(0, str(SOURCE / "backbone_model/GEARS"))
from gears.model import GEARS_Model
from gears.utils import loss_fct


def graph_from_training(y, rows, cfg):
    mask = rows.split.isin(["train", "pretrain_train"]).to_numpy()
    correlation = np.nan_to_num(np.abs(np.corrcoef(y[mask].astype(np.float64), rowvar=False)))
    np.fill_diagonal(correlation, 1.0)
    indices = np.argsort(correlation, axis=1, kind="stable")[:, -(cfg["graph_k"] + 1):]
    edges, weights = [], []
    for target, sources in enumerate(indices):
        for source in sources:
            value = correlation[target, source]
            if value > cfg["graph_threshold"]:
                edges.append((int(source), target))
                weights.append(float(value))
    return torch.tensor(edges, dtype=torch.long).T, torch.tensor(weights, dtype=torch.float32)


def make_batch(reps, control, device):
    n, genes = len(reps), control.shape[-1]
    return SimpleNamespace(x=torch.as_tensor(control, dtype=torch.float32, device=device).reshape(-1, 1),
                           batch=torch.arange(n, device=device).repeat_interleave(genes),
                           unipert_rep=reps[:, None, :].tolist())


def predict(model, reps, control, cfg, device):
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(reps), cfg["batch_size"]):
            current = reps[start:start + cfg["batch_size"]]
            controls = np.repeat(control[None], len(current), axis=0)
            output.append(model(make_batch(current, controls, device)).cpu().numpy())
    return np.concatenate(output)


def validation_data(rows, indices, y, reps):
    # The published validation metric is a mean over perturbagens, not signatures.
    groups = list(rows.loc[indices].groupby("condition", sort=True).groups.values())
    features = np.asarray([reps[index].mean(0) for index in groups])
    if any(not np.allclose(reps[index], reps[index[0]], rtol=0, atol=1e-6) for index in groups):
        raise RuntimeError("UniPert representation varies within a perturbagen")
    truth = np.asarray([y[index].mean(0) for index in groups])
    return features, truth


def run(cell, seed, smoke=False):
    cfg = plan()
    verify_data()
    torch.set_num_threads(2)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    device = torch.device("cuda")
    directory = RUN / ("smoke" if smoke else "models") / f"{cell}_seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "completed.json").exists():
        print(f"Already complete: {directory}", flush=True)
        return
    rows = pd.read_csv(DATA / f"{cell}_rows.csv", keep_default_na=False)
    data = np.load(DATA / f"{cell}.npz", allow_pickle=False)
    y, reps, controls = data["y"], data["representations"], data["controls"]
    control = controls.mean(0)
    edge, weight = graph_from_training(y, rows, cfg)
    args = dict(hidden_size=cfg["hidden_size"], uncertainty=False, decoder_hidden_size=16,
                num_gene_gnn_layers=1, no_perturb=False, G_coexpress=edge, G_coexpress_weight=weight,
                device=str(device), num_genes=y.shape[1], num_perts=rows.condition.nunique())
    model = GEARS_Model(args).to(device)
    filter_indices = {pert: np.arange(y.shape[1]) for pert in rows.condition.unique()}
    control_tensor = torch.as_tensor(control, device=device)
    rng = np.random.default_rng(seed)
    started, logs = time.perf_counter(), []
    state = {"state": "running", "cell": cell, "seed": seed, "smoke": smoke, "started_utc": now(),
             "device": torch.cuda.get_device_name(0), "test_used_for_selection": False}
    try:
        for stage, train_split, val_split in (("pretrain", "pretrain_train", "pretrain_val"),
                                               ("finetune", "train", "val")):
            train = np.flatnonzero(rows.split.eq(train_split))
            val = np.flatnonzero(rows.split.eq(val_split))
            if smoke:
                train, val = train[:128], val[:64]
            if not len(train) or not len(val):
                raise RuntimeError("Empty training or validation split")
            vr, vy = validation_data(rows, val, y, reps)
            optimizer = torch.optim.Adam(model.parameters(), lr=cfg[f"{stage}_lr"],
                                         weight_decay=cfg[f"{stage}_weight_decay"])
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=cfg["lr_decay"])
            best_loss, best_state, best_epoch = float("inf"), None, None
            for epoch in range(1, (1 if smoke else cfg[f"{stage}_epochs"]) + 1):
                epoch_start = time.perf_counter()
                model.train()
                order = rng.permutation(train)
                total, count = 0.0, 0
                for start in range(0, len(order), cfg["batch_size"]):
                    index = order[start:start + cfg["batch_size"]]
                    input_control = controls[rng.integers(0, len(controls), len(index))]
                    batch = make_batch(reps[index], input_control, device)
                    optimizer.zero_grad(set_to_none=True)
                    predicted = model(batch)
                    target = torch.as_tensor(y[index], device=device)
                    loss = loss_fct(predicted, target, rows.condition.iloc[index].tolist(), ctrl=control_tensor,
                                    direction_lambda=cfg["direction_lambda"], dict_filter=filter_indices)
                    if not torch.isfinite(loss):
                        raise RuntimeError("Nonfinite training loss")
                    loss.backward()
                    torch.nn.utils.clip_grad_value_(model.parameters(), 1.0)
                    optimizer.step()
                    total += loss.item() * len(index)
                    count += len(index)
                scheduler.step()
                predicted = predict(model, vr, control, cfg, device)
                val_mse = float(np.square(predicted.astype(np.float64) - vy).mean(axis=1).mean())
                entry = {"stage": stage, "epoch": epoch, "train_loss": total / count, "val_mse": val_mse,
                         "seconds": time.perf_counter() - epoch_start}
                logs.append(entry)
                if not np.isfinite(val_mse):
                    raise RuntimeError("Nonfinite validation MSE")
                if val_mse < best_loss:
                    best_loss, best_epoch = val_mse, epoch
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                state.update(stage=stage, epoch=epoch, epochs=1 if smoke else cfg[f"{stage}_epochs"],
                             val_mse=val_mse, best_epoch=best_epoch, best_val_mse=best_loss,
                             updated_utc=now(), elapsed_seconds=time.perf_counter() - started)
                save_json(directory / "status.json", state)
                save_json(directory / "history.json", logs)
                print(f"{cell} seed={seed} {stage} epoch={epoch:02d} train_loss={total/count:.5f} val_mse={val_mse:.5f} seconds={entry['seconds']:.1f}", flush=True)
            model.load_state_dict(best_state)
            torch.save({"model": best_state, "args": {**args, "device": "cpu"}, "seed": seed,
                        "best_epoch": best_epoch, "best_val_mse": best_loss,
                        "data_audit": record(OUT / "data_audit.json")}, directory / f"{stage}.pt")
        # Save predictions without computing test metrics; final scoring follows weight freezing.
        evaluation = pd.read_csv(OUT / f"{cell}_evaluation.csv")
        vectors = []
        for row in evaluation.itertuples():
            member = rows.loc[rows.pert_type.eq("trt_cp") & rows.canonical_identity.eq(row.canonical_identity)]
            ids = list(member.groupby("pert_id", sort=True).groups.values())
            features = np.asarray([reps[index[0]] for index in ids])
            vectors.append(predict(model, features, control, cfg, device).mean(0))
        np.save(directory / "evaluation_predictions.npy", np.asarray(vectors, dtype=np.float32), allow_pickle=False)
        state.update(state="completed", elapsed_seconds=time.perf_counter() - started, updated_utc=now(),
                     peak_cuda_gib=torch.cuda.max_memory_allocated() / 1024**3,
                     files=[record(directory / name) for name in ("pretrain.pt", "finetune.pt", "history.json", "evaluation_predictions.npy")])
        save_json(directory / "completed.json", state, immutable=True)
    except BaseException as error:
        state.update(state="failed", error=repr(error), updated_utc=now())
        raise
    finally:
        save_json(directory / "status.json", state)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cell", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke", action="store_true")
    options = parser.parse_args()
    run(options.cell, options.seed, options.smoke)
