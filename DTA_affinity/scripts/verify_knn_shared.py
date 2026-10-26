#!/usr/bin/env python3
"""Independently reconstruct sampled predictions with NumPy/SciPy equations."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import erf, expit, softmax
from scipy.spatial.distance import cdist
import torch

PROJECT = Path(__file__).resolve().parents[1]
RUNS = PROJECT / "results/knn_shared_20261025"
REPORT = PROJECT / "reports/knn_shared_20261025"


def encode(x, state, kind):
    x = (x.astype(np.float64) - state[kind + "_mean"]) / state[kind + "_std"]
    prefix = kind + "_encoder."
    x = x @ state[prefix + "0.weight"].T + state[prefix + "0.bias"]
    x = (x - x.mean(1, keepdims=True)) / np.sqrt(x.var(1, keepdims=True) + 1e-5)
    x = x * state[prefix + "1.weight"] + state[prefix + "1.bias"]
    x = .5 * x * (1 + erf(x / np.sqrt(2)))
    return x @ state[prefix + "4.weight"].T + state[prefix + "4.bias"]


def head(drug, protein, state):
    x = np.concatenate([drug, protein], axis=1)
    x = np.tanh(x @ state["head.layers.0.weight"].T + state["head.layers.0.bias"])
    x = (x @ state["head.layers.3.weight"].T + state["head.layers.3.bias"]).ravel()
    return x * state["head.y_std"] + state["head.y_mean"]


def retrieve(query, keys, values, k):
    distances = cdist(query, keys, metric="sqeuclidean")
    order = np.argsort(distances, axis=1, kind="stable")[:, :k]
    return np.take_along_axis(distances, order, axis=1), values[order]


def blend(query, distance, values, parameters):
    weights = softmax(-distance / parameters["temperature"], axis=1)
    if values.ndim == 3:
        weights = weights[:, :, None]
    average = (weights * values).sum(axis=1)
    return parameters["base_weight"] * query + (1 - parameters["base_weight"]) * average


def adaptive_weights(distance, state, kind, scale):
    x = -distance / scale
    x = np.maximum(x @ state[kind + "_network.0.weight"].T + state[kind + "_network.0.bias"], 0.)
    return softmax(x @ state[kind + "_network.2.weight"].T + state[kind + "_network.2.bias"], axis=1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["davis", "kiba"])
    args = parser.parse_args()
    torch.set_num_threads(2)
    config = json.loads((PROJECT / "configs/knn_shared_20261025.json").read_text())
    results = []
    for dataset in ([args.dataset] if args.dataset else config["datasets"]):
        with np.load(RUNS / "features" / f"{dataset}.npz") as content:
            features = {k: content[k] for k in content.files}
        drug_lookup = dict(zip(features["drug_ids"], features["drug_indices"]))
        protein_lookup = dict(zip(features["protein_ids"], features["protein_indices"]))
        for fold, seed in itertools.product(config["folds"], config["seeds"]):
            folder = RUNS / dataset / f"fold_{fold}" / f"seed_{seed}"
            result = json.loads((folder / "result.json").read_text())
            backbone = torch.load(folder / "backbone.pt", map_location="cpu", weights_only=True)["state"]
            adaptive = torch.load(folder / "adaptive.pt", map_location="cpu", weights_only=True)["state"]
            for key, tensor in backbone.items():
                if key.startswith("head.") and not torch.equal(tensor, adaptive[key]):
                    raise ValueError("Adaptive fitting modified the shared affinity head")
            b = {k: v.numpy().astype(np.float64) for k, v in backbone.items()}
            a = {k: v.numpy().astype(np.float64) for k, v in adaptive.items()}
            drugs, proteins = encode(features["drug"], b, "drug"), encode(features["protein"], b, "protein")
            predicted = pd.read_csv(folder / "test_predictions.csv", dtype={"drug_id": str, "protein_id": str})
            sample = predicted.iloc[np.linspace(0, len(predicted) - 1, 16, dtype=int)].copy()
            qd, qp = drugs[sample.drug_id.map(drug_lookup)], proteins[sample.protein_id.map(protein_lookup)]
            input_folder = PROJECT / f"reports/knn_comparison_20261025/inputs/{dataset}/fold_{fold}"
            train = pd.read_csv(input_folder / "train_inputs.csv", dtype={"drug_id": str, "protein_id": str})
            train = train.merge(pd.read_csv(input_folder / "train_labels.csv"), on="row_id", validate="one_to_one")
            di, pi = train.drug_id.map(drug_lookup).to_numpy(), train.protein_id.map(protein_lookup).to_numpy()
            drug_bank, protein_bank = drugs[list(dict.fromkeys(di))], proteins[list(dict.fromkeys(pi))]
            parameters = result["fixed"]
            dl, vl = retrieve(np.concatenate([qd, qp], 1), np.concatenate([drugs[di], proteins[pi]], 1),
                              train.affinity.to_numpy().astype(np.float32).astype(np.float64), parameters["label"]["k"])
            dd, vd = retrieve(qd, drug_bank, drug_bank, parameters["drug"]["k"])
            dp, vp = retrieve(qp, protein_bank, protein_bank, parameters["protein"]["k"])
            base = head(qd, qp, b)
            knn = blend(head(blend(qd, dd, vd, parameters["drug"]), blend(qp, dp, vp, parameters["protein"]), b),
                        dl, vl, parameters["label"])
            wl = adaptive_weights(dl, a, "label", a["distance_scale"][0])
            wd = adaptive_weights(dd, a, "drug", a["distance_scale"][1])
            wp = adaptive_weights(dp, a, "protein", a["distance_scale"][2])
            dr = wd[:, :1] * qd + (wd[:, 1:, None] * vd).sum(1)
            pr = wp[:, :1] * qp + (wp[:, 1:, None] * vp).sum(1)
            ada = wl[:, 0] * head(dr, pr, b) + (wl[:, 1:] * vl).sum(1)
            prior_stem = (f"bib_davis_target_id_f{fold}_frozen_prior" if dataset == "davis"
                          else f"kiba_dtiam_drugcold_f{fold}_frozen_morgan_prior")
            prior = pd.read_csv(PROJECT / f"results/predictions/{prior_stem}_predictions.csv")
            prior = sample[["row_id"]].merge(prior.loc[prior.split.eq("test")], on="row_id", validate="one_to_one")
            g = result["gate"]
            weight = expit(g["a"] + g["b"] * (prior.max_train_drug_tanimoto.to_numpy() - g["mean"]) / g["std"])
            sgf = weight * base + (1 - weight) * prior.prediction_raw.to_numpy()
            errors = {name: float(np.max(np.abs(value - sample[name].to_numpy())))
                      for name, value in (("base", base), ("knn_dta", knn), ("ada_knn_dta", ada), ("sgf", sgf))}
            # Reconstruction uses float64 algebra instead of the original GPU's
            # float32 operations and the CSV predictions have float32 precision.
            if max(errors.values()) > 2e-4:
                raise ValueError(f"Independent replay differs: {dataset}/{fold}/{seed}: {errors}")
            results.append({"dataset": dataset, "fold": fold, "seed": seed, "samples": len(sample),
                            "maximum_absolute_errors": errors, "affinity_head_unchanged": True})
    report = {"passed": True, "runs": len(results), "sampled_pairs": sum(x["samples"] for x in results),
              "method": "independent NumPy/SciPy encoder, head, exact full-distance retrieval, and aggregation",
              "tolerance": 2e-4, "results": results,
              "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    REPORT.mkdir(parents=True, exist_ok=True)
    name = f"independent_replay_{args.dataset}.json" if args.dataset else "independent_replay.json"
    (REPORT / name).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"passed": True, "runs": len(results), "sampled_pairs": report["sampled_pairs"],
                      "max_abs_error": max(max(x["maximum_absolute_errors"].values()) for x in results)}, indent=2))


if __name__ == "__main__":
    main()
