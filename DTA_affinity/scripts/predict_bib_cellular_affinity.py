#!/usr/bin/env python3
"""Apply frozen KIBA fold predictors to the external candidate library."""

import argparse
import gc
import json

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from scipy.special import expit
from torch.utils.data import DataLoader

from bib_common import PROJECT, ROOT, canonical, record, sha256, write_once
from train_nexusdti_regressor import AffinityDataset, NexusDTIAffinityRegressor, collate_fn, predict


CASE = PROJECT / "reports/bib_cellular_case_external_20261018"
TOKENS = PROJECT / "data/processed/bib_cellular_case_external_20261018/tokens"
KIBA = PROJECT / "data/processed/kiba_dtiam_protocol"
MODELS = ("deep", "neighbor_prior", "fixed_blend", "similarity_gate")


def paths(fold):
    stem = f"bib_kiba_identity_f{fold}_seed20261012"
    return {
        "checkpoint": PROJECT / f"results/checkpoints/nexusdti_{stem}.pt",
        "result": PROJECT / f"results/nexusdti_regressor_{stem}.json",
        "prior": PROJECT / f"results/bib_kiba_identity_f{fold}_frozen_prior.json",
        "fusion": PROJECT / f"results/{stem}_ablation.json",
        "manifest": PROJECT / f"data/processed/kiba_identity_20261017/folds/drug_coldstart/fold_{fold}.csv",
        "saved_predictions": PROJECT / f"results/predictions/{stem}_ablation_predictions.csv",
    }


def frozen_inputs():
    sources = [CASE / "plan.json", CASE / "query_table.csv", KIBA / "table.csv",
               PROJECT / "scripts/train_nexusdti_regressor.py"]
    sources.extend(path for f in range(5) for path in paths(f).values())
    return [record(p) for p in sources]


def neighbor_predictions(query, training, hyperparameters):
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fingerprints = {s: generator.GetFingerprint(Chem.MolFromSmiles(s))
                    for s in set(training.Canonical_SMILES) | set(query.Canonical_SMILES)}
    support = {}
    for sequence, group in training.groupby("Sequence", sort=False):
        support[sequence] = ([fingerprints[s] for s in group.Canonical_SMILES],
                             group.kiba_score.to_numpy(dtype=np.float64))
    prior, maximum = [], []
    for row in query.itertuples(index=False):
        fps, labels = support[row.Sequence]
        similarities = np.asarray(DataStructs.BulkTanimotoSimilarity(fingerprints[row.Canonical_SMILES], fps))
        order = np.argsort(-similarities, kind="stable")[:hyperparameters["top_k"]]
        weights = similarities[order] ** hyperparameters["gamma"]
        prior.append(float(weights @ labels[order] / weights.sum()) if weights.sum() else float(labels.mean()))
        maximum.append(float(similarities.max()))
    return np.asarray(prior), np.asarray(maximum)


def deep_predictions(query, tokens, checkpoint, device):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    args, state, result = payload["args"], payload["model"], payload["result"]
    options = {k: args[k] for k in ("latent_dim", "pool_fusion", "drug_attention", "pocket_fusion",
                                  "pocket_residual", "hybrid", "censor_aware", "dropout")}
    if options["censor_aware"] or not options["hybrid"] or not args["target_normalize"]:
        raise RuntimeError("Unexpected model family for the frozen KIBA case")
    model = NexusDTIAffinityRegressor(**options, pooled_mean=state["pooled_mean"].numpy(),
                                     pooled_std=state["pooled_std"].numpy())
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    # The dataset API requires a target field; this placeholder is never used for fitting or scoring.
    dataset = AffinityDataset(query.assign(inference_placeholder=0.0), token_dir=tokens,
                              target_column="inference_placeholder")
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate_fn)
    with torch.inference_mode():
        _, values = predict(model, loader, device, args["k_ratio"])
    values = values.astype(np.float64) * result["target_std"] + result["target_mean"]
    del model, payload, state, loader, dataset
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return values


def combine(deep, prior, similarity, fit):
    alpha = fit["fixed_deep_weight"]
    gate = fit["gate"]
    weight = expit(gate["a"] + gate["b"] * (similarity - gate["sim_mean"]) / gate["sim_std"])
    return {"deep": deep, "neighbor_prior": prior,
            "fixed_blend": alpha * deep + (1 - alpha) * prior,
            "similarity_gate": weight * deep + (1 - weight) * prior,
            "max_train_drug_tanimoto": similarity, "deep_weight": weight}


def fold_inputs(fold, table):
    source = paths(fold)
    manifest = pd.read_csv(source["manifest"])
    joined = manifest[["row_id", "split"]].merge(table, on="row_id", validate="one_to_one", sort=False)
    training = joined.loc[joined.split.eq("train")].copy()
    prior = json.loads(source["prior"].read_text())
    fit = json.loads(source["fusion"].read_text())
    expected = {"data_table": sha256(KIBA / "table.csv"), "manifest": sha256(source["manifest"])}
    if prior["input_sha256"] != expected or not prior["stable_neighbors"]:
        raise RuntimeError("Frozen prior source mismatch")
    if fit["input_sha256"]["manifest"] != expected["manifest"] or fit["input_sha256"]["table"] != expected["data_table"]:
        raise RuntimeError("Frozen gate source mismatch")
    return source, training, prior["selected_hyperparameters"], fit


def smoke_check(table):
    source, training, hyperparameters, fit = fold_inputs(0, table)
    saved = pd.read_csv(source["saved_predictions"])
    chosen = saved.loc[saved.split.eq("test")].head(4)
    query = chosen[["row_id"]].merge(table, on="row_id", validate="one_to_one")
    training = training.loc[training.Sequence.isin(query.Sequence)]
    prior, similarity = neighbor_predictions(query, training, hyperparameters)
    deep = deep_predictions(query, KIBA / "tokens", source["checkpoint"], torch.device("cpu"))
    predictions = combine(deep, prior, similarity, fit)
    errors = {m: float(np.max(np.abs(predictions[m] - chosen[f"prediction_{m}"].to_numpy()))) for m in MODELS}
    if errors["neighbor_prior"] > 2e-6 or errors["deep"] > 0.02:
        raise RuntimeError(f"Reference prediction compatibility failed: {errors}")
    if not np.allclose(similarity, chosen.max_train_drug_tanimoto, atol=1e-12, rtol=0):
        raise RuntimeError("Reference neighbor similarity differs")
    report = {"device": "cpu", "reference_fold": 0, "reference_rows": query.row_id.tolist(),
              "maximum_absolute_errors": errors, "note": "CPU float32 compared to saved CUDA mixed-precision outputs; no retraining",
              "source_checkpoint": record(source["checkpoint"])}
    write_once(CASE / "inference_smoke_check.json", json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-check", action="store_true")
    parser.add_argument("--freeze-inputs", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    sources = frozen_inputs()
    if args.freeze_inputs:
        write_once(CASE / "inference_inputs.json", json.dumps(sources, indent=2) + "\n")
        print(f"Frozen {len(sources)} inference sources", flush=True)
        return
    frozen = json.loads((CASE / "inference_inputs.json").read_text())
    if sources != frozen:
        raise RuntimeError("Frozen inference inputs changed")
    table = pd.read_csv(KIBA / "table.csv")
    if args.smoke_check:
        smoke_check(table)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for queued candidate inference")
    query = pd.read_csv(CASE / "query_table.csv")
    if query.Canonical_SMILES.duplicated().any():
        raise RuntimeError("Candidate library contains duplicate identities")
    if set(query.Canonical_SMILES.map(canonical)) & set(table.Canonical_SMILES.map(canonical)):
        raise RuntimeError("External candidate appears in KIBA")
    records, checks = [], []
    for fold in range(5):
        source, training, hyperparameters, fit = fold_inputs(fold, table)
        training = training.loc[training.Sequence.isin(query.Sequence)]
        if set(query.Sequence) - set(training.Sequence):
            raise RuntimeError("A query target has no training support")
        deep = deep_predictions(query, TOKENS, source["checkpoint"], torch.device("cuda"))
        prior, similarity = neighbor_predictions(query, training, hyperparameters)
        values = combine(deep, prior, similarity, fit)
        frame = query[["row_id", "Canonical_SMILES"]].rename(columns={"Canonical_SMILES": "chemical_identity"}).copy()
        for key, vector in values.items():
            if not np.isfinite(vector).all():
                raise RuntimeError("Non-finite candidate prediction")
            frame[key] = vector
        frame["fold"] = fold
        records.append(frame)
        checks.append({"fold": fold, "target_training_pairs": len(training),
                       "target_training_identities": int(training.Canonical_SMILES.nunique()),
                       "candidate_training_overlap": 0})
        print(f"Completed candidate inference fold {fold + 1}/5: {len(frame)} compounds", flush=True)
    all_folds = pd.concat(records, ignore_index=True)
    write_once(CASE / "affinity_fold_predictions.csv", all_folds.to_csv(index=False))
    averaged = all_folds.groupby(["row_id", "chemical_identity"], sort=True)[list(values)].mean().reset_index()
    write_once(CASE / "affinity_predictions.csv", averaged.to_csv(index=False))
    report = {"source_records": frozen, "fold_checks": checks, "candidates": len(query),
              "aggregation": "Unweighted mean of five frozen fold predictors; gate applied within fold before averaging",
              "endpoint_values_read": False, "training_performed": False,
              "outputs": [record(CASE / n) for n in ("affinity_fold_predictions.csv", "affinity_predictions.csv")]}
    write_once(CASE / "affinity_audit.json", json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
