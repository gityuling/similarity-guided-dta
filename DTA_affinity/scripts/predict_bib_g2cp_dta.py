#!/usr/bin/env python3
"""Obtain frozen DTA target profiles for the audited LINCS compounds."""

import json

import numpy as np
import pandas as pd
import torch

from bib_g2cp_common import KIBA, OUT, TOKENS, record, save_json, verify_data, write_once
from predict_bib_cellular_affinity import MODELS, combine, deep_predictions, fold_inputs
from predict_bib_cellular_affinity import neighbor_predictions


def main():
    torch.set_num_threads(2)
    verify_data()
    query = pd.read_csv(OUT / "query_table.csv")
    table = pd.read_csv(KIBA / "table.csv")
    frames, audit = [], []
    if set(query.Canonical_SMILES) & set(table.Canonical_SMILES):
        raise RuntimeError("External query overlaps KIBA")
    for fold in range(5):
        source, training, hyperparameters, fit = fold_inputs(fold, table)
        output = OUT / f"affinity_fold_{fold}.csv"
        current = {key: record(path) for key, path in source.items()}
        current["query"] = record(OUT / "query_table.csv")
        frozen = OUT / f"affinity_fold_{fold}_inputs.json"
        save_json(frozen, current, immutable=True)
        if output.exists():
            frame = pd.read_csv(output, float_precision="round_trip")
        else:
            historical = pd.read_csv(source["saved_predictions"], float_precision="round_trip")
            historical = historical.loc[historical.split.eq("test")].head(8)
            reference = historical[["row_id"]].merge(table, on="row_id", validate="one_to_one")
            d = deep_predictions(reference, KIBA / "tokens", source["checkpoint"], torch.device("cuda"))
            p, s = neighbor_predictions(reference, training, hyperparameters)
            values = combine(d, p, s, fit)
            errors = {m: float(np.max(np.abs(values[m] - historical[f"prediction_{m}"].to_numpy()))) for m in MODELS}
            if errors["neighbor_prior"] > 2e-6 or max(errors.values()) > 0.02:
                raise RuntimeError(f"Historical compatibility failed: {errors}")
            deep = deep_predictions(query, TOKENS, source["checkpoint"], torch.device("cuda"))
            prior, similarity = neighbor_predictions(query, training, hyperparameters)
            values = combine(deep, prior, similarity, fit)
            frame = query[["row_id"]].copy()
            for key, value in values.items():
                frame[key] = value
            frame["fold"] = fold
            write_once(output, frame.to_csv(index=False))
            save_json(OUT / f"affinity_fold_{fold}_compatibility.json", errors, immutable=True)
        if frame.row_id.tolist() != query.row_id.tolist() or not np.isfinite(frame[list(MODELS)].to_numpy()).all():
            raise RuntimeError("Invalid DTA prediction coverage")
        frames.append(frame)
        audit.append({"fold": fold, "pairs": len(frame), "file": record(output)})
        print(f"DTA fold {fold+1}/5 complete: {len(frame)} pairs", flush=True)
    combined = pd.concat(frames, ignore_index=True)
    means = combined.groupby("row_id", sort=True)[[*MODELS, "max_train_drug_tanimoto", "deep_weight"]].mean().reset_index()
    write_once(OUT / "affinity_predictions.csv", means.to_csv(index=False))
    save_json(OUT / "affinity_audit.json", {"folds": audit, "training_performed": False,
              "endpoint_values_read": False, "output": record(OUT / "affinity_predictions.csv")}, immutable=True)


if __name__ == "__main__":
    main()
