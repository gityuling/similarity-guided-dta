#!/usr/bin/env python3
"""Build features and run frozen KIBA models under the shared single-GPU lock."""

import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from bib_common import PROJECT, ROOT, record, sha256, write_once
from predict_bib_cellular_affinity import MODELS, combine, deep_predictions, fold_inputs, neighbor_predictions
from run_bib_results34 import require_idle_gpu


OUT = PROJECT / "reports/bib_nanobret_20261018"
RUN = PROJECT / "results/bib_nanobret_20261019"
TOKENS = PROJECT / "data/processed/bib_nanobret_20261019/tokens"
KIBA = PROJECT / "data/processed/kiba_dtiam_protocol"


def validate_freeze():
    frozen = json.loads((OUT / "freeze.json").read_text())["files"]
    for item in frozen:
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"Frozen input changed: {item['path']}")


def infer():
    query = pd.read_csv(OUT / "query_table.csv")
    table = pd.read_csv(KIBA / "table.csv")
    compounds = pd.read_csv(OUT / "compound_audit.csv", keep_default_na=False)
    records, checks = [], []
    for fold in range(5):
        subset = query.loc[query.eligible_folds.astype(str).map(lambda s: str(fold) in s.split("|"))].copy()
        if subset.empty:
            continue
        source, training, hyperparameters, fit = fold_inputs(fold, table)
        manifest = pd.read_csv(source["manifest"])
        selected = table.merge(manifest, on="row_id", validate="one_to_one")
        seen_ids = set(selected.loc[selected.split.isin(["train", "val"]), "drug_id"].astype(str))
        for row in compounds.loc[compounds.compound.isin(subset.compound)].itertuples(index=False):
            ids = set(row.kiba_exact_ids.split("|") + row.kiba_parent_ids.split("|")) - {""}
            if ids & seen_ids:
                raise RuntimeError("Query compound present in model training or validation")
        if set(subset.Sequence) - set(training.Sequence):
            raise RuntimeError("Query target missing from training")
        # Reproduce historical held-out predictions before applying the same functions externally.
        historical = pd.read_csv(source["saved_predictions"], float_precision="round_trip")
        historical = historical.loc[historical.split.eq("test")].head(8)
        reference = historical[["row_id"]].merge(table, on="row_id", validate="one_to_one")
        ref_deep = deep_predictions(reference, KIBA / "tokens", source["checkpoint"], torch.device("cuda"))
        ref_prior, ref_sim = neighbor_predictions(reference, training, hyperparameters)
        ref_values = combine(ref_deep, ref_prior, ref_sim, fit)
        errors = {m: float(np.max(np.abs(ref_values[m] - historical[f"prediction_{m}"].to_numpy()))) for m in MODELS}
        if errors["neighbor_prior"] > 2e-6 or max(errors.values()) > 0.02:
            raise RuntimeError(f"Historical compatibility failed: {errors}")
        if not np.allclose(ref_sim, historical.max_train_drug_tanimoto, atol=1e-12, rtol=0):
            raise RuntimeError("Historical similarity mismatch")
        deep = deep_predictions(subset, TOKENS, source["checkpoint"], torch.device("cuda"))
        prior, similarity = neighbor_predictions(subset, training, hyperparameters)
        values = combine(deep, prior, similarity, fit)
        frame = subset[["row_id"]].copy()
        for key, vector in values.items():
            if not np.isfinite(vector).all():
                raise RuntimeError("Nonfinite model prediction")
            frame[key] = vector
        frame["fold"] = fold
        records.append(frame)
        checks.append({"fold": fold, "queries": len(subset), "compounds": subset.compound.nunique(),
                       "train_validation_overlap": 0, "reference_max_errors": errors})
        print(f"Fold {fold + 1}/5: {len(frame)} predictions; historical checks {errors}", flush=True)
    predictions = pd.concat(records, ignore_index=True)
    expected = query.eligible_folds.astype(str).map(lambda s: len(s.split("|")))
    counts = predictions.groupby("row_id").size().reindex(query.row_id).to_numpy()
    if not np.array_equal(counts, expected.to_numpy()):
        raise RuntimeError("Incorrect eligible-fold prediction coverage")
    write_once(OUT / "fold_predictions.csv", predictions.to_csv(index=False))
    averaged = predictions.groupby("row_id", sort=True)[[*MODELS, "max_train_drug_tanimoto", "deep_weight"]].mean().reset_index()
    averaged["fold_count"] = averaged.row_id.map(predictions.groupby("row_id").size())
    write_once(OUT / "predictions.csv", averaged.to_csv(index=False))
    write_once(OUT / "inference_audit.json", json.dumps({"fold_checks": checks, "training_performed": False,
               "endpoint_values_read": False, "device": torch.cuda.get_device_name(0),
               "files": [record(OUT / n) for n in ("fold_predictions.csv", "predictions.csv")]}, indent=2) + "\n")


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    started = time.perf_counter()
    state = {"state": "starting", "pid": os.getpid(), "started_utc": datetime.now(timezone.utc).isoformat(),
             "gpu_jobs_at_once": 1, "completed_stages": []}

    def save():
        state["updated_utc"] = datetime.now(timezone.utc).isoformat()
        tmp = RUN / "status.tmp"
        tmp.write_text(json.dumps(state, indent=2) + "\n")
        tmp.replace(RUN / "status.json")

    sources = [PROJECT / "scripts" / n for n in ("run_bib_nanobret.py", "evaluate_bib_nanobret.py", "predict_bib_cellular_affinity.py", "train_nexusdti_regressor.py", "build_kiba_benchmark_tokens.py", "build_bindingdb_articles_kd_tokens.py")]
    write_once(RUN / "code_freeze.json", json.dumps([record(p) for p in sources], indent=2) + "\n")
    with (PROJECT / "results/bib_validation_20261017/queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save()
        try:
            validate_freeze()
            require_idle_gpu()
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA unavailable")
            for path, expected in json.loads((KIBA / "tokens/build_audit.json").read_text())["model_sha256"].items():
                if sha256(Path(path)) != expected:
                    raise RuntimeError("Encoder weights changed")
            memory = {line.split(":")[0]: int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines()}
            if memory["MemAvailable"] < 12 * 1024 * 1024:
                raise RuntimeError("Less than 12 GiB available host RAM")
            for name, command in (
                ("features", [sys.executable, "-u", "scripts/build_kiba_benchmark_tokens.py", "--data-table", str(OUT / "query_table.csv"),
                              "--cache-dir", str(TOKENS), "--reuse-cache", str(KIBA / "tokens"), "--drug-batch-size", "1", "--protein-batch-size", "1"]),
                ("inference", None),
                ("evaluation", [sys.executable, "-u", "scripts/evaluate_bib_nanobret.py"]),
            ):
                if name != "evaluation":
                    require_idle_gpu()
                state.update(state="running", stage=name)
                save()
                begin = time.perf_counter()
                if command:
                    with (RUN / f"{name}.log").open("a") as handle:
                        handle.write(json.dumps({"command": command}) + "\n")
                        handle.flush()
                        subprocess.run(command, cwd=PROJECT, stdout=handle, stderr=subprocess.STDOUT, check=True)
                else:
                    infer()
                state["completed_stages"].append({"stage": name, "seconds": time.perf_counter() - begin})
                save()
            validate_freeze()
            state.update(state="completed", elapsed_seconds=time.perf_counter() - started)
        except BaseException as error:
            state.update(state="failed", error=repr(error))
            raise
        finally:
            save()


if __name__ == "__main__":
    main()
