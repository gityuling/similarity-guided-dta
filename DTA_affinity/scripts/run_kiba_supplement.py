#!/usr/bin/env python3
"""Resume KIBA preparation, CUDA inference/training and matched ablations."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


PROJECT = Path(__file__).resolve().parents[1]
DATA = PROJECT / "data/processed/kiba_dtiam_protocol"
CONFIG = PROJECT / "configs/kiba_drugcold_20261016.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize_priors(config):
    import pandas as pd
    paths = [PROJECT / "results" / f"kiba_dtiam_drugcold_f{fold}_frozen_morgan_prior.json" for fold in range(5)]
    if not all(path.exists() for path in paths):
        return
    table = pd.read_csv(DATA / "table.csv", usecols=["row_id", "kiba_score", "Canonical_SMILES"])
    fold_results, test_rows = [], []
    for fold, path in enumerate(paths):
        result = json.loads(path.read_text())
        manifest_path = DATA / "folds/drug_coldstart" / f"fold_{fold}.csv"
        if result["input_sha256"] != {"data_table": sha256(DATA / "table.csv"), "manifest": sha256(manifest_path)}:
            raise RuntimeError(f"Prior fold {fold} has stale input hashes")
        manifest = pd.read_csv(manifest_path, usecols=["row_id", "split"])
        prediction_path = PROJECT / "results/predictions" / f"kiba_dtiam_drugcold_f{fold}_frozen_morgan_prior_predictions.csv"
        predictions = pd.read_csv(prediction_path, usecols=["row_id", "split", "affinity_true", "prediction_raw"])
        expected = manifest.loc[manifest.split.isin(["val", "test"])].merge(table, on="row_id", validate="one_to_one")
        checked = expected.merge(predictions, on=["row_id", "split"], validate="one_to_one")
        if len(checked) != len(expected) or len(checked) != len(predictions):
            raise RuntimeError(f"Prior fold {fold} prediction coverage mismatch")
        if not np.allclose(checked.kiba_score, checked.affinity_true, rtol=0, atol=1e-12):
            raise RuntimeError(f"Prior fold {fold} label mismatch")
        test = checked.loc[checked.split.eq("test")]
        mse = float(np.mean((test.kiba_score - test.prediction_raw) ** 2))
        if abs(mse - result["test"]["raw"]["mse"]) > 1e-12:
            raise RuntimeError(f"Prior fold {fold} MSE mismatch")
        train_smiles = set(table.loc[table.row_id.isin(manifest.loc[manifest.split.eq("train"), "row_id"]), "Canonical_SMILES"])
        novel = test.loc[~test.Canonical_SMILES.isin(train_smiles)]
        fold_results.append({"fold": fold, "test": result["test"]["raw"],
                             "test_rows": len(test), "test_identity_novel_rows": len(novel),
                             "test_identity_novel_mse": float(np.mean((novel.kiba_score - novel.prediction_raw) ** 2)),
                             "selected_hyperparameters": result["selected_hyperparameters"],
                             "result_sha256": sha256(path), "prediction_sha256": sha256(prediction_path)})
        test_rows.extend(test.row_id.tolist())
    if sorted(test_rows) != table.row_id.tolist():
        raise RuntimeError("KIBA prior outer predictions do not cover every pair exactly once")
    summary = {"model": "training-only Morgan neighbor prior", "scope": "No deep model or fusion result is included",
               "aggregation": "Unweighted mean and sample SD across official drug-ID folds; raw predictions, no affine calibration",
               "pairs_checked": len(test_rows), "config_sha256": sha256(CONFIG),
               "metrics": {}, "folds": fold_results, "published_reference": config["published_reference"],
               "caveat": "Official drug-ID cold includes duplicate canonical structures; novel-row errors are a subset analysis under this protocol, not retraining under identity-grouped folds."}
    for metric in ("mse", "ci", "pearson"):
        values = [fold["test"][metric] for fold in fold_results]
        summary["metrics"][metric] = {"mean": float(np.mean(values)), "sd_across_folds": float(np.std(values, ddof=1)), "per_fold": values}
    output = PROJECT / "results/kiba_drugcold_prior_5fold.json"
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"prior_summary": str(output), "metrics": summary["metrics"]}, indent=2), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("all", "prepare", "prior", "features", "train", "summary"), default="all")
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(5)))
    args = parser.parse_args()
    if not set(args.folds) <= set(range(5)) or len(set(args.folds)) != len(args.folds):
        raise ValueError("Folds must be unique integers in 0..4")
    config = json.loads(CONFIG.read_text())
    stages = ["prepare", "prior", "features", "train", "summary"] if args.stage == "all" else [args.stage]
    tag = config["run_tag"]
    log_dir = PROJECT / "results/logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    log_path = log_dir / f"kiba_supplement_{args.stage}_{timestamp}.log"
    status_path = PROJECT / "results/kiba_drugcold_supplement_status.json"
    lock_path = log_dir / "kiba_supplement.lock"
    import fcntl
    lock = lock_path.open("a")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as error:
        raise RuntimeError("Another KIBA supplement runner is active; use its log") from error
    status = {"config": str(CONFIG), "config_sha256": sha256(CONFIG), "requested_stages": stages,
              "requested_folds": args.folds, "log": str(log_path), "completed_stages": [], "state": "running"}

    def save_status():
        status["updated_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        status["completed_prior_folds"] = [fold for fold in range(5) if (PROJECT / "results" / f"kiba_dtiam_drugcold_f{fold}_frozen_morgan_prior.json").exists()]
        status["completed_training_folds"] = [fold for fold in range(5) if (PROJECT / "results" / f"kiba_dtiam_drugcold_f{fold}_{tag}_ablation.json").exists()]
        status["experiment_complete"] = len(status["completed_training_folds"]) == 5
        status_path.write_text(json.dumps(status, indent=2) + "\n")

    def execute(script, *arguments):
        command = [sys.executable, "-u", str(PROJECT / "scripts" / script), *map(str, arguments)]
        print(f"Running {script}: {' '.join(map(str, arguments))}", flush=True)
        with log_path.open("a") as log:
            log.write(json.dumps({"command": command}) + "\n")
            process = subprocess.Popen(command, cwd=PROJECT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            try:
                for line in process.stdout:
                    print(line, end="", flush=True)
                    log.write(line)
                    log.flush()
                return_code = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)

    def require_cuda():
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("GPU stage pending: this process cannot access CUDA. Run this same command from a GPU-enabled host session; CPU training is disabled.")
        print(f"GPU: {torch.cuda.get_device_name(0)}; PyTorch {torch.__version__}", flush=True)

    save_status()
    try:
        for stage in stages:
            status["current_stage"] = stage
            save_status()
            if stage == "prepare":
                execute("prepare_dtiam_kiba_protocol.py")
            elif stage == "prior":
                for fold in args.folds:
                    manifest = DATA / "folds/drug_coldstart" / f"fold_{fold}.csv"
                    stem = f"kiba_dtiam_drugcold_f{fold}_frozen_morgan_prior"
                    result_file = PROJECT / "results" / f"{stem}.json"
                    predictions = PROJECT / "results/predictions" / f"{stem}_predictions.csv"
                    expected = {"data_table": sha256(DATA / "table.csv"), "manifest": sha256(manifest)}
                    if result_file.exists() and predictions.exists():
                        result = json.loads(result_file.read_text())
                        if result.get("input_sha256") != expected or not result.get("stable_neighbors"):
                            raise RuntimeError(f"Frozen prior inputs/configuration changed: {result_file}")
                        print(f"Reuse frozen prior fold {fold}", flush=True)
                    else:
                        execute("evaluate_drug_similarity_prior.py", "--data-table", DATA / "table.csv",
                                "--manifest-file", manifest, "--target-column", "kiba_score", "--stable-neighbors",
                                "--output-stem", stem)
                summarize_priors(config)
            elif stage == "features":
                require_cuda()
                # Fresh local-weight inference gives this benchmark an explicit cache provenance.
                execute("build_kiba_benchmark_tokens.py")
            elif stage == "train":
                require_cuda()
                for fold in args.folds:
                    split = f"kiba_dtiam_drugcold_f{fold}"
                    stem = f"{split}_{tag}"
                    manifest = DATA / "folds/drug_coldstart" / f"fold_{fold}.csv"
                    result_file = PROJECT / "results" / f"nexusdti_regressor_{stem}.json"
                    predictions = PROJECT / "results/predictions" / f"nexusdti_{stem}_predictions.csv"
                    checkpoint = PROJECT / "results/checkpoints" / f"nexusdti_{stem}.pt"
                    if result_file.exists() and predictions.exists() and checkpoint.exists():
                        result = json.loads(result_file.read_text())
                        expected = {"data_table": sha256(DATA / "table.csv"), "manifest": sha256(manifest)}
                        expected_options = {key: config[key] for key in ("seed", "batch_size", "lr", "patience", "latent_dim", "k_ratio", "dropout", "pool_fusion", "hybrid", "target_normalize")}
                        if (result.get("input_sha256") != expected
                                or result.get("epochs_requested") != config["epochs"]
                                or result.get("device") != "cuda"
                                or any(result.get(key) != value for key, value in expected_options.items())):
                            raise RuntimeError(f"Completed run does not match frozen inputs/configuration: {result_file}")
                        print(f"Reuse completed training fold {fold}", flush=True)
                    else:
                        if result_file.exists() or checkpoint.exists() or predictions.exists():
                            raise RuntimeError(f"Partial saved run requires inspection before retraining: {stem}")
                        options = []
                        for key in ("epochs", "batch_size", "workers", "lr", "k_ratio", "latent_dim", "dropout", "patience", "seed"):
                            options.extend([f"--{key.replace('_', '-')}", config[key]])
                        options.extend(f"--{key.replace('_', '-')}" for key in ("pool_fusion", "hybrid", "target_normalize") if config[key])
                        execute("train_nexusdti_regressor.py", "--split", split,
                                "--data-table", DATA / "table.csv", "--manifest-file", manifest,
                                "--dataset-label", "kiba_dtiam_protocol", "--target-column", "kiba_score",
                                "--token-dir", DATA / "tokens", "--device", "cuda", "--save-predictions",
                                "--run-tag", tag, *options)
                    execute("evaluate_affinity_ablation.py", "--deep", predictions,
                            "--prior", PROJECT / "results/predictions" / f"{split}_frozen_morgan_prior_predictions.csv",
                            "--data-table", DATA / "table.csv", "--manifest-file", manifest,
                            "--target-column", "kiba_score", "--output-stem", stem)
            elif stage == "summary":
                results = []
                for fold in range(5):
                    path = PROJECT / "results" / f"kiba_dtiam_drugcold_f{fold}_{tag}_ablation.json"
                    if not path.exists():
                        raise RuntimeError(f"Complete five-fold summary is pending: {path.name}")
                    results.append(json.loads(path.read_text()))
                summary = {"config_sha256": sha256(CONFIG), "aggregation": "Unweighted five-fold mean and sample SD; one training seed",
                           "published_reference": config["published_reference"], "models": {}}
                for model in ("deep", "neighbor_prior", "fixed_blend", "similarity_gate"):
                    summary["models"][model] = {
                        metric: {"mean": float(np.mean(values)), "sd_across_folds": float(np.std(values, ddof=1)), "per_fold": values}
                        for metric in ("mse", "ci", "pearson")
                        for values in [[result["test"][model][metric] for result in results]]
                    }
                output = PROJECT / "results/kiba_drugcold_supplement_5fold.json"
                output.write_text(json.dumps(summary, indent=2) + "\n")
                print(json.dumps(summary, indent=2), flush=True)
            status["completed_stages"].append(stage)
            save_status()
        status["state"] = "completed_requested_stages"
    except BaseException as error:
        status["state"] = "incomplete"
        status["error"] = str(error)
        raise
    finally:
        save_status()
        lock.close()
        print(f"Status: {status_path}\nLog: {log_path}", flush=True)


if __name__ == "__main__":
    main()
