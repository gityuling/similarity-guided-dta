#!/usr/bin/env python3
"""Resume the frozen repeated-seed and strict-identity BIB GPU queue."""

import argparse
import fcntl
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from bib_common import (DATASETS, PROJECT, baseline_paths, manifest_path, record,
                        sha256, table_path, verify_freeze, write_once)


CONFIG = PROJECT / "configs/bib_validation_20261017.json"
OUTPUT = PROJECT / "results/bib_validation_20261017"


def jobs(config: dict) -> list[dict]:
    result = []
    for dataset in DATASETS:
        for seed in config["additional_training_seeds"]:
            for fold in range(5):
                result.append({"dataset": dataset, "protocol": "drug_id", "seed": seed, "fold": fold})
    for fold in range(5):
        result.append({"dataset": "kiba", "protocol": "identity", "seed": config["strict_kiba_training_seed"], "fold": fold})
    for job in result:
        job["id"] = f"bib_{job['dataset']}_{job['protocol']}_f{job['fold']}_seed{job['seed']}"
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-jobs", type=int)
    args = parser.parse_args()
    config = json.loads(CONFIG.read_text())
    queue = jobs(config)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(json.dumps({"jobs": queue, "training": config["training"]}, indent=2))
        return
    lock = (OUTPUT / "queue.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    verify_freeze()
    write_once(OUTPUT / "config.json", CONFIG.read_text())
    code_files = [Path(__file__), Path(__file__).with_name("bib_train_worker.py"),
                  Path(__file__).with_name("prepare_bib_identity_protocol.py")]
    write_once(OUTPUT / "queue.json", json.dumps({"config": record(CONFIG), "jobs": queue,
                                                 "code": [record(p) for p in code_files]}, indent=2) + "\n")
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CPU training is disabled")
    status = {"state": "running", "jobs_total": len(queue), "completed": [],
              "gpu": torch.cuda.get_device_name(0), "config_sha256": sha256(CONFIG)}

    def save():
        status["updated_utc"] = datetime.now(timezone.utc).isoformat()
        temp = OUTPUT / "status.tmp"
        temp.write_text(json.dumps(status, indent=2) + "\n")
        temp.replace(OUTPUT / "status.json")

    def execute(script: str, arguments: list, job_id: str, stage: str):
        command = [sys.executable, "-u", str(PROJECT / "scripts" / script), *map(str, arguments)]
        log_path = OUTPUT / f"{job_id}_{stage}.log"
        started = time.perf_counter()
        print(f"{job_id}: {stage}", flush=True)
        with log_path.open("a") as log:
            log.write(json.dumps({"started_utc": datetime.now(timezone.utc).isoformat(), "command": command}) + "\n")
            subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, check=True)
            log.write(json.dumps({"wall_seconds": time.perf_counter() - started}) + "\n")

    save()
    try:
        execute("prepare_bib_identity_protocol.py", [], "identity_protocol", "prepare")
        for index, job in enumerate(queue):
            if args.max_jobs is not None and index >= args.max_jobs:
                status["state"] = "completed_requested_subset"
                break
            verify_freeze(include_features=False)
            dataset, fold = job["dataset"], job["fold"]
            original = job["protocol"] == "drug_id"
            table = table_path(dataset)
            manifest = (manifest_path(dataset, fold) if original else
                        PROJECT / f"data/processed/kiba_identity_20261017/folds/drug_coldstart/fold_{fold}.csv")
            if original:
                prior = baseline_paths(dataset, fold)["prior_predictions"]
            else:
                prior_stem = f"bib_kiba_identity_f{fold}_frozen_prior"
                prior = PROJECT / f"results/predictions/{prior_stem}_predictions.csv"
                prior_result = PROJECT / f"results/{prior_stem}.json"
                if not prior.exists() and not prior_result.exists():
                    execute("evaluate_drug_similarity_prior.py", ["--data-table", table,
                            "--manifest-file", manifest, "--target-column", DATASETS[dataset]["label"],
                            "--stable-neighbors", "--output-stem", prior_stem], job["id"], "prior")
                if not prior.exists() or not prior_result.exists():
                    raise RuntimeError("Partial strict prior requires inspection")
                if json.loads(prior_result.read_text())["input_sha256"] != {"data_table": sha256(table), "manifest": sha256(manifest)}:
                    raise RuntimeError("Strict prior input hash mismatch")
            status["current_job"] = job
            status["job_index"] = index + 1
            save()
            split = job["id"].rsplit("_seed", 1)[0]
            tag = f"seed{job['seed']}"
            stem = f"{split}_{tag}"
            result_path = PROJECT / f"results/nexusdti_regressor_{stem}.json"
            prediction_path = PROJECT / f"results/predictions/nexusdti_{stem}_predictions.csv"
            checkpoint = PROJECT / f"results/checkpoints/nexusdti_{stem}.pt"
            expected = {"data_table": sha256(table), "manifest": sha256(manifest)}
            if result_path.exists() and prediction_path.exists() and checkpoint.exists():
                result = json.loads(result_path.read_text())
                if (result.get("input_sha256") != expected or result["seed"] != job["seed"]
                        or result["device"] != "cuda"
                        or any(result.get(k if k != "epochs" else "epochs_requested") != v
                               for k, v in config["training"].items())):
                    raise RuntimeError(f"Completed result/configuration mismatch: {stem}")
            else:
                if any(p.exists() for p in (result_path, prediction_path, checkpoint)):
                    raise RuntimeError(f"Partial training requires inspection: {stem}")
                options = []
                for key, value in config["training"].items():
                    if isinstance(value, bool):
                        if value:
                            options.append(f"--{key.replace('_', '-')}")
                    else:
                        options.extend([f"--{key.replace('_', '-')}", value])
                execute("bib_train_worker.py", ["--runtime-output", OUTPUT / f"{stem}_runtime.json",
                        "--split", split, "--run-tag", tag, "--seed", job["seed"],
                        "--data-table", table, "--manifest-file", manifest,
                        "--dataset-label", f"{dataset}_{job['protocol']}",
                        "--target-column", DATASETS[dataset]["label"],
                        "--token-dir", DATASETS[dataset]["tokens"], "--device", "cuda",
                        "--save-predictions", *options], job["id"], "train")
            ablation = PROJECT / f"results/{stem}_ablation.json"
            if not ablation.exists():
                execute("evaluate_affinity_ablation.py", ["--deep", prediction_path, "--prior", prior,
                        "--data-table", table, "--manifest-file", manifest,
                        "--target-column", DATASETS[dataset]["label"], "--output-stem", stem], job["id"], "ablation")
            metrics = json.loads(ablation.read_text())
            if metrics["input_sha256"] != {"deep": sha256(prediction_path), "prior": sha256(prior),
                                           "table": expected["data_table"], "manifest": expected["manifest"]}:
                raise RuntimeError("Ablation inputs differ from frozen files")
            status["completed"].append({**job, "result": record(ablation), "test": metrics["test"]})
            save()
            print(f"Completed {index + 1}/{len(queue)}: {job['id']}", flush=True)
        else:
            status["state"] = "completed"
        status.pop("current_job", None)
    except BaseException as error:
        status["state"] = "failed"
        status["error"] = repr(error)
        raise
    finally:
        save()
        lock.close()


if __name__ == "__main__":
    main()
