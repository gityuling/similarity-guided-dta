#!/usr/bin/env python3
"""Run the five preplanned scaffold folds serially with the existing trainer."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone

from bib_common import DATASETS, PROJECT, ROOT, record, sha256, verify_freeze, write_once


PROTOCOL = PROJECT / "data/processed/kiba_scaffold_20261017"
OUTPUT = PROJECT / "results/bib_scaffold_20261018"
MODELS = ("deep", "neighbor_prior", "fixed_blend", "similarity_gate")


def now():
    return datetime.now(timezone.utc).isoformat()


def check_protocol():
    plan = json.loads((PROTOCOL / "plan.json").read_text())
    audit = json.loads((PROTOCOL / "summary.json").read_text())
    if record(PROTOCOL / "plan.json") != audit["plan"]:
        raise RuntimeError("Scaffold plan differs from the audited plan")
    if record(ROOT / plan["table"]["path"]) != plan["table"]:
        raise RuntimeError("Scaffold table changed")
    if (audit["pairs_removed"] or not audit["every_test_row_once"]
            or [f["fold"] for f in audit["folds"]] != list(range(5))):
        raise RuntimeError("Scaffold fold coverage audit failed")
    for fold in audit["folds"]:
        if record(ROOT / fold["manifest"]["path"]) != fold["manifest"]:
            raise RuntimeError("Scaffold manifest changed")
        for key in ("chemical_identity_overlap", "scaffold_overlap", "unsupported_same_target_rows"):
            if any(fold[key].values()):
                raise RuntimeError(f"Scaffold audit failed: {key}")
    return plan, audit


def require_idle_gpu():
    result = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    active = {int(line) for line in result.stdout.splitlines() if line.strip().isdigit()}
    active.discard(os.getpid())
    if active:
        raise RuntimeError(f"Another CUDA process is active: {sorted(active)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--max-jobs", type=int, choices=range(1, 6))
    args = parser.parse_args()
    plan, audit = check_protocol()
    jobs = [{"dataset": "kiba", "protocol": "scaffold", "fold": f,
             "seed": plan["training_seed"],
             "id": f"bib_kiba_scaffold_f{f}_seed{plan['training_seed']}"}
            for f in range(5)]
    if args.dry_run:
        print(json.dumps({"jobs": jobs, "training": plan["training"],
                          "protocol_audit": "passed", "gpu_jobs_at_once": 1}, indent=2))
        return

    OUTPUT.mkdir(parents=True, exist_ok=True)
    # Share the earlier queue's lock so the two DTA queues cannot overlap.
    lock_path = PROJECT / "results/bib_validation_20261017/queue.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; CPU training is disabled")
        require_idle_gpu()
        code = [PROJECT / "scripts" / name for name in (
            "run_bib_scaffold.py", "bib_common.py", "bib_train_worker.py",
            "train_nexusdti_regressor.py", "evaluate_drug_similarity_prior.py",
            "evaluate_affinity_ablation.py")]
        write_once(OUTPUT / "queue.json", json.dumps({
            "plan": record(PROTOCOL / "plan.json"), "audit": record(PROTOCOL / "summary.json"),
            "jobs": jobs, "code": [record(path) for path in code],
            "training": plan["training"], "gpu_jobs_at_once": 1,
        }, indent=2) + "\n")
        status = {"state": "running", "stage": "verify_frozen_inputs", "pid": os.getpid(),
                  "started_utc": now(), "jobs_total": 5, "completed": [],
                  "gpu": torch.cuda.get_device_name(0)}

        def save():
            status["updated_utc"] = now()
            temp = OUTPUT / "status.tmp"
            temp.write_text(json.dumps(status, indent=2) + "\n")
            temp.replace(OUTPUT / "status.json")

        def execute(script, arguments, job_id, stage):
            status["stage"] = stage
            save()
            command = [sys.executable, "-u", str(PROJECT / "scripts" / script),
                       *map(str, arguments)]
            print(f"{job_id}: {stage}", flush=True)
            started = time.perf_counter()
            with (OUTPUT / f"{job_id}_{stage}.log").open("a") as log:
                log.write(json.dumps({"started_utc": now(), "command": command}) + "\n")
                log.flush()
                subprocess.run(command, cwd=PROJECT, stdout=log,
                               stderr=subprocess.STDOUT, check=True)
                log.write(json.dumps({"wall_seconds": time.perf_counter() - started}) + "\n")

        save()
        try:
            verify_freeze()
            for job, fold in zip(jobs, audit["folds"]):
                if args.max_jobs is not None and len(status["completed"]) >= args.max_jobs:
                    status["state"] = "completed_requested_subset"
                    break
                check_protocol()
                verify_freeze(include_features=False)
                status.update(current_job=job, job_index=job["fold"] + 1)
                save()
                table = ROOT / plan["table"]["path"]
                manifest = ROOT / fold["manifest"]["path"]
                expected = {"data_table": sha256(table), "manifest": sha256(manifest)}
                prior_stem = f"bib_kiba_scaffold_f{job['fold']}_frozen_prior"
                prior_result = PROJECT / f"results/{prior_stem}.json"
                prior = PROJECT / f"results/predictions/{prior_stem}_predictions.csv"
                if not any(p.exists() for p in (prior_result, prior)):
                    execute("evaluate_drug_similarity_prior.py", [
                        "--data-table", table, "--manifest-file", manifest,
                        "--target-column", DATASETS["kiba"]["label"],
                        "--stable-neighbors", "--output-stem", prior_stem], job["id"], "prior")
                if not all(p.exists() for p in (prior_result, prior)):
                    raise RuntimeError("Partial scaffold prior requires inspection")
                if json.loads(prior_result.read_text())["input_sha256"] != expected:
                    raise RuntimeError("Scaffold prior inputs changed")

                stem = job["id"]
                result = PROJECT / f"results/nexusdti_regressor_{stem}.json"
                prediction = PROJECT / f"results/predictions/nexusdti_{stem}_predictions.csv"
                checkpoint = PROJECT / f"results/checkpoints/nexusdti_{stem}.pt"
                runtime = OUTPUT / f"{stem}_runtime.json"
                artifacts = (result, prediction, checkpoint, runtime)
                if any(p.exists() for p in artifacts) and not all(p.exists() for p in artifacts):
                    raise RuntimeError(f"Partial training requires inspection: {stem}")
                if not result.exists():
                    require_idle_gpu()
                    options = []
                    for key, value in plan["training"].items():
                        if isinstance(value, bool):
                            if value:
                                options.append(f"--{key.replace('_', '-')}")
                        else:
                            options.extend([f"--{key.replace('_', '-')}", value])
                    execute("bib_train_worker.py", [
                        "--runtime-output", runtime, "--split", stem.rsplit("_seed", 1)[0],
                        "--run-tag", f"seed{job['seed']}", "--seed", job["seed"],
                        "--data-table", table, "--manifest-file", manifest,
                        "--dataset-label", "kiba_scaffold", "--target-column", DATASETS["kiba"]["label"],
                        "--token-dir", DATASETS["kiba"]["tokens"], "--device", "cuda",
                        "--save-predictions", *options], stem, "train")
                trained = json.loads(result.read_text())
                if (trained["input_sha256"] != expected or trained["seed"] != job["seed"]
                        or trained["device"] != "cuda"
                        or any(trained.get(k if k != "epochs" else "epochs_requested") != v
                               for k, v in plan["training"].items())):
                    raise RuntimeError(f"Training configuration mismatch: {stem}")
                ablation = PROJECT / f"results/{stem}_ablation.json"
                if not ablation.exists():
                    execute("evaluate_affinity_ablation.py", [
                        "--deep", prediction, "--prior", prior, "--data-table", table,
                        "--manifest-file", manifest, "--target-column", DATASETS["kiba"]["label"],
                        "--output-stem", stem], stem, "ablation")
                metrics = json.loads(ablation.read_text())
                if metrics["input_sha256"] != {"deep": sha256(prediction), "prior": sha256(prior),
                                               "table": expected["data_table"], "manifest": expected["manifest"]}:
                    raise RuntimeError("Scaffold ablation inputs changed")
                status["completed"].append({**job, "result": record(ablation),
                                            "test": {m: metrics["test"][m] for m in MODELS}})
                save()
                print(f"Completed {len(status['completed'])}/5: {stem}", flush=True)
            else:
                status["state"] = "completed"
            status.pop("current_job", None)
            status["stage"] = "finished"
        except BaseException as error:
            status["state"] = "failed"
            status["error"] = repr(error)
            raise
        finally:
            save()


if __name__ == "__main__":
    main()
