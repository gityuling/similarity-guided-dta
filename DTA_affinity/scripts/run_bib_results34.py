#!/usr/bin/env python3
"""Finish scaffold summaries, then run the cellular case with one GPU job at a time."""

import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from bib_common import PROJECT, ROOT, record, sha256, write_once


OUT = PROJECT / "results/bib_results34_20261018"
CASE = PROJECT / "reports/bib_cellular_case_external_20261018"
SCAFFOLD = PROJECT / "results/bib_scaffold_20261018"


def now():
    return datetime.now(timezone.utc).isoformat()


def require_idle_gpu():
    result = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
                            capture_output=True, text=True, check=True)
    active = [int(p) for p in result.stdout.splitlines() if p.strip().isdigit()]
    if active:
        raise RuntimeError(f"Another CUDA process is active: {active}")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = [CASE / n for n in ("plan.json", "coverage_audit.json", "query_table.csv", "inference_inputs.json",
                                 "functional_scores.csv", "functional_audit.json", "inference_smoke_check.json")]
    inputs += [PROJECT / "scripts" / n for n in ("run_bib_results34.py", "summarize_bib_scaffold.py",
               "build_kiba_benchmark_tokens.py", "build_bindingdb_articles_kd_tokens.py",
               "predict_bib_cellular_affinity.py", "evaluate_bib_cellular_case.py", "bib_common.py")]
    encoder_audit_path = PROJECT / "data/processed/kiba_dtiam_protocol/tokens/build_audit.json"
    inputs.append(encoder_audit_path)
    encoder_audit = json.loads(encoder_audit_path.read_text())
    for name, expected in encoder_audit["model_sha256"].items():
        if sha256(Path(name)) != expected:
            raise RuntimeError("Encoder weights differ from the KIBA feature provenance")
    frozen = [record(p) for p in inputs]
    write_once(OUT / "queue.json", json.dumps({"inputs": frozen, "gpu_jobs_at_once": 1,
                                              "dependency": str(SCAFFOLD / "status.json")}, indent=2) + "\n")
    status = {"state": "waiting", "stage": "scaffold_five_fold", "pid": os.getpid(), "started_utc": now(),
              "completed_stages": [], "gpu_jobs_at_once": 1}

    def save():
        status["updated_utc"] = now()
        temp = OUT / "status.tmp"
        temp.write_text(json.dumps(status, indent=2) + "\n")
        temp.replace(OUT / "status.json")

    def execute(stage, script, options=()):
        for item in frozen:
            if sha256(ROOT / item["path"]) != item["sha256"]:
                raise RuntimeError("Queued input or code changed")
        status.update(state="running", stage=stage)
        save()
        command = [sys.executable, "-u", str(PROJECT / "scripts" / script), *map(str, options)]
        started = time.perf_counter()
        print(stage, flush=True)
        with (OUT / f"{stage}.log").open("a") as log:
            log.write(json.dumps({"started_utc": now(), "command": command}) + "\n")
            log.flush()
            subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, check=True)
        status["completed_stages"].append({"stage": stage, "wall_seconds": time.perf_counter() - started})
        save()

    save()
    try:
        with (OUT / "queue.lock").open("a") as own_lock:
            fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            while True:
                scaffold = json.loads((SCAFFOLD / "status.json").read_text())
                if scaffold["state"] == "completed" and len(scaffold["completed"]) == 5:
                    break
                if scaffold["state"] == "failed":
                    raise RuntimeError("Scaffold queue failed; dependent GPU work stopped")
                os.kill(scaffold["pid"], 0)
                completed = len(scaffold["completed"])
                if status.get("scaffold_completed") != completed:
                    status["scaffold_completed"] = completed
                    save()
                    print(f"Waiting for scaffold queue: {completed}/5 complete", flush=True)
                time.sleep(15)
            with (PROJECT / "results/bib_validation_20261017/queue.lock").open("a") as gpu_lock:
                # The dependency writes completion just before it releases this shared lock.
                fcntl.flock(gpu_lock, fcntl.LOCK_EX)
                require_idle_gpu()
                execute("scaffold_summary", "summarize_bib_scaffold.py")
                memory = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
                if memory["MemAvailable"] < 12 * 1024 * 1024:
                    raise RuntimeError("Less than 12 GiB host RAM available for the Uni-Mol2 encoder; case inference stopped")
                execute("case_features", "build_kiba_benchmark_tokens.py", [
                    "--data-table", CASE / "query_table.csv",
                    "--cache-dir", PROJECT / "data/processed/bib_cellular_case_external_20261018/tokens",
                    "--reuse-cache", PROJECT / "data/processed/kiba_dtiam_protocol/tokens",
                    "--drug-batch-size", "1", "--protein-batch-size", "1"])
                require_idle_gpu()
                execute("case_affinity", "predict_bib_cellular_affinity.py")
                require_idle_gpu()
                execute("case_evaluation", "evaluate_bib_cellular_case.py")
            status.update(state="completed", stage="computations_complete", finished_utc=now(),
                          remaining="Integrate completed Results 3/4 figures, tables and prose; inspect the compiled manuscript")
    except BaseException as error:
        status.update(state="failed", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
