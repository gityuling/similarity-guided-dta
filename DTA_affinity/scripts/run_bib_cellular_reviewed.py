#!/usr/bin/env python3
"""Run the authorized revised case sequentially with the shared GPU lock."""

import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from bib_common import PROJECT, ROOT, record, sha256, write_once
from run_bib_results34 import require_idle_gpu


OUT = PROJECT / "results/bib_cellular_reviewed_20261018"
ORIGINAL = PROJECT / "reports/bib_cellular_case_external_20261018"
CASE = PROJECT / "reports/bib_cellular_case_reviewed_20261018"
GEARS_PYTHON = Path("/home/lsy/miniconda3/envs/gears_env2/bin/python")


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    files = [PROJECT / "scripts" / n for n in ("run_bib_cellular_reviewed.py", "review_bib_cellular_profiles.py",
             "evaluate_bib_cellular_reviewed.py", "predict_bib_cellular_affinity.py", "build_kiba_benchmark_tokens.py",
             "build_bindingdb_articles_kd_tokens.py", "summarize_bib_scaffold.py", "bib_common.py", "run_bib_results34.py",
             "build_bib_cellular_profiles.py", "audit_bib_cellular_case.py", "fetch_bib_case_sources.py",
             "evaluate_bib_cellular_case.py")]
    files += [ORIGINAL / n for n in ("plan.json", "coverage_audit.json", "query_table.csv", "inference_inputs.json", "inference_smoke_check.json")]
    frozen = [record(p) for p in files]
    write_once(OUT / "queue.json", json.dumps({"files": frozen, "gpu_jobs_at_once": 1,
               "authorization": "User explicitly resumed the agreed DTA/LINCS/CTRP case after the pause; preprocessing review precedes endpoint evaluation."}, indent=2) + "\n")
    state = {"state": "starting", "pid": os.getpid(), "started_utc": now(), "gpu_jobs_at_once": 1, "completed_stages": []}

    def save():
        state["updated_utc"] = now()
        temp = OUT / "status.tmp"
        temp.write_text(json.dumps(state, indent=2) + "\n")
        temp.replace(OUT / "status.json")

    def execute(name, script, options=(), python=None):
        for item in frozen:
            if sha256(ROOT / item["path"]) != item["sha256"]:
                raise RuntimeError(f"Queued input changed: {item['path']}")
        state.update(state="running", stage=name)
        save()
        command = [str(python or sys.executable), "-u", str(PROJECT / "scripts" / script), *map(str, options)]
        started = time.perf_counter()
        print(name, flush=True)
        with (OUT / f"{name}.log").open("a") as log:
            log.write(json.dumps({"started_utc": now(), "command": command}) + "\n")
            log.flush()
            subprocess.run(command, cwd=PROJECT, stdout=log, stderr=subprocess.STDOUT, check=True)
        state["completed_stages"].append({"stage": name, "wall_seconds": time.perf_counter() - started})
        save()

    with (OUT / "queue.lock").open("a") as own_lock:
        fcntl.flock(own_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        save()
        try:
            scaffold = json.loads((PROJECT / "results/bib_scaffold_20261018/status.json").read_text())
            if scaffold["state"] != "completed" or len(scaffold["completed"]) != 5:
                raise RuntimeError("Scaffold five-fold dependency is incomplete")
            with (PROJECT / "results/bib_validation_20261017/queue.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                require_idle_gpu()
                execute("scaffold_summary", "summarize_bib_scaffold.py")
                execute("signature_review", "review_bib_cellular_profiles.py", python=GEARS_PYTHON)
                encoder = PROJECT / "data/processed/kiba_dtiam_protocol/tokens/build_audit.json"
                for path, expected in json.loads(encoder.read_text())["model_sha256"].items():
                    if sha256(Path(path)) != expected:
                        raise RuntimeError("Encoder weights changed")
                memory = {line.split(':')[0]: int(line.split()[1]) for line in Path('/proc/meminfo').read_text().splitlines()}
                if memory["MemAvailable"] < 12 * 1024 * 1024:
                    raise RuntimeError("Less than 12 GiB host RAM available for feature extraction")
                require_idle_gpu()
                execute("case_features", "build_kiba_benchmark_tokens.py", [
                    "--data-table", ORIGINAL / "query_table.csv",
                    "--cache-dir", PROJECT / "data/processed/bib_cellular_case_external_20261018/tokens",
                    "--reuse-cache", PROJECT / "data/processed/kiba_dtiam_protocol/tokens",
                    "--drug-batch-size", "1", "--protein-batch-size", "1"])
                require_idle_gpu()
                execute("case_affinity", "predict_bib_cellular_affinity.py")
                require_idle_gpu()
                execute("case_evaluation", "evaluate_bib_cellular_reviewed.py")
            state.update(state="completed", stage="computations_complete", finished_utc=now(),
                         analysis=str(CASE / "analysis.json"), remaining="Review results, update TODO and prepare manuscript figures/prose.")
        except BaseException as error:
            state.update(state="failed", error=repr(error))
            raise
        finally:
            save()


if __name__ == "__main__":
    main()
