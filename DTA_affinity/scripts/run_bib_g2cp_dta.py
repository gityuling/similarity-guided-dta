#!/usr/bin/env python3
"""Run the fixed Results 5 jobs serially, with persistent progress and GPU exclusion."""

import argparse
import fcntl
import json
import os
import subprocess
import time
from pathlib import Path

from bib_g2cp_common import CONFIG, DATA, KIBA, OUT, PROJECT, ROOT, RUN, SOURCE, TOKENS
from bib_g2cp_common import check_records, now, plan, record, save_json, sha256, verify_data
from predict_bib_cellular_affinity import paths
from run_bib_results34 import require_idle_gpu

TRAIN_PYTHON = "/home/lsy/miniconda3/envs/gears_env2/bin/python"
DTA_PYTHON = "/home/lsy/miniconda3/envs/psichic_fp/bin/python"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    options = parser.parse_args()
    cfg = plan()
    RUN.mkdir(parents=True, exist_ok=True)
    status_path = RUN / ("smoke_status.json" if options.smoke else "status.json")
    state = {"state": "starting", "pid": os.getpid(), "started_utc": now(), "gpu_jobs_at_once": 1,
             "completed_jobs": [], "smoke": options.smoke}
    previous = json.loads(status_path.read_text()) if status_path.exists() else {}
    completed = {job["name"]: job for job in previous.get("completed_jobs", [])}
    state["completed_jobs"] = list(completed.values())
    names = ["bib_g2cp_common.py", "prepare_bib_g2cp_dta.py", "train_bib_g2cp.py", "predict_bib_g2cp_dta.py",
             "evaluate_bib_g2cp_dta.py", "verify_bib_g2cp_dta.py", "run_bib_g2cp_dta.py", "bib_common.py",
             "predict_bib_cellular_affinity.py", "train_nexusdti_regressor.py", "run_bib_results34.py",
             "build_kiba_benchmark_tokens.py", "build_bindingdb_articles_kd_tokens.py"]
    sources = [CONFIG, OUT / "data_audit.json", *(PROJECT / "scripts" / n for n in names),
               *(SOURCE / "backbone_model/GEARS/gears" / n for n in ("model.py", "utils.py"))]
    encoder_audit = KIBA / "tokens/build_audit.json"
    sources.append(encoder_audit)
    for path, expected in json.loads(encoder_audit.read_text())["model_sha256"].items():
        if sha256(Path(path)) != expected:
            raise RuntimeError("DTA encoder weights differ from the original KIBA feature cache")
    sources.extend(path for fold in range(5) for path in paths(fold).values())
    frozen = [record(path) for path in sources]
    if not options.smoke:
        save_json(RUN / "code_freeze.json", frozen, immutable=True)
    env = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2",
           "NUMEXPR_NUM_THREADS": "2", "TOKENIZERS_PARALLELISM": "false", "WANDB_MODE": "disabled"}

    def execute(name, command, gpu=False):
        if name in completed:
            print(f"Already completed: {name}", flush=True)
            return
        check_records(frozen)
        if gpu:
            require_idle_gpu()
            available = next(int(line.split()[1]) for line in Path("/proc/meminfo").read_text().splitlines() if line.startswith("MemAvailable:"))
            if available < 10 * 1024 * 1024:
                raise RuntimeError("Less than 10 GiB available system RAM")
        state.update(state="running", stage=name, updated_utc=now())
        save_json(status_path, state)
        began = time.perf_counter()
        with (RUN / f"{name}.log").open("a") as stream:
            stream.write(json.dumps({"started_utc": now(), "command": command}) + "\n")
            stream.flush()
            process = subprocess.Popen(command, cwd=PROJECT, stdout=stream, stderr=subprocess.STDOUT, env=env)
            state["child_pid"] = process.pid
            save_json(status_path, state)
            code = process.wait()
            if code:
                raise RuntimeError(f"{name} exited with {code}; see {RUN / (name+'.log')}")
        job = {"name": name, "seconds": time.perf_counter()-began, "finished_utc": now(),
               "log": record(RUN / f"{name}.log")}
        state["completed_jobs"].append(job)
        completed[name] = job
        state.pop("child_pid", None)
        save_json(status_path, state)

    lock_path = PROJECT / "results/bib_validation_20261017/queue.lock"
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            verify_data()
            execute("verify_prepared", [TRAIN_PYTHON, "-u", "scripts/verify_bib_g2cp_dta.py"])
            if options.smoke:
                execute("smoke_A375", [TRAIN_PYTHON, "-u", "scripts/train_bib_g2cp.py", "--cell", "A375", "--seed", str(cfg["training_seeds"][0]), "--smoke"], True)
            else:
                smoke = json.loads((RUN / "smoke_status.json").read_text())
                if smoke.get("state") != "completed":
                    raise RuntimeError("Complete the isolated smoke run before formal training")
                tokens = [DTA_PYTHON, "-u", "scripts/build_kiba_benchmark_tokens.py", "--data-table", str(OUT / "query_table.csv"),
                          "--cache-dir", str(TOKENS), "--reuse-cache", str(KIBA / "tokens"),
                          "--reuse-cache", str(PROJECT / "data/processed/bib_nanobret_20261019/tokens"),
                          "--reuse-cache", str(PROJECT / "data/processed/bib_cellular_case_external_20261018/tokens"),
                          "--drug-batch-size", "1", "--protein-batch-size", "1"]
                execute("features", tokens, True)
                execute("dta_inference", [DTA_PYTHON, "-u", "scripts/predict_bib_g2cp_dta.py"], True)
                for seed in cfg["training_seeds"]:
                    for cell in cfg["cells"]:
                        execute(f"train_{cell}_seed{seed}", [TRAIN_PYTHON, "-u", "scripts/train_bib_g2cp.py", "--cell", cell, "--seed", str(seed)], True)
                execute("evaluation", [TRAIN_PYTHON, "-u", "scripts/evaluate_bib_g2cp_dta.py"])
                execute("verify_final", [TRAIN_PYTHON, "-u", "scripts/verify_bib_g2cp_dta.py", "--final"])
            state.update(state="completed", updated_utc=now())
        except BaseException as error:
            state.update(state="failed", error=repr(error), updated_utc=now())
            raise
        finally:
            save_json(status_path, state)


if __name__ == "__main__":
    main()
