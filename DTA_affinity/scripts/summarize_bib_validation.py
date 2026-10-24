#!/usr/bin/env python3
"""Describe completed queue jobs without treating fold-seed scores as independent."""

import importlib.metadata
import json
import platform
import statistics
import sys
from collections import defaultdict
from datetime import datetime, timezone

from bib_common import PROJECT, record, verify_freeze

OUTPUT = PROJECT / "results/bib_validation_20261017"
MODELS = ("deep", "neighbor_prior", "fixed_blend", "similarity_gate")
METRICS = ("mse", "pearson", "ci")


def main():
    verify_freeze(include_features=False)
    status = json.loads((OUTPUT / "status.json").read_text())
    queue = json.loads((OUTPUT / "queue.json").read_text())
    groups, costs, inputs = defaultdict(dict), [], []
    original = json.loads((PROJECT / "reports/bib_fusion_20261017/analysis.json").read_text())
    for dataset, data in original["datasets"].items():
        groups[(dataset, "drug_id", 20261012)] = {f["fold"]: f["metrics"] for f in data["folds"]}
    for job in status["completed"]:
        path = PROJECT.parent / job["result"]["path"]
        if record(path) != job["result"]:
            raise RuntimeError("Completed queue artifact changed")
        data = json.loads(path.read_text())
        groups[(job["dataset"], job["protocol"], job["seed"])][job["fold"]] = data["test"]
        inputs.append(record(path))
        runtime = OUTPUT / f"{job['id']}_runtime.json"
        measured = json.loads(runtime.read_text())
        costs.append({"job": job["id"], **measured})
    summaries = []
    for (dataset, protocol, seed), folds in sorted(groups.items()):
        complete = set(folds) == set(range(5))
        entry = {"dataset": dataset, "protocol": protocol, "seed": seed, "folds_complete": sorted(folds),
                 "five_fold_complete": complete, "metrics": {}}
        if complete:
            for model in MODELS:
                entry["metrics"][model] = {metric: {"fold_mean": statistics.mean(folds[f][model][metric] for f in range(5)),
                                                     "fold_sample_sd": statistics.stdev(folds[f][model][metric] for f in range(5)),
                                                     "per_fold": [folds[f][model][metric] for f in range(5)]}
                                            for metric in METRICS}
            entry["gate_lower_mse_folds"] = {m: sum(folds[f]["similarity_gate"]["mse"] < folds[f][m]["mse"] for f in range(5))
                                              for m in MODELS if m != "similarity_gate"}
        summaries.append(entry)
    repeated = []
    for dataset in ("davis", "kiba"):
        seeds = [e for e in summaries if e["dataset"] == dataset and e["protocol"] == "drug_id" and e["five_fold_complete"]]
        if {e["seed"] for e in seeds} != {20261012, 20261017, 20261018}:
            continue
        for model in MODELS:
            for metric in METRICS:
                means = [e["metrics"][model][metric]["fold_mean"] for e in seeds]
                fold_means = [statistics.mean(e["metrics"][model][metric]["per_fold"][f] for e in seeds) for f in range(5)]
                repeated.append({"dataset": dataset, "model": model, "metric": metric,
                                 "mean_over_seeds_and_folds": statistics.mean(means),
                                 "sample_sd_of_three_seed_means": statistics.stdev(means),
                                 "sample_sd_of_five_fold_means_over_seeds": statistics.stdev(fold_means),
                                 "seed_means": {str(e["seed"]): v for e, v in zip(seeds, means)},
                                 "seed_averaged_fold_means": fold_means})
    result = {"updated_utc": datetime.now(timezone.utc).isoformat(), "queue_state": status["state"],
              "new_jobs_completed": len(status["completed"]), "new_jobs_total": len(queue["jobs"]),
              "current_job": status.get("current_job"), "per_seed": summaries, "three_seed_summary": repeated,
              "uncertainty_note": "Seed SD measures variation of three five-fold means; fold SD measures five fold means averaged over seeds. No independence or standard error is inferred from 15 correlated scores.",
              "training_costs": costs, "inputs": inputs}
    (OUTPUT / "progress_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    environment = {"python": sys.version, "executable": sys.executable, "platform": platform.platform(),
                   "packages": dict(sorted((d.metadata["Name"], d.version) for d in importlib.metadata.distributions() if d.metadata.get("Name"))),
                   "cost_note": "Worker runtime includes trainer setup/evaluation/output and peak CUDA allocation/reservation. Historical feature extraction wall time was not instrumented and cannot be reconstructed as a precise benchmark."}
    (OUTPUT / "environment.json").write_text(json.dumps(environment, indent=2) + "\n")
    print(json.dumps({"completed": len(status["completed"]), "total": len(queue["jobs"]), "state": status["state"],
                      "complete_three_seed_datasets": sorted({r["dataset"] for r in repeated})}, indent=2))


if __name__ == "__main__":
    main()
