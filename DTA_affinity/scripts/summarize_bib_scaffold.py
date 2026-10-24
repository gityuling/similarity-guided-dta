#!/usr/bin/env python3
"""Check and summarize completed scaffold folds without repeating training."""

import json
import statistics

from bib_common import PROJECT, ROOT, record, sha256


def main():
    out = PROJECT / "results/bib_scaffold_20261018"
    status = json.loads((out / "status.json").read_text())
    queue = json.loads((out / "queue.json").read_text())
    for item in [queue["plan"], queue["audit"], *queue["code"]]:
        if sha256(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError("Scaffold queue input changed")
    entries = sorted(status["completed"], key=lambda e: e["fold"])
    folds = [e["fold"] for e in entries]
    if len(folds) != len(set(folds)):
        raise RuntimeError("Duplicate completed fold")
    metrics = {m: {k: [] for k in ("mse", "pearson", "ci")}
               for m in ("deep", "neighbor_prior", "fixed_blend", "similarity_gate")}
    for entry in entries:
        item = entry["result"]
        path = ROOT / item["path"]
        if record(path) != item:
            raise RuntimeError("Completed scaffold result changed")
        result = json.loads(path.read_text())
        for model in metrics:
            for metric in metrics[model]:
                value = result["test"][model][metric]
                if value != entry["test"][model][metric]:
                    raise RuntimeError("Scaffold status and result disagree")
                metrics[model][metric].append(value)
    complete = status["state"] == "completed" and folds == list(range(5))
    summary = {"queue_state": status["state"], "folds_complete": folds, "five_fold_complete": complete,
               "metric_values": metrics, "inputs": [e["result"] for e in entries],
               "gate_lower_mse_folds": {m: sum(a < b for a, b in zip(metrics["similarity_gate"]["mse"], metrics[m]["mse"]))
                                        for m in ("deep", "neighbor_prior", "fixed_blend")},
               "five_fold_metrics": {m: {k: {"mean": statistics.mean(v), "sd_across_folds": statistics.stdev(v), "per_fold": v}
                                         for k, v in values.items()} for m, values in metrics.items()} if complete else None}
    temp = out / "progress_summary.tmp"
    temp.write_text(json.dumps(summary, indent=2) + "\n")
    temp.replace(out / "progress_summary.json")
    print(json.dumps({"complete": complete, "folds": folds, "gate_lower_mse_folds": summary["gate_lower_mse_folds"]}), flush=True)


if __name__ == "__main__":
    main()
