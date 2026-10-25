#!/usr/bin/env python3
"""Rebuild Davis reference pools by experimental target ID using saved model outputs.

Original sequence-pool results remain intact. The dated report is the active
manuscript source and records the original evidence and the correction together.
"""
from __future__ import annotations

import copy
import json
import statistics
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from analyze_bib_fusion import MODELS, bootstrap, canonical, fit_controls, metric
from bib_common import (PROJECT, ROOT, FREEZE, baseline_paths, check_predictions,
                        manifest_path, record, sha256, table_path)
from train_nexusdti_regressor import regression_metrics

OUTPUT = PROJECT / "reports/davis_target_id_20261025"
LEGACY = PROJECT / "reports/bib_fusion_20261017"
SEEDS = (20261012, 20261017, 20261018)
REPEATED_MODELS = ("deep", "neighbor_prior", "fixed_blend", "similarity_gate")


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def execute(script, arguments, log):
    with (OUTPUT / log).open("w") as stream:
        subprocess.run([sys.executable, str(PROJECT / "scripts" / script), *map(str, arguments)],
                       cwd=PROJECT, stdout=stream, stderr=subprocess.STDOUT, check=True)


def verify_original_evidence():
    frozen = json.loads((FREEZE / "protocol.json").read_text())
    replaced = "DTA_affinity/scripts/evaluate_drug_similarity_prior.py"
    for item in [*frozen["files"], frozen["feature_inventory"], frozen["current_source_snapshot"]]:
        path = (OUTPUT / "legacy_evaluate_drug_similarity_prior.py"
                if item["path"] == replaced else ROOT / item["path"])
        if sha256(path) != item["sha256"]:
            raise RuntimeError(f"Original evidence changed: {path}")


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    verify_original_evidence()
    original = json.loads((LEGACY / "analysis.json").read_text())
    repeated = json.loads((PROJECT / "results/bib_validation_20261017/progress_summary.json").read_text())
    config = json.loads((PROJECT / "configs/bib_analysis_20261017.json").read_text())
    table = pd.read_csv(table_path("davis"))
    table["chemical_identity"] = table.Canonical_SMILES.map(canonical)
    before_neural = {}
    results, frames, inputs, fits, subgroup_rows, ranking, controls, curves = {}, [], [], [], [], [], [], []
    for fold in range(5):
        paths = baseline_paths("davis", fold)
        manifest = manifest_path("davis", fold)
        prior_stem = f"bib_davis_target_id_f{fold}_frozen_prior"
        prior = PROJECT / f"results/predictions/{prior_stem}_predictions.csv"
        execute("evaluate_target_id_similarity_prior.py", ["--data-table", table_path("davis"),
                "--manifest-file", manifest, "--target-column", "pKd_floor",
                "--reference-key", "protein_id", "--output-stem", prior_stem], f"prior_f{fold}.log")
        for seed in SEEDS:
            deep = (paths["deep_predictions"] if seed == SEEDS[0] else PROJECT /
                    f"results/predictions/nexusdti_bib_davis_drug_id_f{fold}_seed{seed}_predictions.csv")
            before_neural[str(deep)] = sha256(deep)
            stem = f"bib_davis_target_id_f{fold}_seed{seed}"
            execute("evaluate_affinity_ablation.py", ["--deep", deep, "--prior", prior,
                    "--data-table", table_path("davis"), "--manifest-file", manifest,
                    "--target-column", "pKd_floor", "--output-stem", stem], f"fusion_f{fold}_seed{seed}.log")
            path = PROJECT / f"results/{stem}_ablation.json"
            result = json.loads(path.read_text())
            results[seed, fold] = (result, path)
            if seed != SEEDS[0]:
                continue
            check_predictions(deep, table, pd.read_csv(manifest), "pKd_floor")
            check_predictions(prior, table, pd.read_csv(manifest), "pKd_floor")
            inputs.extend([record(deep), record(prior)])
            frame = pd.read_csv(PROJECT / f"results/predictions/{stem}_ablation_predictions.csv")
            frame.rename(columns={"affinity_true": "truth"}, inplace=True)
            fitted = fit_controls(frame)
            saved = pd.read_csv(PROJECT / f"results/predictions/{stem}_ablation_predictions.csv")
            delta = float(np.max(np.abs(frame.prediction_similarity_gate - saved.prediction_similarity_gate)))
            if delta > 1e-7:
                raise RuntimeError("Independent fusion fits disagree")
            frame = frame.merge(table[["row_id", "drug_idx", "drug_id", "protein_id", "chemical_identity", "pKd_floor"]],
                                on="row_id", validate="one_to_one")
            frame["fold"] = fold
            test = frame.loc[frame.split.eq("test")].copy()
            metrics = {m: regression_metrics(test.truth.to_numpy(), test[f"prediction_{m}"].to_numpy()) for m in MODELS}
            fits.append({"fold": fold, "fit": fitted, "metrics": metrics, "saved_gate_max_abs_difference": delta})
            frames.append(test)
            controls.append({"dataset": "davis", "fold": fold, "fixed_deep_weight": fitted["fixed_deep_weight"],
                             **fitted["linear_stacking"], **{f"gate_{k}": v for k, v in fitted["gate"].items()}})
        print(f"Completed Davis target-ID references and three fusion fits: fold {fold}", flush=True)
    oof = pd.concat(frames, ignore_index=True).sort_values("row_id")
    if not oof.row_id.is_unique or set(oof.row_id) != set(table.row_id):
        raise RuntimeError("Incomplete OOF coverage")
    summary = {m: {k: {"mean": float(np.mean(v)), "sd_across_folds": float(np.std(v, ddof=1)), "per_fold": v}
                   for k in ("mse", "ci", "pearson") for v in [[f["metrics"][m][k] for f in fits]]} for m in MODELS}
    masks = [("all", np.ones(len(oof), dtype=bool))]
    for lo, hi in ((0, .3), (.3, .5), (.5, .7), (.7, 1.)):
        mask = oof.max_train_drug_tanimoto.ge(lo) & (oof.max_train_drug_tanimoto.lt(hi) if hi < 1 else oof.max_train_drug_tanimoto.le(1))
        masks.append((f"tanimoto_{lo}_{hi}", mask))
    masks.extend([("pKd_floor_5", oof.pKd_floor.le(5 + 1e-8)), ("pKd_above_5", oof.pKd_floor.gt(5 + 1e-8))])
    subgroups = []
    for label, mask in masks:
        subset = oof.loc[mask]
        entry = {"subgroup": label, "pairs": len(subset), "drug_ids": int(subset.drug_idx.nunique()),
                 "chemical_identities": int(subset.chemical_identity.nunique()),
                 "folds_represented": sorted(map(int, subset.fold.unique())),
                 "deep_weight_mean": float(subset.deep_weight.mean()),
                 "deep_weight_range": [float(subset.deep_weight.min()), float(subset.deep_weight.max())],
                 "pooled_metrics": {m: metric(subset.truth, subset[f"prediction_{m}"]) for m in MODELS}}
        subgroups.append(entry)
        for m, v in entry["pooled_metrics"].items():
            subgroup_rows.append({"dataset": "davis", "subgroup": label, "model": m, "pairs": len(subset),
                                 "drug_ids": entry["drug_ids"], "chemical_identities": entry["chemical_identities"],
                                 "mean_deep_weight": entry["deep_weight_mean"], **v})
    for target, subset in oof.groupby("protein_id", sort=True):
        if subset.chemical_identity.nunique() < 20 or subset.truth.nunique() < 2:
            continue
        for m in MODELS:
            pred = subset[f"prediction_{m}"]
            ranking.append({"dataset": "davis", "target": str(target), "model": m, "pairs": len(subset),
                            "chemical_identities": int(subset.chemical_identity.nunique()),
                            "spearman": float(spearmanr(subset.truth, pred).statistic), **metric(subset.truth, pred)})
    oof_path = OUTPUT / "davis_oof.csv"
    oof.to_csv(oof_path, index=False)
    analysis = copy.deepcopy(original)
    analysis["code"] = record(Path(__file__))
    analysis["reference_pool"] = "Experimental target ID; KIBA IDs and sequences have a one-to-one mapping"
    analysis["scope"] = "Davis target-ID reference-pool correction; neural predictions and all KIBA analyses retained"
    analysis["inputs"] = inputs + [r for r in original["inputs"] if "kiba" in r["path"]]
    analysis["datasets"]["davis"] = {"summary": summary, "folds": fits, "subgroups": subgroups,
        "cluster_bootstrap": bootstrap(oof, config["statistics"]["bootstrap_repeats"], config["statistics"]["bootstrap_seed"]),
        "oof": record(oof_path)}
    analysis["ranking_macro"] = [r for r in original["ranking_macro"] if r["dataset"] == "kiba"]
    for model, group in pd.DataFrame(ranking).groupby("model"):
        analysis["ranking_macro"].append({"dataset": "davis", "model": model, "targets": len(group),
             "spearman_macro": float(group.spearman.mean()), "pearson_macro": float(group.pearson.mean()), "mse_macro": float(group.mse.mean())})
    for name, rows in (("per_target_ranking.csv", ranking), ("fusion_parameters.csv", controls),
                       ("similarity_subgroups.csv", subgroup_rows)):
        old = pd.read_csv(LEGACY / name)
        pd.concat([pd.DataFrame(rows), old.loc[old.dataset.eq("kiba")]], ignore_index=True).to_csv(OUTPUT / name, index=False)
    dump(OUTPUT / "analysis.json", analysis)
    for entry in repeated["per_seed"]:
        if entry["dataset"] != "davis":
            continue
        seed = entry["seed"]
        for m in REPEATED_MODELS:
            for metric_name in ("mse", "pearson", "ci"):
                v = [results[seed, f][0]["test"][m][metric_name] for f in range(5)]
                entry["metrics"][m][metric_name] = {"fold_mean": statistics.mean(v), "fold_sample_sd": statistics.stdev(v), "per_fold": v}
        entry["gate_lower_mse_folds"] = {m: sum(results[seed, f][0]["test"]["similarity_gate"]["mse"] < results[seed, f][0]["test"][m]["mse"] for f in range(5)) for m in REPEATED_MODELS[:-1]}
    for entry in repeated["three_seed_summary"]:
        if entry["dataset"] != "davis":
            continue
        m, k = entry["model"], entry["metric"]
        means = [statistics.mean(results[s, f][0]["test"][m][k] for f in range(5)) for s in SEEDS]
        fold_means = [statistics.mean(results[s, f][0]["test"][m][k] for s in SEEDS) for f in range(5)]
        entry.update(mean_over_seeds_and_folds=statistics.mean(means), sample_sd_of_three_seed_means=statistics.stdev(means),
                     sample_sd_of_five_fold_means_over_seeds=statistics.stdev(fold_means), seed_means=dict(zip(map(str, SEEDS), means)),
                     seed_averaged_fold_means=fold_means)
    repeated["inputs"] = [r for r in repeated["inputs"] if "bib_davis_" not in r["path"]] + [record(p) for _, p in results.values()]
    repeated["updated_utc"] = datetime.now(timezone.utc).isoformat()
    repeated["reference_pool_correction"] = "Davis protein_id; original neural training runtimes retained"
    dump(OUTPUT / "progress_summary.json", repeated)
    if any(sha256(Path(p)) != h for p, h in before_neural.items()):
        raise RuntimeError("Saved neural predictions changed")
    targets = table[["protein_id", "Sequence"]].drop_duplicates()
    sizes = targets.groupby("Sequence").protein_id.nunique()
    amendment = {"created_utc": datetime.now(timezone.utc).isoformat(), "change": "Davis neighbor pools: Sequence -> protein_id",
                 "target_ids": len(targets), "unique_sequences": len(sizes),
                 "shared_sequence_groups": int((sizes > 1).sum()), "target_ids_in_shared_groups": int(sizes[sizes > 1].sum()),
                 "selection": "Original validation k/gamma grid, sorting rule and fusion objectives; no test selection",
                 "neural_predictions_unchanged": [record(Path(p)) for p in before_neural],
                 "original_protocol": record(FREEZE / "protocol.json"), "original_analysis": record(LEGACY / "analysis.json"),
                 "original_repeated_summary": record(PROJECT / "results/bib_validation_20261017/progress_summary.json"),
                 "code": [record(Path(__file__)), record(PROJECT / "scripts/evaluate_target_id_similarity_prior.py"), record(PROJECT / "scripts/evaluate_affinity_ablation.py")],
                 "outputs": [record(OUTPUT / "analysis.json"), record(OUTPUT / "progress_summary.json")],
                 "reference_mse_before": {m: original["datasets"]["davis"]["summary"][m]["mse"]["mean"] for m in MODELS},
                 "reference_mse_after": {m: summary[m]["mse"]["mean"] for m in MODELS}}
    dump(OUTPUT / "amendment.json", amendment)
    print(json.dumps(amendment["reference_mse_after"], indent=2), flush=True)


if __name__ == "__main__":
    main()
