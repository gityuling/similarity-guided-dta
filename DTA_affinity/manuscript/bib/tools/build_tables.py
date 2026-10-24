#!/usr/bin/env python3
"""Build manuscript tables from completed, traceable local metric summaries."""

from __future__ import annotations

import hashlib
import csv
import json
import math
import statistics
from pathlib import Path


MANUSCRIPT = Path(__file__).resolve().parents[1]
PROJECT = MANUSCRIPT.parents[1]
SOURCES = {
    "davis": PROJECT / "reports/davis_supplement_20261016.json",
    "kiba": PROJECT / "results/kiba_drugcold_supplement_5fold.json",
    "published": PROJECT / "reports/dtiam_published_comparison_20261016.json",
    "kiba_status": PROJECT / "results/kiba_drugcold_supplement_status.json",
    "kiba_config": PROJECT / "configs/kiba_drugcold_20261016.json",
    "fusion": PROJECT / "reports/bib_fusion_20261017/analysis.json",
    "comparator_audit": PROJECT / "reports/bib_comparator_audit_20261017/audit.json",
    "analysis_config": PROJECT / "configs/bib_analysis_20261017.json",
    "repeated": PROJECT / "results/bib_validation_20261017/progress_summary.json",
    "identity_audit": PROJECT / "data/processed/kiba_identity_20261017/summary.json",
}
LABELS = {
    "deep": "Deep only",
    "neighbor_prior": "Neighbor only",
    "half_blend": "Equal-weight blend",
    "fixed_blend": "Constant blend",
    "linear_stacking": "Linear stacking",
    "similarity_gate": "Similarity gate",
}
REPEATED_MODELS = ("deep", "neighbor_prior", "fixed_blend", "similarity_gate")
SEEDS = (20261012, 20261017, 20261018)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_metric(value: dict) -> dict:
    values = value.get("per_fold", value.get("per_fold_in_source_order"))
    if len(values) != 5 or not all(math.isfinite(x) for x in values):
        raise ValueError("Expected five finite fold values")
    mean, sd = statistics.mean(values), statistics.stdev(values)
    stored_sd = value.get("sd_across_folds", value.get("sd_across_folds_ddof1"))
    if not math.isclose(mean, value["mean"], abs_tol=1e-12):
        raise ValueError("Stored fold mean disagrees with source values")
    if not math.isclose(sd, stored_sd, abs_tol=1e-12):
        raise ValueError("Stored fold SD disagrees with source values")
    return {"mean": mean, "sd": sd, "per_fold": values}


def cell(metric: dict) -> str:
    return f"${metric['mean']:.5f} \\pm {metric['sd']:.5f}$"


def checked_repeats(raw: dict) -> tuple[dict, dict]:
    if raw["queue_state"] != "completed" or raw["new_jobs_completed"] != raw["new_jobs_total"]:
        raise RuntimeError("Repeated-training queue is incomplete")
    for item in raw["inputs"]:
        if digest(PROJECT.parent / item["path"]) != item["sha256"]:
            raise RuntimeError("Repeated-training source changed")
    per_seed = {(e["dataset"], e["seed"]): e for e in raw["per_seed"]
                if e["protocol"] == "drug_id"}
    summary = {(e["dataset"], e["model"], e["metric"]): e
               for e in raw["three_seed_summary"]}
    for dataset in ("davis", "kiba"):
        for model in REPEATED_MODELS:
            for metric in ("mse", "pearson", "ci"):
                record = summary[dataset, model, metric]
                means, fold_values = [], []
                for seed in SEEDS:
                    entry = per_seed[dataset, seed]
                    if not entry["five_fold_complete"] or entry["folds_complete"] != list(range(5)):
                        raise RuntimeError("Expected all five folds for every seed")
                    values = entry["metrics"][model][metric]
                    checked = checked_metric({"per_fold": values["per_fold"],
                                              "mean": values["fold_mean"],
                                              "sd_across_folds": values["fold_sample_sd"]})
                    if not math.isclose(checked["mean"], record["seed_means"][str(seed)], abs_tol=1e-12):
                        raise RuntimeError("Seed summary disagrees with fold values")
                    means.append(checked["mean"])
                    fold_values.append(checked["per_fold"])
                fold_means = [statistics.mean(v[f] for v in fold_values) for f in range(5)]
                checks = {
                    "mean_over_seeds_and_folds": statistics.mean(means),
                    "sample_sd_of_three_seed_means": statistics.stdev(means),
                    "sample_sd_of_five_fold_means_over_seeds": statistics.stdev(fold_means),
                }
                for key, value in checks.items():
                    if not math.isclose(value, record[key], abs_tol=1e-12):
                        raise RuntimeError(f"Repeated-training aggregate disagrees: {key}")
                if any(not math.isclose(a, b, abs_tol=1e-12)
                       for a, b in zip(fold_means, record["seed_averaged_fold_means"])):
                    raise RuntimeError("Seed-averaged fold values disagree")
    return summary, per_seed


def checked_identity(raw: dict, audit: dict) -> tuple[dict, dict]:
    entries = [e for e in raw["per_seed"] if e["dataset"] == "kiba"
               and e["protocol"] == "identity" and e["seed"] == SEEDS[0]]
    if len(entries) != 1:
        raise RuntimeError("Expected one strict-identity KIBA summary")
    entry = entries[0]
    if not entry["five_fold_complete"] or entry["folds_complete"] != list(range(5)):
        raise RuntimeError("Strict-identity five-fold evaluation is incomplete")
    if audit["pairs_removed"] != 0 or not audit["every_test_row_once"]:
        raise RuntimeError("Strict-identity evaluation does not retain all pairs exactly once")
    if [f["fold"] for f in audit["folds"]] != list(range(5)):
        raise RuntimeError("Expected five ordered strict-identity audit records")
    for source in [audit["table"], *(f["manifest"] for f in audit["folds"])]:
        if digest(PROJECT.parent / source["path"]) != source["sha256"]:
            raise RuntimeError("Strict-identity audited input changed")
    models = {
        model: {metric: checked_metric({"per_fold": v["per_fold"],
                                      "mean": v["fold_mean"],
                                      "sd_across_folds": v["fold_sample_sd"]})
                for metric, v in entry["metrics"][model].items()}
        for model in REPEATED_MODELS
    }
    for fold in audit["folds"]:
        f = fold["fold"]
        if any(fold["canonical_overlap"].values()) or any(fold["unsupported_same_target_rows"].values()):
            raise RuntimeError("Strict-identity isolation or target support check failed")
        if sum(fold["rows"].values()) != audit["pairs"]:
            raise RuntimeError("Strict-identity fold pair count disagrees")
        path = PROJECT / f"results/bib_kiba_identity_f{f}_seed{SEEDS[0]}_ablation.json"
        original = json.loads(path.read_text())
        expected_hashes = {"table": audit["table"]["sha256"],
                           "manifest": fold["manifest"]["sha256"]}
        if any(original["input_sha256"][key] != value for key, value in expected_hashes.items()):
            raise RuntimeError("Strict-identity result and audit inputs disagree")
        if any(original["rows"][part] != fold["rows"][part] for part in ("val", "test")):
            raise RuntimeError("Strict-identity result and audit row counts disagree")
        for model in REPEATED_MODELS:
            for metric in ("mse", "pearson", "ci"):
                if not math.isclose(models[model][metric]["per_fold"][f],
                                    original["test"][model][metric], rel_tol=0, abs_tol=1e-12):
                    raise RuntimeError("Strict-identity summary disagrees with original fold result")
    if sum(f["rows"]["test"] for f in audit["folds"]) != audit["pairs"]:
        raise RuntimeError("Strict-identity total test count disagrees")
    wins = {}
    for model in REPEATED_MODELS[:-1]:
        wins[model] = sum(a < b for a, b in zip(models["similarity_gate"]["mse"]["per_fold"],
                                               models[model]["mse"]["per_fold"]))
        if wins[model] != entry["gate_lower_mse_folds"][model]:
            raise RuntimeError("Strict-identity fold direction count disagrees")
    return models, wins


def table_file(name: str, caption: str, label: str, columns: str,
               header: str, rows: list[str], note: str, compact: bool = False) -> None:
    text = [
        "% Generated by tools/build_tables.py; do not edit values manually.",
        r"\begin{table}[htbp]", r"\centering", r"\small",
        f"\\caption{{{caption}}}", f"\\label{{{label}}}",
        *([r"\footnotesize\setlength{\tabcolsep}{3pt}"] if compact else []),
        f"\\begin{{tabular}}{{{columns}}}", r"\toprule",
        header + r" \\", r"\midrule", *rows,
        r"\bottomrule", r"\end{tabular}", r"\par\medskip",
        r"\begin{minipage}{0.98\linewidth}\footnotesize " + note,
        r"\end{minipage}", r"\end{table}", "",
    ]
    (MANUSCRIPT / "tables" / name).write_text("\n".join(text))


def fold_table(rows: list[str]) -> None:
    header = r"Dataset & Seed & Fold & Predictor & MSE & Pearson & C-index \\"
    text = [
        "% Generated by tools/build_tables.py; do not edit values manually.",
        r"\begingroup\small\setlength{\tabcolsep}{4pt}",
        r"\begin{longtable}{lrrlrrr}",
        r"\caption{Complete local drug-ID-cold fold and training-seed results.}\label{tab:folds}\\",
        r"\toprule", header, r"\midrule", r"\endfirsthead",
        r"\multicolumn{7}{l}{\tablename\ \thetable{} (continued)}\\",
        r"\toprule", header, r"\midrule", r"\endhead",
        r"\midrule\multicolumn{7}{r}{Continued on next page}\\", r"\endfoot",
        r"\bottomrule", r"\endlastfoot", *rows, r"\end{longtable}",
        r"\noindent\footnotesize Fold indices 0--4 are displayed as 1--5. All three seeds use the same partitions and the same cached neighbor priors. Four matched predictors are available for every seed; equal blending and linear stacking are additionally listed for reference seed 20261012. C-index denotes concordance index. These folds are not paired by row order with published baseline scores.",
        r"\par\endgroup", "",
    ]
    (MANUSCRIPT / "tables/folds.tex").write_text("\n".join(text))


def main() -> None:
    raw = {key: json.loads(path.read_text()) for key, path in SOURCES.items()}
    repeated, per_seed = checked_repeats(raw["repeated"])
    identity, identity_wins = checked_identity(raw["repeated"], raw["identity_audit"])
    if not raw["kiba_status"].get("experiment_complete"):
        raise RuntimeError("KIBA five-fold experiment is incomplete")
    if raw["kiba"]["config_sha256"] != digest(SOURCES["kiba_config"]):
        raise RuntimeError("KIBA result/config hash mismatch")
    if not raw["davis"]["integrity_checks"]["passed"]:
        raise RuntimeError("Davis prediction integrity checks did not pass")
    archived_data_checked = 0
    for record in raw["davis"]["inputs"]:
        path = PROJECT / record["path"]
        if path.suffix in (".json", ".csv"):
            if digest(path) != record["sha256"]:
                raise RuntimeError(f"Archived Davis input changed: {path}")
            archived_data_checked += 1
    local = {
        "Davis": raw["davis"]["five_fold_metrics"],
        "KIBA": raw["kiba"]["models"],
    }
    local = {dataset: {model: {metric: checked_metric(values[metric])
                              for metric in ("mse", "pearson", "ci")}
                       for model, values in models.items()}
             for dataset, models in local.items()}
    if not raw["comparator_audit"]["verified"]:
        raise RuntimeError("Reported-score audit did not pass")
    if raw["fusion"]["analysis_config"]["sha256"] != digest(SOURCES["analysis_config"]):
        raise RuntimeError("Fusion analysis/config hash mismatch")
    for item in raw["fusion"]["inputs"]:
        if digest(PROJECT.parent / item["path"]) != item["sha256"]:
            raise RuntimeError("Fusion input predictions changed")
    for dataset, models in local.items():
        additional = raw["fusion"]["datasets"][dataset.lower()]["summary"]
        for model, values in additional.items():
            checked = {k: checked_metric(v) for k, v in values.items()}
            if model in models:
                for metric in ("mse", "pearson", "ci"):
                    if not math.isclose(models[model][metric]["mean"], checked[metric]["mean"], abs_tol=1e-12):
                        raise RuntimeError("Reanalysis changed an original metric")
            models[model] = checked
    reported = {
        dataset: {model: {metric: checked_metric(values[metric])
                          for metric in ("mse", "pearson")}
                  for model, values in raw["published"]["published_figure3_results"]
                  [dataset.lower()]["drug_cold"].items()}
        for dataset in local
    }
    (MANUSCRIPT / "tables").mkdir(parents=True, exist_ok=True)
    macros, benchmark, ablation, folds, uncertainty, strata, rankings = [], [], [], [], [], [], []
    stability, gate_parameters, stacking_parameters, fold_records = [], [], [], []
    gains = {}
    for dataset, models in local.items():
        prefix = "Davis" if dataset == "Davis" else "Kiba"
        for model in REPEATED_MODELS:
            for metric in ("mse", "pearson", "ci"):
                reference_values = per_seed[dataset.lower(), SEEDS[0]]["metrics"][model][metric]["per_fold"]
                if any(not math.isclose(a, b, rel_tol=0, abs_tol=1e-12)
                       for a, b in zip(models[model][metric]["per_fold"], reference_values)):
                    raise RuntimeError("Reference-seed metrics disagree between original and repeated summaries")
        gate = models["similarity_gate"]
        gains[dataset] = {}
        references = {
            "Reported": reported[dataset]["DTIAM"]["mse"]["mean"],
            "Fixed": models["fixed_blend"]["mse"]["mean"],
            "Prior": models["neighbor_prior"]["mse"]["mean"],
            "Deep": models["deep"]["mse"]["mean"],
        }
        for key, reference in references.items():
            gain = 100 * (1 - gate["mse"]["mean"] / reference)
            gains[dataset][key] = gain
            macros.append(f"\\newcommand{{\\{prefix}{key}Gain}}{{{gain:.2f}}}")
        for model, alias in (("similarity_gate", "Gate"), ("deep", "Deep"),
                             ("neighbor_prior", "Prior"), ("fixed_blend", "Fixed"),
                             ("half_blend", "Half"), ("linear_stacking", "Stack")):
            for metric, metric_alias in (("mse", "MSE"), ("pearson", "Pearson")):
                mean = models[model][metric]["mean"]
                macros.append(f"\\newcommand{{\\{prefix}{alias}{metric_alias}}}{{{mean:.5f}}}")
        for model in ("DeepDTA", "MONN", "BACPI", "GraphDTA", "DTIAM"):
            values = reported[dataset][model]
            benchmark.append(f"{dataset} & {model} & Reported & {cell(values['mse'])} & {cell(values['pearson'])}" + r" \\")
        benchmark.append(f"{dataset} & Fusion (reference seed) & Local & {cell(gate['mse'])} & {cell(gate['pearson'])}" + r" \\")
        combined = {}
        for metric, alias in (("mse", "MSE"), ("pearson", "Pearson")):
            entry = repeated[dataset.lower(), "similarity_gate", metric]
            combined[metric] = {"mean": entry["mean_over_seeds_and_folds"],
                                "sd": entry["sample_sd_of_five_fold_means_over_seeds"]}
            macros.append(f"\\newcommand{{\\{prefix}Three{alias}}}{{{entry['mean_over_seeds_and_folds']:.5f}}}")
            macros.append(f"\\newcommand{{\\{prefix}Three{alias}SeedSD}}{{{entry['sample_sd_of_three_seed_means']:.5f}}}")
        benchmark.append(f"{dataset} & Fusion (3 seeds) & Local & {cell(combined['mse'])} & {cell(combined['pearson'])}" + r" \\")
        wins = sum(per_seed[dataset.lower(), seed]["gate_lower_mse_folds"]["fixed_blend"] for seed in SEEDS)
        macros.append(f"\\newcommand{{\\{prefix}GateFixedWins}}{{{wins}}}")
        for model, label in LABELS.items():
            values = models[model]
            ablation.append(f"{dataset} & {label} & {cell(values['mse'])} & {cell(values['pearson'])} & {values['ci']['mean']:.5f}" + r" \\")
        for model in REPEATED_MODELS:
            for metric, metric_label in (("mse", "MSE"), ("pearson", "Pearson"), ("ci", "C-index")):
                entry = repeated[dataset.lower(), model, metric]
                stability.append(f"{dataset} & {LABELS[model]} & {metric_label} & "
                                 f"{entry['mean_over_seeds_and_folds']:.5f} & "
                                 f"{entry['sample_sd_of_three_seed_means']:.5f} & "
                                 f"{entry['sample_sd_of_five_fold_means_over_seeds']:.5f}" + r" \\")
        for seed in SEEDS:
            for fold in range(5):
                available = LABELS if seed == SEEDS[0] else REPEATED_MODELS
                for model in available:
                    values = (models[model] if seed == SEEDS[0]
                              else per_seed[dataset.lower(), seed]["metrics"][model])
                    metrics = {metric: values[metric]["per_fold"][fold] for metric in ("mse", "pearson", "ci")}
                    folds.append(f"{dataset} & {seed} & {fold + 1} & {LABELS[model]} & "
                                 + " & ".join(f"{metrics[m]:.5f}" for m in ("mse", "pearson", "ci")) + r" \\")
                    fold_records.append({"dataset": dataset.lower(), "protocol": "drug_id", "seed": seed,
                                         "fold": fold, "model": model, **metrics})
        analysis = raw["fusion"]["datasets"][dataset.lower()]
        for contrast in analysis["cluster_bootstrap"]["comparisons"]:
            model = contrast["contrast"].removeprefix("similarity_gate_minus_")
            lo, hi = contrast["percentile_95"]
            delta = contrast["mse_difference"]
            uncertainty.append(f"{dataset} & {LABELS[model]} & {delta:.5f} & {lo:.5f} & {hi:.5f}" + r" \\")
            alias = {"deep": "Deep", "neighbor_prior": "Prior", "half_blend": "Half", "fixed_blend": "Fixed", "linear_stacking": "Stack"}[model]
            for suffix, value in (("Diff", delta), ("Lo", lo), ("Hi", hi)):
                macros.append(f"\\newcommand{{\\{prefix}{alias}{suffix}}}{{{value:.5f}}}")
        for group in analysis["subgroups"]:
            if group["subgroup"] == "all":
                continue
            label = {"tanimoto_0_0.3": "$[0,0.3)$", "tanimoto_0.3_0.5": "$[0.3,0.5)$",
                     "tanimoto_0.5_0.7": "$[0.5,0.7)$", "tanimoto_0.7_1.0": "$[0.7,1]$",
                     "pKd_floor_5": "pKd floor", "pKd_above_5": "Above floor"}[group["subgroup"]]
            strata.append(f"{dataset} & {label} & {group['pairs']:,} & {group['chemical_identities']:,} & "
                          + " & ".join(f"{group['pooled_metrics'][m]['mse']:.5f}" for m in LABELS) + r" \\")
        for entry in analysis["folds"]:
            fit, fold = entry["fit"], entry["fold"]
            gate_parameters.append(f"{dataset} & {fold + 1} & " + " & ".join(f"{v:.5f}" for v in (
                fit["fixed_deep_weight"], fit["gate"]["a"], fit["gate"]["b"],
                fit["gate"]["sim_mean"], fit["gate"]["sim_std"])) + r" \\")
            stacking_parameters.append(f"{dataset} & {fold + 1} & " + " & ".join(f"{fit['linear_stacking'][key]:.5f}"
                                       for key in ("intercept", "deep_coefficient", "prior_coefficient")) + r" \\")
        for entry in raw["fusion"]["ranking_macro"]:
            if entry["dataset"] != dataset.lower():
                continue
            rankings.append(f"{dataset} & {LABELS[entry['model']]} & {entry['targets']} & {entry['spearman_macro']:.5f} & {entry['pearson_macro']:.5f}" + r" \\")
        if dataset == "Davis":
            benchmark.append(r"\midrule")
            ablation.append(r"\midrule")
            folds.append(r"\midrule")
    identity_rows, identity_folds, identity_records, audit_rows = [], [], [], []
    identity_gains = {}
    for model, alias in (("deep", "Deep"), ("neighbor_prior", "Prior"),
                         ("fixed_blend", "Fixed"), ("similarity_gate", "Gate")):
        values = identity[model]
        identity_rows.append(f"{LABELS[model]} & {cell(values['mse'])} & {cell(values['pearson'])} & "
                             f"{values['ci']['mean']:.5f}" + r" \\")
        for metric, suffix in (("mse", "MSE"), ("pearson", "Pearson")):
            macros.append(f"\\newcommand{{\\KibaIdentity{alias}{suffix}}}{{{values[metric]['mean']:.5f}}}")
        if model != "similarity_gate":
            gain = 100 * (1 - identity["similarity_gate"]["mse"]["mean"] / values["mse"]["mean"])
            identity_gains[model] = gain
            macros.append(f"\\newcommand{{\\KibaIdentity{alias}Gain}}{{{gain:.2f}}}")
    for fold in raw["identity_audit"]["folds"]:
        f = fold["fold"]
        counts = [fold[k][p] for k in ("rows", "identities") for p in ("train", "val", "test")]
        audit_rows.append(f"{f + 1} & " + " & ".join(f"{n:,}" for n in counts)
                          + f" & {max(fold['canonical_overlap'].values())}" + r" \\")
        for model in REPEATED_MODELS:
            metrics = {m: identity[model][m]["per_fold"][f] for m in ("mse", "pearson", "ci")}
            identity_folds.append(f"{f + 1} & {LABELS[model]} & "
                                  + " & ".join(f"{metrics[m]:.5f}" for m in ("mse", "pearson", "ci")) + r" \\")
            identity_records.append({"dataset": "kiba", "protocol": "identity", "seed": SEEDS[0],
                                     "fold": f, "model": model, **metrics})
    table_file("identity.tex", "KIBA performance with disjoint canonical chemical identities.",
               "tab:identity", "lrrr", "Predictor & MSE & Pearson & Mean CI", identity_rows,
               r"MSE and Pearson give the unweighted five-fold mean $\pm$ sample fold SD; CI denotes concordance index. All predictors share the same folds, deep learning model outputs and training-only neighbor evidence, with training seed 20261012. Blend and gate parameters use validation predictions only. The gate has lower MSE in all five folds than each listed control; these direction counts are descriptive, without a new significance test. Published drug-ID baseline scores are not evaluated under this protocol.")
    table_file("identity_folds.tex", "Complete strict-identity KIBA fold results, training seed 20261012.",
               "tab:identity_folds", "rlrrr", "Fold & Predictor & MSE & Pearson & C-index", identity_folds,
               r"Fold indices 0--4 are displayed as 1--5. All planned folds are retained. C-index denotes concordance index. These results use different partitions from the drug-ID-cold benchmark.")
    table_file("identity_audit.tex", "Chemical-identity-disjoint KIBA partition audit.",
               "tab:identity_audit", "rrrrrrrr",
               "Fold & Train pairs & Val pairs & Test pairs & Train IDs & Val IDs & Test IDs & Shared IDs",
               audit_rows,
               r"IDs denote canonical isomeric SMILES identities, retaining salt, charge and tautomer states. Shared IDs is the maximum of the train/validation, train/test and validation/test identity intersections; all three are zero in every fold. All 118,254 pairs and 2,068 identities are retained, and each pair is tested once. Every validation and test query has same-target training support. Chemical-identity separation does not impose scaffold separation.", compact=True)
    with (MANUSCRIPT / "tables/identity_fold_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(identity_records[0]))
        writer.writeheader()
        writer.writerows(identity_records)
    (MANUSCRIPT / "tables/metrics.tex").write_text("\n".join(macros) + "\n")
    table_file("benchmark.tex", "Drug-ID-cold performance with fold variation and repeated-training summaries.",
               "tab:benchmark", "lllcc", "Dataset & Model & Source & MSE & Pearson",
               benchmark,
               r"Reported rows are taken from the DTIAM study's Figure 3 source data \citep{dtiam2025,dtiamSource}, not local reproductions. Reported and reference-seed local rows give the five-fold mean $\pm$ sample fold SD; the local reference seed is 20261012. Three-seed rows average seeds 20261012, 20261017 and 20261018 within each fold, then report the mean $\pm$ SD of those five fold averages. Seed SD is reported separately in Table~\ref{tab:stability}. Local inner validation differs from the reported workflows; no paired significance is implied.", compact=True)
    table_file("ablation.tex", "Matched local ablations under the original drug-ID protocol.",
               "tab:ablation", "llrrr", "Dataset & Predictor & MSE & Pearson & Mean CI",
               ablation,
               "MSE and Pearson are mean plus/minus sample SD across five folds for reference training seed 20261012; CI denotes concordance index. All six predictors use the same test rows, deep outputs and saved training-only priors. Constant blend, linear stacking and gate parameters are fitted using validation data only.")
    fold_table(folds)
    with (MANUSCRIPT / "tables/fold_seed_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fold_records[0]))
        writer.writeheader()
        writer.writerows(fold_records)
    table_file("stability.tex", "Separate training-seed and outer-fold variation for the four repeated predictors.",
               "tab:stability", "lllrrr", "Dataset & Predictor & Metric & Mean & Seed SD & Fold SD",
               stability,
               r"Mean averages five folds equally within each seed, then averages the three seed means. Seed SD is the sample SD of these three five-fold means. Fold SD is the sample SD of the five fold means after averaging seeds within each fold. Neighbor predictions are reused and therefore have zero seed SD. Neither SD is a standard error; the 15 correlated fold--seed results are not independent datasets. All per-fold values are in Table~\ref{tab:folds}.")
    table_file("bootstrap.tex", "Paired chemical-identity bootstrap intervals for gate-minus-control MSE differences.",
               "tab:bootstrap", "llrrr", "Dataset & Control & Difference & Lower 95\\% & Upper 95\\%",
               uncertainty,
               r"Negative differences favor gated fusion. Percentile intervals use 10,000 global chemical-identity resamples, sharing identity multiplicities across models and folds (seed 20261017). The estimand is the unweighted five-fold MSE difference, conditional on saved predictions. Intervals do not include retraining uncertainty or multiplicity adjustment; no published baseline is included.")
    table_file("subgroups.tex", "Similarity and Davis-label subgroup composition, training seed 20261012.",
               "tab:subgroups", "llrrrrrrrr", "Dataset & Subgroup & Pairs & IDs & Deep & Neighbor & Equal & Constant & Stacking & Gate",
               strata,
               r"All six predictor columns give pooled MSE, unlike the fold-mean benchmark metric. IDs counts distinct canonical chemical identities. Similarity refers to maximum same-target training Tanimoto; an identity can occur in multiple intervals. The highest-similarity Davis interval contains only one identity. The pKd subsets are separate from the similarity bins; above-floor labels are not automatically high-affinity binding. Full subgroup metrics and mean weights remain in the archived analysis JSON.", compact=True)
    table_file("gate_parameters.tex", "Validation-fitted constant-blend and similarity-gate parameters, reference seed 20261012.",
               "tab:gate_parameters", "lrrrrrr", r"Dataset & Fold & $\alpha$ & $a$ & $b$ & $\mu_s$ & $\sigma_s$",
               gate_parameters,
               r"$\alpha$ is the deep weight in the constant blend; $a,b$ define $w=\operatorname{sigmoid}(a+b(s-\mu_s)/\sigma_s)$. Similarity normalization and all fitted parameters use validation data only. Fold indices 0--4 are displayed as 1--5.")
    table_file("stacking_parameters.tex", "Validation-fitted unconstrained OLS stacking coefficients, reference seed 20261012.",
               "tab:stacking_parameters", "lrrrr", r"Dataset & Fold & $\beta_0$ & $\beta_d$ & $\beta_n$",
               stacking_parameters,
               r"Stacking predicts $\beta_0+\beta_d\widehat y_{\mathrm{deep}}+\beta_n\widehat y_{\mathrm{nbr}}$. Coefficients are not constrained to a convex combination. Every reference-seed design matrix has rank three; fits use only the corresponding validation predictions.")
    table_file("ranking.tex", "Descriptive target-wise ranking, macro-averaged over all eligible target IDs.",
               "tab:ranking", "llrrr", "Dataset & Predictor & Targets & Spearman & Pearson",
               rankings,
               r"Eligibility requires at least 20 distinct test chemical identities and two distinct labels. Out-of-fold predictions are pooled within each target. Davis includes 442 benchmark target IDs, some with identical sequences; KIBA includes 215 of 229 target IDs. These are retrospective ranking summaries, not prospective screening or independent experimental validation.")
    snapshot = {
        "created_for": "BIB Results 1 and 2 plus strict-identity Results 3 integration, 2026-10-18",
        "sources": {key: {"path": str(path.relative_to(PROJECT)), "sha256": digest(path)}
                    for key, path in SOURCES.items()},
        "archived_davis_data_files_checked": archived_data_checked,
        "aggregation": {"reference": "Unweighted five-fold mean and sample fold SD; seed 20261012",
                        "repeated": raw["repeated"]["uncertainty_note"]},
        "local": local, "reported_by_dtiam_study": reported, "relative_mse_reduction_percent": gains,
        "repeated_training": {"per_seed": [e for e in raw["repeated"]["per_seed"] if e["protocol"] == "drug_id"],
                              "three_seed_summary": raw["repeated"]["three_seed_summary"]},
        "fusion_analysis": raw["fusion"],
        "strict_identity": {"models": identity, "gate_lower_mse_folds": identity_wins,
                            "relative_mse_reduction_percent": identity_gains,
                            "audit": raw["identity_audit"]},
        "limitations": [
            "Published baselines are not local reproductions; inner validation differs.",
            "Public per-fold ordering is not independently verified for paired tests.",
            "KIBA drug-ID splits are not strictly canonical-identity-disjoint.",
            "Strict-identity KIBA results are integrated; identity separation does not ensure scaffold separation.",
            "Scaffold evaluation and independent candidate screening with cell-response evidence remain incomplete.",
            "Bootstrap intervals condition on fixed predictions and do not model scaffold dependence.",
            "Davis was inspected during development; additional seeds do not create a new holdout.",
        ],
    }
    (MANUSCRIPT / "evidence_snapshot.json").write_text(json.dumps(snapshot, indent=2) + "\n")
    print(json.dumps({"tables": 12, "fold_seed_rows": len(fold_records), "identity_fold_rows": len(identity_records),
                      "identity_gains": identity_gains, "verified_davis_files": archived_data_checked,
                      "gains": gains, "snapshot": str(MANUSCRIPT / "evidence_snapshot.json")}, indent=2))


if __name__ == "__main__":
    main()
