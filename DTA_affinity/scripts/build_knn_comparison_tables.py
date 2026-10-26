#!/usr/bin/env python3
"""Export the overall local comparison and matched-backbone comparison."""
import hashlib
import json
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
REPORT = PROJECT / "reports/knn_shared_20261025"


def write_tex(path, frame, caption, label, methods):
    lines = [r"\begin{table}[htbp]", r"\centering", "\\caption{" + caption + "}",
             "\\label{" + label + "}", r"\small", r"\begin{tabular}{lrrrr}", r"\toprule",
             r"& \multicolumn{2}{c}{Davis} & \multicolumn{2}{c}{KIBA} \\",
             r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
             r"Method & MSE & Pearson & MSE & Pearson \\", r"\midrule"]
    for method in methods:
        parts = [method]
        for dataset in ("davis", "kiba"):
            row = frame.loc[frame.dataset.eq(dataset) & frame.method.eq(method)].iloc[0]
            for metric in ("mse", "pearson"):
                parts.append(f"${row[metric + '_mean']:.5f} \\pm {row[metric + '_fold_sd']:.5f}$")
        lines.append(" & ".join(parts) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    path.write_text("\n".join(lines) + "\n")


def main():
    shared = pd.read_csv(REPORT / "summary.csv")
    native_path = PROJECT / "reports/davis_target_id_20261025/progress_summary.json"
    native = json.loads(native_path.read_text())
    original = {(r["dataset"], r["metric"]): r for r in native["three_seed_summary"]
                if r["model"] == "similarity_gate" and r["metric"] in ("mse", "pearson")}
    overall = []
    for dataset in ("davis", "kiba"):
        row = {"dataset": dataset, "method": "SGF-DTA", "backbone": "SGF-DTA token-interaction backbone"}
        for metric in ("mse", "pearson"):
            value = original[dataset, metric]
            row[metric + "_mean"] = value["mean_over_seeds_and_folds"]
            row[metric + "_fold_sd"] = value["sample_sd_of_five_fold_means_over_seeds"]
        overall.append(row)
        for key, name in (("knn_dta", "kNN-DTA"), ("ada_knn_dta", "Ada-kNN-DTA")):
            row = shared.loc[shared.dataset.eq(dataset) & shared.method.eq(key)].iloc[0].to_dict()
            row.update(method=name, backbone="shared vector backbone; local aggregation implementation")
            overall.append(row)
    overall = pd.DataFrame(overall)
    overall.to_csv(REPORT / "overall_comparison.csv", index=False)
    labels = {"base": "Base model", "knn_dta": "+ kNN-DTA", "ada_knn_dta": "+ Ada-kNN-DTA", "sgf": "+ SGF"}
    controlled = shared.loc[shared.method.isin(labels)].copy()
    controlled["method"] = controlled.method.map(labels)
    controlled["backbone"] = "same trained vector backbone within each fold and seed"
    controlled.to_csv(REPORT / "controlled_comparison.csv", index=False)
    caption = ("Local method comparison under fivefold drug-ID-cold evaluation. "
               "Values are means and sample SDs across folds after averaging three training seeds within each fold. "
               "SGF-DTA uses its full prediction backbone; the kNN methods use the vector backbone specified for the local implementations.")
    write_tex(REPORT / "overall_comparison.tex", overall, caption, "tab:knn_overall", ["kNN-DTA", "Ada-kNN-DTA", "SGF-DTA"])
    caption = ("Retrieval-method comparison on the same trained backbone within each fold and seed. "
               "Values are fivefold means and sample SDs after averaging three training seeds within each fold.")
    write_tex(REPORT / "controlled_comparison.tex", controlled, caption, "tab:knn_controlled", list(labels.values()))
    sources = {str(path.relative_to(PROJECT)): hashlib.sha256(path.read_bytes()).hexdigest()
               for path in (native_path, REPORT / "summary.csv")}
    (REPORT / "table_provenance.json").write_text(json.dumps({"sources": sources, "scope": "overall local implementations plus matched-backbone mechanism comparison"}, indent=2) + "\n")
    print("Wrote overall_comparison and controlled_comparison as CSV and LaTeX")


if __name__ == "__main__":
    main()
