#!/usr/bin/env python3
"""Render the frozen target-wise error analysis used by the manuscript."""
import hashlib
import json
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from check_overlap import assert_figure_clean

M = Path(__file__).resolve().parents[1]
SOURCE = M / "tables/source/target_analysis_20261026"
PROTOCOLS = ["davis_drug_id", "kiba_drug_id", "kiba_identity", "kiba_scaffold"]
LABELS = ["Davis drug-ID cold", "KIBA drug-ID cold", "KIBA identity-disjoint", "KIBA scaffold-disjoint"]
CONTROLS = ["deep", "neighbor_prior", "fixed_blend"]
NAMES = ["Deep", "Neighbor", "Constant"]
COLORS = {"deep": "#515D69", "fixed_blend": "#598A83", "zero": "#A6AEB5"}


def load_verified():
    manifest = json.loads((SOURCE / "manifest.json").read_text())
    for name, digest in manifest.items():
        assert hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() == digest, name
    summary = pd.read_csv(SOURCE / "target_summary.csv")
    data = {}
    for protocol in PROTOCOLS:
        data[protocol] = d = pd.read_csv(SOURCE / f"{protocol}_targets.csv")
        seeds = json.loads((SOURCE / f"{protocol}_analysis.json").read_text())["seeds"]
        assert seeds == ([20261012, 20261017, 20261018] if protocol.endswith("drug_id") else [20261012])
        for minimum in [0, 20]:
            selected = d[d.identities >= minimum]
            for control in CONTROLS:
                row = summary[(summary.protocol == protocol) &
                    (summary.minimum_identities == minimum) & (summary.control == control)].iloc[0]
                delta = selected.mse_similarity_gate - selected["mse_" + control]
                assert row.targets == len(selected)
                assert row.improved == (delta < 0).sum()
                assert row.worsened == (delta > 0).sum()
                assert np.isclose(row.median_mse_difference, delta.median())
                best = selected.assign(delta=delta).sort_values("delta").head(int(np.ceil(.1 * len(selected)))).target
                rest = selected[~selected.target.isin(best)]
                remaining = np.average(rest.mse_similarity_gate - rest["mse_" + control], weights=rest.pairs)
                assert np.isclose(row.remaining_pooled_mse_difference, remaining, atol=1e-12)
    return summary, data


def draw(data):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
        "axes.labelsize": 8, "xtick.labelsize": 7, "ytick.labelsize": 7,
        "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42, "svg.fonttype": "none", "axes.linewidth": .6})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.7))
    fig.subplots_adjust(left=.10, right=.98, bottom=.10, top=.84, hspace=.68, wspace=.33)
    for i, (ax, protocol, title) in enumerate(zip(axes.flat, PROTOCOLS, LABELS)):
        d = data[protocol]
        for control, style in [("deep", "-"), ("fixed_blend", "--")]:
            x = np.sort(100 * (d.mse_similarity_gate - d["mse_" + control]) / d["mse_" + control])
            assert x.min() >= -100 and x.max() <= 100
            ax.step(np.r_[-100, x, 100], np.r_[0, 100 * np.arange(1, len(x) + 1) / len(x), 100],
                    where="post", color=COLORS[control], ls=style, lw=1.3)
        ax.axvline(0, color=COLORS["zero"], ls=":", lw=.8)
        ax.set(xlim=(-100, 100), ylim=(0, 102), xticks=[-100, -50, 0, 50, 100],
               yticks=[0, 25, 50, 75, 100], xlabel="Target MSE change (%)",
               ylabel="Cumulative targets (%)" if i % 2 == 0 else "")
        ax.text(0, 1.13, "abcd"[i], transform=ax.transAxes, weight="bold", fontsize=10)
        ax.text(.09, 1.13, title, transform=ax.transAxes, fontsize=8.5)
    handles = [Line2D([], [], color=COLORS[c], ls=s, label=l)
               for c, s, l in [("deep", "-", "SGF vs deep learning model"),
                              ("fixed_blend", "--", "SGF vs constant blend")]]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.5, .97),
               ncol=2, frameon=False, columnspacing=2)
    assert_figure_clean(fig, include_ticks=True)
    (M / "figures").mkdir(parents=True, exist_ok=True)
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(M / f"figures/target_error_distribution.{ext}", dpi=300)
    plt.close(fig)


def table(summary, minimum):
    suffix = "all" if minimum == 0 else "min20"
    caption = ("Target-wise MSE comparisons for all target IDs." if minimum == 0 else
               "Sensitivity analysis restricted to targets with at least 20 test chemical identities.")
    out = [r"\begin{table}[htbp]\centering\small",
        rf"\caption{{{caption}}}\label{{tab:target_errors_{suffix}}}",
        r"\begin{tabular}{llrrr}\toprule",
        r"Protocol & Control & Improved / total & Median $\Delta$MSE & Remaining $\Delta$MSE \\ \midrule"]
    for protocol, label in zip(PROTOCOLS, ["Davis / ID", "KIBA / ID", "KIBA / identity", "KIBA / scaffold"]):
        for control, name in zip(CONTROLS, NAMES):
            r = summary[(summary.protocol == protocol) & (summary.minimum_identities == minimum)
                        & (summary.control == control)].iloc[0]
            out.append(f"{label} & {name} & {int(r.improved)}/{int(r.targets)} & "
                       f"{r.median_mse_difference:+.5f} & {r.remaining_pooled_mse_difference:+.5f} "+r"\\")
    out += [r"\bottomrule\end{tabular}",
        r"\par\smallskip\begin{minipage}{0.98\linewidth}\footnotesize "
        r"$\Delta$MSE is SGF-DTA minus the specified control. Improved counts denote strictly lower target-wise MSE. "
        r"Remaining $\Delta$MSE is the pair-weighted difference after removing the 10\% of eligible targets with "
        r"the largest MSE reductions for that comparison. The number removed is rounded up. Negative values favor SGF-DTA."
        r"\end{minipage}\end{table}", r"\FloatBarrier"]
    return "\n".join(out)


def main():
    summary, data = load_verified()
    draw(data)
    lines = [r"\section{Target-wise Prediction Errors}", r"\label{sec:target_errors}",
        r"For each test pair, squared errors are averaged across the available training seeds; "
        r"these errors are then pooled over all five test folds and averaged within each experimental target ID. "
        r"The drug-ID-cold analyses use seeds 20261012, 20261017 and 20261018. Chemical-identity and scaffold "
        r"analyses use seed 20261012, matching the corresponding main-text experiments. These statistics describe "
        r"prediction error within targets, separately from the target-wise Spearman ranking analysis.",
        r"All 442 Davis and 229 KIBA target IDs are included in the primary analysis. "
        r"The minimum-sample sensitivity retains all Davis IDs and 215 KIBA IDs with at least 20 test chemical identities. "
        r"The contribution of the strongest improvements is assessed by removing the top 10\% of targets, ranked "
        r"by control MSE minus SGF-DTA MSE, separately for each comparator and eligibility rule. "
        r"The remaining difference is recomputed by weighting each target by its test-pair count.",
        r"\begin{figure}[htbp]\centering",
        r"\includegraphics[width=0.98\linewidth]{figures/target_error_distribution.pdf}",
        r"\caption{Distribution of target-wise MSE changes. (a,b) Drug-ID-cold evaluation on Davis and KIBA. "
        r"(c,d) KIBA chemical-identity and scaffold separation. Curves show the cumulative fraction of target IDs "
        r"as a function of $100(\mathrm{MSE}_{\mathrm{SGF}}-\mathrm{MSE}_{\mathrm{control}})/\mathrm{MSE}_{\mathrm{control}}$. "
        r"Negative changes favor SGF-DTA; the fraction at zero indicates the proportion of targets with reduced error. "
        r"Each curve includes every target ID in its dataset. Solid and dashed curves compare SGF-DTA with the "
        r"deep learning model and constant blending, respectively. SGF abbreviates SGF-DTA.}",
        r"\label{fig:target_errors}\end{figure}", r"\FloatBarrier",
        table(summary, 0), table(summary, 20),
        r"Improvements over the deep learning model occur in most targets under all four protocols and remain "
        r"after removing the strongest-improving decile. The comparisons with constant blending show a different "
        r"pattern: Davis improves in 191 of 442 targets and has a median difference of $+0.00242$, whereas "
        r"the three KIBA protocols improve in a majority of targets and have negative median differences. "
        r"Complete target identifiers, pair counts, chemical-identity counts and model errors accompany the source tables."]
    (M / "sections").mkdir(parents=True, exist_ok=True)
    (M / "sections/target_error_supplement.tex").write_text("\n\n".join(lines) + "\n")
    (M / "figures/target_error_distribution_qa.json").write_text(json.dumps({
        "artist_check": "passed including ticks",
        "input_manifest_sha256": hashlib.sha256((SOURCE / "manifest.json").read_bytes()).hexdigest(),
        "claim": "Distribution of improvements over the deep learning model; constant-blend comparisons retained.",
        "independent_summary_recalculation": "counts, medians and trimmed pooled differences verified for all rows"
    }, indent=2) + "\n")
    print("Verified target-wise summaries, supplementary figure and tables generated.")


if __name__ == "__main__":
    main()
