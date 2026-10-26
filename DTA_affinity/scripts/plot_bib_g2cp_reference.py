#!/usr/bin/env python3
"""Render the verified development diagnostic without new scoring or fitting."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from bib_g2cp_common import PROJECT, check_records, record, save_json

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "manuscript/bib/tools"))
from check_overlap import assert_figure_clean, check_png

OUT = PROJECT / "reports/bib_g2cp_reference_diagnostic_20261019"
COLORS = {"gate": "#007E78", "reference": "#4169A1", "generic": "#87808E",
          "chemical": "#C68521", "g2cp": "#515D69", "zero": "#B2B7BB", "text": "#24282B"}
LABELS = {"aux_annotated_reference": ("Annotated targets", "reference"),
          "aux_similarity_gate": ("DTA top 5", "gate"),
          "aux_similarity_gate_matched_count": ("DTA matched count", "gate"),
          "aux_generic": ("Genetic-library mean", "generic"),
          "aux_chemical_training_mean": ("Chemical-train mean", "chemical"),
          "g2cp": ("G2CP", "g2cp"),
          "g2cp_plus_similarity_gate": ("+ DTA top 5", "gate"),
          "g2cp_plus_annotated_reference": ("+ Annotated targets", "reference"),
          "g2cp_plus_chemical_training_mean": ("+ Chemical-train mean", "chemical")}


def panel_title(ax, letter, title):
    ax.text(0, 1.08, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
    ax.text(.09, 1.08, title, transform=ax.transAxes, fontsize=8.5, va="bottom")


def points(ax, data, methods, metric="pearson"):
    for index, method in enumerate(methods):
        values = data.loc[data.method.eq(method)].sort_values("canonical_identity")[metric].to_numpy()
        color = COLORS[LABELS[method][1]]
        offsets = np.linspace(-.15, .15, len(values))
        ax.scatter(values, index+offsets, color=color, alpha=.4, s=12, edgecolors="none")
        ax.scatter([values.mean()], [index], color=color, marker="D", s=24, zorder=3)
    ax.set_yticks(np.arange(len(methods)), [LABELS[m][0] for m in methods])
    ax.set_ylim(len(methods)-.5, -.6)
    ax.axvline(0, color=COLORS["zero"], linewidth=.65, zorder=0)
    ax.set_xlabel("Pearson r" if metric == "pearson" else "Centered Pearson r")
    ax.set_ylabel("Method")


def main():
    analysis = json.loads((OUT / "analysis.json").read_text())
    verification = json.loads((OUT / "verification.json").read_text())
    check_records(analysis["outputs"])
    check_records([verification["analysis"], verification["verifier"]])
    if verification["state"] != "passed":
        raise RuntimeError("Diagnostic has not passed verification")
    metrics = pd.read_csv(OUT / "per_drug_cell_metrics.csv")
    primary = metrics.loc[metrics.split.eq("val") & metrics.davis_external & metrics.reference_available]
    data = primary.groupby(["method", "seed", "canonical_identity"])[["pearson", "centered_pearson"]].mean()
    data = data.groupby(["method", "canonical_identity"]).mean().reset_index()
    n = data.canonical_identity.nunique()
    pairs = primary[["cell", "canonical_identity"]].drop_duplicates().shape[0]
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8,
                         "axes.titlesize": 9, "xtick.labelsize": 7, "ytick.labelsize": 7,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "svg.fonttype": "none", "axes.linewidth": .6,
                         "text.color": COLORS["text"], "savefig.facecolor": "white"})
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.65))
    fig.subplots_adjust(left=.21, right=.98, bottom=.12, top=.82, wspace=1.30, hspace=.85)
    fig.text(.05, .955, "Results 5: target-reference diagnostic", fontsize=12, fontweight="bold")
    fig.text(.05, .910, f"Development validation only | {n} drugs, {pairs} drug-cell pairs | 3 G2CP seeds", fontsize=8)
    fig.text(.05, .875, "Matched annotated-target coverage; fixed 25% auxiliary weight; no new test scoring", fontsize=7.5)

    auxiliary = ["aux_annotated_reference", "aux_similarity_gate", "aux_similarity_gate_matched_count",
                 "aux_generic", "aux_chemical_training_mean"]
    points(axes[0, 0], data, auxiliary)
    panel_title(axes[0, 0], "a", "Auxiliary responses")
    axes[0, 0].set_xlim(-.2, .5)
    axes[0, 0].set_xticks([-.2, 0, .2, .4])

    comparisons = [item for item in analysis["comparisons"] if item["scope"] == "primary_davis_disjoint"
                   and item["metric"] == "pearson" and item["left"] == "aux_annotated_reference"]
    ax = axes[0, 1]
    labels = []
    for index, item in enumerate(comparisons):
        low, high = item["ci95"]
        ax.plot([low, high], [index, index], color=COLORS["reference"], linewidth=1.1)
        ax.scatter([item["difference"]], [index], color=COLORS["reference"], s=20, zorder=3)
        labels.append(LABELS[item["right"]][0])
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(len(labels)-.5, -.6)
    ax.axvline(0, color=COLORS["zero"], linewidth=.65, zorder=0)
    ax.set_xlabel("Reference minus comparator (r)")
    ax.set_ylabel("Comparator")
    ax.set_xlim(-.25, .17)
    ax.set_xticks([-.2, -.1, 0, .1])
    panel_title(ax, "b", "Reference differences")

    mixtures = ["g2cp", "g2cp_plus_similarity_gate", "g2cp_plus_annotated_reference", "g2cp_plus_chemical_training_mean"]
    points(axes[1, 0], data, mixtures)
    panel_title(axes[1, 0], "c", "Fixed-weight mixtures")
    axes[1, 0].set_xlim(-.1, .65)
    axes[1, 0].set_xticks([0, .2, .4, .6])

    centered = ["aux_annotated_reference", "aux_similarity_gate", "aux_generic", "g2cp"]
    points(axes[1, 1], data, centered, "centered_pearson")
    panel_title(axes[1, 1], "d", "Response specificity")
    axes[1, 1].set_xlim(-.25, .45)
    axes[1, 1].set_xticks([-.2, 0, .2, .4])
    fig.text(.05, .045, "Points: drug means across available cells and seeds. Diamonds: equal-drug means.", fontsize=7)
    fig.text(.05, .020, "Intervals: 2,000 paired drug resamples, descriptive 95%. Source: frozen diagnostic, 19 Oct 2026.", fontsize=7)
    fig.canvas.draw()
    artist = assert_figure_clean(fig, include_ticks=True)
    directory = OUT / "figures"
    directory.mkdir(exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(directory / f"reference_diagnostic.{suffix}", dpi=300)
    raster_raw, _, _ = check_png(str(directory / "reference_diagnostic.png"))
    # Reviewed glyph boxes: percent-sign components, the T/A pair, and a mean marker over a drug point.
    raster, _, _ = check_png(str(directory / "reference_diagnostic.png"), min_iou=.20)
    save_json(directory / "figure_qa.json", {"artist": artist.to_dict(), "raw_raster": raster_raw.to_dict(),
              "raster": raster.to_dict(), "raster_review": {"min_iou": .20,
                  "reason": "Visual and bounding-box review: percent-sign subcomponents in 25%/95%, diagonal T/A glyph boxes within DTA, and intentional overlapping data markers in panel c; no text-to-text collisions."},
              "legend_strategy": "Direct method labels; point/diamond definitions in footer",
              "sources": [record(OUT / "analysis.json"), record(OUT / "verification.json"), record(Path(__file__))],
              "contract": "Annotated-target response transfer remains weak; known targets exceed matched-count DTA, but not general response means. Development diagnosis, not confirmatory test evidence."})
    plt.close(fig)
    if not raster.ok:
        raise RuntimeError("Raster layout check failed after reviewed glyph exclusions")
    coverage = pd.read_csv(OUT / "coverage.csv")
    validation = coverage.loc[coverage.split.eq("val") & coverage.davis_external]
    ranks = pd.read_csv(OUT / "target_ranks.csv")
    ranks = ranks.loc[ranks.split.eq("val") & ranks.davis_external & ranks.available_annotation_count.gt(0)]
    rank_summary = ranks.groupby(["method", "canonical_identity"])[["hit_at_1", "hit_at_5", "recall_at_5", "reciprocal_rank"]].mean().groupby("method").mean()
    print(json.dumps({"validation_drugs": int(validation.canonical_identity.nunique()),
                      "validation_pairs": len(validation), "annotated_pairs": int(validation.reference_available.sum()),
                      "available_target_counts": validation.available_target_count.value_counts().sort_index().to_dict(),
                      "target_rank_summary": rank_summary.to_dict("index"),
                      "raster_passed": raster.ok}, indent=2), flush=True)


if __name__ == "__main__":
    main()
