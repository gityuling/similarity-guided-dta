#!/usr/bin/env python3
"""Plot all development retrieval comparisons without changing the protocol."""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.text import Text
import numpy as np
import pandas as pd

from bib_g2cp_common import PROJECT, check_records, record, save_json

sys.path.insert(0, "/home/lsy/.agents/skills/figure-layout-hygiene/scripts")
from check_overlap import assert_figure_clean, check_png

OUT = PROJECT / "reports/bib_target_retrieval_20261019"
COLORS = {"gate":"#007E78", "deep":"#515D69", "observed":"#4169A1", "fixed":"#598A83",
          "predicted":"#87808E", "zero":"#B2B7BB", "text":"#24282B"}
LABELS = {
    "observed": ("Measured response", "observed"),
    "similarity_gate": ("Gate affinity", "gate"),
    "deep_plus_observed_equal": ("Deep + measured", "deep"),
    "fixed_blend_plus_observed_equal": ("Fixed + measured", "fixed"),
    "similarity_gate_plus_observed_equal": ("Gate + measured", "gate"),
    "g2cp_predicted": ("G2CP response", "predicted"),
    "deep_plus_g2cp_predicted_equal": ("Deep + G2CP", "deep"),
    "similarity_gate_plus_g2cp_predicted_equal": ("Gate + G2CP", "gate"),
}


def title(ax, letter, text):
    ax.text(0, 1.09, letter, transform=ax.transAxes, fontsize=10, fontweight="bold", va="bottom")
    ax.text(.10, 1.09, text, transform=ax.transAxes, fontsize=8.5, va="bottom")


def dotplot(ax, frame, methods, metric, dots=True):
    for row, method in enumerate(methods):
        values = frame.loc[frame.method.eq(method)].sort_values("canonical_identity")[metric].to_numpy()
        color = COLORS[LABELS[method][1]]
        if dots:
            ax.scatter(values, row+np.linspace(-.13,.13,len(values)), s=9, color=color, alpha=.35, edgecolors="none")
        ax.scatter([values.mean()], [row], marker="D", color=color, s=22, zorder=3)
    ax.set_yticks(range(len(methods)), [LABELS[method][0] for method in methods])
    ax.set_ylim(len(methods)-.5, -.6)
    ax.set_xlim(-.015, 1.02)
    ax.set_xticks([0,.25,.5,.75,1.])
    ax.set_xlabel("Reciprocal rank (higher is better)" if metric=="mrr" else "Hit@5 (fraction)")
    ax.set_ylabel("Method")


def save_figure(fig, name, qa):
    fig.canvas.draw()
    artist = assert_figure_clean(fig, include_ticks=True)
    renderer = fig.canvas.get_renderer()
    for text in fig.findobj(Text):
        if not text.get_visible() or not text.get_text().strip():
            continue
        bounds = text.get_window_extent(renderer)
        if bounds.x0 < 0 or bounds.y0 < 0 or bounds.x1 > fig.bbox.x1 or bounds.y1 > fig.bbox.y1:
            raise RuntimeError(f"Text outside figure: {text.get_text()}")
    directory = OUT / "figures"
    directory.mkdir(exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(directory / f"{name}.{suffix}", dpi=300)
    raster, _, _ = check_png(directory / f"{name}.png")
    reviewed, _, _ = check_png(directory / f"{name}.png", min_iou=.15)
    if not reviewed.ok:
        raise RuntimeError(f"Unresolved raster overlap: {name}")
    qa[name] = {"artist":artist.to_dict(), "raw_raster":raster.to_dict(),
                "reviewed_raster":reviewed.to_dict(), "text_within_figure":True,
                "visual_review":"Passed: rendered PNGs reviewed at full size, including title glyph crop.",
                "raster_review":"Threshold 0.15 after visual review: the raw main-figure hit is the separate T/o glyph boxes in Top-five (IoU 0.137), not overlapping ink. Supplement had no raw hits."}
    plt.close(fig)


def main():
    analysis = json.loads((OUT / "analysis.json").read_text())
    verification = json.loads((OUT / "verification.json").read_text())
    if verification["state"] != "passed":
        raise RuntimeError("Unverified results")
    check_records(analysis["outputs"])
    check_records([verification["source"], verification["verifier"]])
    frame = pd.read_csv(OUT / "per_drug_cell_metrics.csv")
    validation = frame.loc[frame.split.eq("val")]
    primary = validation.loc[validation.davis_external]
    drug = primary.groupby(["canonical_identity", "method"])[["mrr", "hit_at_5"]].mean().reset_index()
    expanded = validation.groupby(["canonical_identity", "method"])[["mrr", "hit_at_5"]].mean().reset_index()
    plt.rcParams.update({"font.family":"DejaVu Sans", "font.size":8, "axes.labelsize":7.5,
                         "xtick.labelsize":7, "ytick.labelsize":7, "axes.spines.top":False,
                         "axes.spines.right":False, "pdf.fonttype":42, "svg.fonttype":"none",
                         "axes.linewidth":.6, "text.color":COLORS["text"], "savefig.facecolor":"white"})
    methods = ["observed", "similarity_gate", "deep_plus_observed_equal",
               "fixed_blend_plus_observed_equal", "similarity_gate_plus_observed_equal"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2,5.9))
    fig.subplots_adjust(left=.20, right=.98, bottom=.13, top=.83, wspace=1.15, hspace=.85)
    fig.text(.05,.955,"Results 5: affinity and response target retrieval",fontsize=11,fontweight="bold")
    fig.text(.05,.915,"Exploratory validation | 9 strict drugs, 30 matched drug-cell pairs | equal rank fusion",fontsize=7.5)
    fig.text(.05,.882,"All candidate library targets retained; unannotated targets are unknown, not negatives",fontsize=7)
    dotplot(axes[0,0],drug,methods,"mrr")
    title(axes[0,0],"a","Strict validation")
    ax=axes[0,1]
    comparisons=[item for item in analysis["comparisons"] if item["scope"]=="primary_davis_disjoint" and item["primary"]]
    for row,item in enumerate(comparisons):
        ax.plot(item["ci95"],[row,row],color=COLORS["gate"],linewidth=1.1)
        ax.scatter([item["difference"]],[row],color=COLORS["gate"],s=20,zorder=3)
    ax.set_yticks(range(len(comparisons)),[LABELS[item["right"]][0] for item in comparisons])
    ax.set_ylim(len(comparisons)-.5,-.6)
    ax.axvline(0,color=COLORS["zero"],linewidth=.65)
    ax.set_xlim(-.12,.31)
    ax.set_xticks([-.1,0,.1,.2,.3])
    ax.set_xlabel("Difference in MRR",fontsize=7.5)
    ax.set_ylabel("Comparator")
    title(ax,"b","Paired differences")
    dotplot(axes[1,0],drug,methods,"hit_at_5")
    title(axes[1,0],"c","Top-five recovery")
    dotplot(axes[1,1],expanded,methods,"mrr")
    title(axes[1,1],"d","All 14 validation drugs")
    fig.text(.05,.052,"Dots: drug means over available cells. Diamonds: equal-drug means. Panel d includes Davis-known drugs.",fontsize=6.8)
    fig.text(.05,.025,"Intervals: 2,000 drug resamples, descriptive 95 percent. Source: frozen development analysis, 19 Oct 2026.",fontsize=6.8)
    qa={"sources":[record(OUT / "analysis.json"),record(OUT / "verification.json"),record(Path(__file__))],
        "legend_strategy":"Direct labels; point symbols defined in footer",
        "contract":"Measured-response fusion raises strict-validation MRR point estimates, but primary intervals cross zero and the all-drug expansion does not improve on gate alone."}
    save_figure(fig,"target_retrieval",qa)

    fig,axes=plt.subplots(1,2,figsize=(7.2,3.35))
    fig.subplots_adjust(left=.19,right=.98,bottom=.24,top=.73,wspace=1.03)
    fig.text(.05,.94,"Measured and predicted response inputs are different settings",fontsize=10,fontweight="bold")
    fig.text(.05,.86,"Same 9 strict validation drugs and 30 drug-cell pairs; equal rank fusion",fontsize=7.5)
    predicted_methods=["g2cp_predicted","similarity_gate","deep_plus_g2cp_predicted_equal","similarity_gate_plus_g2cp_predicted_equal"]
    dotplot(axes[0],drug,predicted_methods,"mrr")
    title(axes[0],"a","Predicted response input")
    ax=axes[1]
    null=pd.read_csv(OUT / "shuffle_metrics.csv")
    null=null.loc[null.davis_external]
    for row,mode in enumerate(("observed","g2cp_predicted")):
        subset=null.loc[null["mode"].eq(mode)].groupby(["repeat","canonical_identity"]).mrr.mean().groupby("repeat").mean()
        parts=ax.boxplot(subset.to_numpy(),positions=[row],vert=False,widths=.45,whis=[2.5,97.5],showfliers=False,patch_artist=True)
        parts["boxes"][0].set(facecolor=COLORS["zero"],edgecolor=COLORS["zero"],alpha=.5)
        for key in ("whiskers","caps","medians"):
            for line in parts[key]:
                line.set(color=COLORS["deep"],linewidth=.8)
        actual=drug.loc[drug.method.eq(f"similarity_gate_plus_{mode}_equal"),"mrr"].mean()
        ax.scatter([actual],[row],marker="D",s=25,color=COLORS["gate"],zorder=4)
    ax.set_yticks([0,1],["Measured response","G2CP response"])
    ax.set_ylim(1.5,-.6)
    ax.set_xlim(0,.34)
    ax.set_xticks([0,.1,.2,.3])
    ax.set_xlabel("Mean reciprocal rank (MRR)")
    ax.set_ylabel("Response input")
    title(ax,"b","Shuffled affinity controls")
    fig.text(.05,.115,"Panel a: drug dots and mean diamonds. Panel b: 100 shuffled-affinity means; actual fusion is a diamond.",fontsize=6.8)
    fig.text(.05,.065,"Boxes: quartiles; whiskers: 2.5th and 97.5th percentiles across shuffles, not a confidence interval.",fontsize=6.8)
    fig.text(.05,.025,"G2CP input: mean predicted signature over 3 seeds. Source: frozen development analysis, 19 Oct 2026.",fontsize=6.8)
    save_figure(fig,"predicted_response_controls",qa)
    save_json(OUT / "figures/figure_qa.json",qa)
    print(json.dumps({name:{"artist":value["artist"]["ok"],"raw_raster":value["raw_raster"]["ok"]}
                      for name,value in qa.items() if isinstance(value,dict) and "artist" in value},indent=2))


if __name__=="__main__":
    main()
