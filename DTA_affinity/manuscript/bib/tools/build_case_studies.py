#!/usr/bin/env python3
"""Build the two retrospective candidate panels from frozen predictions."""
from pathlib import Path
import hashlib
import json
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from check_overlap import assert_figure_clean

MS = Path(__file__).resolve().parents[1]
PROJECT = MS.parents[1]
ABL = PROJECT / "reports/bib_downstream_20261022"
ADRA = PROJECT / "reports/adra2b_screening_20261026"
COLORS = {"Active": "#007E78", "Inactive": "#687580",
          "Inconclusive": "#C68521", "rule": "#A6AEB5", "text": "#24323D",
          "guide": "#E4E8EB", "background": "#FFFFFF"}
MARKERS = {"Active": "o", "Inactive": "o", "Inconclusive": "D"}
METHODS = ["deep", "neighbor_prior", "fixed_blend", "similarity_gate",
           "equal_blend", "pure_similarity"]
NAMES = ["Deep", "Neighbor", "Constant blend", "SGF-DTA", "Equal blend", "Similarity only"]


def verified_inputs(folder, name):
    audit = json.loads((folder / name).read_text())
    assert audit["state"] == "passed"
    for filename, digest in audit["files"].items():
        assert hashlib.sha256((folder / filename).read_bytes()).hexdigest() == digest, filename
    return audit["files"]


def candidate_records():
    hashes = {"ABL1": verified_inputs(ABL, "evaluation_verification.json"),
              "ADRA2B": verified_inputs(ADRA, "verification.json")}
    p = pd.read_csv(ABL / "predictions.csv").merge(
        pd.read_csv(ABL / "query_table.csv")[["row_id", "task", "source_id"]], on="row_id")
    p = p.loc[p.task.eq("A")].sort_values(
        ["similarity_gate", "source_id"], ascending=[False, True]).head(10)
    p = p.merge(pd.read_csv(ABL / "labels.csv")[["row_id", "outcome", "strict_outcome"]], on="row_id")
    p["rank"] = np.arange(1, 11)
    p["sid"] = p.source_id.str.extract(r"_sid(\d+)").astype(int)
    source = pd.read_csv(ABL / "sources/pubchem_1963317_data.csv", low_memory=False)
    source["sid"] = pd.to_numeric(source.PUBCHEM_SID, errors="coerce")
    fields = ["PUBCHEM_CID"] + [f"Rep{i}ABLNHIBITKGP-Potency" for i in range(1, 5)]
    source = source.dropna(subset=["sid"])
    p = p.merge(source[["sid"] + fields], on="sid", validate="one_to_one")
    p["cid"] = pd.to_numeric(p.PUBCHEM_CID).astype(int)
    assert (p.outcome == "Active").sum() == 8
    assert p.strict_outcome.value_counts().to_dict() == {"Active": 5, "Inconclusive": 3, "Inactive": 2}
    # The first two compounds have four active replicate-level outcome codes.
    for sid in p.head(2).sid:
        row = source.loc[source.sid.eq(sid)].iloc[0]
        assert all(float(row[f"Rep{i}ABLNHIBITKGP-Outcome"]) == 2 for i in range(1, 5))
    a = pd.read_csv(ADRA / "candidate_scores.csv").sort_values(
        ["similarity_gate", "source_id"], ascending=[False, True]).head(10)
    names = pd.read_csv(ADRA / "top10_candidates.csv").query("model == 'similarity_gate'")
    a = a.merge(names[["source_id", "compound_name", "rank"]], on="source_id", validate="one_to_one").sort_values("rank")
    assert (a.outcome == "Active").sum() == 3
    assert a.loc[a.outcome.eq("Active"), "compound_name"].tolist() == ["rasagiline", "pergolide", "risperidone"]
    assert a["rank"].tolist() == list(range(1, 11))
    output = MS / "tables/source"
    output.mkdir(parents=True, exist_ok=True)
    p.to_csv(output / "abl1_case_top10.csv", index=False)
    a.to_csv(output / "adra2b_case_top10.csv", index=False)
    return p, a, hashes


def draw_cases(p, a, hashes):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8,
        "axes.labelsize": 8, "axes.titlesize": 10, "pdf.fonttype": 42,
        "svg.fonttype": "none", "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": .6, "text.color": COLORS["text"]})
    fig = plt.figure(figsize=(7.2, 3.55), facecolor=COLORS["background"])
    records = []
    for letter, title, x0, data in [("a", "ABL1", .02, p), ("b", "ADRA2B", .525, a)]:
        ax = fig.add_axes([x0, .15, .455, .72])
        ax.set_xlim(0, 1)
        ax.set_ylim(10.7, -.65)
        ax.axis("off")
        fig.text(x0, .93, letter, fontsize=11, weight="bold")
        fig.text(x0 + .035, .93, title, fontsize=10, weight="bold")
        # Identical column positions and row spacing in both candidate panels.
        columns = {"rank": .045, "candidate": .13, "activity": .60, "value": .835}
        for key, label, align in [("rank", "Rank", "center"),
                                  ("candidate", "Candidate", "left"),
                                  ("activity", "Activity", "center")]:
            ax.text(columns[key], -.05, label, ha=align, va="center", fontsize=8)
        metric = "Reported potency\nrange (μM)" if title == "ABL1" else "Experimental\nAC₅₀ (μM)"
        ax.text(columns["value"], -.05, metric, ha="center", va="center",
                fontsize=8, linespacing=1.2)
        ax.hlines(.45, 0, 1, color=COLORS["rule"], linewidth=.7)
        ax.hlines(10.5, 0, 1, color=COLORS["rule"], linewidth=.7)
        for row in data.itertuples():
            rank, status = row.rank, row.outcome
            ax.hlines(rank + .5, 0, 1, color=COLORS["guide"], linewidth=.4, zorder=0)
            candidate = f"SID {row.sid}" if title == "ABL1" else row.compound_name.capitalize()
            if title == "ABL1":
                source = data.loc[data["rank"].eq(rank)].iloc[0]
                values = [float(source[f"Rep{i}ABLNHIBITKGP-Potency"]) for i in range(1, 5)]
                low, high = min(values), max(values)
                value = f"{low:.4f}" if low == high else f"{low:.4f}-{high:.4f}"
                unresolved = row.strict_outcome == "Inconclusive"
            else:
                value = (">" if row.relation == ">" else "") + f"{row.AC50_nM / 1000:g}"
                unresolved = False
            ax.text(columns["rank"], rank, str(rank), ha="center", va="center")
            ax.text(columns["candidate"], rank, candidate, ha="left", va="center")
            ax.plot(columns["activity"], rank, marker=MARKERS[status], markersize=5.5,
                    markerfacecolor=COLORS["background"] if status == "Inactive" else COLORS[status],
                    markeredgecolor=COLORS[status],
                    markeredgewidth=1, linestyle="None")
            if unresolved:
                ax.text(columns["activity"] + .035, rank - .08, "†",
                        color=COLORS["Inconclusive"], ha="left", va="center", fontsize=9)
            ax.text(columns["value"], rank, value, ha="center", va="center", fontsize=7.8)
            records.append({"target": title, "rank": int(rank), "candidate": candidate,
                            "activity": status, "concentration": value,
                            "repeat_inconclusive": unresolved})
    handles = [Line2D([], [], marker=MARKERS[s], linestyle="None", markersize=5,
               markerfacecolor=COLORS["background"] if s == "Inactive" else COLORS[s],
               markeredgecolor=COLORS[s], label=s) for s in ["Active", "Inactive"]]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .072),
               ncol=2, frameon=False, handletextpad=.5, columnspacing=2)
    fig.text(.5, .035, "† ABL1 repeat classification: inconclusive.    > indicates a strict lower bound.",
             fontsize=7.5, ha="center", va="center")
    assert_figure_clean(fig, include_ticks=True)
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(MS / f"figures/candidate_case_studies.{ext}", dpi=300)
    plt.close(fig)
    (MS / "figures/candidate_case_studies_qa.json").write_text(json.dumps({
        "artist_check": "passed including ticks",
        "contract": "Two matched candidate panels: predicted rank, compound, activity and reported concentration.",
        "legend": "Shared activity legend; active filled, inactive open. Dagger marks inconclusive ABL1 repeat classification.",
        "displayed_records": records,
        "input_sha256": hashes,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    }, indent=2) + "\n")


def supplementary_cases(p, a):
    lines = [r"\subsection{Candidate identities and experimental support}",
        r"\begin{table}[htbp]\centering\small",
        r"\caption{The ten highest-ranked SGF-DTA ABL1 candidates. Potency ranges span all four deposited repeat values in $\mu$M; these values are shown with the corresponding activity classifications.}",
        r"\begin{tabular}{rrrrll}\hline",
        r"Rank & PubChem CID & PubChem SID & Potency range & Deposited & Repeat \\ \hline"]
    for row in p.to_dict("records"):
        values = [float(row[f"Rep{i}ABLNHIBITKGP-Potency"]) for i in range(1, 5)]
        label = lambda x: "Unresolved" if x == "Inconclusive" else x
        lines.append(f'{row["rank"]} & {row["cid"]} & {row["sid"]} & {min(values):.4f}--{max(values):.4f} & '
                     f'{label(row["outcome"])} & {label(row["strict_outcome"])} '+r"\\")
    lines += [r"\hline\end{tabular}\end{table}", r"\FloatBarrier",
        r"\subsection{ADRA2B data and implementation}",
        r"Human ADRA2B ($\alpha_{2B}$-adrenergic receptor; UniProt P18089) references were obtained from ChEMBL target CHEMBL1942\citep{chembl2024}. "
        r"Human binding assays with target confidence 9, literature provenance, no annotated variant and exact positive $K_i$ values "
        r"provided 314 retained measurements from 67 documents and 69 assays. Median $pK_i$ per standardized stereochemical "
        r"structure yielded 294 compounds in 272 connectivity groups. All parent structures in the external source, including "
        r"excluded candidate records, and source document CHEMBL5291721 were removed from training.",
        r"The external binding panel is PubChem AID 1961694 / ChEMBL CHEMBL5291737 from Sutherland et al.\citep{sutherland2023spd,pubchem2025} "
        r"The authors' Table S2 (Zenodo 7378746) identifies preferred assay 3159 as human ADRA2B binding by scintillation proximity assay. "
        r"This source annotation establishes the assay target; ChEMBL retains a homologous-target mapping flag for this record. "
        r"Of 1,220 activity records for 1,209 source compounds, 37 repeated-parent records and six records without usable structures "
        r"were excluded, yielding 1,177 unique candidate parents. Activity is assigned from the numerical AC$_{50}$ and relation sign: "
        r"exact values $\leq10$ $\mu$M are active, exact values $>10$ $\mu$M and strict lower bounds $\geq10$ $\mu$M are inactive. "
        r"Threshold-ambiguous bounds are excluded. The final pool contains 160 active and 1,017 inactive candidates.",
        r"The full affinity network was fitted to ADRA2B $pK_i$ using fixed Uni-Mol2 and ESM-2 features. Five parent-group "
        r"validation partitions (partition seed 0) used training seed 20261012. The latent dimension was 512, dropout 0.1, "
        r"batch size 64, learning rate $10^{-4}$ and weight decay $10^{-4}$. Training used up to 20 epochs with patience 5. "
        r"Neighbor counts $\{1,3,5,10,20,40\}$ and similarity exponents $\{0.5,1,2,4\}$ were selected on validation data; "
        r"the gate slope penalty was 0.001. Each partition independently selected the checkpoint and fusion parameters. "
        r"Five-model mean scores determined the external ranking, with source identifiers resolving ties. External AC$_{50}$ "
        r"measurements were used only in evaluation. Saved checkpoints reproduced all five sets of neural predictions to "
        r"a maximum absolute difference below $10^{-12}$; independent replay verified neighbor estimates, fusion and ranking metrics.",
        r"\begin{table}[htbp]\centering\small",
        r"\caption{The ten highest-ranked SGF-DTA ADRA2B candidates. Concentrations are experimental AC$_{50}$ values in $\mu$M; $>$ denotes a strict lower bound.}",
        r"\begin{tabular}{rlrrl}\hline",
        r"Rank & Compound & PubChem CID & AC$_{50}$ & Outcome \\ \hline"]
    for r in a.itertuples():
        value = f'{r.AC50_nM / 1000:g}'
        if r.relation == ">": value = "$>$" + value
        lines.append(f'{r.rank} & {r.compound_name.capitalize()} & {int(r.cid)} & {value} & {r.outcome} '+r"\\")
    lines += [r"\hline\end{tabular}\end{table}", r"\FloatBarrier",
        r"\subsection{Complete exploratory non-kinase comparisons}",
        r"Non-kinase cases were explored in the order ESR1, ADORA2A and ADRA2B. ADRA2B was selected before its predictions "
        r"were generated for its larger direct-binding panel and available $K_i$ references. The settings established "
        r"for ESR1 were carried into the subsequent cases. Each case fits target-specific affinity models and holds "
        r"candidate parent structures out of training. The ESR1 and ADORA2A reports, configurations, candidate records "
        r"and complete rankings remain in the reproducibility materials. These exploratory cases use different panels "
        r"and endpoints; their retrieval rates are reported within each case.",
        r"ESR1 uses ChEMBL target CHEMBL206 $K_i$ references and the deposited outcomes of the competitive-binding "
        r"confirmation assay PubChem AID 1223. ADORA2A uses ChEMBL target CHEMBL251 $K_i$ references and the "
        r"Secondary Pharmacology Database binding panel PubChem AID 1961683, with the same 10 $\mu$M AC$_{50}$ "
        r"cutoff as ADRA2B. The corresponding reference sets contain 360 and 4,126 compounds after feature preparation.",
        r"\begin{table}[htbp]\centering\small",
        r"\caption{All methods in the exploratory non-kinase screens. Candidate pools contain 63 compounds (17 active) for ESR1, 1,170 (15 active) for ADORA2A and 1,177 (160 active) for ADRA2B.}",
        r"\begin{tabular}{llrrr}\hline",
        r"Target & Method & Top-10 hits & AP & ROC-AUC \\ \hline"]
    for name, folder in [("ESR1", "nonkinase_screening_20261026"),
                         ("ADORA2A", "adora2a_screening_20261026"),
                         ("ADRA2B", "adra2b_screening_20261026")]:
        summary = pd.read_csv(PROJECT / "reports" / folder / "summary.csv").set_index("model")
        for method, label in zip(METHODS, NAMES):
            r = summary.loc[method]
            lines.append(f"{name} & {label} & {int(r.top10_hits)} & {r.average_precision:.4f} & {r.roc_auc:.4f} "+r"\\")
    lines += [r"\hline\end{tabular}\end{table}", r"\FloatBarrier",
        "In the ADRA2B panel, structure-only ranking retrieved eight active compounds, "
        "and the neighbor and constant-blend methods each retrieved three, as did SGF-DTA. "
        "The main-text case therefore illustrates candidate selection and experimental support; "
        "comparative performance is documented by the complete results above."]
    (MS / "sections/candidate_case_supplement.tex").write_text("\n\n".join(lines) + "\n")


def main():
    for folder in ["figures", "sections"]:
        (MS / folder).mkdir(parents=True, exist_ok=True)
    p, a, hashes = candidate_records()
    draw_cases(p, a, hashes)
    supplementary_cases(p, a)
    print("Generated verified ABL1 and ADRA2B case studies.")


if __name__ == "__main__":
    main()
