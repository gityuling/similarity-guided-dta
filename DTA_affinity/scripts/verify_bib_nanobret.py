#!/usr/bin/env python3
"""Independently check source cells, exclusion rules and saved model comparisons."""

import json

import numpy as np
import openpyxl
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score

from bib_common import PROJECT, ROOT, record, sha256, write_once
from prepare_bib_nanobret import parent_key, canonical


OUT = PROJECT / "reports/bib_nanobret_20261018"
RUN = PROJECT / "results/bib_nanobret_20261019"
MODELS = ("deep", "neighbor_prior", "fixed_blend", "similarity_gate")


def read(name):
    return pd.read_csv(OUT / name, float_precision="round_trip")


def close(actual, expected, tol=1e-11):
    if not np.allclose(actual, expected, atol=tol, rtol=0, equal_nan=True):
        raise AssertionError(f"Numerical mismatch, max error {np.nanmax(np.abs(np.asarray(actual) - expected))}")


def main():
    frozen = json.loads((OUT / "freeze.json").read_text())["files"]
    frozen += json.loads((RUN / "code_freeze.json").read_text())
    for item in frozen:
        assert sha256(ROOT / item["path"]) == item["sha256"], item["path"]
    measurements, query = read("measurements.csv"), read("query_table.csv")
    folds, averaged = read("fold_predictions.csv"), read("predictions.csv")
    scores, metrics = read("scores.csv"), read("per_compound_metrics.csv")
    plan = json.loads((OUT / "plan.json").read_text())
    report = json.loads((OUT / "analysis.json").read_text())
    source = OUT / "sources/PMC12951567/jm5c02916_si_004.xlsx"
    workbook = openpyxl.load_workbook(source, read_only=False, data_only=True)
    for row in measurements.itertuples(index=False):
        sheet = workbook[row.source_sheet]
        assert str(sheet[row.source_target_cell].value).strip() == row.target_name
        raw = str(sheet[row.source_value_cell].value).strip()
        value = float(raw[1:] if raw.startswith((">", "<")) else raw)
        close(row.value, value)
        score = 9 - np.log10(value) if row.endpoint == "ic50_nm" else value
        close(row.endpoint_score, score)
        assert bool(row.active) == (value < 1000 if row.endpoint == "ic50_nm" else value > 50)
    workbook.close()
    assert set(query.measurement_id) == set(measurements.loc[measurements.included, "measurement_id"])
    assert not query.duplicated(["compound", "protein_id"]).any()
    assert not folds.duplicated(["row_id", "fold"]).any()
    assert set(scores.row_id) == set(query.row_id) == set(averaged.row_id)
    table = pd.read_csv(PROJECT / "data/processed/kiba_dtiam_protocol/table.csv")
    drugs = table[["drug_id", "Canonical_SMILES"]].drop_duplicates()
    drugs["identity"] = drugs.Canonical_SMILES.map(canonical)
    drugs["parent"] = drugs.Canonical_SMILES.map(parent_key)
    for compound, group in query.groupby("compound"):
        sm = group.Canonical_SMILES.iloc[0]
        matching = set(drugs.loc[drugs.identity.eq(canonical(sm)) | drugs.parent.eq(parent_key(sm)), "drug_id"])
        assert bool(group.kiba_external.iloc[0]) == (len(matching) == 0)
        for fold in map(int, str(group.eligible_folds.iloc[0]).split("|")):
            manifest = pd.read_csv(PROJECT / f"data/processed/kiba_identity_20261017/folds/drug_coldstart/fold_{fold}.csv")
            joined = table.merge(manifest, on="row_id", validate="one_to_one")
            assert not matching.intersection(joined.loc[joined.split.isin(["train", "val"]), "drug_id"])
            assert set(group.Sequence) <= set(joined.loc[joined.split.eq("train"), "Sequence"])
    for fold, group in folds.groupby("fold"):
        parameters = json.loads((PROJECT / f"results/bib_kiba_identity_f{fold}_seed20261012_ablation.json").read_text())
        a = parameters["fixed_deep_weight"]
        close(group.fixed_blend, a * group.deep + (1 - a) * group.neighbor_prior)
        gate = parameters["gate"]
        weight = expit(gate["a"] + gate["b"] * (group.max_train_drug_tanimoto - gate["sim_mean"]) / gate["sim_std"])
        close(group.deep_weight, weight)
        close(group.similarity_gate, weight * group.deep + (1 - weight) * group.neighbor_prior)
    means = folds.groupby("row_id")[list(MODELS)].mean().reindex(averaged.row_id)
    close(averaged[list(MODELS)].to_numpy(), means.to_numpy())
    close(averaged.fold_count, folds.groupby("row_id").size().reindex(averaged.row_id).to_numpy())
    expected_count = query.eligible_folds.astype(str).map(lambda x: len(x.split("|")))
    close(expected_count, averaged.set_index("row_id").loc[query.row_id, "fold_count"].to_numpy())
    max_error = 0.0
    for row in metrics.itertuples(index=False):
        group = scores.loc[scores.compound.eq(row.compound)].copy()
        if row.variant == "occupancy_clipped":
            group["endpoint_score"] = group.endpoint_score.clip(0, 100)
        if row.variant == "ic50_interior_only":
            group = group.loc[~group.reported_censor & ~group.possible_assay_boundary]
        assert len(group) == row.pairs
        assert int(group.active.sum()) == row.active_pairs
        if len(group) >= plan["minimum_pairs_per_compound"] and group.endpoint_score.nunique() > 1 and group[row.model].nunique() > 1:
            rho = float(np.corrcoef(rankdata(group.endpoint_score), rankdata(group[row.model]))[0, 1])
            close(rho, row.spearman)
            max_error = max(max_error, abs(rho - row.spearman))
        else:
            assert np.isnan(row.spearman)
        if len(group) >= plan["minimum_pairs_per_compound"] and group.active.nunique() == 2:
            close(roc_auc_score(group.active, group[row.model]), row.auroc)
            close(average_precision_score(group.active, group[row.model]), row.ap)
        else:
            assert np.isnan(row.auroc) and np.isnan(row.ap)
        top = group.sort_values([row.model, "protein_id"], ascending=[False, True], kind="stable").head(10)
        assert int(top.active.sum()) == row.top10_hits
        if group.active.any():
            close(top.active.mean() / group.active.mean(), row.ef10)
    for item in report["macro_metrics"] + report["paired_differences"]:
        selected = metrics.loc[metrics.variant.eq(item["variant"]) & metrics.endpoint.eq(item["endpoint"])]
        if item["stratum"] == "davis_disjoint":
            selected = selected.loc[selected.davis_external]
        assert item["stratum"] in ("all", "davis_disjoint")
        wide = selected.pivot(index="compound", columns="model", values=item["metric"]).dropna()
        assert len(wide) == item["compounds"]
        if "macro_mean" in item:
            close(wide[item["model"]].mean(), item["macro_mean"])
        else:
            comparator = item["comparison"].split(" - ")[1]
            delta = (wide.similarity_gate - wide[comparator]).to_numpy()
            rng = np.random.default_rng(plan["bootstrap_seed"])
            samples = rng.choice(delta, size=(plan["bootstrap_n"], len(delta)), replace=True).mean(axis=1)
            close(delta.mean(), item["delta"])
            close(np.quantile(samples, [0.025, 0.975]), item["ci95"])
    cache = PROJECT / "data/processed/bib_nanobret_20261019/tokens/build_audit.json"
    cache_audit = json.loads(cache.read_text())
    assert cache_audit["complete"]
    for copied in cache_audit["copied"]:
        from pathlib import Path
        assert sha256(Path(copied["source"])) == sha256(Path(copied["destination"])) == copied["sha256"]
    result = {"passed": True, "frozen_files_checked": len(frozen), "source_measurements_checked": len(measurements),
              "included_queries_checked": len(query), "fold_predictions_checked": len(folds),
              "metric_rows_checked": len(metrics), "maximum_spearman_recalculation_error": max_error,
              "copied_feature_files_verified": len(cache_audit["copied"]),
              "checks": ["source values and endpoint directions", "training and validation parent overlap", "same-target support", "original fusion parameters", "fold averaging and coverage", "all per-compound metrics", "macro statistics and compound bootstrap intervals", "encoder cache copy integrity"],
              "artifacts": [record(OUT / n) for n in ("scores.csv", "per_compound_metrics.csv", "analysis.json")],
              "verification_code": record(PROJECT / "scripts/verify_bib_nanobret.py")}
    write_once(OUT / "verification.json", json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
