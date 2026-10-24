"""Read existing DTA artifacts and reproduce the submission-readiness audit.

No models are trained and no existing experiment outputs are overwritten.
The fixed-weight blend is a post-hoc diagnostic fitted on validation only.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.optimize import minimize
from scipy.special import expit, logit


PROJECT = Path(__file__).resolve().parents[1]
RESULTS = PROJECT / "results"
FAMILIES = {
    "legacy_deep": "nexusdti_regressor_dtiam_drugcold_f{fold}_hybrid_e12",
    "legacy_gated": "dtiam_drugcold_f{fold}_similarity_gated_ensemble",
    "neighbor_only": "dtiam_drugcold_f{fold}_morgan_neighbor_prior",
    "scratch_deep": "nexusdti_regressor_dtiam_drugcold_f{fold}_scratch_b64_lr1e4_e20",
    "scratch_gated": "dtiam_drugcold_f{fold}_scratch_b64_lr1e4_e20_similarity_gated_ensemble",
    "transfer_deep": "nexusdti_regressor_dtiam_drugcold_f{fold}_bindingdb_kd_e10_transfer_b64_lr1e4_e20",
    "transfer_gated": "dtiam_drugcold_f{fold}_bindingdb_kd_similarity_gated_ensemble",
}


def stem_for(family, fold):
    stem = FAMILIES[family].format(fold=fold)
    if family == "scratch_gated" and fold == 0 and not (RESULTS / f"{stem}.json").exists():
        return "dtiam_drugcold_f0_scratch_b64_similarity_gated_ensemble"
    return stem


def aggregate(values):
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()),
        "sd_across_folds": float(array.std(ddof=1)),
        "median": float(np.median(array)),
        "per_fold": array.tolist(),
    }


def mse(frame, column):
    return float(np.mean((frame.pKd_true - frame[column]) ** 2))


def canonical(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Invalid SMILES in audit input")
    return Chem.MolToSmiles(molecule)


def main():
    table = pd.read_csv(PROJECT / "data/processed/davis_dtiam_protocol/table.csv")
    table["canonical"] = table.Canonical_SMILES.map(canonical)
    assert table.row_id.is_unique
    summaries = {}
    input_paths = []
    for family in FAMILIES:
        metrics = []
        for fold in range(5):
            path = RESULTS / f"{stem_for(family, fold)}.json"
            data = json.loads(path.read_text())
            metrics.append(data["metrics_test"] if "metrics_test" in data else data["test"]["raw"])
            input_paths.append(str(path.relative_to(PROJECT)))
        summaries[family] = {
            metric: aggregate([run[metric] for run in metrics])
            for metric in ("mse", "rmse", "ci", "pearson", "spearman")
        }

    prediction_sets = {}
    fixed_blends = {}
    for family in ("legacy_gated", "scratch_gated", "transfer_gated"):
        test_frames, fixed_rows = [], []
        for fold in range(5):
            path = RESULTS / "predictions" / f"{stem_for(family, fold)}_predictions.csv"
            predictions = pd.read_csv(path)
            input_paths.append(str(path.relative_to(PROJECT)))
            assert not predictions.duplicated(["split", "row_id"]).any()
            manifest = pd.read_csv(PROJECT / f"data/processed/davis_dtiam_protocol/folds/drug_coldstart/fold_{fold}.csv")
            expected = manifest.loc[manifest.split.isin(["val", "test"]), ["row_id", "split"]]
            joined = expected.merge(predictions, on=["row_id", "split"], how="outer", validate="one_to_one", indicator=True)
            assert joined._merge.eq("both").all()
            joined = joined.drop(columns="_merge").merge(
                table[["row_id", "drug_idx", "protein_id", "is_censored_floor", "pKd_floor"]],
                on="row_id", validate="one_to_one",
            )
            assert np.allclose(joined.pKd_true, joined.pKd_floor, atol=1e-6, rtol=0)
            assert np.isfinite(joined[["prediction_deep", "prediction_prior", "prediction_gated_raw"]]).all().all()
            test = joined.loc[joined.split == "test"].copy()
            val = joined.loc[joined.split == "val"]
            assert np.isclose(mse(test, "prediction_gated_raw"), summaries[family]["mse"]["per_fold"][fold], atol=1e-7)
            direction = val.prediction_deep - val.prediction_prior
            denominator = float(np.dot(direction, direction))
            alpha = float(np.clip(np.dot(direction, val.pKd_true - val.prediction_prior) / denominator, 0, 1)) if denominator else 0.0
            test["prediction_fixed"] = alpha * test.prediction_deep + (1 - alpha) * test.prediction_prior
            test["fold"] = fold
            fixed_rows.append({"fold": fold, "validation_fitted_deep_weight": alpha, "test_mse": mse(test, "prediction_fixed")})
            test_frames.append(test)
        out_of_fold = pd.concat(test_frames, ignore_index=True).sort_values("row_id")
        assert len(out_of_fold) == len(table) and out_of_fold.row_id.is_unique
        prediction_sets[family] = out_of_fold
        fixed_blends[family] = {
            "protocol": "One scalar deep weight in [0,1], fitted on each fold's validation rows only; post-hoc diagnostic.",
            "mse": aggregate([row["test_mse"] for row in fixed_rows]),
            "per_fold": fixed_rows,
        }

    split_audit = {}
    for setting, identity in (("drug_coldstart", "canonical"), ("protein_coldstart", "Sequence")):
        rows = []
        for fold in range(5):
            manifest = pd.read_csv(PROJECT / f"data/processed/davis_dtiam_protocol/folds/{setting}/fold_{fold}.csv")
            frame = manifest[["row_id", "split"]].merge(table[["row_id", identity]], on="row_id", validate="one_to_one")
            train = set(frame.loc[frame.split == "train", identity])
            val = set(frame.loc[frame.split == "val", identity])
            test = frame.loc[frame.split == "test"]
            affected = int(test[identity].isin(train).sum())
            rows.append({
                "fold": fold, "train_test_shared_exact_identities": len(train & set(test[identity])),
                "train_val_shared_exact_identities": len(train & val),
                "affected_test_rows": affected, "test_rows": len(test),
                "affected_test_fraction": affected / len(test),
            })
        split_audit[setting] = rows

    scratch = prediction_sets["scratch_gated"]
    transfer = prediction_sets["transfer_gated"]
    assert np.array_equal(scratch.row_id, transfer.row_id)
    prior_drift, common_prior_frames, common_prior_mse = [], [], []
    for fold in range(5):
        scratch_path = RESULTS / "predictions" / f"{stem_for('scratch_gated', fold)}_predictions.csv"
        transfer_path = RESULTS / "predictions" / f"{stem_for('transfer_gated', fold)}_predictions.csv"
        a, b = pd.read_csv(scratch_path), pd.read_csv(transfer_path)
        joined = a.merge(b, on=["row_id", "split"], validate="one_to_one", suffixes=("_scratch", "_transfer"))
        prior_drift.append({
            "fold": fold,
            "max_absolute_prior_difference": float(np.abs(joined.prediction_prior_scratch - joined.prediction_prior_transfer).max()),
            "max_absolute_similarity_difference": float(np.abs(joined.max_train_drug_tanimoto_scratch - joined.max_train_drug_tanimoto_transfer).max()),
        })
        shared = a[["row_id", "split", "pKd_true", "prediction_prior", "max_train_drug_tanimoto"]].merge(
            b[["row_id", "split", "prediction_deep"]], on=["row_id", "split"], validate="one_to_one",
        )
        val = shared.loc[shared.split == "val"]
        test = shared.loc[shared.split == "test"].copy()
        sim_mean, sim_std = val.max_train_drug_tanimoto.mean(), val.max_train_drug_tanimoto.std()

        def predict_gate(params, frame):
            z = (frame.max_train_drug_tanimoto.to_numpy() - sim_mean) / sim_std
            weight = expit(params[0] + params[1] * z)
            return weight * frame.prediction_deep.to_numpy() + (1 - weight) * frame.prediction_prior.to_numpy()

        def objective(params):
            return float(np.mean((val.pKd_true.to_numpy() - predict_gate(params, val)) ** 2) + 1e-3 * params[1] ** 2)

        optimized = minimize(objective, np.array([logit(0.335), 0.0]), method="L-BFGS-B")
        if not optimized.success:
            raise RuntimeError(optimized.message)
        test["prediction_gated_raw"] = predict_gate(optimized.x, test)
        common_prior_mse.append(mse(test, "prediction_gated_raw"))
        common_prior_frames.append(test)
    recomputed_transfer = pd.concat(common_prior_frames).sort_values("row_id").merge(
        table[["row_id", "drug_idx"]], on="row_id", validate="one_to_one",
    ).sort_values("row_id")
    assert np.array_equal(scratch.row_id, recomputed_transfer.row_id)
    delta = recomputed_transfer.copy()
    delta["delta_squared_error"] = (
        (recomputed_transfer.pKd_true.to_numpy() - recomputed_transfer.prediction_gated_raw.to_numpy()) ** 2
        - (scratch.pKd_true.to_numpy() - scratch.prediction_gated_raw.to_numpy()) ** 2
    )
    drug_deltas = delta.groupby("drug_idx").delta_squared_error.mean().to_numpy()
    rng = np.random.default_rng(20261016)
    sampled = rng.choice(drug_deltas, size=(20000, len(drug_deltas)), replace=True).mean(axis=1)
    paired_fold_deltas = np.asarray(summaries["transfer_gated"]["mse"]["per_fold"]) - summaries["scratch_gated"]["mse"]["per_fold"]
    common_prior_deltas = np.asarray(common_prior_mse) - summaries["scratch_gated"]["mse"]["per_fold"]

    strata = []
    for label, subset in (
        ("all", scratch),
        ("pKd_floor_5", scratch.loc[scratch.is_censored_floor]),
        ("pKd_above_5", scratch.loc[~scratch.is_censored_floor]),
        ("max_Tanimoto_below_0.3", scratch.loc[scratch.max_train_drug_tanimoto < 0.3]),
        ("max_Tanimoto_0.3_to_0.5", scratch.loc[scratch.max_train_drug_tanimoto.between(0.3, 0.5, inclusive="left")]),
        ("max_Tanimoto_0.5_to_0.7", scratch.loc[scratch.max_train_drug_tanimoto.between(0.5, 0.7, inclusive="left")]),
        ("max_Tanimoto_at_least_0.7", scratch.loc[scratch.max_train_drug_tanimoto >= 0.7]),
    ):
        if subset.empty:
            continue
        strata.append({
            "stratum": label, "pairs": len(subset), "drugs": int(subset.drug_idx.nunique()),
            "mse_pair_pooled": {name: mse(subset, column) for name, column in (
                ("deep", "prediction_deep"), ("prior", "prediction_prior"),
                ("fixed_blend", "prediction_fixed"), ("similarity_gate", "prediction_gated_raw"),
            )},
        })

    source_audits = []
    for fold in range(5):
        path = RESULTS / f"nexusdti_bindingdb_kd_strict_f{fold}_e10_b64.json"
        data = json.loads(path.read_text())
        source = pd.read_csv(data["audit"]["source_table"])
        manifest = pd.read_csv(PROJECT / f"data/processed/davis_dtiam_protocol/folds/drug_coldstart/fold_{fold}.csv")
        held = set(table.loc[table.row_id.isin(manifest.loc[manifest.split != "train", "row_id"]), "canonical"])
        overlap = set(source.canonical_smiles.map(canonical)) & held
        assert not overlap
        source_audits.append({"fold": fold, "exact_held_drug_overlap": len(overlap), **data["audit"]})

    collisions = table.groupby(["canonical", "Sequence"]).pKd_floor.nunique()
    report = {
        "date": "2026-10-16",
        "scope": "Read-only assessment of existing experiments; no retraining. New fixed-blend and subgroup analyses are exploratory.",
        "aggregation": "Primary summaries are unweighted means across five folds, SD uses ddof=1; subgroup metrics pool out-of-fold pairs.",
        "five_fold_metrics": summaries,
        "existing_artifact_transfer_comparison": {
            "caveat": "Architecture and training settings match, but prior predictions differ on folds 1-4. This does not isolate pretraining.",
            "mse_relative_reduction": 1 - summaries["transfer_gated"]["mse"]["mean"] / summaries["scratch_gated"]["mse"]["mean"],
            "transfer_wins_folds": int((paired_fold_deltas < 0).sum()),
            "scratch_wins_folds": int((paired_fold_deltas > 0).sum()),
            "delta_transfer_minus_scratch": aggregate(paired_fold_deltas),
            "prior_prediction_drift": prior_drift,
        },
        "common_prior_transfer_comparison": {
            "protocol": "Use the same scratch prior for both deep branches; refit the identical similarity gate on each fold's validation rows. Exploratory post-hoc recomputation.",
            "recomputed_transfer_mse": aggregate(common_prior_mse),
            "mse_relative_reduction": 1 - float(np.mean(common_prior_mse)) / summaries["scratch_gated"]["mse"]["mean"],
            "transfer_wins_folds": int((common_prior_deltas < 0).sum()),
            "scratch_wins_folds": int((common_prior_deltas > 0).sum()),
            "delta_transfer_minus_scratch": aggregate(common_prior_deltas),
            "drug_cluster_bootstrap": {
                "n_drugs": len(drug_deltas), "replicates": 20000,
                "delta_mse_equal_drug_mean": float(drug_deltas.mean()),
                "percentile_95_interval": np.quantile(sampled, [0.025, 0.975]).tolist(),
                "limitation": "Conditional on these fitted models, observed target panel and development folds; ignores retraining and overlapping training-set uncertainty. Not confirmatory inference.",
            },
        },
        "fixed_blend_diagnostic": fixed_blends,
        "scratch_gated_subgroups": strata,
        "data_audit": {
            "pairs": len(table), "drug_ids": int(table.drug_id.nunique()),
            "unique_canonical_smiles": int(table.canonical.nunique()),
            "target_ids": int(table.protein_id.nunique()), "unique_sequences": int(table.Sequence.nunique()),
            "floor_pairs": int(table.is_censored_floor.sum()), "floor_fraction": float(table.is_censored_floor.mean()),
            "same_smiles_sequence_groups_with_multiple_labels": int((collisions > 1).sum()),
            "split_identity_audit": split_audit,
        },
        "source_pretraining_audits": source_audits,
        "input_files": input_paths,
        "known_reporting_issues": [
            "The matched-summary script expects a fold-0 filename that differs from the existing artifact. This audit explicitly resolves that alias.",
            "Earlier summary compares batch-4 scratch to batch-64 transfer, and predates completed matched scratch folds 1-4.",
            "Historical summary states calibration was excluded based on held-out fold performance; freeze the analysis protocol before a new confirmation dataset.",
            "Scratch and transfer artifacts use different neighbor predictions on folds 1-4. Common-prior recomputation is included, but the cause of drift is not established.",
        ],
    }
    output = Path(__file__).with_name("ijms_readiness_audit_20261016.json")
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({key: report[key] for key in (
        "existing_artifact_transfer_comparison", "common_prior_transfer_comparison", "fixed_blend_diagnostic", "scratch_gated_subgroups", "data_audit"
    )}, indent=2))
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
