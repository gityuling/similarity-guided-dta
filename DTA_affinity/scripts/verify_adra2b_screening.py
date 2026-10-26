#!/usr/bin/env python3
"""Independently replay ADRA2B neighbor arithmetic, fusion and screening metrics."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / 'reports/adra2b_screening_20261026'
RUN = PROJECT / 'results/adra2b_screening_20261026'
TOKENS = PROJECT / 'data/processed/adra2b_screening_20261026/tokens'
MODELS = ['deep', 'neighbor_prior', 'equal_blend', 'fixed_blend', 'similarity_gate', 'pure_similarity']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--replay-network', action='store_true')
    args = parser.parse_args()
    checks = {}
    for relative, expected in json.loads((RUN / 'frozen_inputs.json').read_text()).items():
        assert hashlib.sha256((PROJECT / relative).read_bytes()).hexdigest() == expected
    checks['frozen_inputs'] = True
    query = pd.read_csv(OUT / 'query_table.csv')
    data = pd.read_csv(OUT / 'training_table.csv')
    parts = pd.read_csv(OUT / 'partitions.csv')
    pred = pd.read_csv(OUT / 'predictions.csv')
    folds = pd.read_csv(OUT / 'fold_predictions.csv')
    labels = pd.read_csv(OUT / 'labels.csv')
    assert not set(data.parent_key) & set(query.parent_key)
    prep = json.loads((OUT / 'preparation.json').read_text())
    assert len(query) == len(labels) == prep['external_compounds']
    assert len(data) == prep['training_compounds']
    assert query.parent_key.is_unique and query.source_id.is_unique
    original = pd.read_csv(OUT / 'sources/pubchem_1961694_data.csv')
    original = original.loc[pd.to_numeric(original.PUBCHEM_RESULT_TAG, errors='coerce').notna()]
    original = original.set_index(original.PUBCHEM_RESULT_TAG.astype(int))
    for row in labels.itertuples():
        raw = original.loc[row.raw_result_tag]
        assert row.sid == int(raw.PUBCHEM_SID)
        v = float(raw['Standard Value'])
        assert raw['Standard Type'] == 'AC50' and raw['Standard Units'] == 'nM'
        assert row.AC50_nM == v and row.relation == raw['Standard Relation']
        expected = 'Active' if row.relation == '=' and v <= 10000 else 'Inactive'
        assert row.relation in ('=', '>')
        if row.relation == '>':
            assert v >= 10000
        assert row.outcome == expected
    records = pd.read_csv(OUT / 'training_record_audit.csv', keep_default_na=False)
    eligible = records.loc[records.exclusion_reason.eq('')]
    expected_labels = eligible.groupby('Canonical_SMILES').Ki_nM.apply(
        lambda v: float(np.median(9 - np.log10(pd.to_numeric(v)))))
    assert np.allclose(data.pKi, data.Canonical_SMILES.map(expected_labels), atol=1e-12)
    assert not eligible.document_id.eq('CHEMBL5291721').any()
    candidate_audit = pd.read_csv(OUT / 'candidate_audit.csv', keep_default_na=False)
    assert not set(data.parent_key) & set(candidate_audit.parent_key)
    checks['source_labels_and_parent_separation'] = True
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp = {s: generator.GetFingerprint(Chem.MolFromSmiles(s))
          for s in set(data.Canonical_SMILES) | set(query.Canonical_SMILES)}
    errors = []
    for fold in range(5):
        part = parts.loc[parts.fold.eq(fold)]
        train = data[data.row_id.isin(part.loc[part.split.eq('train'), 'row_id'])]
        val = data[data.row_id.isin(part.loc[part.split.eq('val'), 'row_id'])]
        assert not set(train.parent_key) & set(val.parent_key)
        saved = folds[folds.fold.eq(fold)].sort_values('row_id')
        selection = json.loads((RUN / f'fold_{fold}/selected.json').read_text())
        fps = [fp[s] for s in train.Canonical_SMILES]
        reconstructed, maxsim = [], []
        k, gamma = selection['neighbor']['k'], selection['neighbor']['gamma']
        for s in query.Canonical_SMILES:
            similarities = np.array(DataStructs.BulkTanimotoSimilarity(fp[s], fps))
            order = np.argsort(-similarities, kind='stable')[:k]
            weight = similarities[order] ** gamma
            value = np.dot(weight, train.pKi.to_numpy()[order]) / sum(weight) if sum(weight) else train.pKi.mean()
            reconstructed.append(value)
            maxsim.append(max(similarities))
        assert np.allclose(reconstructed, saved.neighbor_prior, atol=1e-11, rtol=0)
        assert np.allclose(maxsim, saved.pure_similarity, atol=1e-12, rtol=0)
        fit = selection['fusion']
        gate = fit['gate']
        w = 1 / (1 + np.exp(-(gate['a'] + gate['b'] * (np.array(maxsim) - gate['sim_mean']) / gate['sim_std'])))
        assert np.allclose(w, saved.deep_weight, atol=1e-12, rtol=0)
        d, n = saved.deep.to_numpy(), np.array(reconstructed)
        assert np.allclose(w*d+(1-w)*n, saved.similarity_gate, atol=1e-11, rtol=0)
        assert np.allclose((d+n)/2, saved.equal_blend, atol=1e-11, rtol=0)
        a = fit['fixed_deep_weight']
        assert np.allclose(a*d+(1-a)*n, saved.fixed_blend, atol=1e-11, rtol=0)
        v = pd.read_csv(RUN / f'fold_{fold}/validation_predictions.csv')
        delta = v.prediction_deep - v.prediction_neighbor_prior
        alpha = np.clip(np.dot(v.truth-v.prediction_neighbor_prior, delta) / np.dot(delta, delta), 0, 1)
        assert abs(alpha - a) < 1e-10
        if args.replay_network:
            import torch
            from torch.utils.data import DataLoader
            from train_nexusdti_regressor import NexusDTIAffinityRegressor, AffinityDataset, collate_fn, predict
            torch.set_num_threads(2)
            payload = torch.load(RUN / f'fold_{fold}/model.pt', map_location='cpu', weights_only=False)
            state = payload['state']
            model = NexusDTIAffinityRegressor(**selection['architecture'], pooled_mean=state['pooled_mean'].numpy(),
                                             pooled_std=state['pooled_std'].numpy())
            model.load_state_dict(state, strict=True)
            model.to('cuda').eval()
            ds = AffinityDataset(query.assign(placeholder=0), token_dir=TOKENS, target_column='placeholder')
            loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=collate_fn)
            _, neural = predict(model, loader, torch.device('cuda'), .15)
            neural = neural.astype(float)*selection['target_std']+selection['target_mean']
            error = float(np.max(abs(neural-d)))
            assert error < 1e-5, error
            errors.append(error)
            del model, payload
            torch.cuda.empty_cache()
    checks['five_fold_neighbor_and_fusion_replay'] = True
    average = folds.groupby('row_id')[MODELS].mean()
    assert np.allclose(average, pred.set_index('row_id')[MODELS], atol=1e-12, rtol=0)
    summary = pd.read_csv(OUT / 'summary.csv').set_index('model')
    joined = pred.merge(labels, on=['row_id', 'source_id'], validate='one_to_one')
    truth = joined.outcome.eq('Active').to_numpy()
    for model in MODELS:
        ordered = joined.sort_values([model, 'source_id'], ascending=[False, True])
        hits = int(ordered.head(10).outcome.eq('Active').sum())
        assert hits == summary.loc[model, 'top10_hits']
        s = joined[model].to_numpy()
        comparisons = s[truth, None] - s[~truth][None, :]
        auc = float(np.mean((comparisons > 0) + .5*(comparisons == 0)))
        grouped = joined.assign(positive=truth.astype(int)).groupby(model).positive.agg(['sum','count']).sort_index(ascending=False)
        precision = grouped['sum'].cumsum() / grouped['count'].cumsum()
        ap = float(np.sum(precision * grouped['sum']) / truth.sum())
        assert abs(auc-summary.loc[model, 'roc_auc']) < 1e-12
        assert abs(ap-summary.loc[model, 'average_precision']) < 1e-12
    checks['independent_ranking_metrics'] = True
    files = ['summary.csv', 'analysis.json', 'predictions.csv', 'fold_predictions.csv', 'labels.csv',
             'budget_curves.csv', 'fold_metrics.csv', 'preparation.json', 'candidate_scores.csv']
    result = dict(state='passed', checks=checks, network_replayed=args.replay_network,
                  network_max_abs_errors=errors,
                  files={n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in files})
    (OUT/'verification.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
