#!/usr/bin/env python3
"""Recalculate shared-backbone, ADRA2B and target-wise results without training."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import pearsonr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
P = ROOT / 'DTA_affinity'


def frame(path):
    return pd.read_csv(P / path, float_precision='round_trip')


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=0)


def shared_backbone():
    summary = frame('reports/knn_shared_20261025/summary.csv')
    values = []
    for dataset in ['davis', 'kiba']:
        for fold in range(5):
            labels = frame(f'reports/knn_comparison_20261025/inputs/{dataset}/fold_{fold}/test_labels.csv')
            for seed in [20261012, 20261017, 20261018]:
                data = frame(f'results/knn_shared_20261025/{dataset}/fold_{fold}/seed_{seed}/test_predictions.csv')
                assert data.row_id.is_unique and set(data.row_id) == set(labels.row_id)
                close(data.set_index('row_id').affinity.sort_index(), labels.set_index('row_id').affinity.sort_index())
                for method in summary.loc[summary.dataset.eq(dataset), 'method']:
                    values.append(dict(dataset=dataset, fold=fold, seed=seed, method=method,
                        mse=float(np.mean((data.affinity-data[method])**2)),
                        pearson=float(pearsonr(data.affinity, data[method]).statistic)))
    folds = pd.DataFrame(values).groupby(['dataset', 'method', 'fold'])[['mse', 'pearson']].mean()
    for row in summary.itertuples():
        part = folds.loc[row.dataset, row.method]
        for metric in ['mse', 'pearson']:
            close(part[metric].mean(), getattr(row, metric+'_mean'))
            close(part[metric].std(ddof=1), getattr(row, metric+'_fold_sd'))
    return dict(check='shared_backbone', runs=30, methods=sorted(summary.method.unique()))


def adra2b():
    folder = 'reports/adra2b_screening_20261026'
    # Preserve the CSV parsing used by the original ranking analysis, including ties.
    read_case = lambda name: pd.read_csv(P/folder/name)
    train, query = read_case('training_table.csv'), read_case('query_table.csv')
    assert not set(train.parent_key) & set(query.parent_key)
    predictions, folds = read_case('predictions.csv'), read_case('fold_predictions.csv')
    methods = ['deep', 'neighbor_prior', 'equal_blend', 'fixed_blend', 'similarity_gate', 'pure_similarity']
    for fold, rows in folds.groupby('fold'):
        selected = json.loads((P/f'results/adra2b_screening_20261026/fold_{fold}/selected.json').read_text())
        fit = selected['fusion']; gate = fit['gate']
        weight = expit(gate['a']+gate['b']*(rows.max_train_drug_tanimoto-gate['sim_mean'])/gate['sim_std'])
        close(weight, rows.deep_weight)
        close(weight*rows.deep+(1-weight)*rows.neighbor_prior, rows.similarity_gate)
        close((rows.deep+rows.neighbor_prior)/2, rows.equal_blend)
        a = fit['fixed_deep_weight']
        close(a*rows.deep+(1-a)*rows.neighbor_prior, rows.fixed_blend)
    close(folds.groupby('row_id')[methods].mean().sort_index(), predictions.set_index('row_id')[methods].sort_index())
    data = predictions.merge(read_case('labels.csv'), on=['row_id','source_id'], validate='one_to_one')
    assert len(data) == len(query) == 1177 and data.outcome.eq('Active').sum() == 160
    summary = read_case('summary.csv')
    truth = data.outcome.eq('Active')
    for row in summary.itertuples():
        ordered = data.sort_values([row.model,'source_id'],ascending=[False,True])
        assert ordered.head(10).outcome.eq('Active').sum() == row.top10_hits
        close(average_precision_score(truth,data[row.model]),row.average_precision)
        close(roc_auc_score(truth,data[row.model]),row.roc_auc)
    hits = data.sort_values(['similarity_gate','source_id'],ascending=[False,True]).head(10).outcome.eq('Active').sum()
    return dict(check='adra2b',training_compounds=len(train),candidate_compounds=len(query),top10_active=int(hits))


def target_errors():
    directory = P/'manuscript/bib/tables/source/target_analysis_20261026'
    for name, expected in json.loads((directory/'manifest.json').read_text()).items():
        assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == expected, name
    summary = pd.read_csv(directory/'target_summary.csv')
    for row in summary.itertuples():
        data = pd.read_csv(directory/f'{row.protocol}_targets.csv')
        data = data.loc[data.identities.ge(row.minimum_identities)]
        delta = data.mse_similarity_gate-data['mse_'+row.control]
        assert len(data) == row.targets and delta.lt(0).sum() == row.improved
        assert delta.gt(0).sum() == row.worsened
        close(delta.median(),row.median_mse_difference)
        best = data.assign(delta=delta).sort_values('delta').head(int(np.ceil(.1*len(data)))).target
        rest = data.loc[~data.target.isin(best)]
        close(np.average(rest.mse_similarity_gate-rest['mse_'+row.control],weights=rest.pairs),
              row.remaining_pooled_mse_difference)
    primary = summary.loc[summary.minimum_identities.eq(0)&summary.control.eq('deep')]
    return dict(check='target_wise_errors',comparisons=len(summary),improved=primary.set_index('protocol').improved.to_dict())


def run_checks():
    return [shared_backbone(), adra2b(), target_errors()]


if __name__ == '__main__':
    report = dict(status='passed',checks=run_checks())
    output = ROOT/'verification/extensions.json'
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
