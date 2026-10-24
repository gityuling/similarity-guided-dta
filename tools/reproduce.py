#!/usr/bin/env python3
"""Recalculate the main study results from saved predictions on CPU."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / 'DTA_affinity'
MODELS = ['deep', 'neighbor_prior', 'fixed_blend', 'similarity_gate']
CHECKS = []


def read(path):
    return json.loads((PROJECT / path).read_text())


def frame(path):
    return pd.read_csv(PROJECT / path, float_precision='round_trip')


def close(actual, expected, label, atol=1e-10):
    np.testing.assert_allclose(actual, expected, rtol=0, atol=atol, equal_nan=True, err_msg=label)


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def verify_files(all_files):
    entries = json.loads((ROOT / 'manifests/files.json').read_text())
    groups = {r['group'] for r in entries} if all_files else {'repository', 'reproduction-data'}
    count = 0
    for item in entries:
        if item['group'] in groups:
            path = ROOT / item['path']
            if not path.is_file() or path.stat().st_size != item['bytes'] or sha256(path) != item['sha256']:
                raise RuntimeError(f'Missing or changed input: {item["path"]}. Run tools/load_assets.py first.')
            count += 1
    CHECKS.append({'check': 'file_checksums', 'files': count, 'groups': sorted(groups)})


def benchmark():
    analysis = read('reports/bib_fusion_20261017/analysis.json')
    repeated = read('results/bib_validation_20261017/progress_summary.json')
    scaffold = read('results/bib_scaffold_20261018/progress_summary.json')
    results = {}
    for ds in ['davis', 'kiba']:
        data = frame(f'reports/bib_fusion_20261017/{ds}_oof.csv')
        if not data.row_id.is_unique or set(data.split) != {'test'}:
            raise RuntimeError('Invalid out-of-fold coverage')
        metrics = {}
        for row in analysis['datasets'][ds]['folds']:
            f = data.loc[data.fold.eq(row['fold'])]
            gate = row['fit']['gate']
            weights = expit(gate['a'] + gate['b'] * (f.max_train_drug_tanimoto - gate['sim_mean']) / gate['sim_std'])
            close(weights, f.deep_weight, 'reference gate weights')
            close(weights*f.prediction_deep + (1-weights)*f.prediction_neighbor_prior,
                  f.prediction_similarity_gate, 'reference gate predictions')
            for model, expected in row['metrics'].items():
                y, p = f.truth.to_numpy(), f[f'prediction_{model}'].to_numpy()
                mse, rho = np.mean((y-p)**2), pearsonr(y,p).statistic
                close(mse, expected['mse'], f'{ds} fold MSE')
                close(rho, expected['pearson'], f'{ds} fold Pearson')
                metrics[model, row['fold']] = mse
        results[f'{ds}_reference_gate_mse'] = float(np.mean([metrics['similarity_gate', f] for f in range(5)]))
    verified_folds = 0
    for path in sorted((PROJECT / 'results').glob('bib_*_ablation.json')):
        record = json.loads(path.read_text())
        data = frame(f'results/predictions/{path.stem}_predictions.csv')
        gate = record['gate']
        weight = expit(gate['a'] + gate['b']*(data.max_train_drug_tanimoto-gate['sim_mean'])/gate['sim_std'])
        close(weight, data.deep_weight, 'repeated gate weights')
        close(weight*data.prediction_deep+(1-weight)*data.prediction_neighbor_prior,
              data.prediction_similarity_gate, 'repeated gate predictions')
        for split, key in [('val','validation'),('test','test')]:
            group = data.loc[data.split.eq(split)]
            for model in MODELS:
                y, p = group.affinity_true.to_numpy(), group[f'prediction_{model}'].to_numpy()
                close(np.mean((y-p)**2), record[key][model]['mse'], f'{path.name} {split} MSE')
                close(pearsonr(y,p).statistic, record[key][model]['pearson'], f'{path.name} {split} Pearson')
        verified_folds += 1
    spec = importlib.util.spec_from_file_location('study_tables', PROJECT / 'manuscript/bib/tools/build_tables.py')
    tables = importlib.util.module_from_spec(spec); spec.loader.exec_module(tables)
    summaries, _ = tables.checked_repeats(repeated)
    tables.checked_identity(repeated, read('data/processed/kiba_identity_20261017/summary.json'))
    for ds in ['davis', 'kiba']:
        results[f'{ds}_three_seed_gate_mse'] = summaries[ds,'similarity_gate','mse']['mean_over_seeds_and_folds']
    for model in MODELS:
        values = [read(f'results/bib_kiba_scaffold_f{f}_seed20261012_ablation.json')['test'][model]['mse'] for f in range(5)]
        close(values, scaffold['metric_values'][model]['mse'], 'scaffold fold summary')
    results['kiba_scaffold_gate_mse'] = float(np.mean(scaffold['metric_values']['similarity_gate']['mse']))
    CHECKS.append({'check':'benchmark_predictions', 'new_fold_results':verified_folds,
                   'reference_folds':10, 'results':results})


def splits():
    total = 0
    for dataset, protocol in [('davis','drug_id'),('kiba','drug_id'),('kiba','identity'),('kiba','scaffold')]:
        data = frame(f'data/processed/{dataset}_dtiam_protocol/table.csv')
        folder = f'{dataset}_dtiam_protocol' if protocol == 'drug_id' else f'kiba_{protocol}_20261017'
        all_tests = []
        if protocol != 'drug_id':
            summary = read(f'data/processed/{folder}/summary.json')
            if summary['pairs_removed'] or not summary['every_test_row_once']:
                raise RuntimeError('Incomplete strict-partition coverage')
        for fold in range(5):
            manifest = frame(f'data/processed/{folder}/folds/drug_coldstart/fold_{fold}.csv')
            if not manifest.row_id.is_unique or set(manifest.row_id) != set(data.row_id):
                raise RuntimeError('Partition rows differ from benchmark rows')
            joined = data.merge(manifest[['row_id','split']],on='row_id',validate='one_to_one')
            supported = set(joined.loc[joined.split.eq('train'),'Sequence'])
            if not set(joined.loc[joined.split.ne('train'),'Sequence']) <= supported:
                raise RuntimeError('Query target lacks training support')
            if protocol == 'drug_id':
                ids = {s:set(joined.loc[joined.split.eq(s),'drug_idx']) for s in ['train','val','test']}
                if ids['train']&ids['val'] or ids['train']&ids['test'] or ids['val']&ids['test']:
                    raise RuntimeError('Drug-ID overlap')
            else:
                row = summary['folds'][fold]
                keys = ['canonical_overlap'] if protocol == 'identity' else ['chemical_identity_overlap','scaffold_overlap']
                for key in keys:
                    if any(row[key].values()):
                        raise RuntimeError('Strict partition audit contains overlap')
            all_tests.extend(manifest.loc[manifest.split.eq('test'),'row_id'])
            total += 1
        if sorted(all_tests) != sorted(data.row_id):
            raise RuntimeError('Test coverage is not exactly once')
    CHECKS.append({'check':'partition_coverage_and_support','folds':total})


def external_fusion(directory):
    folded = frame(f'{directory}/fold_predictions.csv')
    averaged = frame(f'{directory}/predictions.csv')
    for fold, group in folded.groupby('fold'):
        fit = read(f'results/bib_kiba_identity_f{fold}_seed20261012_ablation.json')
        gate = fit['gate']
        weight = expit(gate['a']+gate['b']*(group.max_train_drug_tanimoto-gate['sim_mean'])/gate['sim_std'])
        close(weight,group.deep_weight,'external gate weights')
        close(weight*group.deep+(1-weight)*group.neighbor_prior,group.similarity_gate,'external fusion')
        close(fit['fixed_deep_weight']*group.deep+(1-fit['fixed_deep_weight'])*group.neighbor_prior,
              group.fixed_blend,'external fixed blend')
    means = folded.groupby('row_id')[MODELS].mean().reindex(averaged.row_id)
    close(means.to_numpy(),averaged[MODELS].to_numpy(),'external fold averaging')


def nanobret():
    base = 'reports/bib_nanobret_20261018'
    external_fusion(base)
    scores, expected = frame(f'{base}/scores.csv'), frame(f'{base}/per_compound_metrics.csv')
    plan, report = read(f'{base}/plan.json'), read(f'{base}/analysis.json')
    for row in expected.itertuples():
        group = scores.loc[scores.compound.eq(row.compound)].copy()
        if row.variant == 'occupancy_clipped':
            group['endpoint_score'] = group.endpoint_score.clip(0,100)
        if row.variant == 'ic50_interior_only':
            group = group.loc[~group.reported_censor & ~group.possible_assay_boundary]
        if len(group) != row.pairs or int(group.active.sum()) != row.active_pairs:
            raise RuntimeError('NanoBRET subgroup counts differ')
        if len(group) >= plan['minimum_pairs_per_compound'] and group.endpoint_score.nunique()>1 and group[row.model].nunique()>1:
            close(spearmanr(group.endpoint_score,group[row.model]).statistic,row.spearman,'NanoBRET Spearman')
        if len(group) >= plan['minimum_pairs_per_compound'] and group.active.nunique()==2:
            close(average_precision_score(group.active,group[row.model]),row.ap,'NanoBRET AP')
            close(roc_auc_score(group.active,group[row.model]),row.auroc,'NanoBRET ROC-AUC')
    for item in report['macro_metrics'] + report['paired_differences']:
        selected = expected.loc[expected.variant.eq(item['variant']) & expected.endpoint.eq(item['endpoint'])]
        if item['stratum'] == 'davis_disjoint': selected=selected.loc[selected.davis_external]
        wide=selected.pivot(index='compound',columns='model',values=item['metric']).dropna()
        if 'macro_mean' in item:
            close(wide[item['model']].mean(),item['macro_mean'],'NanoBRET macro')
        else:
            comparator=item['comparison'].split(' - ')[1]
            delta=(wide.similarity_gate-wide[comparator]).to_numpy()
            samples=np.random.default_rng(plan['bootstrap_seed']).choice(delta,size=(plan['bootstrap_n'],len(delta))).mean(axis=1)
            close(np.quantile(samples,[.025,.975]),item['ci95'],'NanoBRET bootstrap')
    CHECKS.append({'check':'nanobret_rankings','pairs':len(scores),'metric_rows':len(expected),
                   'bootstrap_intervals_recomputed':True})


def abl1():
    base='reports/bib_downstream_20261022'
    external_fusion(base)
    labels,pred=frame(f'{base}/labels.csv'),frame(f'{base}/predictions.csv')
    data=labels.merge(pred,on='row_id',validate='one_to_one')
    data['pure_similarity']=data.max_train_drug_tanimoto
    data=data.loc[data.task.eq('A')]
    curves=frame(f'{base}/A_budget_curves.csv')
    for row in curves.itertuples():
        selected=data.sort_values([row.model,'source_id'],ascending=[False,True],kind='stable').head(row.k)
        hits=int(selected[row.label_rule].eq('Active').sum())
        unknown=int(selected[row.label_rule].eq('Inconclusive').sum())
        if hits != row.confirmed_hits or unknown != row.unknown_selected:
            raise RuntimeError('ABL1 retrieval result differs')
    for row in frame(f'{base}/A_known_label_metrics.csv').itertuples():
        known=data.loc[data[row.label_rule].isin(['Active','Inactive'])]
        truth=known[row.label_rule].eq('Active')
        close(average_precision_score(truth,known[row.model]),row.average_precision,'ABL1 AP')
        close(roc_auc_score(truth,known[row.model]),row.roc_auc,'ABL1 ROC-AUC')
    top=curves.loc[curves.k.eq(10)&curves.model.isin(['deep','similarity_gate'])]
    CHECKS.append({'check':'abl1_screen','candidates':len(data),'budget_rows':len(curves),
        'top10':top[['label_rule','model','confirmed_hits','unknown_selected']].to_dict('records')})


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--all-files',action='store_true',help='Also hash checkpoint and feature assets')
    parser.add_argument('--output',type=Path,default=ROOT/'verification/reproduction.json')
    args=parser.parse_args()
    for name,call in [('files',lambda:verify_files(args.all_files)),('partitions',splits),
                      ('benchmarks',benchmark),('NanoBRET',nanobret),('ABL1',abl1)]:
        print(f'Checking {name}...',flush=True)
        call()
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps({'status':'passed','checks':CHECKS,
        'scope':'CPU recalculation from saved predictions; no model retraining.'},indent=2)+'\n')
    print(f'Passed. Report: {args.output}',flush=True)


if __name__=='__main__':
    main()
