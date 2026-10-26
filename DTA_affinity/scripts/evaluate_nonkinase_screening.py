#!/usr/bin/env python3
"""Evaluate the frozen ESR1 candidate ranking against deposited assay labels."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / 'reports/nonkinase_screening_20261026'
RUN = PROJECT / 'results/nonkinase_screening_20261026'
MODELS = ['deep', 'neighbor_prior', 'equal_blend', 'fixed_blend', 'similarity_gate', 'pure_similarity']


def score(frame):
    truth = frame.outcome.eq('Active').astype(int).to_numpy()
    assert set(frame.outcome) <= {'Active', 'Inactive'}
    summaries, curves = [], []
    for model in MODELS:
        ranked = frame.sort_values([model, 'source_id'], ascending=[False, True], kind='stable')
        hits = ranked.outcome.eq('Active').astype(int).cumsum().to_numpy()
        prevalence = float(truth.mean())
        for k in range(1, len(frame) + 1):
            curves.append(dict(model=model, k=k, hits=int(hits[k-1]), hit_rate=float(hits[k-1]/k),
                               recall=float(hits[k-1]/truth.sum()), enrichment_factor=float(hits[k-1]/k/prevalence)))
        summaries.append(dict(model=model, n=len(frame), actives=int(truth.sum()),
                              top10_hits=int(hits[9]), top10_hit_rate=float(hits[9]/10),
                              top10_enrichment=float(hits[9]/10/prevalence),
                              average_precision=float(average_precision_score(truth, frame[model])),
                              roc_auc=float(roc_auc_score(truth, frame[model]))))
    return summaries, curves


def main():
    state = json.loads((RUN / 'status.json').read_text())
    assert state['state'] == 'completed'
    assert hashlib.sha256((OUT / 'predictions.csv').read_bytes()).hexdigest() == state['prediction_sha256']
    labels = pd.read_csv(OUT / 'labels.csv')
    pred = pd.read_csv(OUT / 'predictions.csv')
    merged = pred.merge(labels, on=['row_id', 'source_id'], validate='one_to_one')
    summary, curves = score(merged)
    pd.DataFrame(summary).to_csv(OUT / 'summary.csv', index=False)
    pd.DataFrame(curves).to_csv(OUT / 'budget_curves.csv', index=False)
    merged.to_csv(OUT / 'candidate_scores.csv', index=False)
    folds = pd.read_csv(OUT / 'fold_predictions.csv').merge(labels, on=['row_id', 'source_id'], validate='many_to_one')
    fold_records = []
    for f, frame in folds.groupby('fold'):
        values, _ = score(frame)
        fold_records.extend([dict(fold=int(f), **v) for v in values])
    pd.DataFrame(fold_records).to_csv(OUT / 'fold_metrics.csv', index=False)
    report = dict(case='ESR1_AID1223', metrics=summary,
                  random_top10_expected_hits=10 * float(merged.outcome.eq('Active').mean()),
                  endpoint='deposited active/inactive outcome from competitive estrogen binding in permeabilized cells',
                  prediction_selection='parameters frozen on ChEMBL validation compounds; no PubChem outcomes used',
                  primary_budget=10,
                  prediction_sha256=state['prediction_sha256'],
                  labels_sha256=hashlib.sha256((OUT / 'labels.csv').read_bytes()).hexdigest())
    (OUT / 'analysis.json').write_text(json.dumps(report, indent=2) + '\n')
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == '__main__':
    main()
