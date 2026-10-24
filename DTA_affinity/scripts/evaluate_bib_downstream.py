#!/usr/bin/env python3
"""Fixed-budget and analog-series evaluation of frozen external predictions."""
import itertools
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr, hypergeom
from sklearn.metrics import average_precision_score, roc_auc_score
from bib_common import PROJECT, record, write_once

OUT = PROJECT/'reports/bib_downstream_20261022'
MODELS = ['deep','neighbor_prior','fixed_blend','similarity_gate','equal_blend','pure_similarity']


def ranking(group, model):
    return group.sort_values([model,'source_id'], ascending=[False,True], kind='stable')


def pair_accuracy(y, scores):
    i,j = np.triu_indices(len(y),1)
    dy, dp = y[i]-y[j], scores[i]-scores[j]
    use = np.abs(dy)>=1
    correct = (dy[use]*dp[use]>0).astype(float) + 0.5*(dp[use]==0)
    return float(correct.mean()) if len(correct) else np.nan, int(use.sum())


def series_metrics(group, model):
    y = group.endpoint_score.to_numpy(float)
    s = group[model].to_numpy(float)
    rho = float(spearmanr(y,s).statistic) if len(set(s))>1 else 0.0
    pair, n_pairs = pair_accuracy(y,s)
    threshold = float(np.quantile(y,.75))
    good = group.endpoint_score>=threshold
    chosen = ranking(group,model).head(3)
    hits = int((chosen.endpoint_score>=threshold).sum())
    return dict(n=len(group),spearman=rho,pair_accuracy=pair,n_pairs=n_pairs,
                top3_precision=hits/3,top3_recall=hits/int(good.sum()),good_n=int(good.sum()),
                top3_hits=hits,good_threshold=threshold,constant_prediction=len(set(s))==1)


def selection_metrics(group, model, label_col, k):
    chosen = ranking(group,model).head(k)
    a,u = int((chosen[label_col]=='Active').sum()), int((chosen[label_col]=='Inconclusive').sum())
    total = int((group[label_col]=='Active').sum())
    prevalence = total/len(group)
    return dict(k=k,n=len(group),active_total=total,confirmed_hits=a,unknown_selected=u,
                precision_lower=a/k,precision_upper=(a+u)/k,recall=a/total,
                enrichment=(a/k)/prevalence,random_expected_hits=k*prevalence,
                random_tail_p=float(hypergeom.sf(a-1,len(group),total,k)))


def interval(values):
    return [float(v) for v in np.quantile(values,[.025,.975])]


def main():
    labels = pd.read_csv(OUT/'labels.csv')
    prediction = pd.read_csv(OUT/'predictions.csv')
    assert labels.row_id.is_unique and prediction.row_id.is_unique
    data = labels.merge(prediction,on='row_id',how='outer',validate='one_to_one',indicator=True)
    assert data._merge.eq('both').all()
    data['pure_similarity'] = data.max_train_drug_tanimoto
    assert np.isfinite(data[MODELS]).all().all()
    a = data.loc[data.task.eq('A')]
    a_metrics, selected = [], []
    for label_col in ('outcome','strict_outcome'):
        for model in MODELS:
            for k in range(1,len(a)+1):
                a_metrics.append(dict(model=model,label_rule=label_col,**selection_metrics(a,model,label_col,k)))
            pick = ranking(a,model).head(10).copy()
            pick['model'], pick['label_rule'],pick['rank'] = model,label_col,np.arange(1,len(pick)+1)
            selected.append(pick[['row_id','source_id','rank','model','label_rule',label_col]].rename(columns={label_col:'observed_outcome'}))
    a_metrics=pd.DataFrame(a_metrics)
    a_known=[]
    for col in ('outcome','strict_outcome'):
        known=a.loc[a[col].isin(['Active','Inactive'])]
        y=(known[col]=='Active').astype(int)
        for model in MODELS:
            a_known.append(dict(label_rule=col,model=model,n=len(known),active_n=int(y.sum()),
                average_precision=float(average_precision_score(y,known[model])),
                roc_auc=float(roc_auc_score(y,known[model]))))
    b_rows=[]
    for cohort,g in data.loc[data.task.eq('B')].groupby('cohort',sort=True):
        for model in MODELS:
            b_rows.append(dict(cohort=cohort,model=model,**series_metrics(g,model)))
    b=pd.DataFrame(b_rows)
    rng=np.random.default_rng(20261022)
    cohorts=sorted(b.cohort.unique()); indices=rng.integers(0,len(cohorts),size=(10000,len(cohorts)))
    macro,contrasts=[],[]
    for metric in ('spearman','pair_accuracy','top3_precision','top3_recall'):
        matrix=b.pivot(index='cohort',columns='model',values=metric).reindex(cohorts)
        for model in MODELS:
            v=matrix[model].to_numpy()
            macro.append(dict(model=model,metric=metric,mean=float(v.mean()),ci95=interval(v[indices].mean(axis=1))))
        for baseline in [m for m in MODELS if m!='similarity_gate']:
            delta=(matrix.similarity_gate-matrix[baseline]).to_numpy()
            flips=np.array(list(itertools.product([-1,1],repeat=len(delta))))
            p=float((np.abs((flips*delta).mean(axis=1))>=abs(delta.mean())-1e-12).mean())
            contrasts.append(dict(metric=metric,contrast='similarity_gate - '+baseline,
                mean=float(delta.mean()),ci95=interval(delta[indices].mean(axis=1)),exact_sign_flip_two_sided_p=p))
    random_rows=[]
    for cohort,g in data.loc[data.task.eq('B')].groupby('cohort'):
        random_rows.append(dict(cohort=cohort,spearman=0,pair_accuracy=.5,
            top3_precision=float((g.endpoint_score>=g.endpoint_score.quantile(.75)).mean()),top3_recall=3/len(g)))
    fold=pd.read_csv(OUT/'fold_predictions.csv').merge(labels,on='row_id',validate='many_to_one')
    fold['pure_similarity']=fold.max_train_drug_tanimoto
    stability=[]
    for (f,task,cohort),g in fold.groupby(['fold','task','cohort']):
        for model in MODELS:
            pooled=data.loc[data.cohort.eq(cohort)]
            k=10 if task=='A' else 3
            selected_fold=set(ranking(g,model).head(k).row_id)
            selected_ensemble=set(ranking(pooled,model).head(k).row_id)
            stability.append(dict(fold=f,task=task,cohort=cohort,model=model,
                top_k=k,overlap_with_ensemble=len(selected_fold&selected_ensemble)/k,
                spearman_with_ensemble=float(spearmanr(g.sort_values('row_id')[model],pooled.sort_values('row_id')[model]).statistic)))
    for name,frame in [('A_budget_curves.csv',a_metrics),('A_top10_selections.csv',pd.concat(selected)),
                       ('A_known_label_metrics.csv',pd.DataFrame(a_known)),('B_series_metrics.csv',b),
                       ('B_random_expectations.csv',pd.DataFrame(random_rows)),('fold_stability.csv',pd.DataFrame(stability))]:
        write_once(OUT/name,frame.to_csv(index=False))
    report=dict(A_primary_budget=10,A_targets=1,B_series=len(cohorts),B_macro=macro,B_paired_contrasts=contrasts,
        bootstrap_unit='series',bootstrap_replicates=10000,seed=20261022,
        inputs=[record(OUT/n) for n in ['labels.csv','predictions.csv','fold_predictions.csv','EVALUATION_PROTOCOL.md']])
    write_once(OUT/'evaluation.json',json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(a_metrics.loc[a_metrics.k.eq(10)].to_string(index=False))
    print(b.groupby('model')[['spearman','pair_accuracy','top3_precision']].mean().to_string())


if __name__=='__main__':main()
