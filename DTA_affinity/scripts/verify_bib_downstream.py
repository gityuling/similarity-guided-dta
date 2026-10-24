#!/usr/bin/env python3
"""Independently recompute saved downstream summaries and integrity checks."""
import itertools
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit
from bib_common import PROJECT, ROOT, sha256, record

OUT=PROJECT/'reports/bib_downstream_20261022'


def main():
    checked=0
    for manifest in ['evaluation_freeze.json','preparation_inputs.json']:
        for item in json.loads((OUT/manifest).read_text()):
            assert sha256(ROOT/item['path'])==item['sha256'], item['path']
            checked+=1
    labels=pd.read_csv(OUT/'labels.csv')
    query=pd.read_csv(OUT/'query_table.csv')
    all_rows=pd.read_csv(OUT/'source_compound_audit.csv')
    included=all_rows.loc[all_rows.included]
    assert not included.filter(regex='^seen_').any().any()
    assert set(included.source_id)==set(query.source_id)==set(labels.source_id)
    assert len(query)==228 and query.drug_id.is_unique
    assert not set(query.columns)&{'outcome','strict_outcome','endpoint_score'}
    fold=pd.read_csv(OUT/'fold_predictions.csv')
    pred=pd.read_csv(OUT/'predictions.csv')
    cols=['deep','neighbor_prior','fixed_blend','similarity_gate','equal_blend','max_train_drug_tanimoto','deep_weight']
    assert fold.groupby('row_id').fold.nunique().eq(5).all()
    np.testing.assert_allclose(fold.groupby('row_id')[cols].mean().to_numpy(),pred.set_index('row_id')[cols].to_numpy(),atol=1e-12)
    for f,g in fold.groupby('fold'):
        fit=json.loads((PROJECT/f'results/bib_kiba_identity_f{f}_seed20261012_ablation.json').read_text())
        gate=fit['gate']; weight=expit(gate['a']+gate['b']*(g.max_train_drug_tanimoto-gate['sim_mean'])/gate['sim_std'])
        np.testing.assert_allclose(weight,g.deep_weight,atol=1e-12)
        for name,w in [('similarity_gate',weight),('fixed_blend',fit['fixed_deep_weight']),('equal_blend',.5)]:
            np.testing.assert_allclose(w*g.deep+(1-w)*g.neighbor_prior,g[name],atol=1e-12)
    data=labels.merge(pred,on='row_id',validate='one_to_one');data['pure_similarity']=data.max_train_drug_tanimoto
    a=data.loc[data.task.eq('A')]
    curves=pd.read_csv(OUT/'A_budget_curves.csv')
    for row in curves.itertuples():
        ordered=sorted(a.to_dict('records'),key=lambda x:(-x[row.model],x['source_id']))
        selected=ordered[:row.k]
        hits=sum(x[row.label_rule]=='Active' for x in selected)
        unknown=sum(x[row.label_rule]=='Inconclusive' for x in selected)
        assert row.confirmed_hits==hits and row.unknown_selected==unknown
        assert np.isclose(row.precision_lower,hits/row.k) and np.isclose(row.precision_upper,(hits+unknown)/row.k)
    b=pd.read_csv(OUT/'B_series_metrics.csv')
    errors=[]
    for row in b.itertuples():
        g=data.loc[data.cohort.eq(row.cohort)]
        rho=np.corrcoef(g.endpoint_score.rank(),g[row.model].rank())[0,1] if g[row.model].nunique()>1 else 0.
        pairs=[]
        for left,right in itertools.combinations(g.to_dict('records'),2):
            dy=left['endpoint_score']-right['endpoint_score']
            if abs(dy)<1:continue
            ds=left[row.model]-right[row.model]
            pairs.append(.5 if ds==0 else float(ds*dy>0))
        assert len(pairs)==row.n_pairs
        errors.extend([abs(rho-row.spearman),abs(np.mean(pairs)-row.pair_accuracy)])
        threshold=np.quantile(g.endpoint_score,.75)
        ordered=sorted(g.to_dict('records'),key=lambda x:(-x[row.model],x['source_id']))
        assert sum(x['endpoint_score']>=threshold for x in ordered[:3])==row.top3_hits
    report=json.loads((OUT/'evaluation.json').read_text())
    rng=np.random.default_rng(20261022);series=sorted(b.cohort.unique());idx=rng.integers(0,len(series),size=(10000,len(series)))
    for r in report['B_macro']:
        v=b.loc[b.model.eq(r['model'])].set_index('cohort').loc[series,r['metric']].to_numpy()
        assert np.isclose(v.mean(),r['mean'])
        np.testing.assert_allclose(np.quantile(np.mean(v[idx],axis=1),[.025,.975]),r['ci95'],atol=1e-12)
    for r in report['B_paired_contrasts']:
        baseline=r['contrast'].split(' - ')[1]; mat=b.pivot(index='cohort',columns='model',values=r['metric']).loc[series]
        d=(mat.similarity_gate-mat[baseline]).to_numpy()
        assert np.isclose(d.mean(),r['mean'])
        np.testing.assert_allclose(np.quantile(d[idx].mean(axis=1),[.025,.975]),r['ci95'],atol=1e-12)
        possible=[abs(np.mean(d*np.array(signs))) for signs in itertools.product([-1,1],repeat=len(d))]
        assert np.isclose(np.mean(np.array(possible)>=abs(d.mean())-1e-12),r['exact_sign_flip_two_sided_p'])
    assert max(errors)<1e-12
    files=['predictions.csv','fold_predictions.csv','evaluation.json','A_budget_curves.csv','B_series_metrics.csv','A_top10_selections.csv','labels.csv','fold_stability.csv']
    result=dict(state='passed',frozen_input_checks=checked,query_rows=len(query),fold_rows=len(fold),
        selection_lists_checked=len(curves),series_method_rows_checked=len(b),max_metric_error=max(errors),
        intervals_recomputed=True,fusion_formulas_verified=True,
        limitations='No independent GPU rerun; source assay measurements are not re-fitted from dose curves.',
        files={n:sha256(OUT/n) for n in files})
    (OUT/'evaluation_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
