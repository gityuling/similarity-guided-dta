#!/usr/bin/env python3
"""Independent metric, source-boundary, coefficient optimality and CI checks."""
import json
import math
import numpy as np
import pandas as pd
from scipy.stats import hypergeom
from bib_g2cp_common import DATA, PROJECT, check_records, save_json, record

DEST=PROJECT/'reports/bib_mechanism_20261021'
REF=PROJECT/'reports/bib_g2cp_reference_diagnostic_20261019'


def main():
    freeze=json.loads((DEST/'prototype_freeze.json').read_text());check_records(freeze['inputs'])
    analysis=json.loads((DEST/'prototype_analysis.json').read_text());check_records(analysis['outputs'])
    cfg=freeze['config'];alpha=analysis['alpha']
    cases=pd.read_csv(DEST/'prototype_cases.csv')
    assert cases.split.isin(['train','val']).all()
    assert cases.groupby('parent_key').split.nunique().max()==1
    partition=pd.read_csv(DEST/'gene_modules.csv')
    assert partition.groupby(['cell','module']).fold.nunique().max()==1
    coefficients=pd.read_csv(DEST/'prototype_target_scores.csv',float_precision='round_trip')
    retrieval=pd.read_csv(DEST/'prototype_retrieval_metrics.csv',float_precision='round_trip')
    response=pd.read_csv(DEST/'prototype_response_metrics.csv',float_precision='round_trip')
    maxerr=0.;rankerr=0.;checked=0
    with np.load(DEST/'prototype_vectors.npz') as z:
        for row in response.itertuples():
            y=z[f'{row.case_index}_truth'];pred=z[f'{row.case_index}_{row.method}'];bg=z[f'{row.case_index}_background']
            def correlation(x,y):
                if np.linalg.norm(x-x.mean())*np.linalg.norm(y-y.mean())<=1e-12:return 0.
                return float(np.corrcoef(x,y)[0,1])
            values=[np.mean((y-pred)**2),correlation(pred,y),correlation(pred-bg,y-bg)]
            maxerr=max(maxerr,*[abs(a-b) for a,b in zip(values,[row.mse,row.pearson,row.residual_pearson])])
    for (cell,parent,method),g in coefficients.groupby(['cell','parent_key','method']):
        pos=g.known_positive.to_numpy(bool)
        if not pos.any():continue
        score=g.coefficient.to_numpy();assert (score>=0).all()
        value=score[pos].max();ahead=int((score>value).sum());tied=score==value
        n=int(tied.sum());p=int((pos&tied).sum());previous=0.;rr=0.
        hits={}
        for j in range(1,n+1):
            cumulative=1-hypergeom.pmf(0,n,p,j)
            rr+=(cumulative-previous)/(ahead+j);previous=cumulative
        for k in [1,5,10]:hits[k]=0. if k<=ahead else float(1-hypergeom.pmf(0,n,p,min(n,k-ahead)))
        recall=np.mean([np.clip((5-(score>s).sum())/(score==s).sum(),0,1) for s in score[pos]])
        r=retrieval.loc[retrieval.cell.eq(cell)&retrieval.parent_key.eq(parent)&retrieval.method.eq(method)].iloc[0]
        rankerr=max(rankerr,abs(rr-r.mrr),abs(recall-r.recall_at_5),*[abs(hits[k]-r[f'hit_at_{k}']) for k in hits]);checked+=1
    summaryerr=0.;intervalerr=0.
    for kind,frame,metrics in [('response',response,['mse','pearson','residual_pearson']),
                               ('retrieval',retrieval,['mrr','hit_at_1','hit_at_5','hit_at_10','recall_at_5'])]:
        summary=pd.read_csv(DEST/f'prototype_{kind}_summary.csv',float_precision='round_trip')
        for row in summary.to_dict('records'):
            sub=frame.loc[frame.split.eq(row['split'])&frame.method.eq(row['method'])]
            if row['scope'].startswith('strict'):sub=sub.loc[sub.davis_external]
            grouped=sub.groupby('parent_key')[metrics].mean()
            assert len(grouped)==row['drugs']
            summaryerr=max(summaryerr,*[abs(grouped[m].mean()-row[m]) for m in metrics])
        for contrast in [c for c in analysis['comparisons'] if c['endpoint']==kind]:
            sub=frame.loc[frame.split.eq('val')]
            if contrast['scope'].startswith('strict'):sub=sub.loc[sub.davis_external]
            m=sub.groupby(['parent_key','method'])[contrast['metric']].mean().unstack()
            delta=(m[contrast['left']]-m[contrast['right']]).to_numpy()
            indices=np.random.default_rng(cfg['seed']).integers(0,len(delta),(cfg['bootstrap_repeats'],len(delta)))
            ci=np.quantile(np.mean(delta[indices],axis=1),[.025,.975])
            intervalerr=max(intervalerr,*abs(ci-np.array(contrast['ci95'])),abs(delta.mean()-contrast['difference']))
    grid=pd.read_csv(DEST/'prototype_alpha_grid.csv')
    assert grid.split.eq('train').all() and grid.davis_external.all()
    means=grid.groupby(['alpha','parent_key']).mse.mean().groupby('alpha').mean()
    chosen=max(a for a,v in means.items() if v<=means.min()+1e-12);assert chosen==alpha
    assert maxerr<1e-10 and rankerr<1e-10 and summaryerr<1e-10 and intervalerr<1e-10
    numerical=pd.read_csv(DEST/'prototype_numerical_checks.csv');assert numerical.kkt_error.max()<1e-6
    result={'state':'passed','test_rows_scored':0,'case_count':len(cases),'response_rows_checked':len(response),
            'sparse_retrieval_rows_independently_checked':checked,'metric_max_error':maxerr,'rank_max_error':rankerr,
            'summary_max_error':summaryerr,'interval_max_error':intervalerr,'train_only_alpha_verified':chosen,
            'all_recorded_solver_kkt_below':float(numerical.kkt_error.max()),
            'limitations':'Solver KKT values reviewed from run output; independent optimizer equivalence and heldout-value invariance are tested separately.',
            'analysis':record(DEST/'prototype_analysis.json'),'verifier':record(__import__('pathlib').Path(__file__))}
    save_json(DEST/'prototype_verification.json',result,immutable=True);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
