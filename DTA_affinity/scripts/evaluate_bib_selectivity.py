#!/usr/bin/env python3
"""Evaluate frozen recommendations; no parameter fitting on external endpoints."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/bib_selectivity_20261021'


def choose(target,avoid,ties,k=3,fraction=0.25,strategy='target_filtered_margin'):
    order=np.lexsort((ties,-target))
    if strategy=='target_only':return order[:k]
    retained=order[:int(np.ceil(len(order)*fraction))]
    return retained[np.lexsort((ties[retained],-target[retained],-(target[retained]-avoid[retained])))][:k]


def main():
    cfg=json.loads((ROOT/'configs/bib_selectivity_evaluation_20261021.json').read_text())
    query=pd.read_csv(OUT/'query_table.csv',usecols=['row_id','compound','gene','drug_id'])
    predictions=pd.read_csv(OUT/'predictions.csv')
    x=query.merge(predictions,on='row_id',validate='one_to_one')
    assert len(x)==len(query) and x[cfg['models']].notna().all().all()
    names=sorted(x.compound.unique()); genes=sorted(x.gene.unique())
    gi={g:i for i,g in enumerate(genes)};ni={n:i for i,n in enumerate(names)}
    parent=query.drop_duplicates('compound').set_index('compound').drug_id.to_dict()
    ties=np.array([hashlib.sha256(parent[n].encode()).hexdigest() for n in names])
    matrices={m:x.pivot(index='compound',columns='gene',values=m).reindex(index=names,columns=genes).to_numpy() for m in cfg['models']}
    tasks=pd.read_csv(OUT/'family_tasks.csv');tasks=tasks.loc[tasks.eligible].copy()
    rec=[]
    for t in tasks.itertuples():
        for model in cfg['models']:
            a,b=matrices[model][:,gi[t.target]],matrices[model][:,gi[t.avoid]]
            for strategy in cfg['strategies']:
                ids=choose(a,b,ties,cfg['budget'],cfg['target_prefilter_fraction'],strategy)
                for rank,i in enumerate(ids,1):
                    rec.append(dict(task_id=t.task_id,family=t.family,target=t.target,avoid=t.avoid,
                        model=model,strategy=strategy,method=model+'__'+strategy,rank=rank,compound=names[i],
                        target_prediction=a[i],avoid_prediction=b[i]))
    recommendations=pd.DataFrame(rec)
    recommendations.to_csv(OUT/'recommendations.csv',index=False)
    qc=pd.read_csv(OUT/'qc_measurement_states.csv')
    signals=pd.read_csv(OUT/'measurement_states.csv',usecols=['compound','gene','residual_1000nm'])
    signal=signals.pivot(index='compound',columns='gene',values='residual_1000nm').reindex(index=names,columns=genes).to_numpy()
    evaluated=[];random=[]
    for profile in cfg['qc_profiles']:
        states=qc.loc[qc.profile.eq(profile)].pivot(index='compound',columns='gene',values='state').reindex(index=names,columns=genes).to_numpy()
        def metrics(ids,a,b):
            sa,sb=states[ids,a],states[ids,b]
            known_a=sa!='unknown';known_b=sb!='unknown'
            passed=(sa=='strong_competition')&(sb=='limited_competition')
            failed=np.isin(sa,['limited_competition','intermediate_competition'])|np.isin(sb,['strong_competition','intermediate_competition'])
            unknown=~passed&~failed
            return {'joint_lower':passed.astype(float),'joint_upper':(passed|unknown).astype(float),
                'unknown':unknown.astype(float),'target_strong':(sa=='strong_competition').astype(float),
                'avoid_limited':(sb=='limited_competition').astype(float),
                'target_qc':known_a.astype(float),'avoid_qc':known_b.astype(float),
                'joint_qc':(known_a&known_b).astype(float),
                'target_residual':np.where(known_a,signal[ids,a],np.nan),
                'avoid_residual':np.where(known_b,signal[ids,b],np.nan),
                'residual_margin':np.where(known_a&known_b,signal[ids,b]-signal[ids,a],np.nan)}
        for task,group in recommendations.groupby('task_id',sort=False):
            a,b=gi[group.target.iloc[0]],gi[group.avoid.iloc[0]]
            ids=np.array([ni[c] for c in group.compound])
            frame=group.copy();frame['profile']=profile
            for key,value in metrics(ids,a,b).items():frame[key]=value
            evaluated.append(frame)
            m=metrics(np.arange(len(names)),a,b)
            random.append(dict(profile=profile,task_id=task,family=group.family.iloc[0],method='random_expected',
                **{k:float(np.nanmean(v)) if np.isfinite(v).any() else np.nan for k,v in m.items()}))
    scores=pd.concat(evaluated,ignore_index=True)
    scores.to_csv(OUT/'recommendation_outcomes.csv',index=False)
    metric_cols=['joint_lower','joint_upper','unknown','target_strong','avoid_limited','target_qc','avoid_qc','joint_qc','target_residual','avoid_residual','residual_margin']
    task_scores=scores.groupby(['profile','method','family','task_id'])[metric_cols].mean().reset_index()
    task_scores=pd.concat([task_scores,pd.DataFrame(random)],ignore_index=True)
    task_scores.to_csv(OUT/'task_scores.csv',index=False)
    family=task_scores.groupby(['profile','method','family'])[metric_cols].mean().reset_index()
    family.to_csv(OUT/'family_scores.csv',index=False)
    summary=family.groupby(['profile','method'])[metric_cols].mean().reset_index()
    summary.to_csv(OUT/'evaluation_summary.csv',index=False)
    methods=sorted(recommendations.method.unique());families=sorted(tasks.family.unique())
    mi={m:i for i,m in enumerate(methods)};fi={f:i for i,f in enumerate(families)}
    numerator=np.zeros((len(methods),len(families),len(names)));denominator=np.zeros_like(numerator)
    for r in scores.loc[scores.profile.eq(cfg['primary_qc'])].itertuples():
        ix=(mi[r.method],fi[r.family],ni[r.compound]);numerator[ix]+=r.joint_lower;denominator[ix]+=1
    rng=np.random.default_rng(cfg['seed']);B=cfg['bootstrap_repetitions']
    family_values=numerator.sum(axis=2)/denominator.sum(axis=2)
    family_draw=rng.integers(0,len(families),size=(B,len(families)))
    family_samples=family_values[:,family_draw].mean(axis=2)
    cw=rng.exponential(size=(len(names),B));fw=rng.exponential(size=(len(families),B))
    crossed=((numerator.reshape(-1,len(names))@cw)/(denominator.reshape(-1,len(names))@cw)).reshape(len(methods),len(families),B)
    crossed=(crossed*fw[None,:,:]).sum(axis=1)/fw.sum(axis=0)
    gate='similarity_gate__target_filtered_margin'
    comparisons=[]
    for right in methods:
        if right==gate:continue
        i,j=mi[gate],mi[right]
        comparisons.append(dict(left=gate,right=right,difference=float(family_values[i].mean()-family_values[j].mean()),
            family_bootstrap_ci95=np.quantile(family_samples[i]-family_samples[j],[.025,.975]).tolist(),
            crossed_weight_ci95=np.quantile(crossed[i]-crossed[j],[.025,.975]).tolist()))
    result={'stage':cfg['stage'],'tasks':len(tasks),'families':len(families),'candidates':len(names),
        'recommendation_count':len(recommendations),'comparisons_reference_qc':comparisons,
        'limitations':[cfg['interpretation'],cfg['dti_claim'],cfg['uncertainty'],
            'Continuous metrics use available QC records and can have different missingness; inspect coverage.',
            'All intervals exploratory and unadjusted; no inference of reliable negative binding.']}
    (OUT/'evaluation.json').write_text(json.dumps(result,indent=2)+'\n')
    print(summary.loc[summary.profile.eq('reference'),['method','joint_lower','unknown','target_strong','avoid_limited']].to_string(index=False))


if __name__=='__main__':main()
