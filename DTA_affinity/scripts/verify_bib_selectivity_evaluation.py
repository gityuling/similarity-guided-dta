#!/usr/bin/env python3
"""Independently check all chosen identities and operational labels."""
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/bib_selectivity_20261021'


def read(p):
    with p.open() as f:return list(csv.DictReader(f))


def main():
    cfg=json.loads((ROOT/'configs/bib_selectivity_evaluation_20261021.json').read_text())
    for item in json.loads((OUT/'evaluation_freeze.json').read_text()):
        assert hashlib.sha256((ROOT.parent/item['path']).read_bytes()).hexdigest()==item['sha256']
    preds={r['row_id']:r for r in read(OUT/'predictions.csv')}
    scores=defaultdict(dict);parents={}
    for r in read(OUT/'query_table.csv'):
        scores[r['gene']][r['compound']]=preds[r['row_id']];parents[r['compound']]=r['drug_id']
    identities=sorted(parents)
    recommendations=read(OUT/'recommendations.csv');actual=defaultdict(list)
    for r in recommendations:actual[(r['task_id'],r['model'],r['strategy'])].append(r['compound'])
    tasks=[r for r in read(OUT/'family_tasks.csv') if r['eligible']=='True']
    checks=0
    for task in tasks:
        for model in cfg['models']:
            target=lambda c:float(scores[task['target']][c][model])
            avoid=lambda c:float(scores[task['avoid']][c][model])
            tie=lambda c:hashlib.sha256(parents[c].encode()).hexdigest()
            order=sorted(identities,key=lambda c:(-target(c),tie(c)))
            expected=order[:cfg['budget']]
            assert actual[task['task_id'],model,'target_only']==expected
            retained=order[:math.ceil(len(order)*cfg['target_prefilter_fraction'])]
            expected=sorted(retained,key=lambda c:(-(target(c)-avoid(c)),-target(c),tie(c)))[:cfg['budget']]
            assert actual[task['task_id'],model,'target_filtered_margin']==expected
            checks+=2
    qc={(r['profile'],r['compound'],r['gene']):r['state'] for r in read(OUT/'qc_measurement_states.csv')}
    signals={(r['compound'],r['gene']):float(r['residual_1000nm'] or 'nan') for r in read(OUT/'measurement_states.csv')}
    outcome=read(OUT/'recommendation_outcomes.csv'); errors=[]
    for r in outcome:
        a=qc[r['profile'],r['compound'],r['target']];b=qc[r['profile'],r['compound'],r['avoid']]
        passed=a=='strong_competition' and b=='limited_competition'
        failed=a in ['limited_competition','intermediate_competition'] or b in ['strong_competition','intermediate_competition']
        unknown=not passed and not failed
        for k,v in [('joint_lower',passed),('joint_upper',passed or unknown),('unknown',unknown),
                    ('target_strong',a=='strong_competition'),('avoid_limited',b=='limited_competition'),
                    ('target_qc',a!='unknown'),('avoid_qc',b!='unknown'),('joint_qc',a!='unknown' and b!='unknown')]:
            errors.append(abs(float(r[k])-int(v)))
        va=signals[r['compound'],r['target']] if a!='unknown' else float('nan')
        vb=signals[r['compound'],r['avoid']] if b!='unknown' else float('nan')
        for key,value in [('target_residual',va),('avoid_residual',vb),('residual_margin',vb-va)]:
            observed=float(r[key] or 'nan')
            assert (math.isnan(value) and math.isnan(observed)) or abs(value-observed)<1e-12
    assert max(errors)==0
    frame=pd.DataFrame(outcome)
    for col in ['joint_lower','joint_upper','unknown']:frame[col]=frame[col].astype(float)
    independent=frame.groupby(['profile','method','family'])[['joint_lower','joint_upper','unknown']].mean().groupby(['profile','method']).mean()
    reported=pd.read_csv(OUT/'evaluation_summary.csv').set_index(['profile','method'])
    summary_error=float(np.max(np.abs(independent.to_numpy()-reported.loc[independent.index,independent.columns].to_numpy())))
    assert summary_error<1e-12
    # Independently aggregate exact random expectations from all candidates.
    random_family=defaultdict(list)
    for profile in cfg['qc_profiles']:
        for task in tasks:
            passed=unknown=0
            for compound in identities:
                a=qc[profile,compound,task['target']];b=qc[profile,compound,task['avoid']]
                success=a=='strong_competition' and b=='limited_competition'
                failure=a in ['limited_competition','intermediate_competition'] or b in ['strong_competition','intermediate_competition']
                passed+=success;unknown+=not success and not failure
            random_family[profile,task['family']].append(np.array([passed,passed+unknown,unknown])/len(identities))
    random_error=0.0
    for profile in cfg['qc_profiles']:
        expected=np.mean([np.mean(v,axis=0) for (p,f),v in random_family.items() if p==profile],axis=0)
        observed=reported.loc[(profile,'random_expected'),['joint_lower','joint_upper','unknown']].to_numpy(dtype=float)
        random_error=max(random_error,float(np.max(np.abs(expected-observed))))
    assert random_error<1e-12
    full=pd.read_csv(OUT/'fold_predictions.csv');mean=full.groupby('row_id')[cfg['models']].mean()
    averaged=pd.read_csv(OUT/'predictions.csv').set_index('row_id')
    assert (full.groupby('row_id').fold.nunique()==5).all()
    averaging_error=float(np.max(np.abs(mean.to_numpy()-averaged.loc[mean.index,cfg['models']].to_numpy())))
    assert averaging_error<1e-10
    result={'state':'passed','selection_lists_independently_checked':checks,
        'outcome_rows_independently_checked':len(outcome),'label_max_error':max(errors),
        'summary_max_error':summary_error,'fold_average_max_error':averaging_error,
        'random_expectation_max_error':random_error,'continuous_outcomes_checked':True,
        'files':{n:hashlib.sha256((OUT/n).read_bytes()).hexdigest() for n in ['predictions.csv','recommendations.csv','recommendation_outcomes.csv','evaluation_summary.csv','evaluation.json']},
        'limitations':'Frozen-input hashes, rankings, operational labels, per-recommendation continuous signals, summary lower/upper/unknown metrics, random expectations and fold means checked. Resampling intervals not independently reimplemented. No independent retraining or biological label validation.'}
    (OUT/'evaluation_verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))


if __name__=='__main__':main()
