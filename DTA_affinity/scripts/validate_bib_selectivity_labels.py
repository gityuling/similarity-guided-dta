#!/usr/bin/env python3
"""Compare published full-dose observations to two-dose signals; no model effects."""
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from openpyxl import load_workbook

from qc_bib_selectivity import classify

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'reports/bib_selectivity_20261021'
SOURCE=ROOT/'reports/bib_mechanism_20261021/sources/reinecke_targets.xlsx'


def number(x):
    try:
        x=float(x)
        return x if math.isfinite(x) else None
    except (ValueError,TypeError):return None


def category(x):
    if x is None or x<0:return 'unknown'
    return 'strong_competition' if x<=0.5 else ('limited_competition' if x>=0.8 else 'intermediate_competition')


def main():
    config=json.loads((ROOT/'configs/bib_selectivity_qc_20261021.json').read_text())
    with (OUT/'measurement_states.csv').open() as f:
        measurements={(r['compound'],r['gene']):r for r in csv.DictReader(f)}
    book=load_workbook(SOURCE,read_only=True,data_only=True)
    matched=[];allrows=[]
    for sheet in ['Kinobeads - eight dose','SYK inhibitors']:
        values=list(book[sheet].values);headers=[str(h).strip() for h in values[0]]
        for index,values in enumerate(values[1:],start=2):
            r=dict(zip(headers,values));key=(r['Compound'],r['Gene Name'])
            one=number(r['Relative Intensity 1000nM']);hundred=number(r['Relative Intensity 100nM'])
            item=dict(compound=key[0],gene=key[1],sheet=sheet,source_row=index,
                full_dose_residual_100nm=hundred,full_dose_residual_1000nm=one,
                full_dose_class=category(one),full_dose_r2=number(r['R2']),
                full_dose_kd=number(r['apparent Kd']),matched_to_current_panel=key in measurements)
            allrows.append(item.copy())
            if key not in measurements:continue
            raw=measurements[key];two=number(raw['residual_1000nm'])
            item.update(two_dose_residual_1000nm=two,two_dose_class=category(two),
                        absolute_residual_difference=abs(two-one) if two is not None and one is not None else None)
            for name,profile in config['profiles'].items():
                item[name+'_qc_class']=classify(raw,profile)[0]
            matched.append(item)
    for filename,rows in [('full_dose_available_rows.csv',allrows),('full_dose_label_comparison.csv',matched)]:
        with (OUT/filename).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    summaries={}
    for profile in ['all_numeric']+list(config['profiles']):
        chosen=[r for r in matched if r['full_dose_class']!='unknown' and
            (r['two_dose_class']!='unknown' if profile=='all_numeric' else r[profile+'_qc_class']!='unknown')]
        confusion=Counter(r['full_dose_class']+' -> '+r['two_dose_class'] for r in chosen)
        summaries[profile]={'pairs':len(chosen),'compounds':len({r['compound'] for r in chosen}),
            'class_agreements':sum(r['full_dose_class']==r['two_dose_class'] for r in chosen),
            'mean_absolute_residual_difference':sum(r['absolute_residual_difference'] for r in chosen)/len(chosen) if chosen else None,
            'confusion_full_dose_to_two_dose':dict(confusion),
            'full_dose_r2_below_0_8':sum(r['full_dose_r2'] is not None and r['full_dose_r2']<0.8 for r in chosen)}
    report={'available_rows':len(allrows),'matched_rows':len(matched),'summaries':summaries,
        'limitations':['Full-dose follow-up compounds were selected by original authors, not a random negative-validation panel.',
            'Same publication and assay technology: technical cross-check, not an independent external confirmation.',
            'Direct 1uM observations compared; R2 is reported, not used to cherry-pick rows.',
            'No full-dose replicate error estimates are used; observed agreement is not a negative-label certification.',
            'Unmatched compound names/protein groups excluded without fuzzy matching.'],
        'workbook_sha256':hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        'model_scores_loaded':False}
    (OUT/'full_dose_label_validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
