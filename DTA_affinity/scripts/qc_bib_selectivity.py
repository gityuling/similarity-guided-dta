#!/usr/bin/env python3
"""Development QC sensitivity audit; no model predictions or candidate rankings."""
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'reports/bib_selectivity_20261021'
CONFIG = ROOT/'configs/bib_selectivity_qc_20261021.json'


def classify(row, profile, on_cut=0.5, off_cut=0.8):
    if row['state'] != 'finite_signal_not_qc_certified':
        return 'unknown', row['state']
    keys = ['dmso_variance','variance_100nm','variance_1000nm','residual_100nm','residual_1000nm']
    try:
        v = {k:float(row[k]) for k in keys}
    except (ValueError,KeyError,TypeError):
        return 'unknown', 'missing_number'
    if any(not math.isfinite(x) or x < 0 for x in v.values()):
        return 'unknown', 'invalid_number'
    if any(math.sqrt(v[k]) > profile['max_sd'] for k in keys[:3]):
        return 'unknown', 'signal_variability'
    if v['residual_1000nm']-v['residual_100nm'] > profile['max_dose_inversion']:
        return 'unknown', 'dose_inversion'
    residual = v['residual_1000nm']
    if residual <= on_cut:
        return 'strong_competition', 'eligible'
    if residual >= off_cut:
        return 'limited_competition', 'eligible'
    return 'intermediate_competition', 'eligible'


def panel_decision(own, others):
    # Fail takes precedence only when at least one condition is observed to fail.
    if own in ('limited_competition','intermediate_competition'):
        return 'fail'
    if any(s in ('strong_competition','intermediate_competition') for s in others):
        return 'fail'
    if own == 'strong_competition' and all(s == 'limited_competition' for s in others):
        return 'pass'
    return 'unknown'


def main():
    config=json.loads(CONFIG.read_text())
    source=OUT/'measurement_states.csv'
    with source.open() as f: rows=list(csv.DictReader(f))
    genes=sorted({r['gene'] for r in rows})
    grouped=defaultdict(dict)
    for r in rows:
        assert r['gene'] not in grouped[r['compound']]
        grouped[r['compound']][r['gene']]=r
    assert all(set(v)==set(genes) for v in grouped.values())
    summaries={}; target_rows=[]
    with (OUT/'qc_measurement_states.csv').open('w',newline='') as f, (OUT/'qc_panel_decisions.csv').open('w',newline='') as d:
        writer=csv.DictWriter(f,fieldnames=['profile','compound','gene','state','reason'])
        decision_writer=csv.DictWriter(d,fieldnames=['profile','compound','target','decision'])
        writer.writeheader();decision_writer.writeheader()
        for name,profile in config['profiles'].items():
            counts=Counter(); reasons=Counter(); decisions=Counter(); per_target=defaultdict(Counter)
            for compound,entries in grouped.items():
                classified={g:classify(entries[g],profile,config['target_max_residual'],config['off_target_min_residual']) for g in genes}
                state_counts=Counter(s for s,_ in classified.values())
                for g,(state,reason) in classified.items():
                    writer.writerow(dict(profile=name,compound=compound,gene=g,state=state,reason=reason))
                    counts[state]+=1; reasons[reason]+=1
                    others=state_counts.copy();others[state]-=1
                    decision=panel_decision(state,[s for s,n in others.items() if n>0])
                    decision_writer.writerow(dict(profile=name,compound=compound,target=g,decision=decision))
                    decisions[decision]+=1;per_target[g][decision]+=1
                    per_target[g]['qc_eligible']+=state!='unknown'
                    per_target[g]['strong_competition']+=state=='strong_competition'
            for g,c in per_target.items():
                target_rows.append(dict(profile=name,gene=g,**{k:c[k] for k in ['qc_eligible','strong_competition','pass','fail','unknown']}))
            summaries[name]={'states':dict(counts),'reasons':dict(reasons),'panel_decisions':dict(decisions),
                'targets_with_any_panel_pass':sum(c['pass']>0 for c in per_target.values()),
                'targets_with_three_panel_passes':sum(c['pass']>=3 for c in per_target.values()),
                'targets_with_at_least_three_strong_candidates':sum(c['strong_competition']>=3 for c in per_target.values())}
    with (OUT/'qc_target_feasibility.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(target_rows[0]));writer.writeheader();writer.writerows(target_rows)
    report={'config':config,'profiles':summaries,'candidate_count':len(grouped),'gene_count':len(genes),
        'model_scores_loaded':False,'inputs':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source,CONFIG,Path(__file__)]}}
    (OUT/'qc_feasibility.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(summaries,indent=2))


if __name__=='__main__':main()
