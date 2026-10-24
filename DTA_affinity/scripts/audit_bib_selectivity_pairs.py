#!/usr/bin/env python3
"""Report all ordered target pairs; coverage eligibility never uses success counts."""
import csv
import json
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parents[1]/'reports/bib_selectivity_20261021'


def main():
    by_profile={}
    with (OUT/'qc_measurement_states.csv').open() as f:
        for r in csv.DictReader(f):
            by_profile.setdefault(r['profile'],{}).setdefault(r['compound'],{})[r['gene']]=r['state']
    rows=[];summary={}
    for profile,compounds in by_profile.items():
        genes=sorted(next(iter(compounds.values())))
        x=np.array([[compounds[c][g] for g in genes] for c in sorted(compounds)])
        valid=(x!='unknown').astype(np.int64)
        strong=(x=='strong_competition').astype(np.int64)
        limited=(x=='limited_competition').astype(np.int64)
        joint=valid.T@valid;passing=strong.T@limited
        passing_with_unknown=strong.T@(x=='unknown').astype(np.int64)
        # A success needs observed target action and observed limited action at B.
        unknown=((x=='unknown').astype(np.int64).T@limited)+passing_with_unknown+((x=='unknown').astype(np.int64).T@(x=='unknown').astype(np.int64))
        for i,a in enumerate(genes):
            for j,b in enumerate(genes):
                if i==j:continue
                rows.append(dict(profile=profile,target=a,avoid=b,joint_qc_candidates=int(joint[i,j]),
                    coverage_eligible_50=bool(joint[i,j]>=50),
                    operational_passes=int(passing[i,j]),unknown=int(unknown[i,j]),
                    operational_failures=len(compounds)-int(passing[i,j])-int(unknown[i,j])))
        eligible=[r for r in rows if r['profile']==profile and r['coverage_eligible_50']]
        summary[profile]={'ordered_pairs':len(genes)*(len(genes)-1),'coverage_eligible_pairs':len(eligible),
            'eligible_pairs_with_three_operational_passes':sum(r['operational_passes']>=3 for r in eligible)}
    with (OUT/'qc_pair_feasibility.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (OUT/'qc_pair_feasibility.json').write_text(json.dumps({'profiles':summary,
        'limitation':'All-pair development inventory, not a frozen biological avoidance panel or a model efficacy result. Coverage criterion >=50 is descriptive.'},indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__':main()
