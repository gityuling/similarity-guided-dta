#!/usr/bin/env python3
"""Independent coverage checks and processed-file pilot inventory."""
import csv
import hashlib
import json
import math
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/bib_selectivity_20261021'


def read(path):
    with path.open(newline='') as f:
        return list(csv.DictReader(f))


def main():
    audit = json.loads((OUT/'audit.json').read_text())
    compounds = read(OUT/'compound_audit.csv')
    candidates = {r['compound'] for r in compounds if r['primary_candidate'] == 'True'}
    genes = {r['gene'] for r in read(OUT/'target_coverage.csv')}
    rows = read(ROOT/'reports/bib_mechanism_20261021/sources/reinecke_matrix.csv')
    finite = {r['Gene Name']: {c for c in candidates if r[c] and
              math.isfinite(float(r[c])) and float(r[c]) > 0} for r in rows if r['Gene Name'] in genes}
    for r in read(OUT/'target_coverage.csv'):
        assert len(finite[r['gene']]) == int(r['finite_measurements'])
    for r in read(OUT/'pair_coverage.csv'):
        assert len(finite[r['target_a']] & finite[r['target_b']]) == int(r['joint_finite_candidates'])
    assert sum(map(len, finite.values())) == audit['finite_cells']
    assert len(candidates)*len(genes)-sum(map(len,finite.values())) == audit['unknown_cells']
    for r in compounds:
        if r['primary_candidate'] == 'True':
            assert all(r[f] == 'False' for f in ['kiba_seen','davis_seen','old_results5_68','old_braf','old_nanobret','parent_duplicate'])
    manifest = read(OUT/'processed_file_candidates.csv')
    assert {r['name'][:-10] for r in manifest} == candidates
    files = {r['name']: r for r in manifest}
    pilots = []
    for path in sorted((OUT/'sources').glob('*_final.csv')):
        assert path.stat().st_size == int(files[path.name]['size'])
        rows = read(path)
        assert len(rows) > 0
        required = ['Gene.names','Picked.intensity','Target.probability','Target.annotation',
                    'Median.Normalized.LFQ.intensity.100nM','Median.Normalized.LFQ.intensity.1000nM']
        assert all(c in rows[0] for c in required)
        pilots.append({'file':path.name,'rows':len(rows),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'columns':list(rows[0]),'annotation_counts':dict(Counter(r['Target.annotation'] for r in rows)),
            'exact_single_gene_overlap':len({r['Gene.names'] for r in rows}&genes)})
    result = {'state':'passed','target_counts_checked':len(genes),
        'pair_counts_checked':len(read(OUT/'pair_coverage.csv')),
        'candidate_file_matches':len(manifest),'matched_file_bytes':sum(int(r['size']) for r in manifest),
        'downloaded_files_checked':len(pilots),'files':pilots,
        'all_candidate_files_present':len(pilots)==len(manifest),
        'not_verified':['Processed measurements have not been QC certified.',
            'Target.annotation integer code semantics not established; no automatic negative labeling.',
            'Canonical mappings and historical structure exclusions reused from previous audit, not independently rederived here.']}
    (OUT/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))


if __name__ == '__main__':
    main()
