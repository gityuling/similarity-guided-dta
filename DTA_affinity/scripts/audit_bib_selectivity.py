#!/usr/bin/env python3
"""Audit selectivity feasibility; no model loading, scoring, or candidate ranking."""
from pathlib import Path
import hashlib
import json
import itertools
import numpy as np
import pandas as pd
from openpyxl import load_workbook

ROOT = Path(__file__).resolve().parents[1]
PREV = ROOT / 'reports/bib_mechanism_20261021'
OUT = ROOT / 'reports/bib_selectivity_20261021'


def record(path):
    return {'path': str(path.relative_to(ROOT)),
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    OUT.mkdir(exist_ok=True)
    workbook = PREV / 'sources/reinecke_targets.xlsx'
    w = load_workbook(workbook, read_only=True, data_only=True)
    values = list(w['Kinobeads Drugmatrix - all'].values)
    matrix = pd.DataFrame(values[1:], columns=values[0]).set_index('Gene Name').astype(float)
    old = pd.read_csv(PREV / 'sources/reinecke_matrix.csv').set_index('Gene Name')
    assert matrix.index.equals(old.index) and matrix.columns.equals(old.columns)
    np.testing.assert_allclose(matrix.values, old.values, rtol=1e-12, equal_nan=True)
    invalid = matrix.stack()
    invalid = invalid[(invalid <= 0) | ~np.isfinite(invalid)]
    invalid.rename('source_value').to_csv(OUT / 'invalid_affinities.csv')
    matrix = matrix.where((matrix > 0) & np.isfinite(matrix))
    annotations = list(w['Target annotation'].values)
    ann = pd.DataFrame(annotations[1:], columns=annotations[0]).set_index('Gene')
    cov = pd.read_csv(PREV / 'external_compound_coverage.csv', keep_default_na=False)
    mapping = pd.read_csv(PREV / 'external_target_mapping.csv', keep_default_na=False)
    for col in ['valid_structure', 'kiba_seen', 'davis_seen', 'kiba_scaffold_seen',
                'davis_scaffold_seen', 'old_results5_68', 'old_braf', 'old_nanobret']:
        cov[col] = cov[col].astype(str).str.lower().eq('true')
    # Parent-level exclusions prevent aliases from evading historical exposure.
    flags = ['kiba_seen', 'davis_seen', 'kiba_scaffold_seen', 'davis_scaffold_seen',
             'old_results5_68', 'old_braf', 'old_nanobret']
    grouped = cov.groupby('parent_key')[flags].transform('max')
    cov[flags] = grouped
    cov['affinity_external'] = cov.valid_structure & ~cov.kiba_seen & ~cov.davis_seen
    cov['development_external'] = cov.affinity_external & ~cov[flags[4:]].any(axis=1)
    cov['scaffold_external'] = cov.development_external & ~cov[flags[2:4]].any(axis=1)
    cov['parent_duplicate'] = cov.parent_key.duplicated(keep=False)
    # Exclude ambiguous same-parent series rather than select a favorable measurement.
    cov['primary_candidate'] = cov.development_external & ~cov.parent_duplicate
    cov.to_csv(OUT / 'compound_audit.csv', index=False)
    genes = mapping.loc[mapping.status.eq('unique_canonical_match'), 'source_gene']
    genes = sorted(g for g in genes if g in ann.index and
                   ann.loc[g, 'DirectBinder'] == 'x' and
                   ann.loc[g, 'Protein/lipid Kinase'] == 'x')
    names = cov.loc[cov.primary_candidate, 'compound'].tolist()
    x = matrix.loc[genes, names]
    measured = x.notna()
    # Only coverage, not potency thresholds, determines feasibility labels.
    targets = []
    for g in genes:
        targets.append({'gene': g, 'protein_id': mapping.set_index('source_gene').loc[g, 'protein_id'],
                        'candidate_count': len(names), 'finite_measurements': int(measured.loc[g].sum()),
                        'unknown_measurements': int((~measured.loc[g]).sum())})
    pd.DataFrame(targets).to_csv(OUT / 'target_coverage.csv', index=False)
    pairs = []
    for a, b in itertools.combinations(genes, 2):
        n = int((measured.loc[a] & measured.loc[b]).sum())
        pairs.append({'target_a': a, 'target_b': b, 'joint_finite_candidates': n,
                      'coverage_ge_10': n >= 10, 'coverage_ge_20': n >= 20})
    pairs = pd.DataFrame(pairs)
    pairs.to_csv(OUT / 'pair_coverage.csv', index=False)
    # Full fixed modeled panel; finite entries alone do not establish broad selectivity.
    complete = measured.all(axis=0)
    compound_counts = pd.DataFrame({'compound': names,
        'finite_modeled_kinases': measured.sum(axis=0).values,
        'unknown_modeled_kinases': (~measured).sum(axis=0).values,
        'complete_modeled_panel': complete.values})
    compound_counts.to_csv(OUT / 'candidate_measurement_coverage.csv', index=False)
    legend = [' | '.join(str(v) for v in row if v is not None) for row in w['Legend'].values]
    (OUT / 'source_legend.txt').write_text('\n'.join(legend))
    summary = {'stage': 'coverage audit only; not a confirmatory cohort freeze',
        'source_compound_rows': len(cov), 'unique_parents': cov.parent_key.nunique(),
        'nonpositive_or_nonfinite_source_values': len(invalid),
        'duplicate_parent_rows': int(cov.parent_duplicate.sum()),
        'affinity_external_rows': int(cov.affinity_external.sum()),
        'development_external_rows': int(cov.development_external.sum()),
        'scaffold_external_rows': int(cov.scaffold_external.sum()),
        'primary_unique_parent_candidates': len(names), 'mapped_direct_kinases': len(genes),
        'targets_with_10_finite_candidates': sum(t['finite_measurements'] >= 10 for t in targets),
        'targets_with_20_finite_candidates': sum(t['finite_measurements'] >= 20 for t in targets),
        'target_pairs_with_10_joint_candidates': int(pairs.coverage_ge_10.sum()),
        'target_pairs_with_20_joint_candidates': int(pairs.coverage_ge_20.sum()),
        'complete_modeled_panel_candidates': int(complete.sum()),
        'finite_cells': int(measured.sum().sum()), 'unknown_cells': int((~measured).sum().sum()),
        'raw_workbook_csv_equality_verified': True, 'model_predictions_computed': False,
        'limitations': ['Main matrix contains identified-target apparent Kd estimates, not an explicit negative-assay matrix.',
            'Zero/nonfinite values retained in invalid_affinities.csv and treated as unknown pending source clarification, not weak binding.',
            'Two-dose apparent Kd values are approximate, especially weak interactions.',
            'DirectBinder denotes Kinobead binding, not proven causal drug mechanism.',
            'Mapping reused from canonical-sequence-verified KIBA target audit; no mutant/isoform pooling.',
            'Upstream representation pretraining and complete KIBA source lineage unresolved.',
            'Pair coverage-selected subsets are conditional retrospective evaluations, not open-library screening.',
            'Source targets and measurements have been inspected; no prospective/blind claim.'],
        'inputs': [record(p) for p in [workbook, PREV/'sources/reinecke_matrix.csv',
                    PREV/'external_compound_coverage.csv', PREV/'external_target_mapping.csv', Path(__file__)]]}
    (OUT/'audit.json').write_text(json.dumps(summary, indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k not in ['inputs','limitations']}, indent=2))


if __name__ == '__main__':
    main()
