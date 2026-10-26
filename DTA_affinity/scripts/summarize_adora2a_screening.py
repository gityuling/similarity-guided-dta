#!/usr/bin/env python3
"""Write the completed ADORA2A case record from verified predictions."""
import json
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
REPORT = PROJECT / 'reports/adora2a_screening_20261026'


def main():
    verification = json.loads((REPORT / 'verification.json').read_text())
    assert verification['state'] == 'passed' and verification['network_replayed']
    prep = json.loads((REPORT / 'preparation.json').read_text())
    analysis = json.loads((REPORT / 'analysis.json').read_text())
    scores = pd.read_csv(REPORT / 'candidate_scores.csv')
    names = json.loads((REPORT / 'drug_names_by_parent.json').read_text())
    scores['compound_name'] = scores.parent_key.map(lambda x: ' | '.join(names.get(x, [])))
    methods = analysis['metrics']
    records = []
    for method in methods:
        name = method['model']
        top = scores.sort_values([name, 'source_id'], ascending=[False, True]).head(10).copy()
        top['rank'] = range(1, len(top) + 1)
        top['model'] = name
        top['prediction'] = top[name]
        records.append(top[['model', 'rank', 'source_id', 'compound_name', 'cid', 'sid', 'parent_key',
                            'prediction', 'outcome', 'relation', 'AC50_nM']])
    pd.concat(records, ignore_index=True).to_csv(REPORT / 'top10_candidates.csv', index=False)
    metric_lines = '\n'.join(
        f"- {r['model']}: Top-10 {r['top10_hits']}/10; enrichment "
        f"{r['top10_enrichment']:.3f}; AP {r['average_precision']:.5f}; "
        f"ROC-AUC {r['roc_auc']:.5f}." for r in methods)
    text = f'''# ADORA2A candidate prioritization

This second exploratory non-kinase case evaluates SGF-DTA on human ADORA2A,
a GPCR. The target was selected for its larger public candidate panel and
direct binding endpoint after the ESR1 case. The ESR1 results remain in
`reports/nonkinase_screening_20261026/` and `case_history.json`.

## Data and protocol

ChEMBL CHEMBL251 supplies exact positive Ki measurements from human binding
assays with confidence 9, literature provenance and no annotated variant.
Repeated measurements of each standardized stereochemical structure are
summarized by median pKi. The final training references contain
{prep['training_compounds']} compounds, {prep['eligible_measurements']} measurements,
{prep['training_parent_groups']} connectivity groups, {prep['documents']} documents
and {prep['assays']} assays.

The external panel is PubChem AID 1961683 / ChEMBL CHEMBL5291726, from the
Secondary Pharmacology Database (Sutherland et al., Nature Communications,
2023; DOI 10.1038/s41467-023-40064-9). The original author annotation in
Zenodo 7378746, `SI_tables_release.xlsx`, Table S2 row 8, identifies assay
3177 as human ADORA2A binding measured by scintillation proximity assay.
`target_annotation.json` retains both this annotation and the ChEMBL
homologous-target mapping flag.

The activity threshold is AC50 <= 10 micromolar, following the concentration
cutoff in the PubChem assay presentation. Exact measurements above this
threshold and strict lower bounds of at least 10 micromolar form the
negative class. Threshold-ambiguous bounds are excluded. The deposited
`Unspecified` category is resolved using concentration measurements and
their relation signs. Missing structures and repeated standardized parent
groups are excluded without choosing representatives by activity.

The final candidate pool contains {prep['external_compounds']} compounds:
{prep['external_outcomes'].get('Active', 0)} active and
{prep['external_outcomes'].get('Inactive', 0)} inactive at the threshold.
All candidate parent identities, including excluded candidate records, are
held out from the training references. The source study is also excluded
from training. Standard conformer failures affect
{prep['conformer_failure_compounds']} training structures; complete identities
and exclusion reasons are recorded in the configuration and curation audit.

The full affinity network, frozen Uni-Mol2 and ESM-2 representations, training
settings and fusion search follow the ESR1 experiment. Five validation
partitions split chemical parent groups; training seed 20261012 is fixed.
All affinity heads are trained on ADORA2A Ki labels. Each partition selects
its checkpoint, neighbor settings and fusion parameters using its validation
compounds. External predictions are the mean of the five fitted models.
Top-10 is the primary screening budget; budgets 5 and 20 are secondary.
External AC50 labels enter only the separate evaluation step.

## Results

Random Top-10 selection yields {analysis['random_top10_expected_hits']:.5f}
active compounds on average.

{metric_lines}

At the primary Top-10 budget, the deep learning model retrieved two active
compounds and SGF retrieved one. SGF achieved higher average precision
(0.08676 versus 0.06674) and ROC-AUC (0.73812 versus 0.70413) over the full
candidate pool. Early retrieval and overall ranking therefore favor
different predictors in this case.

The complete rankings, all screening budgets and individual-model results
are in `candidate_scores.csv`, `budget_curves.csv`, `top10_candidates.csv`
and `fold_metrics.csv`. These five models share the same external screen;
their variation describes fitting stability. The experiment is a
retrospective evaluation of a compound panel, with Ki predictions used for
ranking AC50 outcomes.

## Reproduction

Run from the directory containing `DTA_affinity/`:

```bash
python DTA_affinity/scripts/prepare_adora2a_screening.py
python DTA_affinity/scripts/build_kiba_benchmark_tokens.py --data-table DTA_affinity/reports/adora2a_screening_20261026/feature_table.csv --cache-dir DTA_affinity/data/processed/adora2a_screening_20261026/tokens --drug-batch-size 2 --protein-batch-size 1
python DTA_affinity/scripts/run_adora2a_screening.py
python DTA_affinity/scripts/evaluate_adora2a_screening.py
python DTA_affinity/scripts/verify_adora2a_screening.py --replay-network
python DTA_affinity/scripts/summarize_adora2a_screening.py
```

`sources/provenance.json` records source URLs and SHA-256 hashes. The runner
freezes configurations and input hashes before fitting, and resumes completed
folds. All five checkpoints are retained in
`results/adora2a_screening_20261026/`. Use an empty output directory for a
fresh training run. `verification.json` records independent reconstruction
of neighbor predictions, fusion arithmetic, screening metrics and replay
of all saved neural predictions.
'''
    (REPORT / 'README.md').write_text(text)
    history = json.loads((REPORT / 'case_history.json').read_text())
    history['cases'][-1].update(status='completed', metrics=methods)
    (REPORT / 'case_history.json').write_text(json.dumps(history, indent=2) + '\n')
    print(metric_lines)


if __name__ == '__main__':
    main()
