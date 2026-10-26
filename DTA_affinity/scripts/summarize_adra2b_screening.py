#!/usr/bin/env python3
"""Write the completed ADRA2B case record from verified predictions."""
import json
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[1]
REPORT = PROJECT / 'reports/adra2b_screening_20261026'


def main():
    verification = json.loads((REPORT / 'verification.json').read_text())
    assert verification['state'] == 'passed' and verification['network_replayed']
    prep = json.loads((REPORT / 'preparation.json').read_text())
    analysis = json.loads((REPORT / 'analysis.json').read_text())
    scores = pd.read_csv(REPORT / 'candidate_scores.csv')
    names = json.loads((REPORT / 'drug_names_by_parent.json').read_text())
    scores['compound_name'] = scores.parent_key.map(lambda x: ' | '.join(names.get(x, [])))
    methods = analysis['metrics']
    metric_map = {r['model']: r for r in methods}
    deep, sgf = metric_map['deep'], metric_map['similarity_gate']
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
    text = f'''# ADRA2B candidate prioritization

This third exploratory non-kinase case evaluates SGF-DTA on human ADRA2B,
a GPCR. The target was selected for its larger public candidate panel and
direct binding endpoint after the ESR1 and ADORA2A cases. Both earlier cases
remain in their report directories and in `case_history.json`.

## Data and protocol

ChEMBL CHEMBL1942 supplies exact positive Ki measurements from human binding
assays with confidence 9, literature provenance and no annotated variant.
Repeated measurements of each standardized stereochemical structure are
summarized by median pKi. The final training references contain
{prep['training_compounds']} compounds, {prep['eligible_measurements']} measurements,
{prep['training_parent_groups']} connectivity groups, {prep['documents']} documents
and {prep['assays']} assays.

The external panel is PubChem AID 1961694 / ChEMBL CHEMBL5291737, from the
Secondary Pharmacology Database (Sutherland et al., Nature Communications,
2023; DOI 10.1038/s41467-023-40064-9). The original author annotation in
Zenodo 7378746, `SI_tables_release.xlsx`, Table S2 row 19, identifies assay
3159 as human ADRA2B binding measured by scintillation proximity assay.
`target_annotation.json` retains both this annotation and the ChEMBL
homologous-target mapping flag.

The activity threshold is AC50 <= 10 micromolar, using the same prespecified concentration
cutoff as the preceding ADORA2A experiment. Exact measurements above this
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
All affinity heads are trained on ADRA2B Ki labels. Each partition selects
its checkpoint, neighbor settings and fusion parameters using its validation
compounds. External predictions are the mean of the five fitted models.
Top-10 is the primary screening budget; budgets 5 and 20 are secondary.
External AC50 labels enter only the separate evaluation step.

## Results

Random Top-10 selection yields {analysis['random_top10_expected_hits']:.5f}
active compounds on average.

{metric_lines}

At the primary Top-10 budget, the deep learning model retrieved
{deep['top10_hits']} active compounds and SGF retrieved {sgf['top10_hits']}.
Average precision was {deep['average_precision']:.5f} for the deep learning
model and {sgf['average_precision']:.5f} for SGF; corresponding ROC-AUC values
were {deep['roc_auc']:.5f} and {sgf['roc_auc']:.5f}.

The fixed-weight fusion and neighbor predictor retrieved
{metric_map['fixed_blend']['top10_hits']} and
{metric_map['neighbor_prior']['top10_hits']} actives, respectively. Ranking by
structural similarity alone retrieved {metric_map['pure_similarity']['top10_hits']}
actives, with average precision {metric_map['pure_similarity']['average_precision']:.5f}.
SGF provides a small improvement over the deep learning model in this case;
structural similarity gives the strongest retrieval among the six methods.

The complete rankings, all screening budgets and individual-model results
are in `candidate_scores.csv`, `budget_curves.csv`, `top10_candidates.csv`
and `fold_metrics.csv`. These five models share the same external screen;
their variation describes fitting stability. The experiment is a
retrospective evaluation of a compound panel, with Ki predictions used for
ranking AC50 outcomes.

## Reproduction

Run from the directory containing `DTA_affinity/`. Public source files and
all fitted checkpoints are retained locally. Molecular tokens reused from
earlier experiments have identical structures, encoder weight hashes and
extractor code hashes; `tokens/verified_reuse.json` records each copied matrix.
An empty cache regenerates the complete feature set.

Commands:

```bash
python DTA_affinity/scripts/fetch_adra2b_screening.py
python DTA_affinity/scripts/prepare_adra2b_screening.py
python DTA_affinity/scripts/build_kiba_benchmark_tokens.py --data-table DTA_affinity/reports/adra2b_screening_20261026/feature_table.csv --cache-dir DTA_affinity/data/processed/adra2b_screening_20261026/tokens --drug-batch-size 2 --protein-batch-size 1
python DTA_affinity/scripts/run_adra2b_screening.py
python DTA_affinity/scripts/evaluate_adra2b_screening.py
python DTA_affinity/scripts/verify_adra2b_screening.py --replay-network
python DTA_affinity/scripts/summarize_adra2b_screening.py
```

`sources/provenance.json` records source URLs and SHA-256 hashes. The runner
freezes configurations and input hashes before fitting, and resumes completed
folds. All five checkpoints are retained in
`results/adra2b_screening_20261026/`. Use an empty output directory for a
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
