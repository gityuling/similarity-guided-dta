# ADRA2B candidate prioritization

This third exploratory non-kinase case evaluates SGF-DTA on human ADRA2B,
a GPCR. The target was selected for its larger public candidate panel and
direct binding endpoint after the ESR1 and ADORA2A cases. Both earlier cases
remain in their report directories and in `case_history.json`.

## Data and protocol

ChEMBL CHEMBL1942 supplies exact positive Ki measurements from human binding
assays with confidence 9, literature provenance and no annotated variant.
Repeated measurements of each standardized stereochemical structure are
summarized by median pKi. The final training references contain
294 compounds, 314 measurements,
272 connectivity groups, 67 documents
and 69 assays.

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

The final candidate pool contains 1177 compounds:
160 active and
1017 inactive at the threshold.
All candidate parent identities, including excluded candidate records, are
held out from the training references. The source study is also excluded
from training. Standard conformer failures affect
0 training structures; complete identities
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

Random Top-10 selection yields 1.35939
active compounds on average.

- deep: Top-10 2/10; enrichment 1.471; AP 0.17689; ROC-AUC 0.58274.
- neighbor_prior: Top-10 3/10; enrichment 2.207; AP 0.17335; ROC-AUC 0.59446.
- equal_blend: Top-10 3/10; enrichment 2.207; AP 0.18618; ROC-AUC 0.60242.
- fixed_blend: Top-10 3/10; enrichment 2.207; AP 0.18229; ROC-AUC 0.59534.
- similarity_gate: Top-10 3/10; enrichment 2.207; AP 0.18331; ROC-AUC 0.59224.
- pure_similarity: Top-10 8/10; enrichment 5.885; AP 0.37686; ROC-AUC 0.78732.

At the primary Top-10 budget, the deep learning model retrieved
2 active compounds and SGF retrieved 3.
Average precision was 0.17689 for the deep learning
model and 0.18331 for SGF; corresponding ROC-AUC values
were 0.58274 and 0.59224.

The fixed-weight fusion and neighbor predictor retrieved
3 and
3 actives, respectively. Ranking by
structural similarity alone retrieved 8
actives, with average precision 0.37686.
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
