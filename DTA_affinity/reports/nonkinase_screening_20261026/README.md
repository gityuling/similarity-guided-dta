# ESR1 candidate prioritization

This case extends Results 5 to the nuclear receptor ESR1. ChEMBL Ki records
provide affinity training data; PubChem AID 1223 provides the independent
competitive-binding screening outcomes.

The final references contain 360 compounds (380 measurements, 43 documents,
45 assays). All 63 screening compounds are held out by standardized parent
identity. One training compound failed the standard conformer generator;
its identity is recorded in the configuration and curation audit. The screen
contains 17 actives and 46 inactives.

The existing full SGF-DTA network was trained from scratch on the ESR1 affinity
labels with fixed Uni-Mol2 and ESM-2 representations. Five grouped validation
partitions select checkpoints and fusion parameters. The external result is
the mean prediction across the five fitted models. Validation scores serve
model selection; the screening set provides the reported evaluation.

At Top-10, the deep model, neighbor predictor, equal blend, constant blend,
similarity gate, and structure-only ranking retrieve 2, 4, 2, 3, 3, and 4
actives, respectively. The gate has AP 0.30413 versus 0.22139 for the deep
model; structure-only ranking has the highest AP, 0.42161. This case therefore
shows modest gains over the deep model and stronger simpler reference-based
rankings. It is not evidence of universal cross-family superiority.

## Reproduction

Run commands from the directory containing `DTA_affinity/`. Frozen data,
configuration, fitting outputs, and source metadata are retained here and in
`results/nonkinase_screening_20261026/`.

```bash
python DTA_affinity/scripts/evaluate_nonkinase_screening.py
python DTA_affinity/scripts/verify_nonkinase_screening.py
```

The first command needs NumPy, pandas, and scikit-learn. The independent
verification additionally needs RDKit. Network replay uses PyTorch and CUDA:

```bash
python DTA_affinity/scripts/verify_nonkinase_screening.py --replay-network
```

The companion `nonkinase_screening_20261026.tar.gz` contains all five model
checkpoints, the cached molecular/protein features, and their supporting
scripts. A fresh training run requires a new output directory or an archived
copy of `results/nonkinase_screening_20261026/`; the runner resumes completed
folds by default. Keep the supplied configuration and input files unchanged
when replaying these results.

The preparation and execution order is:

1. `fetch_nonkinase_screening.py` retrieves the ChEMBL records and assay metadata;
   the original PubChem and UniProt downloads are cached under `sources/`.
2. `prepare_nonkinase_screening.py` creates the reference table, folds, candidate
   identities, and separate outcome file.
3. `build_kiba_benchmark_tokens.py --data-table
   DTA_affinity/reports/nonkinase_screening_20261026/feature_table.csv
   --cache-dir DTA_affinity/data/processed/nonkinase_screening_20261026/tokens
   --drug-batch-size 1 --protein-batch-size 1` creates frozen encoder features.
   Its historical filename is retained; the table argument selects ESR1.
4. `run_nonkinase_screening.py` fits the five affinity models and fusion rules.
5. The evaluation and verification commands above compute the reported results.

`verification.json` records independent neighbor/fusion/metric checks and replay
of all 315 external fold predictions. `sources/provenance.json` records URLs
and hashes. `training_record_audit.csv` retains every source record and its
inclusion or exclusion reason. Source data retain their original attribution:
ChEMBL (EMBL-EBI; CC BY-SA 3.0), PubChem AID 1223 (University of New Mexico
Molecular Libraries Screening Center), and UniProt P03372.
