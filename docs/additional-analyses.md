# Additional analyses

Run commands from the repository root after loading `reproduction-data`.
The default `python tools/reproduce.py` checks the shared-backbone results,
ADRA2B candidate ranking and target-wise summaries alongside the main benchmarks.

## Shared-backbone comparison

The local kNN-DTA, Ada-kNN-DTA and SGF implementations use the same trained
affinity backbone within each fold and seed. Saved test predictions cover
five drug-ID folds and three seeds on each of Davis and KIBA.

To regenerate the summary tables:

```bash
python DTA_affinity/scripts/summarize_knn_shared.py
python DTA_affinity/scripts/build_knn_comparison_tables.py
```

To reconstruct sampled predictions from trained checkpoints, install PyTorch,
then run:

```bash
python tools/load_assets.py --groups shared-backbone-models
python DTA_affinity/scripts/verify_knn_shared.py
```

The verifier compares independent NumPy/SciPy calculations with saved predictions
and checks that the affinity head remains unchanged during adaptive fitting.
Configurations, selected parameters and source hashes are preserved with each run.

## ADRA2B case study

`DTA_affinity/reports/adra2b_screening_20261026/` contains the ChEMBL source
records, training compounds, external assay records, exclusions, labels and
predictions. The external panel contains 1,177 compounds, including 160 active
compounds under the recorded AC50 threshold. The selected top ten contain
rasagiline, pergolide and risperidone as active compounds.

With RDKit installed, the original case verifier checks source labels, parent
structure separation, chemical-neighbor arithmetic, fusion and ranking metrics:

```bash
python DTA_affinity/scripts/verify_adra2b_screening.py
```

The `case-study-models` asset supplies the ADRA2B and ESR1 checkpoints and cached
features. Reports for the exploratory ESR1 and ADORA2A analyses are also retained.

## Case and target-wise figures

Install Matplotlib and Pillow in addition to the replay dependencies:

```bash
pip install matplotlib pillow
python DTA_affinity/manuscript/bib/tools/build_case_studies.py
python DTA_affinity/manuscript/bib/tools/build_target_report.py
```

The builders verify their frozen inputs and export editable SVG and PDF figures.
The target-wise snapshot uses three training seeds for drug-ID partitions and
the original seed for KIBA identity and scaffold partitions, matching the paper.
The original target-level calculations are in the accompanying CSV files.

These analysis commands regenerate derived files. For an unchanged release
inventory, run the default checksum and metric check before regenerating reports.
