# SGF-DTA

Code and reproduction materials for **Similarity-Guided Fusion of Deep Learning
and Chemical Neighbors for Drug–Target Affinity Prediction**.

SGF-DTA combines a deep learning model with measured affinities of chemical
neighbors at the query target. Morgan-fingerprint similarity selects the
references and controls a validation-fitted, two-parameter fusion gate.

## Reproduce the results

Python 3.10 or newer is required. From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-replay.txt
python tools/load_assets.py
python tools/reproduce.py
```

The loader uses the GitHub CLI (`gh`) to download the release data. For a private
repository, sign in with an account that has access. To use previously downloaded
assets, pass `--asset-dir /path/to/assets`.

The CPU check verifies file hashes and partition coverage, recalculates benchmark
MSE and Pearson correlations from individual predictions, checks fusion weights
and fold averaging, and recalculates NanoBRET and ABL1 metrics. Its report is
written to `verification/reproduction.json`.

The main benchmark gate MSEs are 0.58806 for Davis and 0.29783 for KIBA, averaged
over three training seeds. The KIBA scaffold result is 0.36721. ABL1 Top-10
retrieval is 8 versus 5 under deposited labels and 5 versus 5 under the repeat
rule; the latter leaves three additional fusion selections unresolved.

Davis references are grouped by experimental target ID. The correction record
in `DTA_affinity/reports/davis_target_id_20261025/` preserves the original
sequence-pool results and confirms unchanged neural predictions. The gate's
reference-seed Davis MSE is 0.58843; equal blending obtains 0.58402.

## Models and feature caches

Release assets contain the 40 trained affinity checkpoints and the exact feature
arrays used for the benchmark and external analyses. Download them with:

```bash
python tools/load_assets.py --groups all
python tools/reproduce.py --all-files
```

See [the training and inference instructions](docs/training.md) for using these
assets. [Feature provenance](docs/provenance.md) records the encoder sources and
the remaining Davis provenance limitation. The original environment is saved in
`DTA_affinity/results/bib_validation_20261017/environment.json`.

## Files

- `DTA_affinity/scripts/`: training, neighbor retrieval, fusion and evaluation code.
- `DTA_affinity/configs/`: recorded training and analysis settings.
- `DTA_affinity/data/processed/`: benchmark tables, partitions and cache metadata.
- `DTA_affinity/reports/`: per-compound predictions, results and source records.
- `DTA_affinity/results/`: fold results, training records and model checkpoints.
- `manifests/`: release file lists, sizes and SHA-256 hashes.
- `tools/`: portable asset loading and result checks.

The directory layout and historical filenames are retained so recorded paths
and checksums remain usable. `deep` in saved data denotes the deep learning
model; `similarity_gate` denotes SGF-DTA.

## Data sources

Davis and KIBA use the benchmark inputs and drug-ID partition convention from
[DTIAM](https://github.com/CSUBioGroup/DTIAM). Reported baseline scores come from
the [DTIAM source data](https://doi.org/10.6084/m9.figshare.27896790.v1).
The external analyses use the [NanoBRET panel](https://doi.org/10.1021/acs.jmedchem.5c02916)
and [PubChem AID 1963317](https://pubchem.ncbi.nlm.nih.gov/bioassay/1963317).
Source identifiers and measurement-level mappings accompany the saved results.

Third-party source files retain their original notices. The manuscript and
author information are maintained separately from this repository.
