# Training and inference

The default reproduction check uses saved predictions and runs on CPU. Training
uses the original model code and the released feature caches. The recorded
training environment used Python 3.10.20, PyTorch 2.11.0 with CUDA 12.8, NumPy
1.26.4, SciPy 1.15.2 and RDKit 2022.09.5. The complete package inventory is in
`DTA_affinity/results/bib_validation_20261017/environment.json`.

## Train one affinity model

Install PyTorch for your CUDA environment and RDKit in addition to the replay
dependencies. Load the data and feature assets, then run this command from the
repository root:

```bash
python tools/load_assets.py --groups reproduction-data kiba-features
python DTA_affinity/scripts/train_nexusdti_regressor.py \
  --data-table data/processed/kiba_dtiam_protocol/table.csv \
  --manifest-file data/processed/kiba_dtiam_protocol/folds/drug_coldstart/fold_0.csv \
  --token-dir data/processed/kiba_dtiam_protocol/tokens \
  --target-column kiba_score --dataset-label kiba_drug_id \
  --split reproduction_kiba_f0 --run-tag seed20261012 --seed 20261012 \
  --epochs 20 --batch-size 64 --workers 4 --latent-dim 512 \
  --lr 0.0001 --k-ratio 0.15 --dropout 0.1 --patience 5 \
  --pool-fusion --hybrid --target-normalize --device cuda --save-predictions
```

The new run name writes separate outputs. Repeat for folds 0–4 and seeds
20261012, 20261017 and 20261018. For Davis, use its table and drug-cold manifests,
`--target-column pKd_floor`, and the released cache at
`../dataset/processed_DTI_Token/token_features_davis` (the trainer resolves this
argument relative to `DTA_affinity/`). The strict KIBA protocols use
the manifests under `kiba_identity_20261017` and `kiba_scaffold_20261017` with
training seed 20261012.

## Neighbor prediction and fusion

`evaluate_target_id_similarity_prior.py` retrieves training measurements with the
same experimental target ID, using radius-2, 2,048-bit Morgan fingerprints. It selects
the number of references and similarity exponent on the validation partition.
Use `--stable-neighbors` for KIBA to reproduce its tie ordering.

```bash
python DTA_affinity/scripts/evaluate_target_id_similarity_prior.py \
  --data-table DTA_affinity/data/processed/kiba_dtiam_protocol/table.csv \
  --manifest-file DTA_affinity/data/processed/kiba_dtiam_protocol/folds/drug_coldstart/fold_0.csv \
  --target-column kiba_score --reference-key protein_id --stable-neighbors --output-stem reproduction_kiba_f0_prior
```

`evaluate_affinity_ablation.py` combines the neural and neighbor prediction CSVs
using `--deep`, `--prior`, `--data-table`, `--manifest-file`, `--target-column`
and a new `--output-stem`. Both the constant weight and the similarity gate are
fitted on validation rows. Saved fold JSON files contain the fitted parameters.

For the Davis correction, all three seeds reuse their existing neural outputs:

```bash
python DTA_affinity/scripts/recalculate_davis_target_id.py
python DTA_affinity/scripts/verify_davis_target_id.py
```

This rebuilds five target-ID neighbor pools, refits 15 fusion models and
recomputes the matched analyses. It writes to the dated correction directory;
the original sequence-pool results are retained. `--reference-key Sequence`
is available for reproducing the legacy neighbor calculation.

## Check a saved model

After loading `affinity-checkpoints` and `kiba-features`, run:

```bash
python tools/check_inference.py
```

This loads the first KIBA identity-fold checkpoint, runs CPU inference on four
saved test examples, and compares the predicted affinities with the archived
values. It also checks neighbor and fusion predictions. Full neural retraining
is a separate computation from the CPU result-recalculation check.

Original queue scripts and extraction scripts are retained as source records.
The single-run command above is the portable entry point for a new experiment.
