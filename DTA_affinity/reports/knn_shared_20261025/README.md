# Retrieval-method comparison on a shared affinity model

This experiment compares the base affinity model, kNN-DTA, Ada-kNN-DTA, and SGF using the same trained backbone within each fold and seed. The chemical-neighbor predictor is also retained as a reference. The two kNN methods implement the label and representation aggregation equations in [Pei et al., CIKM 2024](https://arxiv.org/html/2407.15202v1). Their aggregation modules and all experimental choices are available in the accompanying code and configuration.

The comparison uses the existing Davis and KIBA drug-ID-cold partitions: five outer folds and training seeds 20261012, 20261017, and 20261018. Drug IDs, experimental target IDs, input structures/sequences, and labels are preserved. Molecular and protein features come from the existing frozen Uni-Mol2 and ESM-2 caches. This controlled experiment uses a common vector-based predictor, with independent molecular and protein projections, so each method can aggregate representations at the same interface.

## Model and selection

Each modality is mean-pooled and standardized with training-pair-weighted statistics. Its projection has a 256-unit hidden layer, layer normalization, GELU, dropout of 0.1, and a 128-dimensional output. The concatenated representations enter a regression head with 256 hidden units, tanh, dropout of 0.1, and one output. Labels use training-set standardization. AdamW uses a learning rate and weight decay of 0.0001, batch size 256, and gradient clipping at 1. Training runs for at most 100 epochs, with validation-MSE selection and patience 10.

All retrieval stores contain training rows only. Pair retrieval uses concatenated drug and protein vectors. Representation stores contain unique molecular inputs and unique protein sequences; the original experimental target IDs remain attached to each affinity record. Squared Euclidean distances are searched exactly, with stable ordering for ties. Pair distances are computed as the sum of drug and protein squared distances, which is algebraically equivalent to concatenation.

kNN-DTA uses the published grids for neighbor counts, temperatures, and blend weights. Neighbor counts are capped at the available datastore size. Separate component searches are followed by two coordinate passes over the drug, protein, and label components. Ada-kNN-DTA uses three two-layer distance-to-weight networks with 32 hidden units and the effective neighbor counts selected for fixed aggregation. The affinity head is frozen. Distance inputs are scaled by their validation-neighbor RMS; final network weights start at zero with half of the probability mass on the query/model and half on neighbors. Adam runs for 5,000 steps with learning rate 0.001 and batch size 256. The lowest full-validation fitting MSE, evaluated every 250 steps and at initialization, selects the state.

SGF reuses the existing validation-selected Morgan neighbor estimates for each fold. It fits the two-parameter similarity gate against the same common model's validation predictions. The reference pool is the experimental target ID, including the corrected 442 separate Davis target pools. Test labels are loaded after all parameters have been selected.

The common backbone, coordinate-search schedule, adaptive optimizer, initialization, distance scaling, and stopping rules are specified implementation choices. The configuration is fixed across all folds and seeds.

## Outputs and verification

- `overall_comparison.csv` and `.tex`: the full SGF-DTA model and the two locally implemented kNN methods, with the backbone identified for each row.
- `controlled_comparison.csv` and `.tex`: the four methods sharing exactly the same backbone within each fold and seed.
- `analysis.json`: aggregate metrics, paired differences, conditional bootstrap intervals, and provenance.
- `summary.csv`: seed-averaged, five-fold means and sample standard deviations.
- `per_run_metrics.csv` and `seed_averaged_folds.csv`: every planned run and fold.
- `oof_predictions.csv`: test predictions with row IDs, chemical identities, and experimental target IDs.
- `paired_comparisons.csv`: matched MSE differences and win counts.
- `target_rankings.csv`: within-target Spearman correlations from full out-of-fold predictions; targets need at least 20 distinct chemical identities and two distinct labels.
- `independent_replay.json`: sampled predictions reconstructed separately with NumPy/SciPy and confirmation that adaptive fitting preserved the affinity head.

Seed-specific MSEs are averaged within each fold before computing the five-fold mean and SD. Bootstrap intervals resample chemical identities within each fixed outer test fold, using seed-averaged squared errors and 10,000 replicates. They describe uncertainty conditional on the fitted models.

## Commands

Run from the project parent directory with the `psichic_fp` Python environment and host GPU access:

```bash
python DTA_affinity/scripts/run_knn_shared.py
python DTA_affinity/scripts/summarize_knn_shared.py
python DTA_affinity/scripts/verify_knn_shared.py
python DTA_affinity/scripts/build_knn_comparison_tables.py
python -m unittest discover -s DTA_affinity/scripts -p 'test_knn*.py' -v
```

The configuration is `DTA_affinity/configs/knn_shared_20261025.json`. Checkpoints, parameter searches, and per-run predictions are in `DTA_affinity/results/knn_shared_20261025`.

## Concise methods wording

To compare retrieval strategies under a common predictive model, we implemented kNN-DTA and Ada-kNN-DTA following their published label and representation aggregation formulations. Both methods and SGF used the same trained affinity backbone, pretrained features, data partitions, and evaluation metrics within each fold and training seed. Retrieval stores contained training measurements, and aggregation parameters were selected using validation data.
