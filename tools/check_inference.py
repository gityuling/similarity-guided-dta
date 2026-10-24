#!/usr/bin/env python3
"""Check one released checkpoint against four archived test predictions."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'DTA_affinity/scripts'))

import numpy as np
import pandas as pd
import torch
from predict_bib_cellular_affinity import fold_inputs, deep_predictions, neighbor_predictions, combine


if __name__ == '__main__':
    table = pd.read_csv(ROOT / 'DTA_affinity/data/processed/kiba_dtiam_protocol/table.csv')
    source, training, hyperparameters, fit = fold_inputs(0, table)
    saved = pd.read_csv(source['saved_predictions'])
    chosen = saved.loc[saved.split.eq('test')].head(4)
    query = chosen[['row_id']].merge(table, on='row_id', validate='one_to_one')
    training = training.loc[training.Sequence.isin(query.Sequence)]
    prior, similarity = neighbor_predictions(query, training, hyperparameters)
    neural = deep_predictions(query, ROOT / 'DTA_affinity/data/processed/kiba_dtiam_protocol/tokens',
                              source['checkpoint'], torch.device('cpu'))
    predictions = combine(neural, prior, similarity, fit)
    errors = {m:float(np.max(np.abs(predictions[m]-chosen[f'prediction_{m}'].to_numpy())))
              for m in ['deep','neighbor_prior','fixed_blend','similarity_gate']}
    if errors['neighbor_prior'] > 2e-6 or max(errors.values()) > 0.02:
        raise RuntimeError(f'Inference mismatch: {errors}')
    np.testing.assert_allclose(similarity,chosen.max_train_drug_tanimoto,atol=1e-12,rtol=0)
    result = {'status':'passed','device':'cpu','reference_fold':0,'test_rows':query.row_id.tolist(),
              'maximum_absolute_errors':errors,
              'comparison':'CPU float32 against archived CUDA mixed-precision predictions',
              'tolerances':{'neighbor':2e-6,'neural_and_fused':0.02}}
    output = ROOT / 'verification/inference.json'
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))
