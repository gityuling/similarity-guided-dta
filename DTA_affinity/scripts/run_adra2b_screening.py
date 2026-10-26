#!/usr/bin/env python3
"""Fit ADRA2B affinity predictors and freeze predictions for the external screen."""
import gc
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from torch.utils.data import DataLoader
from analyze_bib_fusion import fit_controls
from predict_bib_cellular_affinity import combine
from train_nexusdti_regressor import (
    AffinityDataset, NexusDTIAffinityRegressor, build_pooled_stats,
    collate_fn, predict, set_seed,
)

PROJECT = Path(__file__).resolve().parents[1]
REPORT = PROJECT / 'reports/adra2b_screening_20261026'
RUN = PROJECT / 'results/adra2b_screening_20261026'
TOKENS = PROJECT / 'data/processed/adra2b_screening_20261026/tokens'
CONFIG = PROJECT / 'configs/adra2b_screening_20261026.json'
MODELS = ['deep', 'neighbor_prior', 'equal_blend', 'fixed_blend', 'similarity_gate', 'pure_similarity']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')


def neighbor_cache(training, query):
    assert training.protein_id.nunique() == query.protein_id.nunique() == 1
    assert training.protein_id.iloc[0] == query.protein_id.iloc[0]
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fps = {s: gen.GetFingerprint(Chem.MolFromSmiles(s))
           for s in set(training.Canonical_SMILES) | set(query.Canonical_SMILES)}
    source = [fps[s] for s in training.Canonical_SMILES]
    scores = np.array([DataStructs.BulkTanimotoSimilarity(fps[s], source)
                       for s in query.Canonical_SMILES], dtype=float)
    order = np.argsort(-scores, axis=1, kind='stable')[:, :40]
    return np.take_along_axis(scores, order, axis=1), training.pKi.to_numpy()[order], float(training.pKi.mean())


def neighbor_estimate(cache, k, gamma):
    sim, values, fallback = cache
    weights = sim[:, :k] ** gamma
    denom = weights.sum(axis=1)
    return np.divide((weights * values[:, :k]).sum(axis=1), denom,
                     out=np.full(len(sim), fallback), where=denom > 0)


def loader(frame, batch, shuffle=False, placeholder=False):
    if placeholder:
        frame = frame.assign(pKi=0.)
    dataset = AffinityDataset(frame, token_dir=TOKENS, target_column='pKi')
    return DataLoader(dataset, batch_size=batch, shuffle=shuffle, drop_last=shuffle,
                      num_workers=0, pin_memory=True, collate_fn=collate_fn)


def main():
    RUN.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')
    cfg = json.loads(CONFIG.read_text())
    assert json.loads((TOKENS / 'build_audit.json').read_text())['complete']
    protected = [CONFIG, Path(__file__),
                 PROJECT / 'scripts/train_nexusdti_regressor.py',
                 PROJECT / 'scripts/analyze_bib_fusion.py',
                 PROJECT / 'scripts/predict_bib_cellular_affinity.py']
    protected += [REPORT / n for n in ['training_table.csv', 'query_table.csv', 'partitions.csv',
                                       'preparation_inputs.json']]
    protected += [TOKENS / 'build_audit.json']
    if (TOKENS / 'verified_reuse.json').exists():
        protected += [TOKENS / 'verified_reuse.json']
    frozen = {str(p.relative_to(PROJECT)): sha(p) for p in protected}
    freeze = RUN / 'frozen_inputs.json'
    if freeze.exists():
        assert json.loads(freeze.read_text()) == frozen, 'Inputs changed after freeze'
    else:
        dump(freeze, frozen)
    state = dict(pid=os.getpid(), state='running', started_utc=datetime.now(timezone.utc).isoformat())
    def status(**kw):
        state.update(kw)
        state['updated_utc'] = datetime.now(timezone.utc).isoformat()
        dump(RUN / 'status.json', state)
    data = pd.read_csv(REPORT / 'training_table.csv')
    query = pd.read_csv(REPORT / 'query_table.csv')
    partitions = pd.read_csv(REPORT / 'partitions.csv')
    assert not set(data.parent_key) & set(query.parent_key)
    assert not set(query.columns) & {'outcome', 'pKi', 'activity_score'}
    device = torch.device('cuda')
    outputs = []
    start = time.monotonic()
    try:
        for fold in range(cfg['validation_folds']):
            folder = RUN / f'fold_{fold}'
            folder.mkdir(exist_ok=True)
            final = folder / 'external_predictions.csv'
            if final.exists():
                outputs.append(pd.read_csv(final))
                continue
            set_seed(cfg['training_seed'])
            part = partitions.loc[partitions.fold.eq(fold)]
            train = data.loc[data.row_id.isin(part.loc[part.split.eq('train'), 'row_id'])].copy()
            val = data.loc[data.row_id.isin(part.loc[part.split.eq('val'), 'row_id'])].copy()
            assert not set(train.parent_key) & set(val.parent_key)
            train_loader = loader(train, cfg['batch_size'], shuffle=True)
            val_loader = loader(val, 16)
            target_mean, target_std = float(train.pKi.mean()), float(train.pKi.std(ddof=1))
            pooled_mean, pooled_std = build_pooled_stats(train, token_dir=TOKENS)
            options = {k: cfg[k] for k in ['latent_dim', 'pool_fusion', 'hybrid', 'dropout',
                                           'drug_attention', 'pocket_fusion']}
            model = NexusDTIAffinityRegressor(**options, pooled_mean=pooled_mean,
                                             pooled_std=pooled_std).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=cfg['learning_rate'],
                                           weight_decay=cfg['weight_decay'])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg['epochs'], eta_min=1e-7)
            scaler = torch.amp.GradScaler('cuda')
            loss_fn = torch.nn.SmoothL1Loss(beta=.25)
            best, stale, history, best_state = np.inf, 0, [], None
            for epoch in range(cfg['epochs']):
                status(stage='training', fold=fold, epoch=epoch + 1)
                model.train()
                losses = []
                for batch in train_loader:
                    protein, drug, phys, pocket, pmask, dmask, labels = [x.to(device) for x in batch]
                    optimizer.zero_grad(set_to_none=True)
                    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        pred = model(protein, drug, phys, pmask, dmask,
                                     pocket_selector=pocket, k_ratio=cfg['k_ratio'])
                        loss = loss_fn(pred, (labels - target_mean) / target_std)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
                    scaler.step(optimizer)
                    scaler.update()
                    losses.append(float(loss.detach()))
                scheduler.step()
                y, pred = predict(model, val_loader, device, cfg['k_ratio'])
                pred = pred.astype(float) * target_std + target_mean
                mse = float(np.mean((pred - y) ** 2))
                history.append(dict(epoch=epoch + 1, loss=float(np.mean(losses)), val_mse=mse))
                print(json.dumps(dict(fold=fold, **history[-1])), flush=True)
                if mse < best:
                    best, stale = mse, 0
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    stale += 1
                    if stale >= cfg['patience']:
                        break
            assert best_state is not None
            model.load_state_dict(best_state)
            _, val_deep = predict(model, val_loader, device, cfg['k_ratio'])
            val_deep = val_deep.astype(float) * target_std + target_mean
            vcache = neighbor_cache(train, val)
            search = []
            for k in cfg['neighbor_k']:
                for gamma in cfg['neighbor_gamma']:
                    prior = neighbor_estimate(vcache, k, gamma)
                    search.append(dict(k=k, gamma=gamma, val_mse=float(np.mean((prior - val.pKi) ** 2))))
            selected = min(search, key=lambda x: x['val_mse'])
            val_prior = neighbor_estimate(vcache, selected['k'], selected['gamma'])
            controls = pd.DataFrame(dict(row_id=val.row_id.to_numpy(), split='val', truth=val.pKi.to_numpy(),
                                         prediction_deep=val_deep, prediction_neighbor_prior=val_prior,
                                         max_train_drug_tanimoto=vcache[0][:, 0]))
            fit = fit_controls(controls)
            selection = dict(fold=fold, train_n=len(train), val_n=len(val),
                             target_mean=target_mean, target_std=target_std,
                             history=history, neighbor_search=search, neighbor=selected, fusion=fit,
                             config_sha256=sha(CONFIG), architecture=options)
            dump(folder / 'selected.json', selection)
            controls.to_csv(folder / 'validation_predictions.csv', index=False)
            torch.save(dict(state=best_state, selection=selection), folder / 'model.pt')
            # External labels are read only by the separate evaluation script.
            status(stage='external_inference', fold=fold)
            _, external_deep = predict(model, loader(query, 16, placeholder=True), device, cfg['k_ratio'])
            external_deep = external_deep.astype(float) * target_std + target_mean
            qcache = neighbor_cache(train, query)
            external_prior = neighbor_estimate(qcache, selected['k'], selected['gamma'])
            values = combine(external_deep, external_prior, qcache[0][:, 0], fit)
            values['equal_blend'] = (external_deep + external_prior) / 2
            values['pure_similarity'] = qcache[0][:, 0]
            frame = query[['row_id', 'source_id']].copy()
            for name, value in values.items():
                assert np.isfinite(value).all()
                frame[name] = value
            frame['fold'] = fold
            frame.to_csv(final, index=False)
            outputs.append(frame)
            del model, optimizer, scaler, train_loader, val_loader, best_state
            gc.collect()
            torch.cuda.empty_cache()
        full = pd.concat(outputs, ignore_index=True)
        assert not full.duplicated(['row_id', 'fold']).any()
        assert len(full) == len(query) * cfg['validation_folds']
        full.to_csv(REPORT / 'fold_predictions.csv', index=False)
        full.groupby(['row_id', 'source_id'])[MODELS + ['deep_weight', 'max_train_drug_tanimoto']].mean().reset_index().to_csv(
            REPORT / 'predictions.csv', index=False)
        assert {str(p.relative_to(PROJECT)): sha(p) for p in protected} == frozen
        status(state='completed', stage='predictions_frozen', elapsed_seconds=time.monotonic() - start,
               device=torch.cuda.get_device_name(0), external_outcomes_read_by_training=False,
               prediction_sha256=sha(REPORT / 'predictions.csv'))
    except BaseException as error:
        status(state='failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
