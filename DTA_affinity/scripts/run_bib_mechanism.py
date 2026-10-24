#!/usr/bin/env python3
"""Development-only, DTA-penalized nonnegative sparse response decomposition."""
import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
from sklearn.linear_model import Lasso
from sklearn.exceptions import ConvergenceWarning

from bib_g2cp_common import DATA, OUT, PROJECT, record, check_records, save_json, write_once
from evaluate_bib_target_retrieval import percentile, retrieval_metrics

CFG = PROJECT / 'configs/bib_mechanism_20261021.json'
DEST = PROJECT / 'reports/bib_mechanism_20261021'
REF = PROJECT / 'reports/bib_g2cp_reference_diagnostic_20261019'
OLD = PROJECT / 'reports/bib_target_retrieval_20261019'
DTA = ('deep', 'neighbor_prior', 'fixed_blend', 'similarity_gate')


def gene_partition(library, modules=20, folds=5):
    # Use only genetic TRAIN signatures; no query expression or annotation.
    gene = library.T.copy()
    gene -= gene.mean(1, keepdims=True)
    norm = np.linalg.norm(gene, axis=1, keepdims=True)
    unit = gene / np.maximum(norm, 1e-12)
    dist = np.clip(pdist(unit, metric='sqeuclidean') / 2, 0, 2)
    labels = fcluster(linkage(dist, method='complete'), modules, criterion='maxclust')
    sizes = {m: int((labels == m).sum()) for m in np.unique(labels)}
    totals = [0]*folds
    result = np.zeros(len(labels), dtype=int)
    for module in sorted(sizes, key=lambda m: (-sizes[m], m)):
        fold = int(np.argmin(totals))
        result[labels == module] = fold
        totals[fold] += sizes[module]
    if min(totals) < 2:
        raise RuntimeError('Degenerate gene partition')
    return labels, result


def sparse_fit(library, truth, background, fit, alpha, rank=None, strength=2):
    """All query-dependent scaling uses fit genes only. Returns full prediction."""
    dictionary = (library-library.mean(0)).T
    mu = dictionary[fit].mean(0)
    sd = np.sqrt(np.mean((dictionary[fit]-mu)**2, axis=0))
    valid = sd > 1e-10
    sd = np.where(valid, sd, 1.)
    x = (dictionary-mu)/sd
    x[:, ~valid] = 0
    residual = truth-background
    intercept = residual[fit].mean()
    scale = max(float(np.sqrt(np.mean((residual[fit]-intercept)**2))), 1e-10)
    penalty = np.ones(library.shape[0]) if rank is None else 1+strength*(1-rank)
    penalty /= penalty.mean()
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        model = Lasso(alpha=alpha, positive=True, fit_intercept=False,
                      max_iter=30000, tol=1e-8, selection='cyclic')
        model.fit(x[fit]/penalty, (residual[fit]-intercept)/scale)
    coefficients = model.coef_/penalty
    coefficients[coefficients < 1e-10] = 0
    predicted = background+intercept+scale*(x@coefficients)
    grad = x[fit].T @ (x[fit]@coefficients-(residual[fit]-intercept)/scale)/fit.sum()+alpha*penalty
    active = coefficients > 0
    kkt = max(float(np.max(np.abs(grad[active]))) if active.any() else 0.,
              float(np.max(np.maximum(-grad[~active], 0))) if (~active).any() else 0.)
    return predicted, coefficients, {'kkt_error': kkt, 'active_targets': int(active.sum()),
                                   'iterations': int(model.n_iter_), 'dual_gap': float(model.dual_gap_)}


def corr(x, y):
    a, b = x-x.mean(), y-y.mean()
    denom = np.linalg.norm(a)*np.linalg.norm(b)
    return float(a@b/denom) if denom > 1e-12 else 0.


def cases(config):
    candidates = pd.read_csv(OUT/'candidates.csv').set_index('canonical_identity')
    old_scores = pd.read_csv(OLD/'target_scores.csv', float_precision='round_trip')
    result, partition_rows, similarity_rows, inputs = [], [], [], []
    for cell in config['cells']:
        paths = [REF/f'{cell}_development_rows.csv', REF/f'{cell}_development_vectors.npz', DATA/f'{cell}.npz']
        inputs.extend(record(p) for p in paths)
        meta = pd.read_csv(paths[0])
        if not meta.split.isin(['train','val']).all():
            raise RuntimeError('Test rows are forbidden')
        with np.load(paths[1], allow_pickle=False) as z:
            truth, background = z['truth'].copy(), z['training_mean'].copy()
        with np.load(paths[2], allow_pickle=False) as z:
            # Do not load the test expression or y arrays.
            library, proteins = z['library'].astype(float), z['proteins'].astype(str)
        labels, folds = gene_partition(library, config['gene_modules'], config['gene_folds'])
        partition_rows.extend({'cell':cell,'gene_index':i,'module':int(labels[i]),'fold':int(folds[i])}
                              for i in range(len(labels)))
        similarity = np.corrcoef(library)
        for i,j in zip(*np.where(np.triu(similarity >= .9, 1))):
            similarity_rows.append({'cell':cell,'target1':proteins[i],'target2':proteins[j],
                                    'genetic_pearson':float(similarity[i,j])})
        for i,item in enumerate(meta.itertuples()):
            details = {k:getattr(item,k) for k in ['cell','canonical_identity','parent_key','names','split','davis_external']}
            rows = old_scores.loc[old_scores.cell.eq(cell)&old_scores.canonical_identity.eq(item.canonical_identity)].set_index('protein_id').loc[proteins]
            rank = {m:percentile(rows[m].to_numpy()) for m in DTA}
            seed = int(hashlib.sha256(f"{config['seed']}:{cell}:{item.parent_key}".encode()).hexdigest()[:16],16)
            rank['shuffled_gate'] = np.random.default_rng(seed).permutation(rank['similarity_gate'])
            annotated = set(candidates.loc[item.canonical_identity,'mapped_kiba_targets'].split('|'))
            result.append({'details':details,'library':library,'proteins':proteins,'truth':truth[i],
                           'background':background[i],'folds':folds,'rank':rank,
                           'positive':np.array([p in annotated for p in proteins])})
    return result, partition_rows, similarity_rows, inputs


def heldout(case, alpha, rank=None, strength=2):
    yhat = np.zeros_like(case['truth'])
    stats=[]
    for fold in np.unique(case['folds']):
        fit=case['folds']!=fold
        pred, coef, info=sparse_fit(case['library'],case['truth'],case['background'],fit,alpha,rank,strength)
        yhat[~fit]=pred[~fit]
        stats.append(info)
    return yhat, stats


def summarize(frame, metrics):
    output=[]
    for scope in ('strict_davis_disjoint','all_kiba_external'):
        current=frame.loc[frame.davis_external] if scope.startswith('strict') else frame
        for (split, method),g in current.groupby(['split','method']):
            drug=g.groupby('parent_key')[metrics].mean()
            output.append({'scope':scope,'split':split,'method':method,'drugs':len(drug),
                           'drug_cell_pairs':len(g),**drug.mean().to_dict()})
    return pd.DataFrame(output)


def main():
    cfg=json.loads(CFG.read_text())
    if cfg['splits']!=['train','val']: raise RuntimeError('Development-only protocol')
    check_records(json.loads((REF/'analysis.json').read_text())['outputs'])
    check_records(json.loads((OLD/'analysis.json').read_text())['outputs'])
    examples,parts,similar,inputs=cases(cfg)
    inputs += [record(p) for p in [CFG,Path(__file__),OUT/'candidates.csv',OLD/'target_scores.csv',OLD/'per_drug_cell_metrics.csv']]
    save_json(DEST/'prototype_freeze.json',{'config':cfg,'inputs':inputs,'test_rows_scored':0},immutable=True)
    write_once(DEST/'gene_modules.csv',pd.DataFrame(parts).to_csv(index=False))
    write_once(DEST/'ambiguous_genetic_targets.csv',pd.DataFrame(similar,columns=['cell','target1','target2','genetic_pearson']).to_csv(index=False))
    grid=[]
    for alpha in cfg['alpha_grid']:
        for case in examples:
            d=case['details']
            if d['split']!='train' or not d['davis_external']: continue
            pred,_=heldout(case,alpha)
            grid.append({**d,'alpha':alpha,'mse':float(np.mean((pred-case['truth'])**2))})
    grid=pd.DataFrame(grid)
    means=grid.groupby(['alpha','parent_key']).mse.mean().groupby('alpha').mean()
    best=means.min(); alpha=max(float(a) for a,v in means.items() if v<=best+1e-12)
    write_once(DEST/'prototype_alpha_grid.csv',grid.to_csv(index=False))
    save_json(DEST/'prototype_weights.json',{'alpha':alpha,'curve':{str(k):v for k,v in means.items()},
              'selection':cfg['selection'],'validation_scored':False},immutable=True)
    print('Frozen alpha',alpha,flush=True)
    response_rows=[]; retrieval_rows=[]; score_rows=[]; numerical=[]; vectors={}
    for idx,case in enumerate(examples):
        d=case['details']; y=case['truth']; full=np.ones(len(y),dtype=bool)
        methods={'sparse_response':None,**{f'sparse_{m}':r for m,r in case['rank'].items()}}
        # Fixed-background and intercept-fitted-background controls.
        bg=case['background'].copy(); bgfit=bg.copy()
        for fold in np.unique(case['folds']):
            fit=case['folds']!=fold
            bgfit[~fit]+=(y[fit]-bg[fit]).mean()
        for name,pred in [('background_fixed',bg),('background_intercept',bgfit)]:
            response_rows.append({**d,'case_index':idx,'method':name,'mse':float(np.mean((pred-y)**2)),'pearson':corr(pred,y),
                                  'residual_pearson':corr(pred-bg,y-bg)})
            vectors[f'{idx}_{name}']=pred
        for name,rank in methods.items():
            pred,info=heldout(case,alpha,rank,cfg['prior_strength'])
            _,coef,fullinfo=sparse_fit(case['library'],y,bg,full,alpha,rank,cfg['prior_strength'])
            response_rows.append({**d,'case_index':idx,'method':name,'mse':float(np.mean((pred-y)**2)),'pearson':corr(pred,y),
                                  'residual_pearson':corr(pred-bg,y-bg)})
            vectors[f'{idx}_{name}']=pred
            numerical.extend({**d,'method':name,'fit':'full' if j==len(info) else f'gene_fold_{j}',**x}
                             for j,x in enumerate([*info,fullinfo]))
            if case['positive'].any():
                retrieval_rows.append({**d,'method':name,**retrieval_metrics(coef,case['positive'])})
            score_rows.extend({**d,'method':name,'protein_id':p,'coefficient':float(coef[j]),
                               'known_positive':bool(case['positive'][j])} for j,p in enumerate(case['proteins']))
        vectors[f'{idx}_truth']=y
        vectors[f'{idx}_background']=bg
        if idx%20==0: print('case',idx+1,'/',len(examples),flush=True)
    old=pd.read_csv(OLD/'per_drug_cell_metrics.csv',float_precision='round_trip')
    retained=['observed','observed_centered',*DTA,*[f'{m}_plus_observed_equal' for m in DTA]]
    old=old.loc[old.method.isin(retained)]
    retrieval=pd.concat([pd.DataFrame(retrieval_rows),old],ignore_index=True)
    response=pd.DataFrame(response_rows)
    response_metrics=['mse','pearson','residual_pearson']
    retrieval_metrics_names=['mrr','hit_at_1','hit_at_5','hit_at_10','recall_at_5']
    comparisons=[]
    for name,frame,metric,controls in [
        ('response',response,'mse',['sparse_response','background_fixed','background_intercept','sparse_deep','sparse_fixed_blend','sparse_shuffled_gate']),
        ('retrieval',retrieval,'mrr',['sparse_response','similarity_gate','observed','similarity_gate_plus_observed_equal','sparse_deep','sparse_fixed_blend','sparse_shuffled_gate'])]:
        for scope in ['strict_davis_disjoint','all_kiba_external']:
            sub=frame.loc[frame.split.eq('val')]
            if scope.startswith('strict'): sub=sub.loc[sub.davis_external]
            matrix=sub.groupby(['parent_key','method'])[metric].mean().unstack()
            for control in controls:
                pair=matrix[['sparse_similarity_gate',control]]
                if pair.isna().any().any(): raise RuntimeError('Unequal comparison coverage')
                delta=(pair.iloc[:,0]-pair.iloc[:,1]).to_numpy()
                rng=np.random.default_rng(cfg['seed']); draws=rng.integers(0,len(delta),(cfg['bootstrap_repeats'],len(delta)))
                comparisons.append({'endpoint':name,'scope':scope,'metric':metric,'left':'sparse_similarity_gate','right':control,
                                    'difference':float(delta.mean()),'ci95':np.quantile(delta[draws].mean(1),[.025,.975]).tolist(),
                                    'drugs':len(delta),'interpretation':'conditional development comparison; unadjusted'})
    files={'prototype_response_metrics.csv':response,'prototype_retrieval_metrics.csv':retrieval,
           'prototype_response_summary.csv':summarize(response,response_metrics),
           'prototype_retrieval_summary.csv':summarize(retrieval,retrieval_metrics_names),
           'prototype_target_scores.csv':pd.DataFrame(score_rows),'prototype_numerical_checks.csv':pd.DataFrame(numerical),
           'prototype_cases.csv':pd.DataFrame([{'case_index':i,**c['details']} for i,c in enumerate(examples)])}
    for name,frame in files.items(): write_once(DEST/name,frame.to_csv(index=False))
    vectorpath=DEST/'prototype_vectors.npz'
    if vectorpath.exists():
        with np.load(vectorpath) as z:
            if set(z.files)!=set(vectors) or any(not np.array_equal(z[k],v) for k,v in vectors.items()):
                raise RuntimeError('Existing vectors differ')
    else: np.savez_compressed(vectorpath,**vectors)
    check_records(inputs)
    save_json(DEST/'prototype_analysis.json',{'alpha':alpha,'cases':len(examples),'test_rows_scored':0,
              'comparisons':comparisons,'max_kkt_error':max(r['kkt_error'] for r in numerical),
              'outputs':[record(DEST/n) for n in files]+[record(vectorpath)]},immutable=True)
    print(files['prototype_retrieval_summary.csv'].query("scope=='strict_davis_disjoint' and split=='val'").to_string(index=False),flush=True)


if __name__=='__main__': main()
