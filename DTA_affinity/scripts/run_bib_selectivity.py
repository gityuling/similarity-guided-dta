#!/usr/bin/env python3
"""Locked, resumable frozen-model feature generation and inference for Results 5."""
import fcntl
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from bib_common import PROJECT, ROOT, record, sha256, write_once
from predict_bib_cellular_affinity import MODELS, combine, deep_predictions, fold_inputs, neighbor_predictions, paths
from run_bib_results34 import require_idle_gpu

OUT=PROJECT/'reports/bib_selectivity_20261021'
RUN=PROJECT/'results/bib_selectivity_20261021'
TOKENS=PROJECT/'data/processed/bib_selectivity_20261021/tokens'
KIBA=PROJECT/'data/processed/kiba_dtiam_protocol'


def main():
    RUN.mkdir(exist_ok=True);torch.set_num_threads(2)
    sources=[OUT/n for n in ['query_table.csv','family_tasks.csv','task_candidate_pool.csv','task_definition.json',
        'qc_measurement_states.csv','measurement_states.csv','EVALUATION_PROTOCOL.md']]
    sources+=[PROJECT/'configs/bib_selectivity_evaluation_20261021.json',KIBA/'table.csv']
    sources += [PROJECT/'scripts'/n for n in ['run_bib_selectivity.py','evaluate_bib_selectivity.py',
        'predict_bib_cellular_affinity.py','train_nexusdti_regressor.py','build_kiba_benchmark_tokens.py',
        'build_bindingdb_articles_kd_tokens.py']]
    sources += [p for fold in range(5) for p in paths(fold).values()]
    frozen=[record(p) for p in sources]
    write_once(OUT/'evaluation_freeze.json',json.dumps(frozen,indent=2)+'\n')
    state={'state':'starting','pid':os.getpid(),'started_utc':datetime.now(timezone.utc).isoformat()}
    def status(**values):
        state.update(values);state['updated_utc']=datetime.now(timezone.utc).isoformat()
        tmp=RUN/'status.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(RUN/'status.json')
    with (PROJECT/'results/bib_validation_20261017/queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            require_idle_gpu()
            if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
            for path,expected in json.loads((KIBA/'tokens/build_audit.json').read_text())['model_sha256'].items():
                if sha256(Path(path))!=expected:raise RuntimeError('Encoder weights changed')
            status(state='running',stage='features')
            command=[sys.executable,'-u',str(PROJECT/'scripts/build_kiba_benchmark_tokens.py'),
                '--data-table',str(OUT/'query_table.csv'),'--cache-dir',str(TOKENS),
                '--reuse-cache',str(KIBA/'tokens'),'--drug-batch-size','1','--protein-batch-size','1']
            with (RUN/'features.log').open('a') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True,cwd=PROJECT)
            require_idle_gpu()
            query=pd.read_csv(OUT/'query_table.csv');table=pd.read_csv(KIBA/'table.csv')
            assert len(query)==112506 and query.row_id.is_unique
            eligibility=pd.read_csv(OUT/'task_candidate_pool.csv')
            chosen=set(query.compound)
            eligible=eligibility.loc[eligibility.compound.isin(chosen)]
            assert len(eligible)==1103 and not eligible[['kiba_seen','davis_seen','label_validation_used']].any().any()
            chunks=RUN/'chunks';chunks.mkdir(exist_ok=True)
            checks=[];frames=[]
            for fold in range(5):
                source,training,hyperparameters,fit=fold_inputs(fold,table)
                if set(query.Sequence)-set(training.Sequence):raise RuntimeError('Unsupported target')
                historical=pd.read_csv(source['saved_predictions'],float_precision='round_trip')
                historical=historical.loc[historical.split.eq('test')].head(8)
                reference=historical[['row_id']].merge(table,on='row_id',validate='one_to_one')
                rd=deep_predictions(reference,KIBA/'tokens',source['checkpoint'],torch.device('cuda'))
                rp,rs=neighbor_predictions(reference,training,hyperparameters);rv=combine(rd,rp,rs,fit)
                errors={m:float(np.max(np.abs(rv[m]-historical['prediction_'+m].to_numpy()))) for m in MODELS}
                if errors['neighbor_prior']>2e-6 or max(errors.values())>0.02:raise RuntimeError(str(errors))
                if not np.allclose(rs,historical.max_train_drug_tanimoto,atol=1e-12,rtol=0):raise RuntimeError('Historical similarity mismatch')
                checks.append(dict(fold=fold,reference_errors=errors))
                for start in range(0,len(query),8192):
                    stop=min(start+8192,len(query));subset=query.iloc[start:stop].copy()
                    dest=chunks/f'fold_{fold}_{start:06d}.csv'
                    status(stage='inference',fold=fold,chunk_start=start,total_rows=len(query))
                    if dest.exists():
                        frame=pd.read_csv(dest)
                        if frame.row_id.tolist()!=subset.row_id.tolist():raise RuntimeError('Cached chunk identity mismatch')
                    else:
                        # The inherited Dataset requires numeric ids; preserve external ids in saved output.
                        deep=deep_predictions(subset.assign(row_id=np.arange(len(subset))),TOKENS,source['checkpoint'],torch.device('cuda'))
                        prior,similarity=neighbor_predictions(subset,training,hyperparameters)
                        values=combine(deep,prior,similarity,fit)
                        frame=subset[['row_id']].copy()
                        for k,v in values.items():
                            if not np.isfinite(v).all():raise RuntimeError('Nonfinite predictions')
                            frame[k]=v
                        frame['equal_blend']=(deep+prior)/2;frame['fold']=fold
                        tmp=dest.with_suffix('.tmp');frame.to_csv(tmp,index=False);tmp.replace(dest)
                    frames.append(frame)
                    print(f'fold={fold} rows={stop}/{len(query)}',flush=True)
            full=pd.concat(frames,ignore_index=True)
            assert (full.groupby('row_id').size()==5).all() and len(full)==5*len(query)
            full.to_csv(OUT/'fold_predictions.csv',index=False)
            model_cols=list(MODELS)+['equal_blend','max_train_drug_tanimoto','deep_weight']
            average=full.groupby('row_id')[model_cols].mean().reset_index();average['fold_count']=5
            average.to_csv(OUT/'predictions.csv',index=False)
            (OUT/'inference_audit.json').write_text(json.dumps({'checks':checks,'training_performed':False,
                'external_endpoint_values_read_by_inference':False,'device':torch.cuda.get_device_name(0),
                'files':[record(OUT/n) for n in ['predictions.csv','fold_predictions.csv']]},indent=2)+'\n')
            if [record(p) for p in sources]!=frozen:raise RuntimeError('Frozen input changed during execution')
            status(stage='evaluation')
            with (RUN/'evaluation.log').open('w') as f:
                subprocess.run([sys.executable,str(PROJECT/'scripts/evaluate_bib_selectivity.py')],stdout=f,stderr=subprocess.STDOUT,check=True,cwd=PROJECT)
            status(state='completed',stage='done')
        except BaseException as exc:
            status(state='failed',error=repr(exc));raise


if __name__=='__main__':main()
