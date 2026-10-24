#!/usr/bin/env python3
"""Frozen, single-GPU execution for the pre-specified A/B downstream cohorts."""
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
from bib_common import PROJECT, record, sha256, write_once
from predict_bib_cellular_affinity import MODELS, combine, deep_predictions, fold_inputs, neighbor_predictions, paths
from run_bib_results34 import require_idle_gpu

OUT=PROJECT/'reports/bib_downstream_20261022'
RUN=PROJECT/'results/bib_downstream_20261022'
TOKENS=PROJECT/'data/processed/bib_downstream_20261022/tokens'
KIBA=PROJECT/'data/processed/kiba_dtiam_protocol'


def main():
    RUN.mkdir(exist_ok=True); torch.set_num_threads(2)
    sources=[OUT/n for n in ['query_table.csv','labels.csv','source_compound_audit.csv',
        'cohort_coverage.csv','preparation_inputs.json','experimental_label_verification.csv',
        'EVALUATION_PROTOCOL.md','sources/pinned_source_verification.json']]
    sources += [KIBA/'table.csv']
    sources += [PROJECT/'scripts'/n for n in ['run_bib_downstream.py','prepare_bib_downstream.py',
        'evaluate_bib_downstream.py','predict_bib_cellular_affinity.py','train_nexusdti_regressor.py',
        'build_kiba_benchmark_tokens.py','build_bindingdb_articles_kd_tokens.py']]
    sources += [p for f in range(5) for p in paths(f).values()]
    frozen=[record(p) for p in sources]
    write_once(OUT/'evaluation_freeze.json',json.dumps(frozen,indent=2)+'\n')
    state=dict(state='starting',pid=os.getpid(),started_utc=datetime.now(timezone.utc).isoformat())
    def status(**kw):
        state.update(kw);state['updated_utc']=datetime.now(timezone.utc).isoformat()
        tmp=RUN/'status.tmp';tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(RUN/'status.json')
    with (PROJECT/'results/bib_validation_20261017/queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            require_idle_gpu()
            if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
            for p,h in json.loads((KIBA/'tokens/build_audit.json').read_text())['model_sha256'].items():
                if sha256(Path(p))!=h:raise RuntimeError('Encoder changed')
            status(state='running',stage='features')
            cmd=[sys.executable,'-u',str(PROJECT/'scripts/build_kiba_benchmark_tokens.py'),
                '--data-table',str(OUT/'query_table.csv'),'--cache-dir',str(TOKENS),
                '--reuse-cache',str(KIBA/'tokens'),'--drug-batch-size','1','--protein-batch-size','1']
            with (RUN/'features.log').open('a') as stream:
                subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,check=True,cwd=PROJECT)
            require_idle_gpu()
            query=pd.read_csv(OUT/'query_table.csv'); table=pd.read_csv(KIBA/'table.csv')
            assert len(query)==228 and query.row_id.is_unique and query.drug_id.is_unique
            assert not set(query.columns)&{'endpoint_score','outcome','strict_outcome'}
            chunks=RUN/'folds';chunks.mkdir(exist_ok=True)
            frames=[]; checks=[]
            for fold in range(5):
                status(stage='inference',fold=fold)
                source,training,hp,fit=fold_inputs(fold,table)
                assert not set(query.Sequence)-set(training.Sequence)
                saved=pd.read_csv(source['saved_predictions'],float_precision='round_trip')
                saved=saved.loc[saved.split.eq('test')].head(8)
                ref=saved[['row_id']].merge(table,on='row_id',validate='one_to_one')
                d=deep_predictions(ref,KIBA/'tokens',source['checkpoint'],torch.device('cuda'))
                p,s=neighbor_predictions(ref,training,hp); values=combine(d,p,s,fit)
                errors={m:float(np.max(np.abs(values[m]-saved['prediction_'+m].to_numpy()))) for m in MODELS}
                assert errors['neighbor_prior']<2e-6 and max(errors.values())<.02, errors
                assert np.allclose(s,saved.max_train_drug_tanimoto,atol=1e-12,rtol=0)
                checks.append(dict(fold=fold,reference_errors=errors))
                path=chunks/f'fold_{fold}.csv'
                if path.exists():
                    frame=pd.read_csv(path)
                    assert frame.row_id.tolist()==query.row_id.tolist()
                else:
                    d=deep_predictions(query,TOKENS,source['checkpoint'],torch.device('cuda'))
                    p,s=neighbor_predictions(query,training,hp); values=combine(d,p,s,fit)
                    frame=query[['row_id']].copy()
                    for key,value in values.items():
                        assert np.isfinite(value).all()
                        frame[key]=value
                    frame['equal_blend']=(d+p)/2;frame['fold']=fold
                    write_once(path,frame.to_csv(index=False))
                frames.append(frame)
                print(f'Completed fold {fold+1}/5: {len(frame)} queries',flush=True)
            full=pd.concat(frames,ignore_index=True)
            assert len(full)==5*len(query) and not full.duplicated(['row_id','fold']).any()
            cols=list(MODELS)+['equal_blend','max_train_drug_tanimoto','deep_weight']
            average=full.groupby('row_id')[cols].mean().reset_index();average['fold_count']=5
            write_once(OUT/'fold_predictions.csv',full.to_csv(index=False))
            write_once(OUT/'predictions.csv',average.to_csv(index=False))
            write_once(OUT/'inference_audit.json',json.dumps(dict(checks=checks,training_performed=False,
                endpoint_values_read_by_predictor=False,device=torch.cuda.get_device_name(0),
                outputs=[record(OUT/n) for n in ['fold_predictions.csv','predictions.csv']]),indent=2)+'\n')
            assert [record(p) for p in sources]==frozen, 'Frozen input changed'
            status(stage='evaluation')
            with (RUN/'evaluation.log').open('w') as stream:
                subprocess.run([sys.executable,str(PROJECT/'scripts/evaluate_bib_downstream.py')],stdout=stream,stderr=subprocess.STDOUT,check=True,cwd=PROJECT)
            status(state='completed',stage='done')
        except BaseException as exc:
            status(state='failed',error=repr(exc));raise


if __name__=='__main__':main()
