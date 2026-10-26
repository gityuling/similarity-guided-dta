#!/usr/bin/env python3
"""Archive ESR1 sources, predictions, checkpoints and cached encoder features."""
import hashlib
import json
import tarfile
from pathlib import Path

P=Path(__file__).resolve().parents[1]
OUT=P/'release/assets'
NAME='nonkinase_screening_20261026'


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    files=set()
    for folder in [P/'reports'/NAME,P/'results'/NAME,P/'data/processed'/NAME]:
        files.update(f for f in folder.rglob('*') if f.is_file())
    files.add(P/'configs'/f'{NAME}.json')
    scripts=['fetch_nonkinase_screening.py','prepare_nonkinase_screening.py','run_nonkinase_screening.py',
             'evaluate_nonkinase_screening.py','verify_nonkinase_screening.py','package_nonkinase_screening.py',
             'train_nexusdti_regressor.py','analyze_bib_fusion.py','predict_bib_cellular_affinity.py',
             'bib_common.py','build_kiba_benchmark_tokens.py','build_bindingdb_articles_kd_tokens.py']
    files.update(P/'scripts'/n for n in scripts)
    records=[]
    for f in sorted(files):
        records.append(dict(path=str(Path('DTA_affinity')/f.relative_to(P)),bytes=f.stat().st_size,
                            sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
    manifest=OUT/f'{NAME}_manifest.json'
    manifest.write_text(json.dumps(dict(files=records),indent=2)+'\n')
    archive=OUT/f'{NAME}.tar.gz'
    with tarfile.open(archive,'w:gz',compresslevel=3) as tar:
        for f in sorted(files):tar.add(f,arcname=str(Path('DTA_affinity')/f.relative_to(P)))
        tar.add(manifest,arcname='manifest.json')
    with tarfile.open(archive,'r:gz') as tar:
        for record in records:
            member=tar.extractfile(record['path'])
            assert member is not None
            digest=hashlib.sha256();size=0
            for chunk in iter(lambda:member.read(4*1024*1024),b''):
                digest.update(chunk);size+=len(chunk)
            assert digest.hexdigest()==record['sha256'] and size==record['bytes']
    verified=dict(state='passed',archive=str(archive.relative_to(P)),files=len(records),
                  bytes=archive.stat().st_size,sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
    (OUT/f'{NAME}_verification.json').write_text(json.dumps(verified,indent=2)+'\n')
    print(json.dumps(verified,indent=2))


if __name__=='__main__':main()
