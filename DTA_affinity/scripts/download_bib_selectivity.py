#!/usr/bin/env python3
"""Resume public MassIVE CSV downloads; validate each file before atomic install."""
import csv
import hashlib
import json
import os
import threading
import time
from ftplib import FTP_TLS
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / 'reports/bib_selectivity_20261021'
DEST = OUT / 'sources'
REQUIRED = {'Gene.names', 'Picked.intensity', 'Target.probability',
            'Target.annotation', 'CorrectionFactor', 'Kd'}
LOCAL = threading.local()


def connection():
    if getattr(LOCAL, 'ftp', None) is None:
        ftp = FTP_TLS('massive-ftp.ucsd.edu', timeout=60)
        ftp.login()
        ftp.prot_p()
        LOCAL.ftp = ftp
    return LOCAL.ftp


def validate(path, expected):
    if not path.exists() or path.stat().st_size != expected:
        raise ValueError('missing or byte-size mismatch')
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        if not REQUIRED.issubset(reader.fieldnames or []):
            raise ValueError('required CSV fields missing')
        n = 0
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError('malformed CSV row')
            n += 1
    if not n:
        raise ValueError('empty CSV')
    return n, hashlib.sha256(path.read_bytes()).hexdigest()


def fetch(row):
    name = row['name']
    if Path(name).name != name or not name.endswith('_final.csv'):
        raise ValueError('invalid manifest filename')
    remote = '/v05/' + row['file_descriptor'].removeprefix('f.')
    url = 'ftps://massive-ftp.ucsd.edu' + remote
    path = DEST / name
    result = {'file': name, 'url': url, 'expected_bytes': int(row['size'])}
    try:
        n, digest = validate(path, int(row['size']))
        return dict(result, state='verified_existing', rows=n, sha256=digest)
    except (ValueError, UnicodeError):
        pass
    part = path.with_suffix('.csv.part')
    for attempt in range(4):
        try:
            with part.open('wb') as stream:
                connection().retrbinary('RETR ' + remote, stream.write, blocksize=65536)
            break
        except Exception as exc:
            ftp = getattr(LOCAL, 'ftp', None)
            if ftp is not None:
                ftp.close()
            LOCAL.ftp = None
            if attempt == 3:
                return dict(result, state='failed', error=str(exc))
            time.sleep(min(60, 5 * 3**attempt))
    try:
        n, digest = validate(part, int(row['size']))
        os.replace(part, path)
        return dict(result, state='downloaded_verified', rows=n, sha256=digest)
    except (ValueError, UnicodeError) as exc:
        return dict(result, state='failed', error=str(exc))


def main():
    DEST.mkdir(exist_ok=True)
    with (OUT/'processed_file_candidates.csv').open() as f:
        manifest = list(csv.DictReader(f))
    assert len({r['name'] for r in manifest}) == len(manifest)
    results = []
    with (OUT/'download_progress.jsonl').open('w') as log:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(fetch, row) for row in manifest]
            for future in as_completed(futures):
                r = future.result()
                results.append(r)
                log.write(json.dumps(r)+'\n'); log.flush()
                if len(results) % 50 == 0 or r['state'] == 'failed':
                    print(json.dumps({'completed':len(results), 'total':len(manifest),
                        'failures':sum(x['state']=='failed' for x in results)}), flush=True)
    results.sort(key=lambda r:r['file'])
    report = {'files':results, 'total':len(manifest),
        'verified':sum(r['state']!='failed' for r in results),
        'failed':sum(r['state']=='failed' for r in results),
        'expected_bytes':sum(int(r['size']) for r in manifest),
        'integrity_note':'Repository byte size plus local CSV validation; SHA256 computed locally, not compared with a publisher digest.'}
    (OUT/'download_manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='files'}),flush=True)
    if report['failed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
