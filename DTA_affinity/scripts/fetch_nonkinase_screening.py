#!/usr/bin/env python3
"""Download public source records for the ESR1 screening case."""
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import requests

PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / 'reports/nonkinase_screening_20261026/sources'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    provenance = []
    def get(name, url):
        path = OUT / name
        if not path.exists():
            for attempt in range(4):
                try:
                    response = requests.get(url, timeout=90)
                    response.raise_for_status()
                    response.json()
                    path.write_bytes(response.content)
                    break
                except requests.RequestException:
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
        value = json.loads(path.read_text())
        provenance.append(dict(file=name, url=url, bytes=path.stat().st_size,
                               sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        print(name, value.get('page_meta', {}).get('total_count', ''), flush=True)
        return value
    target = get('chembl_esr1_target.json',
                 'https://www.ebi.ac.uk/chembl/api/data/target.json?target_components__accession=P03372&limit=100')
    matches = [t for t in target['targets'] if t['target_chembl_id'] == 'CHEMBL206']
    assert len(matches) == 1 and matches[0]['target_type'] == 'SINGLE PROTEIN'
    assert matches[0]['tax_id'] == 9606
    activities = []
    url = 'https://www.ebi.ac.uk/chembl/api/data/activity.json?target_chembl_id=CHEMBL206&standard_type=Ki&limit=1000&offset=0'
    page = 0
    while url:
        data = get(f'chembl_esr1_ki_page{page}.json', url)
        activities.extend(data['activities'])
        nxt = data['page_meta']['next']
        url = 'https://www.ebi.ac.uk' + nxt if nxt else None
        page += 1
    assert len(activities) == data['page_meta']['total_count']
    assays = sorted({a['assay_chembl_id'] for a in activities})
    for start in range(0, len(assays), 50):
        ids = ','.join(assays[start:start + 50])
        result = get(f'chembl_esr1_assays_{start // 50}.json',
                     'https://www.ebi.ac.uk/chembl/api/data/assay.json?assay_chembl_id__in=' + ids + '&limit=1000')
        assert result['page_meta']['next'] is None
        assert set(a['assay_chembl_id'] for a in result['assays']) == set(ids.split(','))
    for name, url in json.loads((OUT / 'download_urls.json').read_text()).items():
        path = OUT / name
        provenance.append(dict(file=name, url=url, bytes=path.stat().st_size,
                               sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    (OUT / 'provenance.json').write_text(json.dumps(dict(
        retrieved_utc=datetime.now(timezone.utc).isoformat(), files=provenance), indent=2) + '\n')


if __name__ == '__main__':
    main()
