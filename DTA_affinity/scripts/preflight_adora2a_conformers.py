#!/usr/bin/env python3
"""Check uncached molecules with the encoder's unchanged RDKit embedding calls."""
import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

PROJECT = Path(__file__).resolve().parents[1]
REPORT = PROJECT / 'reports/adora2a_screening_20261026'
CACHE = PROJECT / 'data/processed/adora2a_screening_20261026/tokens'


def check(smiles):
    RDLogger.DisableLog('rdApp.warning')
    start = time.monotonic()
    molecule = AllChem.AddHs(Chem.MolFromSmiles(smiles), addCoords=True)
    params = AllChem.ETKDG()
    params.randomSeed = 20261016
    first = AllChem.EmbedMolecule(molecule, params)
    fallback = None
    if first == -1:
        fallback = AllChem.EmbedMolecule(molecule, randomSeed=20261016, useRandomCoords=True)
    return dict(smiles=smiles, md5=hashlib.md5(smiles.encode()).hexdigest(),
                first_status=first, fallback_status=fallback,
                success=first != -1 or fallback != -1,
                elapsed_seconds=time.monotonic() - start)


def main():
    table_path = REPORT / 'feature_table.csv'
    table = pd.read_csv(table_path)
    pending = [s for s in sorted(table.Canonical_SMILES.unique())
               if not (CACHE / 'drugs' / (hashlib.md5(s.encode()).hexdigest() + '.npy')).exists()]
    output = REPORT / 'conformer_preflight.json'
    result = dict(table_sha256=hashlib.sha256(table_path.read_bytes()).hexdigest(),
                  encoder_algorithm='ETKDG seed 20261016, then EmbedMolecule seed 20261016 useRandomCoords=True',
                  pending_count=len(pending), workers=8, complete=False, checks=[])
    print(f'Checking {len(pending)} uncached molecules', flush=True)
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(check, s): s for s in pending}
        for future in as_completed(futures):
            item = future.result()
            result['checks'].append(item)
            print(json.dumps(dict(done=len(result['checks']), **item)), flush=True)
            output.write_text(json.dumps(result, indent=2) + '\n')
    result['checks'].sort(key=lambda x: x['smiles'])
    result['complete'] = True
    output.write_text(json.dumps(result, indent=2) + '\n')
    print('Failed:', sum(not x['success'] for x in result['checks']), flush=True)


if __name__ == '__main__':
    main()
