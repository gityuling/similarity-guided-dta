#!/usr/bin/env python3
"""Collect the study's saved inputs and outputs without changing their bytes."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
import tarfile
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--reuse-model-assets', action='store_true',
                        help='Reuse checkpoint/feature archives after checking their contents and hashes.')
    args = parser.parse_args()
    project = args.project.resolve()
    workspace = project.parent
    output = args.output.resolve()
    args.assets.mkdir(parents=True, exist_ok=True)
    selected = set()

    def add(path):
        path = path.resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        path.relative_to(workspace)
        selected.add(path)

    def glob(pattern):
        for path in project.glob(pattern):
            if path.is_file():
                add(path)

    frozen = json.loads((project / 'reports/bib_protocol_20261017/protocol.json').read_text())
    for item in frozen['files']:
        add(workspace / item['path'])
    for pattern in ['configs/*.json', 'scripts/*.py', 'reports/bib_protocol_20261017/*',
                    'results/bib_*_ablation.json', 'results/bib_*_frozen_prior.json',
                    'results/nexusdti_regressor_bib_*.json', 'results/checkpoints/nexusdti_bib_*.pt',
                    'results/predictions/*bib_*.csv',
                    'results/bib_validation_20261017/*.json', 'results/bib_scaffold_20261018/*.json',
                    'results/bib_nanobret_20261019/*.json',
                    'data/processed/*dtiam_protocol/summary.json',
                    'data/processed/kiba_identity_20261017/**/*.csv',
                    'data/processed/kiba_identity_20261017/*.json',
                    'data/processed/kiba_scaffold_20261017/**/*.csv',
                    'data/processed/kiba_scaffold_20261017/*.json',
                    'manuscript/bib/tables/*.csv', 'manuscript/bib/tables/*.tex']:
        glob(pattern)
    for name in ['bib_fusion_20261017', 'bib_nanobret_20261018', 'bib_downstream_20261022',
                 'bib_cellular_case_reviewed_20261018', 'bib_comparator_audit_20261017',
                 'davis_target_id_20261025']:
        for suffix in ['csv', 'json']:
            glob(f'reports/{name}/*.{suffix}')
    add(project / 'reports/davis_target_id_20261025/legacy_evaluate_drug_similarity_prior.py')
    for path in ['results/kiba_drugcold_supplement_status.json',
                 'results/kiba_drugcold_supplement_5fold.json',
                 'manuscript/bib/tools/build_tables.py']:
        add(project / path)
    # Preserve the records used by the main external evaluations.
    for path in ['reports/bib_nanobret_20261018/freeze.json',
                 'reports/bib_downstream_20261022/evaluation_freeze.json',
                 'reports/bib_downstream_20261022/preparation_inputs.json']:
        obj = json.loads((project / path).read_text())
        records = obj['files'] if isinstance(obj, dict) else obj
        for item in records:
            add(workspace / item['path'])
    for name in ['kiba_dtiam_protocol', 'bib_nanobret_20261019', 'bib_downstream_20261022']:
        glob(f'data/processed/{name}/tokens/build_audit.json')
    inventory = list(csv.DictReader((project / 'reports/bib_protocol_20261017/feature_inventory.csv').open()))
    expected_feature_hashes = {r['path']: r['sha256'] for r in inventory}
    for row in inventory:
        add(workspace / row['path'])
    for name in ['bib_nanobret_20261019', 'bib_downstream_20261022']:
        glob(f'data/processed/{name}/tokens/**/*.npy')
    for candidate in (project / 'baselines/DTIAM').glob('LICENSE*'):
        add(candidate)

    groups = {name: [] for name in ['repository', 'reproduction-data', 'affinity-checkpoints',
                                   'davis-features', 'kiba-features', 'external-features']}
    for path in sorted(selected):
        rel = path.relative_to(workspace).as_posix()
        if path.suffix == '.npy':
            group = ('davis-features' if rel.startswith('dataset/') else
                     'kiba-features' if '/kiba_dtiam_protocol/' in rel else 'external-features')
        elif path.suffix == '.pt':
            group = 'affinity-checkpoints'
        elif path.stat().st_size > 1_000_000 or path.suffix in ('.xlsx', '.tgz'):
            group = 'reproduction-data'
        else:
            group = 'repository'
        groups[group].append(path)

    entries = []
    assets = []
    previous = {}
    previous_assets = {}
    if args.reuse_model_assets:
        previous = {r['path']: r for r in json.loads((output / 'manifests/files.json').read_text())}
        previous_assets = {r['group']: r for r in json.loads((output / 'manifests/assets.json').read_text())['assets']}
    for group, paths in groups.items():
        print(f'{group}: {len(paths)} files, {sum(p.stat().st_size for p in paths)/1e6:.1f} MB', flush=True)
        if args.reuse_model_assets and group not in ('repository', 'reproduction-data'):
            current = [{'path': p.relative_to(workspace).as_posix(), 'bytes': p.stat().st_size,
                        'sha256': digest(p), 'group': group} for p in paths]
            old = sorted((r for r in previous.values() if r['group'] == group), key=lambda r: r['path'])
            if current != old:
                raise RuntimeError(f'Model asset inputs changed: {group}')
            asset = previous_assets[group]
            archive = args.assets / asset['name']
            if archive.stat().st_size != asset['bytes'] or digest(archive) != asset['sha256']:
                raise RuntimeError(f'Model asset archive changed: {group}')
            entries.extend(current)
            assets.append(asset)
            continue
        archive = None
        if group != 'repository':
            archive = args.assets / f'{group}.tar.gz'
            raw = archive.open('wb')
            zipped = gzip.GzipFile(fileobj=raw, mode='wb', compresslevel=1, mtime=0)
            tar = tarfile.open(fileobj=zipped, mode='w|')
        for path in paths:
            rel = path.relative_to(workspace).as_posix()
            sha = digest(path)
            if rel in expected_feature_hashes and expected_feature_hashes[rel] != sha:
                raise RuntimeError(f'Feature hash differs from frozen inventory: {rel}')
            entries.append({'path': rel, 'bytes': path.stat().st_size, 'sha256': sha, 'group': group})
            if archive is None:
                dest = output / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, dest)
            else:
                info = tar.gettarinfo(str(path), arcname=rel)
                info.uid = info.gid = 0
                info.uname = info.gname = ''
                info.mtime = 0
                with path.open('rb') as source:
                    tar.addfile(info, source)
        if archive is not None:
            tar.close(); zipped.close(); raw.close()
            assets.append({'name': archive.name, 'group': group, 'bytes': archive.stat().st_size,
                           'sha256': digest(archive), 'files': len(paths)})
            print(f'Packed {archive.name}: {archive.stat().st_size/1e6:.1f} MB', flush=True)

    manifests = output / 'manifests'
    manifests.mkdir(parents=True, exist_ok=True)
    (manifests / 'files.json').write_text(json.dumps(sorted(entries, key=lambda x: x['path']), indent=2) + '\n')
    (manifests / 'assets.json').write_text(json.dumps({'repository': 'gityuling/similarity-guided-dta',
        'tag': 'v0.1.0', 'assets': assets}, indent=2) + '\n')
    (args.assets / 'SHA256SUMS').write_text(''.join(f"{a['sha256']}  {a['name']}\n" for a in assets))
    print(f'Collected {len(entries)} files; original feature inventory verified.', flush=True)


if __name__ == '__main__':
    main()
