#!/usr/bin/env python3
"""Add the shared-backbone comparison and final case studies to a base release."""
import argparse
import gzip
import hashlib
import json
import shutil
import tarfile
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--assets', type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    root = Path(__file__).resolve().parents[1]
    rows = {r['path']: r for r in json.loads((root / 'manifests/files.json').read_text())}
    metadata = json.loads((root / 'manifests/assets.json').read_text())
    assets = {a['group']: a for a in metadata['assets']}
    selected = set()

    for name in ['knn_shared_comparison_20261025.manifest.json',
                 'nonkinase_screening_20261026_manifest.json']:
        for row in json.loads((args.assets / name).read_text())['files']:
            source = project.parent / row['path']
            if digest(source) != row['sha256']:
                raise RuntimeError(f'Frozen comparison input changed: {row["path"]}')
            if source.suffix != '.log':
                selected.add(source)
    for folder in ['reports/adra2b_screening_20261026',
                   'results/adra2b_screening_20261026',
                   'data/processed/adra2b_screening_20261026',
                   'reports/adora2a_screening_20261026',
                   'manuscript/bib/tables/source/target_analysis_20261026']:
        selected.update(p for p in (project / folder).rglob('*')
                        if p.is_file() and p.suffix not in {'.log', '.png', '.svg', '.pdf', '.pyc'})
    for pattern in ['scripts/*adra2b*.py', 'scripts/*adora2a*.py',
                    'configs/adra2b*.json', 'configs/adora2a*.json',
                    'manuscript/bib/tables/source/*case_top10.csv']:
        selected.update(project.glob(pattern))
    for name in ['build_case_studies.py', 'build_target_report.py']:
        selected.add(project / 'manuscript/bib/tools' / name)
    selected.add(project / 'reports/bib_downstream_20261022/sources/pubchem_1963317_data.csv')

    changed = set()
    for source in sorted(selected):
        relative = source.relative_to(project.parent).as_posix()
        sha = digest(source)
        portable = None
        if relative in {'DTA_affinity/manuscript/bib/tools/build_case_studies.py',
                        'DTA_affinity/manuscript/bib/tools/build_target_report.py'}:
            portable = source.read_text().replace(
                'sys.path.append("/home/lsy/.agents/skills/figure-layout-hygiene/scripts")\n','').encode()
            sha = hashlib.sha256(portable).hexdigest()
        if relative in rows:
            if rows[relative]['sha256'] != sha:
                raise RuntimeError(f'Base-release input changed: {relative}')
            continue
        if '/knn_shared_20261025/' in relative and source.suffix in {'.pt', '.npz'}:
            group = 'shared-backbone-models'
        elif source.suffix in {'.pt', '.npy'}:
            group = 'case-study-models'
        elif source.stat().st_size > 1_000_000 or source.suffix in {'.xlsx', '.tgz'}:
            group = 'reproduction-data'
        else:
            group = 'repository'
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if portable is None:
            shutil.copy2(source, target)
        else:
            target.write_bytes(portable)
        rows[relative] = dict(path=relative, bytes=target.stat().st_size,
                              sha256=sha, group=group)
        changed.add(group)

    for group in sorted(changed - {'repository'}):
        members = sorted((r for r in rows.values() if r['group'] == group), key=lambda r:r['path'])
        archive = args.assets / f'{group}.tar.gz'
        temporary = archive.with_suffix('.tar.gz.partial')
        with temporary.open('wb') as raw:
            with gzip.GzipFile(fileobj=raw, mode='wb', compresslevel=1, mtime=0) as zipped:
                with tarfile.open(fileobj=zipped, mode='w|') as tar:
                    for row in members:
                        source = root / row['path']
                        if digest(source) != row['sha256']:
                            raise RuntimeError(f'Changed release input: {source}')
                        info = tar.gettarinfo(str(source), arcname=row['path'])
                        info.uid = info.gid = 0
                        info.uname = info.gname = ''
                        info.mtime = 0
                        with source.open('rb') as stream:
                            tar.addfile(info, stream)
        temporary.replace(archive)
        assets[group] = dict(name=archive.name, group=group, bytes=archive.stat().st_size,
                             sha256=digest(archive), files=len(members))
        print(f'{group}: {len(members)} files, {archive.stat().st_size / 1e6:.1f} MB', flush=True)

    (root / 'manifests/files.json').write_text(json.dumps(sorted(rows.values(),key=lambda r:r['path']),indent=2)+'\n')
    metadata['assets'] = list(assets.values())
    (root / 'manifests/assets.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (args.assets / 'SHA256SUMS').write_text(''.join(f"{a['sha256']}  {a['name']}\n" for a in assets.values()))
    ignore = ['.venv/', '__pycache__/', 'downloads/', 'verification/', '*.partial',
              '/DTA_affinity/manuscript/bib/figures/', '/DTA_affinity/manuscript/bib/sections/']
    ignore.extend('/'+r['path'] for r in sorted(rows.values(),key=lambda r:r['path']) if r['group']!='repository')
    (root / '.gitignore').write_text('\n'.join(ignore)+'\n')
    print(f'Release inventory: {len(rows)} files.', flush=True)


if __name__ == '__main__':
    main()
