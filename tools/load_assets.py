#!/usr/bin/env python3
"""Download, verify and extract a selected set of release assets."""
import argparse
import hashlib
import json
import shutil
import subprocess
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--groups', nargs='+', default=['reproduction-data'])
    parser.add_argument('--asset-dir', type=Path, default=ROOT / 'downloads')
    args = parser.parse_args()
    metadata = json.loads((ROOT / 'manifests/assets.json').read_text())
    records = json.loads((ROOT / 'manifests/files.json').read_text())
    available = {a['group'] for a in metadata['assets']}
    groups = available if args.groups == ['all'] else set(args.groups)
    if not groups <= available:
        parser.error(f'Unknown groups: {groups - available}')
    args.asset_dir.mkdir(parents=True, exist_ok=True)
    for asset in metadata['assets']:
        if asset['group'] not in groups:
            continue
        path = args.asset_dir / asset['name']
        if not path.exists():
            subprocess.run(['gh', 'release', 'download', metadata['tag'], '--repo', metadata['repository'],
                            '--pattern', asset['name'], '--dir', str(args.asset_dir)], check=True)
        if path.stat().st_size != asset['bytes'] or digest(path) != asset['sha256']:
            raise RuntimeError(f'Archive checksum mismatch: {path}')
        expected = {r['path']: r for r in records if r['group'] == asset['group']}
        extracted = set()
        with tarfile.open(path, 'r:gz') as tar:
            for member in tar:
                relative = PurePosixPath(member.name)
                if (not member.isfile() or relative.is_absolute() or '..' in relative.parts
                        or member.name not in expected or member.name in extracted):
                    raise RuntimeError(f'Unexpected archive member: {member.name}')
                target = ROOT / member.name
                if not target.resolve().is_relative_to(ROOT):
                    raise RuntimeError(f'Extraction path escapes repository: {target}')
                target.parent.mkdir(parents=True, exist_ok=True)
                temporary = target.with_name(target.name + '.partial')
                with tar.extractfile(member) as source, temporary.open('wb') as output:
                    shutil.copyfileobj(source, output, 8 * 1024 * 1024)
                row = expected[member.name]
                if temporary.stat().st_size != row['bytes'] or digest(temporary) != row['sha256']:
                    temporary.unlink()
                    raise RuntimeError(f'Extracted file checksum mismatch: {member.name}')
                temporary.replace(target)
                extracted.add(member.name)
        if extracted != set(expected):
            raise RuntimeError(f'Incomplete archive: {asset["name"]}')
        print(f'{asset["name"]}: verified and extracted {len(extracted)} files', flush=True)


if __name__ == '__main__':
    main()
