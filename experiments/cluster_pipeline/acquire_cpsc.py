"""Acquire the public CPSC2021 v1.0.0 archive and audit official file hashes."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import shutil
import time
import urllib.request
import zipfile

SOURCE = 'https://physionet.org/content/cpsc2021/get-zip/1.0.0/'

def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < 4 * 1024**3:
        raise RuntimeError('At least 4 GiB free disk space is required')
    archive = root / 'cpsc2021-1.0.0.zip'
    if not archive.exists():
        partial = archive.with_suffix('.zip.part')
        for attempt in range(3):
            try:
                request = urllib.request.Request(SOURCE, headers={'User-Agent': 'CERA-CPD-research/1.0'})
                with urllib.request.urlopen(request, timeout=120) as response, partial.open('wb') as stream:
                    copied = 0
                    for block in iter(lambda: response.read(1024 * 1024), b''):
                        stream.write(block)
                        copied += len(block)
                        if copied % (100 * 1024 * 1024) < len(block):
                            print(json.dumps({'downloaded_bytes': copied}), flush=True)
                os.replace(partial, archive)
                break
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(5)
    extracted = root / 'extracted'
    extracted.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as package:
        for item in package.infolist():
            destination = (extracted / item.filename).resolve()
            if not destination.is_relative_to(extracted) or (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Archive contains an unsafe path or symbolic link')
        package.extractall(extracted)
    hash_files = list(extracted.rglob('SHA256SUMS.txt'))
    if not hash_files:
        raise RuntimeError('Official SHA256SUMS.txt is absent')
    checked, failures = 0, []
    for checksum in hash_files:
        for line in checksum.read_text().splitlines():
            if not line.strip():
                continue
            expected, name = line.split(maxsplit=1)
            path = (checksum.parent / name.lstrip(' *')).resolve()
            if not path.is_relative_to(extracted):
                raise ValueError('Unsafe checksum path')
            if not path.is_file() or digest(path) != expected:
                failures.append(str(path))
            checked += 1
    if failures:
        raise RuntimeError(f'Official file checksum audit failed: {failures[:10]}')
    records = list(extracted.rglob('*.hea'))
    report = {'status': 'downloaded_and_checksum_verified', 'dataset': 'CPSC2021', 'version': '1.0.0',
              'source_url': SOURCE, 'archive_sha256': digest(archive), 'archive_bytes': archive.stat().st_size,
              'official_file_hashes_checked': checked, 'headers': len(records), 'root': str(extracted),
              'note': 'Header count includes any example-entry files; patient split audit is a separate step.'}
    (root / 'acquisition.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)

if __name__ == '__main__':
    main()
