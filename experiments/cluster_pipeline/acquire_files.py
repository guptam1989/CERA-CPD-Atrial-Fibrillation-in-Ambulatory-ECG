"""Checksum-verified static CPSC files, with bounded parallel HTTP downloads."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import argparse
import hashlib
import json
import os
import time
import urllib.request
import urllib.parse

PRIMARY = 'https://physionet.org/files/cpsc2021/1.0.0/'
MIRROR = 'https://physionet-open.s3.amazonaws.com/cpsc2021/1.0.0/'

def request(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'CERA-research/1.0'}), timeout=60)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--dataset', choices=['cpsc2021', 'ltafdb', 'nsrdb', 'shdb-af', 'afdb', 'mitdb', 'nstdb'], default='cpsc2021')
    parser.add_argument('--version', default='1.0.0')
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        raise ValueError('At most 16 HTTP downloads')
    root = args.output.resolve()
    primary = f'https://physionet.org/files/{args.dataset}/{args.version}/'
    mirror = f'https://physionet-open.s3.amazonaws.com/{args.dataset}/{args.version}/'
    root.mkdir(parents=True, exist_ok=True)
    with request(primary + 'SHA256SUMS.txt') as response:
        checksums = response.read().decode('utf-8')
    (root / 'SHA256SUMS.txt').write_text(checksums)
    entries = []
    for line in checksums.splitlines():
        expected, name = line.split(maxsplit=1)
        name = name.lstrip(' *')
        dataset_file = (name.startswith(('Training_set_I/', 'Training_set_II/')) if args.dataset == 'cpsc2021'
                        else name.endswith(('.hea', '.dat', '.atr', '.qrs', '.qrsc', '.csv')) and '/old/' not in '/' + name)
        if dataset_file or name in ['LICENSE.txt', 'README.md', 'RECORDS']:
            path = (root / name).resolve()
            if not path.is_relative_to(root):
                raise ValueError('Unsafe checksum path')
            entries.append((expected, name, path))
    if not entries:
        raise ValueError('No dataset paths in official hash list')
    # Use the published anonymous S3 mirror only if the same hash list is present.
    source = primary
    try:
        with request(mirror + 'SHA256SUMS.txt') as response:
            if response.read().decode('utf-8') == checksums:
                source = mirror
    except Exception:
        pass
    def download(entry):
        expected, name, path = entry
        if path.is_file() and sha(path) == expected:
            return name
        path.parent.mkdir(parents=True, exist_ok=True)
        for attempt in range(3):
            try:
                temporary = path.with_name(path.name + '.part')
                with request(source + urllib.parse.quote(name, safe='/')) as response, temporary.open('wb') as stream:
                    for block in iter(lambda: response.read(1024 * 1024), b''):
                        stream.write(block)
                if sha(temporary) != expected:
                    raise ValueError('Downloaded hash differs from official checksum: ' + name)
                os.replace(temporary, path)
                return name
            except Exception:
                if attempt == 2:
                    raise
                time.sleep(2)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(download, entry) for entry in entries]
        for count, future in enumerate(as_completed(futures), 1):
            future.result()
            if count % 100 == 0:
                print(json.dumps({'files_checked': count, 'total': len(entries), 'source': source}), flush=True)
    report = dict(status='downloaded_and_checksum_verified', dataset=args.dataset, version=args.version,
                  source_url=source, official_checksum_source=primary+'SHA256SUMS.txt',
                  official_file_hashes_checked=len(entries), headers=len(list(root.rglob('*.hea'))), root=str(root))
    (root / 'acquisition.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)

if __name__ == '__main__':
    main()
