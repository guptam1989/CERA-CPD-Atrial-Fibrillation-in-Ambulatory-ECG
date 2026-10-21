"""Audit official CPSC2021 files and freeze patient-group outer splits.

The source subject namespace is verified against the primary research reader
listed below. No two pool-specific namespaces are invented. Exact duplicate
waveforms are quarantined before assigning any role.
"""
from collections import Counter, defaultdict
from pathlib import Path
import argparse
import hashlib
import json
import math
import random
import re

import wfdb

IDENTITY_SOURCE = 'https://raw.githubusercontent.com/DeepPSP/torch_ecg/master/torch_ecg/databases/cpsc_databases/cpsc2021.py'

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def write(path, payload):
    path.write_text(json.dumps(payload, indent=2, allow_nan=False))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    root, output = args.data.resolve(), args.output.resolve()
    acquisition = json.loads((root / 'acquisition.json').read_text())
    if acquisition['status'] != 'downloaded_and_checksum_verified':
        raise ValueError('Official checksum verification must precede splitting')
    output.mkdir(parents=True, exist_ok=False)
    rows, duplicate_ids, digests = [], [], {}
    for pool in ['Training_set_I', 'Training_set_II']:
        for header in sorted((root / pool).glob('*.hea')):
            match = re.fullmatch(r'data_(\d+)_(\d+)', header.stem)
            if not match:
                raise ValueError('Unrecognised record naming: ' + header.name)
            subject, segment = map(int, match.groups())
            if (pool == 'Training_set_I') != (subject <= 53):
                raise ValueError('Official subject namespace contradicts source reader')
            record = header.with_suffix('')
            info = wfdb.rdheader(str(record))
            annotation = wfdb.rdann(str(record), 'atr')
            if info.n_sig != 2 or info.fs != 200 or info.sig_len <= 0:
                raise ValueError('Unexpected CPSC signal metadata')
            signal_paths = [header.parent / name for name in sorted(set(info.file_name))]
            source_hash = hashlib.sha256(''.join(sha(path) for path in signal_paths).encode()).hexdigest()
            if source_hash in digests:
                duplicate_ids.append({'record': header.stem, 'duplicate_of': digests[source_hash], 'source_hash': source_hash})
                continue
            digests[source_hash] = header.stem
            labels = list(info.comments)
            allowed = {'non atrial fibrillation', 'persistent atrial fibrillation', 'paroxysmal atrial fibrillation'}
            if len(labels) != 1 or labels[0] not in allowed:
                raise ValueError('Unexpected official global diagnosis: ' + repr(labels))
            rows.append(dict(recording_id=header.stem, patient_id=f'CPSC2021_subject_{subject:03d}',
                             patient_id_verified=True, patient_identity_source=IDENTITY_SOURCE,
                             subject_code=subject, segment_code=segment, pool=pool,
                             record_path=str(record), rhythm_extension='atr', beat_extension='atr',
                             lead_indices=[0, 1], sampling_hz=float(info.fs), duration_s=info.sig_len/info.fs,
                             global_diagnosis=labels[0], source_signal_sha256=source_hash,
                             source_header_sha256=sha(header), source_annotation_sha256=sha(record.with_suffix('.atr')),
                             reference_beat_annotations=int(len(annotation.sample))))
    patients = defaultdict(list)
    for row in rows:
        patients[row['patient_id']].append(row)
    strata = defaultdict(list)
    for patient, records in patients.items():
        diagnoses = {row['global_diagnosis'] for row in records}
        group = 'paroxysmal_or_mixed' if 'paroxysmal atrial fibrillation' in diagnoses or len(diagnoses)>1 else next(iter(diagnoses))
        strata[group].append(patient)
        for row in records:
            row['stratum'] = group
    rng = random.Random(args.seed)
    groups = [[] for _ in range(5)]
    for stratum in sorted(strata):
        shuffled = sorted(strata[stratum]); rng.shuffle(shuffled)
        for patient in shuffled:
            target = min(range(5), key=lambda i: (sum(p in strata[stratum] for p in groups[i]), len(groups[i]), i))
            groups[target].append(patient)
    folds = []
    for index, heldout in enumerate(groups):
        remaining = [p for p in sorted(patients) if p not in heldout]
        inner_rng = random.Random(args.seed + index + 1)
        # Inner calibration selection is random, not burden/class enriched.
        # The outer stratification and shared fitted models still require
        # empirical CV coverage assessment rather than an unconditional claim.
        shuffled = list(remaining)
        inner_rng.shuffle(shuffled)
        n_cal = max(10, math.ceil(0.15 * len(remaining)))
        n_val = max(5, math.ceil(0.15 * len(remaining)))
        calibration, validation = set(shuffled[:n_cal]), set(shuffled[n_cal:n_cal+n_val])
        test = set(heldout)
        output_rows = []
        for row in rows:
            role = 'test' if row['patient_id'] in test else 'calibration' if row['patient_id'] in calibration else 'validation' if row['patient_id'] in validation else 'train'
            output_rows.append(dict(row, split=role,
                                   primary_burden_record=row['segment_code'] == min(r['segment_code'] for r in patients[row['patient_id']])))
        target = output / f'raw_fold_{index}.json'
        write(target, {'records': output_rows, 'fold': index, 'seed': args.seed,
                       'calibration_selection': 'one lowest numeric segment ID per patient; same primary test-record rule',
                       'identity_source': IDENTITY_SOURCE, 'acquisition': acquisition})
        folds.append({'fold': index, 'manifest_path': target.name,
                      'patient_roles': dict(Counter(r['split'] for r in {r['patient_id']:r for r in output_rows}.values()))})
    write(output / 'raw_fold_index.json', {'folds':folds})
    # Preparation is done once. Immutable features can then be assigned to
    # each split; fitting will open only the roles for the selected fold.
    write(output / 'raw_preparation_manifest.json', {'records':[dict(row, split='test') for row in rows],
                                                   'identity_source':IDENTITY_SOURCE})
    report = dict(dataset='CPSC2021', version='1.0.0', raw_records=len(rows)+len(duplicate_ids),
                  eligible_after_exact_duplicate_audit=len(rows), patient_groups=len(patients),
                  pool_counts=dict(Counter(row['pool'] for row in rows)),
                  record_hours=sum(row['duration_s'] for row in rows)/3600,
                  diagnosis_counts=dict(Counter(row['global_diagnosis'] for row in rows)),
                  exact_duplicates_quarantined=duplicate_ids, identity_source=IDENTITY_SOURCE,
                  split_seed=args.seed, fold_patient_counts=folds,
                  near_duplicate_status='same-source segments are kept in one verified subject group; cross-subject near-duplicates still require waveform review',
                  advertised_count_discrepancy='Official v1.0.0 SHA inventory has 719 stage-I headers, versus 730 in descriptive metadata.')
    write(output / 'dataset_audit.json', report)
    print(json.dumps(report), flush=True)

if __name__ == '__main__':
    main()
