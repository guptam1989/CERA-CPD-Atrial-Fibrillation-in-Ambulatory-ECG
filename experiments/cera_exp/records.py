"""Dataset readers returning a common record structure.

Every reader returns a dict with native-rate two-lead signal, reference beat
positions (oracle R peaks), beat symbols, beat-level AF labels (1 AF/AFL,
0 non-AF, -1 unknown) and record metadata. Labels before the first rhythm
annotation are unknown unless the database documents a single rhythm.
"""
from __future__ import annotations

import os

import numpy as np
import wfdb

BEAT_SYMBOLS = set("NLRBAaJSVrFejnE/fQ?")
ECTOPIC_SYMBOLS = set("AaJSVFE")
AF_RHYTHMS = ("(AFIB", "(AFL")
CPSC_CLASSES = {"non atrial fibrillation": 0, "persistent atrial fibrillation": 1,
                "paroxysmal atrial fibrillation": 2}


def _clean(note: str) -> str:
    return (note or "").replace("\x00", "").strip()


def _signal(path: str):
    rec = wfdb.rdrecord(path)
    sig = np.nan_to_num(rec.p_signal[:, :2].astype(np.float32))
    if sig.shape[1] == 1:
        sig = np.repeat(sig, 2, axis=1)
    return sig, float(rec.fs), list(rec.sig_name[:2])


def _beats(ann) -> tuple[np.ndarray, np.ndarray]:
    sym = np.asarray(ann.symbol)
    keep = np.isin(sym, list(BEAT_SYMBOLS))
    samples = np.asarray(ann.sample)[keep].astype(np.int64)
    symbols = sym[keep]
    order = np.argsort(samples, kind="stable")
    samples, symbols = samples[order], symbols[order]
    uniq = np.concatenate(([True], np.diff(samples) > 0))
    return samples[uniq], symbols[uniq]


def _labels_from_rhythm(beat_samples: np.ndarray, rhythm_samples, rhythm_notes) -> np.ndarray:
    """Beat label = rhythm of last rhythm mark at or before the beat; else unknown."""
    labels = np.full(len(beat_samples), -1, dtype=np.int8)
    if len(rhythm_samples) == 0:
        return labels
    order = np.argsort(rhythm_samples, kind="stable")
    rs = np.asarray(rhythm_samples)[order]
    notes = [rhythm_notes[i] for i in order]
    is_af = np.array([n.startswith(AF_RHYTHMS) for n in notes])
    pos = np.searchsorted(rs, beat_samples, side="right") - 1
    known = pos >= 0
    labels[known] = is_af[pos[known]].astype(np.int8)
    return labels


def read_cpsc(path: str) -> dict:
    """CPSC2021 record. Episodes follow the official scorer pairing of
    ('(AFIB'|'(AFL') and '(N' rhythm marks; all other beats are non-AF."""
    sig, fs, names = _signal(path)
    hdr = wfdb.rdheader(path)
    comment = " ".join(hdr.comments)
    rec_class = next(v for k, v in CPSC_CLASSES.items() if k in comment)
    ann = wfdb.rdann(path, "atr")
    all_samples = np.asarray(ann.sample).astype(np.int64)
    notes = np.array([_clean(n) for n in ann.aux_note])
    starts = np.where((notes == "(AFIB") | (notes == "(AFL"))[0]
    ends = np.where(notes == "(N")[0]
    beats, symbols = _beats(ann)
    labels = np.zeros(len(beats), dtype=np.int8)
    episodes = []
    for s, e in zip(starts, ends):
        a, b = all_samples[s], all_samples[e]
        labels[(beats >= a) & (beats < b)] = 1
        episodes.append((int(a), int(b)))
    flag = np.zeros(len(all_samples), dtype=np.int8)
    flag[starts] = 1
    flag[ends] = 2
    name = os.path.basename(path)
    pid = int(name.split("_")[1])
    return dict(dataset="cpsc2021", record=name, patient=f"cpsc_{pid:03d}", path=path,
                fs=fs, sig=sig, lead_names=names, rpeaks=beats, symbols=symbols, labels=labels,
                rec_class=rec_class, n_samples=len(sig), official_pairs=len(starts) == len(ends),
                ref_episodes=episodes, ann_samples=all_samples, ann_flag=flag)


def read_rhythm_db(path: str, dataset: str, beat_ext: str = "atr", rhythm_ext: str = "atr",
                   all_non_af: bool = False, patient: str | None = None) -> dict:
    sig, fs, names = _signal(path)
    beat_ann = wfdb.rdann(path, beat_ext)
    beats, symbols = _beats(beat_ann)
    if all_non_af:
        labels = np.zeros(len(beats), dtype=np.int8)
    else:
        rann = beat_ann if rhythm_ext == beat_ext else wfdb.rdann(path, rhythm_ext)
        notes = [_clean(n) for n in rann.aux_note]
        idx = [i for i, n in enumerate(notes) if n.startswith("(")]
        labels = _labels_from_rhythm(beats, np.asarray(rann.sample)[idx], [notes[i] for i in idx])
    name = os.path.basename(path)
    return dict(dataset=dataset, record=name, patient=patient or f"{dataset}_{name}", path=path,
                fs=fs, sig=sig, lead_names=names, rpeaks=beats, symbols=symbols, labels=labels,
                rec_class=None, n_samples=len(sig))


def read_afdb(path: str) -> dict:
    beat_ext = "qrsc" if os.path.exists(path + ".qrsc") else "qrs"
    rec = read_rhythm_db(path, "afdb", beat_ext=beat_ext, rhythm_ext="atr")
    rec["beat_source"] = beat_ext
    return rec


def read_ltafdb(path: str) -> dict:
    return read_rhythm_db(path, "ltafdb")


def read_nsrdb(path: str) -> dict:
    # NSRDB documents subjects without significant arrhythmia: known non-AF.
    return read_rhythm_db(path, "nsrdb", all_non_af=True)


def read_mitdb(path: str) -> dict:
    # Records 201 and 202 are the same subject (MIT-BIH directory).
    name = os.path.basename(path)
    patient = "mitdb_201" if name in ("201", "202") else f"mitdb_{name}"
    return read_rhythm_db(path, "mitdb", patient=patient)


READERS = {"cpsc2021": read_cpsc, "afdb": read_afdb, "ltafdb": read_ltafdb,
           "nsrdb": read_nsrdb, "mitdb": read_mitdb}
