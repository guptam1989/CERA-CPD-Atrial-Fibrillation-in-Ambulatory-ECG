"""NSTDB noise stress: baseline wander (bw), electrode motion (em), muscle (ma).

Only the second half of each noise record is used, so that no excerpt can be
shared with any training-time augmentation. SNR follows the NSTDB convention
S = (peak-to-peak QRS amplitude)^2 / 8 with the QRS amplitude estimated as the
median peak-to-peak value of 1-s windows of the clean band-passed lead.
"""
from __future__ import annotations

import os

import numpy as np
import wfdb
from scipy.signal import resample_poly
from fractions import Fraction

from .features import bandpass

_CACHE: dict = {}


def _noise(kind: str, fs: float, root: str) -> np.ndarray:
    key = (kind, fs, root)
    if key not in _CACHE:
        rec = wfdb.rdrecord(os.path.join(root, kind))
        x = rec.p_signal[:, :2].astype(np.float64)
        x = x[len(x) // 2:]
        frac = Fraction(fs / rec.fs).limit_denominator(1000)
        x = resample_poly(x, frac.numerator, frac.denominator, axis=0)
        _CACHE[key] = (x - x.mean(axis=0)).astype(np.float32)
    return _CACHE[key]


def add_noise(sig: np.ndarray, fs: float, kind: str, snr_db: float, noise_root: str,
              seed: int = 0) -> np.ndarray:
    noise = _noise(kind, fs, noise_root)
    rng = np.random.default_rng(seed)
    n = len(sig)
    reps = int(np.ceil(n / len(noise))) + 1
    tiled = np.concatenate([noise] * reps, axis=0)
    start = int(rng.integers(0, len(noise)))
    seg = tiled[start:start + n]
    clean = bandpass(sig, fs)
    w = int(fs)
    m = (n // w) * w
    out = sig.copy()
    for lead in range(sig.shape[1]):
        ptp = np.ptp(clean[:m, lead].reshape(-1, w), axis=1) if m else np.array([1.0])
        s_pow = np.median(ptp) ** 2 / 8.0
        n_pow = float(np.mean(seg[:, lead] ** 2)) + 1e-12
        gain = np.sqrt(s_pow / (n_pow * 10 ** (snr_db / 10.0)))
        out[:, lead] = sig[:, lead] + gain * seg[:, lead]
    return out
