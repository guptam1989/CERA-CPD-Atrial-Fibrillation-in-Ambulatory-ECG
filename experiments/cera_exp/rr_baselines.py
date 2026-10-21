"""Classical RR-interval AF detectors used as reference comparators.

  cv      coefficient of variation of the last w RR intervals (Tateno & Glass)
  cosen   coefficient of sample entropy over 16 RR intervals (Lake & Moorman,
          m=1, r=30 ms, numerator floored at 1/2 match when zero)
  markov  RR-transition Markov log-likelihood ratio over 32 beats
          (Moody & Mark), transitions estimated on training patients

Scores are causal (window ends at the scored beat). Decisions are smoothed by a
causal majority over k beats and boundaries are shifted back by the window lag.
"""
from __future__ import annotations

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def _rr(rp_s: np.ndarray) -> np.ndarray:
    rr = np.diff(rp_s)
    return np.concatenate([[rr[0] if len(rr) else 1.0], rr])  # rr[n] = interval ending at beat n


def cv_score(rp_s, w: int = 32):
    rr = _rr(rp_s)
    n = len(rr)
    if n < w:
        return np.zeros(n)
    win = sliding_window_view(rr, w)
    s = win.std(axis=1) / win.mean(axis=1)
    return np.concatenate([np.full(w - 1, s[0]), s])


def cosen_score(rp_s, w: int = 16, r: float = 0.03, chunk: int = 200000):
    rr = _rr(rp_s)
    n = len(rr)
    if n < w:
        return np.full(n, -3.0)
    win = sliding_window_view(rr, w)
    out = np.empty(len(win))
    iu = np.triu_indices(w - 1, 1)
    for s in range(0, len(win), chunk):
        x = win[s:s + chunk]
        a = x[:, :-1]
        d1 = np.abs(a[:, :, None] - a[:, None, :]) <= r              # length-1 matches (first w-1)
        d2 = d1 & (np.abs(x[:, 1:, None] - x[:, None, 1:]) <= r)      # length-2 matches
        B = d1[:, iu[0], iu[1]].sum(axis=1).astype(float)
        A = d2[:, iu[0], iu[1]].sum(axis=1).astype(float)
        A = np.maximum(A, 0.5)
        B = np.maximum(B, A)
        qse = -np.log(A / B) + np.log(2 * r)
        out[s:s + chunk] = qse - np.log(x.mean(axis=1))
    return np.concatenate([np.full(w - 1, out[0]), out])


def markov_states(rp_s, k: int = 8):
    rr = _rr(rp_s)
    c = np.cumsum(np.concatenate([[0.0], rr]))
    idx = np.arange(len(rr))
    lo = np.maximum(0, idx - k)
    mean = (c[idx] - c[lo]) / np.maximum(idx - lo, 1)
    mean[0] = rr[0]
    ratio = rr / mean
    return np.where(ratio < 0.85, 0, np.where(ratio > 1.15, 2, 1))


def fit_markov(records: list[dict]) -> np.ndarray:
    counts = np.ones((2, 3, 3))
    for rec in records:
        st = markov_states(rec["rp_s"])
        lab = rec["labels"]
        ok = (lab[1:] >= 0) & (lab[1:] == lab[:-1])
        for y in (0, 1):
            m = ok & (lab[1:] == y)
            np.add.at(counts[y], (st[:-1][m], st[1:][m]), 1)
    P = counts / counts.sum(axis=2, keepdims=True)
    return np.log(P[1]) - np.log(P[0])


def markov_score(rp_s, llr: np.ndarray, w: int = 32):
    st = markov_states(rp_s)
    step = np.concatenate([[0.0], llr[st[:-1], st[1:]]])
    c = np.cumsum(step)
    out = c - np.concatenate([np.zeros(w), c[:-w]]) if len(c) > w else c.copy()
    return out


LAGS = {"cv": 16, "cosen": 8, "markov": 16}


def detect_beats(score: np.ndarray, thr: float, k: int, lag: int, rp_s: np.ndarray, min_beats: int = 5):
    """Causal majority over k beats of (score >= thr); boundaries shifted back by lag.
    The onset alarm time is the beat at which the causal state changed."""
    n = len(score)
    d = (score >= thr).astype(np.int32)
    c = np.cumsum(np.concatenate([[0], d]))
    idx = np.arange(n)
    lo = np.maximum(0, idx - k + 1)
    state = ((c[idx + 1] - c[lo]) * 2 > (idx - lo + 1)).astype(np.int8)
    shift = lag + k // 2
    st = np.concatenate([state[shift:], np.full(min(shift, n), state[-1] if n else 0)])[:n]
    x = np.concatenate([[0], st, [0]])
    dd = np.diff(x)
    starts, ends = np.nonzero(dd == 1)[0], np.nonzero(dd == -1)[0]
    eps = []
    for s, e in zip(starts, ends):
        if e - s >= min_beats:
            eps.append(dict(start=int(s), end=int(e), lc=bool(s == 0), rc=bool(e == n),
                            onset_alarm_s=float(rp_s[min(n - 1, s + shift)]), spread_s=0.0,
                            fallback_offset=False))
    return eps
