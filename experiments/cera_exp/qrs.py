"""Causal two-lead Pan--Tompkins-type QRS detector for the end-to-end track.

The detector follows Pan and Tompkins (1985): 5--15 Hz band-pass, five-point
derivative, squaring, 150-ms moving-window integration (MWI), adaptive signal
and noise peak levels, a 200-ms refractory period, T-wave discrimination within
360 ms and search-back after 166% of the regular RR average. The squared
derivatives of the two leads are summed before integration. The band-pass starts
from the steady state of the first sample, and the peak levels are re-learned
from the preceding 2 s whenever no beat has been detected for 3 s, so that an
offset or artefact cannot block detection permanently. Detections whose
0.5--40 Hz peak-to-peak amplitude stays below 0.1 mV on both leads in the QRS
search window are discarded (lead-off safeguard). Every operation is
causal; each detection carries the sample at which it became available:

* an MWI peak is confirmed 100 ms after its maximum (no larger value nearby);
* a search-back beat is confirmed when the missed-beat limit expires;
* peaks inside the 2-s learning period are confirmed at its end.

The R location is the earliest local maximum of the absolute causally
band-passed (0.5--40 Hz) signal, summed over leads, that reaches 70% of its
largest value in the 250 ms preceding the MWI peak, corrected for the integer
group delay of that filter. Only samples available at the confirmation time are
used.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter1d
from scipy.signal import butter, group_delay, sosfilt, sosfilt_zi

from .features import bandpass


def _gd_samples(fs: float) -> int:
    """Integer group delay (samples) of the 0.5--40 Hz feature filter at 10 Hz."""
    high = min(40.0, 0.45 * fs)
    b, a = butter(4, [0.5, high], btype="band", fs=fs)
    _, gd = group_delay((b, a), w=[10.0], fs=fs)
    return int(round(float(gd[0])))


def detect_qrs(sig: np.ndarray, fs: float, learn_s: float = 2.0, refractory_s: float = 0.2,
               twave_s: float = 0.36, confirm_s: float = 0.1, rel_peak: float = 0.7, relearn_s: float = 3.0,
               min_amp_mv: float = 0.1):
    """Return (r_samples, confirm_samples), both int64 and strictly increasing in r."""
    x = np.asarray(sig, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    n = len(x)
    if n < int(fs):
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    sos = butter(2, [5.0, min(15.0, 0.45 * fs)], btype="band", fs=fs, output="sos")
    # steady-state initial conditions from the first sample (causal): a DC offset
    # must not produce a start-up transient that inflates the learned thresholds
    zi = sosfilt_zi(sos)[:, :, None] * x[0][None, None, :]
    bp, _ = sosfilt(sos, x, axis=0, zi=zi)
    d = np.zeros_like(bp)
    d[4:] = (2 * bp[4:] + bp[3:-1] - bp[1:-3] - 2 * bp[:-4]) * (fs / 8.0)
    e = (d ** 2).sum(axis=1)
    w = max(1, int(round(0.15 * fs)))
    c = np.concatenate([[0.0], np.cumsum(e)])
    idx = np.arange(n)
    mwi = (c[idx + 1] - c[np.maximum(0, idx + 1 - w)]) / w
    h = max(1, int(round(confirm_s * fs)))
    mf = maximum_filter1d(mwi, size=2 * h + 1, mode="nearest")
    cand = np.nonzero((mwi >= mf) & (mwi > 0))[0]
    if len(cand):
        cand = cand[np.concatenate([[True], np.diff(cand) > 1])]
    # slope proxy for T-wave discrimination: largest squared derivative in e[i-w+1..i]
    mfc = maximum_filter1d(e, size=w, mode="nearest")      # window [i - w//2, i - w//2 + w - 1]
    ce = mfc[np.maximum(0, idx - (w - 1 - w // 2))]
    L = min(n, int(round(learn_s * fs)))
    spki = 0.25 * float(mwi[:L].max())
    npki = 0.5 * float(mwi[:L].mean())
    refr = int(round(refractory_s * fs))
    twav = int(round(twave_s * fs))
    rr_all: list[int] = []          # last 8 RR intervals (samples)
    rr_reg: list[int] = []          # last 8 RR intervals within the regular limits
    qrs_i: list[int] = []           # MWI peak samples of accepted beats
    qrs_conf: list[int] = []
    last_slope = 0.0
    pending: list[tuple[int, float]] = []   # noise peaks since the last beat (search-back pool)

    def irregular():
        if len(rr_all) < 8 or not rr_reg:
            return False
        m = float(np.mean(rr_reg))
        return any(not (0.92 * m <= r <= 1.16 * m) for r in rr_all)

    def thresholds():
        thr1 = npki + 0.25 * (spki - npki)
        if irregular():
            thr1 *= 0.5
        return thr1, 0.5 * thr1

    def missed_limit():
        if rr_reg and not irregular():
            return 1.66 * float(np.mean(rr_reg))
        if rr_all:
            return 1.66 * float(np.mean(rr_all))
        return 1.66 * fs

    def accept(i: int, height: float, conf: int, searchback: bool):
        nonlocal spki, last_slope
        if qrs_i:
            rr = i - qrs_i[-1]
            rr_all.append(rr)
            del rr_all[:-8]
            if not rr_reg or 0.92 * np.mean(rr_reg) <= rr <= 1.16 * np.mean(rr_reg) or len(rr_reg) < 2:
                rr_reg.append(rr)
                del rr_reg[:-8]
        spki = (0.25 * height + 0.75 * spki) if searchback else (0.125 * height + 0.875 * spki)
        qrs_i.append(i)
        qrs_conf.append(conf)
        last_slope = float(ce[i])
        pending.clear()

    relearn = int(round(relearn_s * fs))
    anchor = L                     # last (re)initialisation of the peak levels
    sb_failed = None               # beat after which search-back already found nothing
    for i in cand.tolist():
        i = int(i)
        height = float(mwi[i])
        conf = max(i + h, L)
        # re-learning: no beat for relearn_s since the last beat or (re)initialisation;
        # the levels are re-estimated from the preceding learn_s of the MWI (causal)
        if i - max(qrs_i[-1] if qrs_i else -1, anchor) > relearn:
            lo = max(0, i - L)
            spki = 0.25 * float(mwi[lo:i + 1].max())
            npki = 0.5 * float(mwi[lo:i + 1].mean())
            anchor, sb_failed = i, None
        # search-back: an expired missed-beat limit before this peak was confirmed
        while qrs_i and sb_failed != qrs_i[-1]:
            limit = qrs_i[-1] + int(missed_limit())
            if conf <= limit:
                break
            _, thr2 = thresholds()
            pool = [(k, v) for k, v in pending if qrs_i[-1] + refr <= k and max(k + h, L) <= limit and v > thr2]
            if not pool:
                sb_failed = qrs_i[-1]
                break
            k, v = max(pool, key=lambda kv: kv[1])
            accept(k, v, max(limit, max(k + h, L)), True)
        if qrs_i and i - qrs_i[-1] < refr:
            continue
        thr1, _ = thresholds()
        if height > thr1:
            if qrs_i and i - qrs_i[-1] < twav and float(ce[i]) < 0.5 * last_slope:
                npki = 0.125 * height + 0.875 * npki
                pending.append((i, height))
                continue
            accept(i, height, conf, False)
        else:
            npki = 0.125 * height + 0.875 * npki
            pending.append((i, height))
    if not qrs_i:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    # R location on the causal feature-band signal, compensated for its group delay
    fb = bandpass(x.astype(np.float32), fs)
    f = np.abs(fb).sum(axis=1)
    gd = _gd_samples(fs)
    look = int(round(0.25 * fs))
    r_out, c_out = [], []
    for i, cf in zip(qrs_i, qrs_conf):
        lo = max(0, i - look)
        # lead-off safeguard: a QRS complex must reach min_amp_mv peak-to-peak on some lead
        if min_amp_mv > 0 and float(np.ptp(fb[lo:i + 1], axis=0).max()) < min_amp_mv:
            continue
        seg = f[lo:i + 1]
        # earliest local maximum reaching rel_peak of the window maximum (wide complexes
        # can have a larger late deflection than the initial R wave)
        top = float(seg.max())
        k = int(np.argmax(seg))
        if rel_peak < 1.0 and len(seg) > 2:
            locmax = np.nonzero((seg[1:-1] >= seg[:-2]) & (seg[1:-1] >= seg[2:]) &
                                (seg[1:-1] >= rel_peak * top))[0] + 1
            if len(locmax):
                k = int(locmax[0])
        r = lo + k - gd
        r = min(max(r, 0), n - 1)
        if r_out and r <= r_out[-1]:
            continue
        r_out.append(r)
        c_out.append(max(cf, r))
    return np.asarray(r_out, np.int64), np.asarray(c_out, np.int64)


def match_beats(ref: np.ndarray, det: np.ndarray, tol: int):
    """One-to-one matching of sorted beat samples within +/- tol samples.
    Returns (tp, fn, fp, matched_det_index_per_ref [-1 if unmatched])."""
    ref = np.asarray(ref, np.int64)
    det = np.asarray(det, np.int64)
    out = np.full(len(ref), -1, np.int64)
    i = j = 0
    while i < len(ref) and j < len(det):
        if det[j] < ref[i] - tol:
            j += 1
        elif det[j] > ref[i] + tol:
            i += 1
        else:
            # prefer the closer of det[j] and det[j+1] for ref[i]
            if j + 1 < len(det) and abs(det[j + 1] - ref[i]) < abs(det[j] - ref[i]) and \
                    (i + 1 >= len(ref) or abs(det[j + 1] - ref[i]) <= abs(det[j + 1] - ref[i + 1])):
                j += 1
            out[i] = j
            i += 1
            j += 1
    tp = int((out >= 0).sum())
    return tp, len(ref) - tp, len(det) - tp, out
