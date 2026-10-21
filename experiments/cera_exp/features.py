"""Beat-synchronous conditioning (Phase I) and block features (Phase II inputs).

All operations are vectorised per record. Signals are filtered causally at the
native rate; waveform windows are linearly interpolated onto a fixed grid and
masked to the block support [R_first, R_next).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
from scipy.signal import butter, sosfilt

from .records import ECTOPIC_SYMBOLS


@dataclass(frozen=True)
class PrepConfig:
    block: int = 8              # beats per block (b)
    grid: int = 64              # samples per waveform window on the neural grid
    q_min: float = 0.5          # abstention threshold on beat quality
    min_rr: float = 0.2         # s; shorter intervals break continuity
    max_rr: float = 3.0         # s; longer intervals break continuity
    confirm_latency: float = 0.15  # s; nominal R-peak confirmation delay
    quality_post_window: float = 0.5  # s; final sample support of flat/saturation SQI
    spur_lo: float = 0.3        # spurious/QRS energy ratio mapped to quality 1
    spur_hi: float = 0.7        # ratio mapped to quality 0 (q_min=0.5 <=> ratio 0.5)


def bandpass(sig: np.ndarray, fs: float) -> np.ndarray:
    high = min(40.0, 0.45 * fs)
    sos = butter(4, [0.5, high], btype="band", fs=fs, output="sos")
    return sosfilt(sos, sig, axis=0).astype(np.float32)


def _interp(x: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Linear interpolation of 1-D x at fractional sample positions (clipped)."""
    n = len(x)
    p = np.clip(pos, 0, n - 1)
    i0 = np.floor(p).astype(np.int64)
    i1 = np.minimum(i0 + 1, n - 1)
    w = (p - i0).astype(np.float32)
    return x[i0] * (1 - w) + x[i1] * w


def block_release_times(rpeaks_s: np.ndarray, block_start: np.ndarray, b: int,
                        confirm_latency: float = 0.15,
                        quality_post_window: float = 0.5) -> np.ndarray:
    """Release only after both next-peak confirmation and every SQI input.

    The flat/saturation window for each accepted beat extends through R+0.5 s.
    Its latest endpoint is that of the block's final beat. This conservative
    endpoint also covers the centred 100-ms slope-energy envelope. Recomputing
    these times for legacy prepared arrays changes timing, not features/models.
    """
    rp = np.asarray(rpeaks_s, dtype=np.float64)
    starts = np.asarray(block_start, dtype=np.int64)
    if b < 1 or confirm_latency < 0 or quality_post_window < 0:
        raise ValueError('Invalid block size or release latency')
    if not len(starts):
        return np.empty(0, dtype=np.float64)
    if np.any(starts < 0) or np.any(starts + b >= len(rp)):
        raise ValueError('Every complete block requires a confirmed following peak')
    return np.maximum(rp[starts + b] + confirm_latency,
                      rp[starts + b - 1] + quality_post_window)


def beat_quality(filt: np.ndarray, rpeaks: np.ndarray, fs: float, cfg: PrepConfig) -> np.ndarray:
    """Spurious-QRS-energy signal-quality index, best of two leads.

    A slope-energy envelope (squared first difference, 100-ms moving average)
    approximates QRS detector input. For beat n, rho is the largest envelope
    value inside the beat cell away from both QRS complexes,
    [R_n+0.3 s, R_{n+1}-0.15 s], divided by the envelope peak within 100 ms of
    R_n. A clean cell has rho well below 0.5 whatever the QRS width; noise or
    artefact produces QRS-like energy (cf. detector-agreement bSQI). Flat or
    saturated windows receive zero. Release additionally waits through R_n+0.5 s
    for that test; block_release_times accounts for both dependencies."""
    n = len(rpeaks)
    best = np.zeros(n, dtype=np.float32)
    if n < 2:
        return best
    w = max(1, int(0.1 * fs))
    kernel = np.ones(w, dtype=np.float32) / w
    nxt = np.append(rpeaks[1:], rpeaks[-1] + int(fs))
    lo = rpeaks + int(0.3 * fs)
    hi = nxt - int(0.15 * fs)
    offs = np.arange(-int(0.1 * fs), int(0.1 * fs) + 1)
    qrs_idx = np.clip(rpeaks[:, None] + offs[None, :], 0, len(filt) - 1)
    for lead in range(filt.shape[1]):
        x = filt[:, lead]
        env = np.convolve(np.diff(x, prepend=x[0]) ** 2, kernel, mode="same")
        peak = env[qrs_idx].max(axis=1)
        # running maximum over [lo, hi) via a sparse-table-free reduceat
        valid = hi > lo
        spur = np.zeros(n, dtype=np.float32)
        if valid.any():
            bounds = np.stack([np.clip(lo[valid], 0, len(env) - 1), np.clip(hi[valid], 0, len(env) - 1)], axis=1).ravel()
            order_ok = bounds[1::2] > bounds[0::2]
            red = np.maximum.reduceat(env, bounds)[0::2]
            spur[np.where(valid)[0][order_ok]] = red[order_ok]
        rho = spur / np.maximum(peak, 1e-12)
        seg = filt[np.clip(rpeaks[:, None] + np.arange(-int(0.3 * fs), int(cfg.quality_post_window * fs))[None, :], 0, len(filt) - 1), lead]
        ok = (seg.std(axis=1) > 5e-3) & (np.ptp(seg, axis=1) < 10.0)
        q = np.clip((cfg.spur_hi - rho) / (cfg.spur_hi - cfg.spur_lo), 0, 1) * ok
        best = np.maximum(best, q.astype(np.float32))
    return best


def make_blocks(rpeaks_s: np.ndarray, accepted: np.ndarray, cfg: PrepConfig):
    """Non-overlapping groups of b consecutive accepted beats with valid
    internal RR and a confirmed next peak. Returns block start beats and
    continuity IDs (a new ID after every rejection or gap)."""
    b = cfg.block
    n = len(rpeaks_s)
    starts, conts = [], []
    cont, pending = 0, 0
    rr_next = np.diff(rpeaks_s)
    for i in range(n - 1):
        ok = accepted[i] and cfg.min_rr <= rr_next[i] <= cfg.max_rr
        if not ok:
            if pending:
                cont += 1
            pending = 0
            continue
        pending += 1
        if pending == b:
            starts.append(i - b + 1)
            conts.append(cont)
            pending = 0
    return np.asarray(starts, dtype=np.int64), np.asarray(conts, dtype=np.int64)


def rr_descriptors(d: np.ndarray) -> np.ndarray:
    """d: [J, b-1] internal RR (s). Returns [J,5]: log median RR, RMSSD/median,
    CV, turning-point ratio, regularised sample-entropy descriptor."""
    med = np.median(d, axis=1)
    rmssd = np.sqrt(np.mean(np.diff(d, axis=1) ** 2, axis=1)) / med
    cv = d.std(axis=1) / d.mean(axis=1)
    dd = np.diff(d, axis=1)
    tpr = ((dd[:, :-1] * dd[:, 1:]) < 0).mean(axis=1)
    m = d.shape[1]
    tol = np.maximum(1e-8, 0.2 * d.std(axis=1))
    starts = range(m - 2)
    n2 = np.zeros(len(d))
    n3 = np.zeros(len(d))
    for i in starts:
        for j in starts:
            if j <= i:
                continue
            c0 = np.abs(d[:, i] - d[:, j]) <= tol
            c1 = np.abs(d[:, i + 1] - d[:, j + 1]) <= tol
            c2 = np.abs(d[:, i + 2] - d[:, j + 2]) <= tol
            n2 += c0 & c1
            n3 += c0 & c1 & c2
    sampen = -np.log((n3 + 1) / (n2 + 1))
    return np.stack([np.log(med), rmssd, cv, tpr, sampen], axis=1).astype(np.float32)


def block_features(rec: dict, cfg: PrepConfig) -> dict:
    fs = rec["fs"]
    filt = bandpass(rec["sig"], fs)
    rp = rec["rpeaks"]
    rp_s = rp / fs
    q = beat_quality(filt, rp, fs, cfg)
    accepted = q >= cfg.q_min
    starts, conts = make_blocks(rp_s, accepted, cfg)
    b, L = cfg.block, cfg.grid
    J = len(starts)
    sym = rec["symbols"]
    sym_code = np.where(sym == "N", 0, np.where(np.isin(sym, list(ECTOPIC_SYMBOLS)), 1, 2)).astype(np.uint8)
    out = dict(rpeaks=rp.astype(np.int64), labels=rec["labels"].astype(np.int8), sym_code=sym_code,
               quality=q.astype(np.float16), block_start=starts.astype(np.int32), continuity=conts.astype(np.int32))
    if J == 0:
        out.update(rr=np.zeros((0, b - 1), np.float32), desc=np.zeros((0, 5), np.float32),
                   atrial=np.zeros((0, b, 4, L), np.float16), mask=np.zeros((0, b, 4, L), bool),
                   available=np.zeros(0, np.float32), block_label=np.zeros(0, np.int8),
                   ectopy=np.zeros(0, bool), boundary=np.zeros(0, bool), qmean=np.zeros(0, np.float32))
        return out
    beat_idx = starts[:, None] + np.arange(b)[None, :]          # [J,b]
    r = rp_s[beat_idx]                                           # [J,b]
    r_next = rp_s[starts + b]                                    # [J]
    sup0, sup1 = r[:, 0], r_next
    d = np.diff(r, axis=1)                                       # [J,b-1]
    med = np.median(d, axis=1)
    rr = (d / med[:, None]).astype(np.float32)
    desc = rr_descriptors(d)
    # Previous RR for pre-QRS length; first beat uses internal median.
    prev = np.concatenate([med[:, None], d], axis=1)             # [J,b]
    pre_len = np.minimum(0.4, 0.7 * prev)
    u = np.linspace(0.0, 1.0, L, dtype=np.float32)
    t_pre = r[..., None] - pre_len[..., None] + u * (pre_len[..., None] - 0.06)   # [J,b,L]
    t_al = r[..., None] + (-0.25 + 0.70 * u)                                       # [J,b,L]
    m_pre = (t_pre >= sup0[:, None, None]) & (t_pre < sup1[:, None, None])
    m_al = (t_al >= sup0[:, None, None]) & (t_al < sup1[:, None, None])
    atrial = np.zeros((J, b, 4, L), np.float32)
    for lead in range(2):
        x = filt[:, lead]
        w_pre = _interp(x, t_pre * fs)
        w_al = _interp(x, t_al * fs)
        al_nan = np.where(m_al, w_al, np.nan)
        with np.errstate(all="ignore"):
            centre = np.nanmedian(al_nan.reshape(J, -1), axis=1)
            mad = np.nanmedian(np.abs(al_nan.reshape(J, -1) - centre[:, None]), axis=1)
            template = np.nanmedian(al_nan, axis=1)               # [J,L]
        centre = np.nan_to_num(centre)
        scale = 1.4826 * np.nan_to_num(mad) + 1e-3
        template = np.nan_to_num(template)
        resid = (w_al - template[:, None, :]) / scale[:, None, None]
        pre = (w_pre - centre[:, None, None]) / scale[:, None, None]
        atrial[:, :, lead] = np.where(m_pre, pre, 0)
        atrial[:, :, 2 + lead] = np.where(m_al, resid, 0)
    atrial = np.clip(atrial, -50, 50)
    mask = np.zeros((J, b, 4, L), bool)
    mask[:, :, 0] = m_pre
    mask[:, :, 1] = m_pre
    mask[:, :, 2] = m_al
    mask[:, :, 3] = m_al
    lab = rec["labels"][beat_idx]
    known = (lab >= 0).all(axis=1)
    block_label = np.where(known, (lab.mean(axis=1) >= 0.5).astype(np.int8), -1).astype(np.int8)
    out.update(rr=rr, desc=desc, atrial=atrial.astype(np.float16), mask=mask,
               available=block_release_times(rp_s, starts, b, cfg.confirm_latency,
                                             cfg.quality_post_window),
               block_label=block_label, ectopy=(sym_code[beat_idx] == 1).any(axis=1),
               boundary=known & (lab.min(axis=1) != lab.max(axis=1)),
               qmean=q[beat_idx].mean(axis=1).astype(np.float32))
    return out


def config_dict(cfg: PrepConfig) -> dict:
    return asdict(cfg)
