"""Online episode delineators operating on frozen block scores (Phase III).

All detectors share the record context (accepted blocks, release times,
beat-level probabilities), the same initial-mode pilot and the same
beat-level retrospective localisation, so that only the decision rule differs.
Each returns a list of Boundary records; ``assemble`` pairs them into
half-open episodes [first AF beat, first non-AF beat).
"""
from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Boundary:
    direction: str      # "onset" or "offset"
    beat: int           # estimated boundary beat (onset: first AF beat; offset: first non-AF beat)
    alarm_s: float      # time at which the alarm was issued (block release time)
    spread_s: float     # heuristic localisation spread (seconds)
    kind: str = "alarm"  # "alarm", "fallback"


@dataclass
class RecordContext:
    rp_s: np.ndarray        # [N] beat times (s)
    block_start: np.ndarray # [J] first beat of each block
    available: np.ndarray   # [J] release time (s)
    logit: np.ndarray       # [J] fused block logit
    pb: np.ndarray          # [J,b] beat probabilities
    b: int
    duration_s: float
    reset_gap_s: float = 30.0

    @property
    def J(self) -> int:
        return len(self.block_start)

    def segments(self) -> np.ndarray:
        """Segment id per block; a new segment after > reset_gap_s unobserved."""
        if self.J == 0:
            return np.zeros(0, int)
        t0 = self.rp_s[self.block_start]
        t1 = self.rp_s[self.block_start + self.b]
        gap = np.concatenate([[0.0], t0[1:] - t1[:-1]])
        return np.cumsum(gap > self.reset_gap_s)


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ---------------------------------------------------------------- e-values

def log_mixture_calibrator(p: float) -> float:
    """log f(p), f(p)=int_0^1 k p^(k-1) dk=(1-p+p ln p)/(p ln^2 p), f(1)=1/2."""
    p = min(max(p, 1e-15), 1.0)
    t = -math.log(p)
    if t < 1e-4:
        return math.log(0.5 + t / 6 + t * t / 24)
    if t <= 1:
        return math.log(math.expm1(t) - t) - 2 * math.log(t)
    return t + math.log1p(-(1 + t) * math.exp(-t)) - 2 * math.log(t)


def log_power_calibrator(p: float, kappa: float) -> float:
    p = min(max(p, 1e-15), 1.0)
    return math.log(kappa) + (kappa - 1) * math.log(p)


def logaddexp(a, b):
    if a == -math.inf:
        return b
    if b == -math.inf:
        return a
    hi, lo = (a, b) if a > b else (b, a)
    return hi + math.log1p(math.exp(lo - hi))


# ---------------------------------------------------------------- localisation

def localize(ctx: RecordContext, active: list[int], cand_pos: range, mode: int) -> tuple[int, float]:
    """Beat-level CUSUM refinement (Eq. 13) over candidate blocks active[cand_pos]
    using beat probabilities of active blocks up to the alarm block only."""
    b = ctx.b
    probs = ctx.pb[active].reshape(-1).astype(np.float64)
    sign = 1.0 - 2.0 * mode
    contrib = (2.0 * probs - 1.0) * sign
    suffix = np.cumsum(contrib[::-1])[::-1]               # S(n) for each beat position
    lo = max(0, cand_pos.start) * b
    hi = min(len(active), cand_pos.stop) * b
    seg = suffix[lo:hi]
    k = int(np.argmax(seg))                               # earliest maximiser
    pos = lo + k
    beat = int(ctx.block_start[active[pos // b]] + pos % b)
    near = np.nonzero(seg >= seg[k] - 1.0)[0] + lo
    beats = ctx.block_start[np.array(active)[near // b]] + near % b
    times = ctx.rp_s[beats]
    rr = np.diff(ctx.rp_s[ctx.block_start[active[0]]:ctx.block_start[active[-1]] + b])
    spread = float(times.max() - times.min() + (np.median(rr) if len(rr) else 0.0))
    return beat, spread


def pilot_mode(ctx: RecordContext, w0: int) -> tuple[int, int]:
    """Initial mode from the first w0 blocks (discarded afterwards)."""
    n = min(w0, ctx.J)
    if n == 0:
        return 0, 0
    return int(_sigmoid(ctx.logit[:n]).mean() >= 0.5), n


# ---------------------------------------------------------------- CERA-CPD

def cera(ctx: RecordContext, A: float = 1121.0, W: int = 4, W0: int = 4, beta: float = 1.0,
         calibrator: str = "mixture", kappa: float = 0.3, seed: int = 0,
         fallback: bool = False, restart_at_change: bool = False,
         beat_refine: bool = True, clip: bool = True, trace: list | None = None,
         seed_bag: dict | None = None) -> tuple[list[Boundary], int]:
    """CERA-CPD. With clip=True the nonconformity score is decision-clipped,
    alpha = max(pi, 1/2) in onset mode and max(1-pi, 1/2) in offset mode (in
    logit space: max(+-logit, 0)). It remains a fixed function of the block,
    so randomised conformal p-values stay exactly uniform under
    exchangeability, but blocks on the current-state side of the decision
    boundary are tied and cannot accumulate evidence."""
    rng = np.random.default_rng(seed)
    mode, start = pilot_mode(ctx, W0)
    m0 = mode
    seg = ctx.segments()
    logA = math.log(A)
    out: list[Boundary] = []
    # Optional population seed: sorted clipped scores of held-out patients in each
    # state. Online conformal p-values against seed + own history are exactly
    # valid when the wearer's current-state blocks are exchangeable with the seed.
    bag: list[float] = list(seed_bag[mode]) if seed_bag else []
    active: list[int] = []
    loge: list[float] = []
    log_sr = -math.inf

    def reset():
        nonlocal bag, active, loge, log_sr
        bag = list(seed_bag[mode]) if seed_bag else []
        active, loge, log_sr = [], [], -math.inf

    for j in range(start, ctx.J):
        if j > start and seg[j] != seg[j - 1]:
            reset()
        score = float(ctx.logit[j]) if mode == 0 else -float(ctx.logit[j])
        if clip:
            score = max(score, 0.0)
        bisect.insort(bag, score)
        n = len(bag)
        greater = n - bisect.bisect_right(bag, score)
        equal = bisect.bisect_right(bag, score) - bisect.bisect_left(bag, score)
        p = (greater + rng.random() * equal) / n
        active.append(j)
        if calibrator == "mixture":
            lf = log_mixture_calibrator(p)
        else:
            lf = log_power_calibrator(p, kappa)
        le = lf if beta == 1.0 else logaddexp(math.log1p(-beta), math.log(beta) + lf)
        if len(active) > W:
            log_sr = logaddexp(0.0, log_sr) + le
            loge.append(le)
        if trace is not None:
            trace.append((j, mode, p, log_sr))
        if fallback and mode == 1 and len(active) == W and W > 0 and \
                _sigmoid(ctx.logit[active]).mean() < 0.5:
            beat, spread = localize(ctx, active, range(0, len(active)), mode)
            out.append(Boundary("offset", beat, float(ctx.available[j]), spread, "fallback"))
            mode = 1 - mode
            reset()
            continue
        if len(active) > W and log_sr >= logA:
            # earliest maximiser of prod_{i>=k} E_i over active (post-warm-up) blocks
            suffix, best, arg = 0.0, -math.inf, len(loge) - 1
            for k in range(len(loge) - 1, -1, -1):
                suffix += loge[k]
                if suffix >= best:
                    best, arg = suffix, k
            khat = W + arg
            if beat_refine:
                beat, spread = localize(ctx, active, range(khat - 1, khat + 2), mode)
            else:
                beat = int(ctx.block_start[active[khat]])
                spread = float(ctx.rp_s[ctx.block_start[active[khat]] + ctx.b] - ctx.rp_s[beat])
            out.append(Boundary("onset" if mode == 0 else "offset", beat, float(ctx.available[j]), spread))
            mode = 1 - mode
            if restart_at_change:
                # bias stress test: reuse alarm-selected blocks from khat as the new reference
                keep = active[khat:]
                reset()
                for jj in keep:
                    s2 = float(ctx.logit[jj]) if mode == 0 else -float(ctx.logit[jj])
                    bisect.insort(bag, max(s2, 0.0) if clip else s2)
                    active.append(jj)
            else:
                reset()
    return out, m0


# ---------------------------------------------------------------- conformal gate

def gated(ctx: RecordContext, drift: float = 2.0, h: float = 10.0, A_c: float = 10.0, W: int = 4,
          W0: int = 4, seed: int = 0, clip: bool = True) -> tuple[list[Boundary], int]:
    """CUSUM proposes, the conformal e-detector certifies.

    Both statistics run on the same blocks in the same mode. An alarm is issued
    only when the signed-logit CUSUM S_j >= h AND the conformal SR statistic
    R_j >= A_c (decision-clipped scores, mixture calibrator, warm-up W). The
    first gated alarm can never precede the first conformal crossing of A_c,
    so under the fixed-mode exchangeable null its expected run length is at
    least that of the conformal detector, i.e. >= A_c active blocks. The
    boundary is localised from the CUSUM change estimate (last zero of S).
    """
    rng = np.random.default_rng(seed)
    mode, start = pilot_mode(ctx, W0)
    m0 = mode
    seg = ctx.segments()
    logA = math.log(A_c)
    out: list[Boundary] = []
    bag: list[float] = []
    active: list[int] = []
    log_sr, S, last_zero = -math.inf, 0.0, 0

    def reset():
        nonlocal bag, active, log_sr, S, last_zero
        bag, active, log_sr, S, last_zero = [], [], -math.inf, 0.0, 0

    for j in range(start, ctx.J):
        if j > start and seg[j] != seg[j - 1]:
            reset()
        l = float(ctx.logit[j])
        score = l if mode == 0 else -l
        if clip:
            score = max(score, 0.0)
        bisect.insort(bag, score)
        n = len(bag)
        hi = bisect.bisect_right(bag, score)
        p = (n - hi + rng.random() * (hi - bisect.bisect_left(bag, score))) / n
        active.append(j)
        if len(active) > W:
            log_sr = logaddexp(0.0, log_sr) + log_mixture_calibrator(p)
        S = max(0.0, S + (1 - 2 * mode) * l - drift)
        if S == 0.0:
            last_zero = len(active)
        if S >= h and len(active) > W and log_sr >= logA:
            lo = max(0, last_zero - 1)
            beat, spread = localize(ctx, active, range(lo, min(len(active), last_zero + 2)), mode)
            out.append(Boundary("onset" if mode == 0 else "offset", beat, float(ctx.available[j]), spread))
            mode = 1 - mode
            reset()
    return out, m0


# ---------------------------------------------------------------- generic conformal gate
#
# Secondary analysis (revision): any causal proposer can be certified. The
# proposer and the conformal statistic share the pilot, mode, active blocks and
# resets; an alarm is issued only at a block where the proposer proposes and the
# conformal SR statistic satisfies R_j >= A_c. As for the CUSUM gate, the first
# gated alarm cannot precede the first conformal crossing of A_c.

class SmoothingProposer:
    """Causal moving average with hysteresis (k=1: percentile thresholds). The
    localisation window is the one smoothing() would use at the proposal block."""

    def __init__(self, ctx: RecordContext, k: int, t_on: float, t_off: float):
        self.k, self.t_on, self.t_off = k, t_on, t_off
        self.p = _sigmoid(ctx.logit.astype(np.float64))
        self.start = None

    def reset(self):
        self.start = None

    def alarm(self, beat: int):
        pass

    def step(self, active: list[int], mode: int):
        s = self.p[active[-self.k:]].mean()
        if not ((mode == 0 and s >= self.t_on) or (mode == 1 and s <= self.t_off)):
            self.start = None
            return None
        if self.start is None:
            self.start = len(active) - 1
        return range(max(0, self.start - self.k), self.start + 1)


class BocpdProposer:
    """bocpd() as a proposer: a detected change is held while the recent blocks
    still indicate the other rhythm, with the window bocpd() would localise in."""

    def __init__(self, ctx: RecordContext, hazard: float, threshold: float, rmax: int = 300, confirm: int = 3):
        from scipy.special import gammaln
        self.gammaln = gammaln
        self.x_all = np.clip(ctx.logit.astype(np.float64), -8, 8)
        self.hazard, self.threshold, self.rmax, self.confirm = hazard, threshold, rmax, confirm
        self.lh, self.l1h = math.log(hazard), math.log1p(-hazard)
        self.reset()

    def reset(self):
        self.logR, self.mu = np.array([0.0]), np.array([0.0])
        self.kap, self.alp, self.bet = np.array([0.1]), np.array([1.0]), np.array([1.0])
        self.latched = None

    def alarm(self, beat: int):
        pass

    def step(self, active: list[int], mode: int):
        x = self.x_all[active[-1]]
        kap, alp, bet, mu = self.kap, self.alp, self.bet, self.mu
        scale2 = bet * (kap + 1) / (alp * kap)
        nu = 2 * alp
        gl = self.gammaln
        lpred = (gl((nu + 1) / 2) - gl(nu / 2) - 0.5 * np.log(nu * math.pi * scale2)
                 - (nu + 1) / 2 * np.log1p((x - mu) ** 2 / (nu * scale2)))
        growth = self.logR + lpred + self.l1h
        cp = np.logaddexp.reduce(self.logR + lpred + self.lh)
        logR = np.concatenate([[cp], growth])
        logR -= np.logaddexp.reduce(logR)
        self.mu = np.concatenate([[0.0], (kap * mu + x) / (kap + 1)])
        self.kap = np.concatenate([[0.1], kap + 1])
        self.alp = np.concatenate([[1.0], alp + 0.5])
        self.bet = np.concatenate([[1.0], bet + kap * (x - mu) ** 2 / (2 * (kap + 1))])
        if len(logR) > self.rmax:
            logR = logR[:self.rmax]
            self.mu, self.kap = self.mu[:self.rmax], self.kap[:self.rmax]
            self.alp, self.bet = self.alp[:self.rmax], self.bet[:self.rmax]
            logR -= np.logaddexp.reduce(logR)
        self.logR = logR
        c = self.confirm
        if len(active) <= c + 1:
            return None
        recent = _sigmoid(self.x_all[active[-c:]]).mean()
        if not ((mode == 0 and recent >= 0.5) or (mode == 1 and recent < 0.5)):
            self.latched = None
            return None
        if self.latched is None and float(np.exp(np.logaddexp.reduce(logR[:c + 1]))) >= self.threshold:
            self.latched = range(max(0, len(active) - c - 2), len(active))
        return self.latched


class BeatStateProposer:
    """A beat-level causal detector with its own model (e.g. the RR-LSTM): it
    proposes when its causal majority state at the newest block's last beat is
    the other rhythm; the boundary is the start of that state run shifted back by
    the window lag, as in rr_baselines.detect_beats."""

    def __init__(self, ctx: RecordContext, beat_prob: np.ndarray, thr: float, k: int, lag: int = 0):
        n = len(beat_prob)
        d = (np.asarray(beat_prob) >= thr).astype(np.int32)
        c = np.cumsum(np.concatenate([[0], d]))
        idx = np.arange(n)
        lo = np.maximum(0, idx - k + 1)
        self.state = ((c[idx + 1] - c[lo]) * 2 > (idx - lo + 1)).astype(np.int8)
        change = np.concatenate([[True], self.state[1:] != self.state[:-1]]) if n else np.zeros(0, bool)
        self.run_start = np.maximum.accumulate(np.where(change, idx, 0)) if n else idx
        self.shift = lag + k // 2
        self.block_start, self.b = ctx.block_start, ctx.b
        self.floor = 0

    def reset(self):
        pass

    def alarm(self, beat: int):
        self.floor = beat + 1

    def step(self, active: list[int], mode: int):
        last = int(self.block_start[active[-1]]) + self.b - 1
        if self.state[last] != 1 - mode:
            return None
        return ("beat", max(int(self.run_start[last]) - self.shift, self.floor))


def gated_generic(ctx: RecordContext, proposer, A_c: float = 10.0, W: int = 4, W0: int = 4,
                  seed: int = 0, clip: bool = True) -> tuple[list[Boundary], int]:
    """Conformal certification of an arbitrary causal proposer (see above)."""
    rng = np.random.default_rng(seed)
    mode, start = pilot_mode(ctx, W0)
    m0 = mode
    seg = ctx.segments()
    logA = math.log(A_c)
    out: list[Boundary] = []
    bag: list[float] = []
    active: list[int] = []
    log_sr = -math.inf
    proposer.reset()
    for j in range(start, ctx.J):
        if j > start and seg[j] != seg[j - 1]:
            bag, active, log_sr = [], [], -math.inf
            proposer.reset()
        l = float(ctx.logit[j])
        score = l if mode == 0 else -l
        if clip:
            score = max(score, 0.0)
        bisect.insort(bag, score)
        n = len(bag)
        hi = bisect.bisect_right(bag, score)
        p = (n - hi + rng.random() * (hi - bisect.bisect_left(bag, score))) / n
        active.append(j)
        if len(active) > W:
            log_sr = logaddexp(0.0, log_sr) + log_mixture_calibrator(p)
        prop = proposer.step(active, mode)
        if prop is not None and len(active) > W and log_sr >= logA:
            if isinstance(prop, range):
                beat, spread = localize(ctx, active, prop, mode)
            else:
                beat, spread = int(prop[1]), 0.0
            out.append(Boundary("onset" if mode == 0 else "offset", beat, float(ctx.available[j]), spread))
            proposer.alarm(beat)
            mode = 1 - mode
            bag, active, log_sr = [], [], -math.inf
            proposer.reset()
    return out, m0


# ---------------------------------------------------------------- wearer memory
#
# Extension (revision): the reference bag of each mode persists for the wearer.
# A mode's bag holds every block the detector has attributed to that mode, across
# episodes and (in CPSC2021) across a wearer's recordings, instead of only the
# blocks since the last alarm. At an alarm the blocks after the estimated change
# leave the old mode's bag and join the new mode's bag; observation gaps reset
# the evidence but keep the memory. Scores are margin-clipped, max(+-l - margin, 0),
# a fixed function of the block (margin 0 is the decision clipping of cera()).
# With reset=True the bags are emptied at every alarm and gap (published policy).

class WearerMemory:
    """Sorted clipped reference scores of one wearer, per mode (0 non-AF, 1 AF)."""

    def __init__(self):
        self.bags = {0: [], 1: []}

    def clear(self):
        self.bags = {0: [], 1: []}


def log_dyadic_calibrator(p: float, n: int) -> float:
    """Resolution-matched calibrator, log g_n(p) with
    g_n(p) = (K+1)^-1 sum_{k=0..K} 2^k 1{p <= 2^-k}, K = ceil(log2 n).
    It integrates to one for every n, and the bag size n is fixed before the
    randomised p-value is drawn, so g_n(p_j) is an exact e-value."""
    K = max(0, math.ceil(math.log2(max(n, 1))))
    k = K if p <= 0 else min(K, int(math.floor(-math.log2(p) + 1e-12)))
    return math.log(2.0 ** (k + 1) - 1.0) - math.log(K + 1.0)


def _memory_step(memory: WearerMemory, mode: int, logit: float, margin: float, rng):
    """Insert the block's clipped score into the mode's bag; returns the score, its
    randomised conformal p-value and the bag size."""
    bag = memory.bags[mode]
    s = float(logit) if mode == 0 else -float(logit)
    s = max(s - margin, 0.0)
    bisect.insort(bag, s)
    n = len(bag)
    hi = bisect.bisect_right(bag, s)
    return s, (n - hi + rng.random() * (hi - bisect.bisect_left(bag, s))) / n, n


def _memory_switch(ctx: RecordContext, memory: WearerMemory, mode: int, active: list[int],
                   scores: list[float], k: int, margin: float, reset: bool):
    """After an alarm in `mode`: blocks active[k:] leave the old bag and join the new one."""
    if reset:
        memory.clear()
        return
    bag = memory.bags[mode]
    for v in scores[k:]:
        i = bisect.bisect_left(bag, v)
        if i < len(bag) and bag[i] == v:
            bag.pop(i)
    new = memory.bags[1 - mode]
    for jj in active[k:]:
        s = float(ctx.logit[jj]) if mode == 1 else -float(ctx.logit[jj])
        bisect.insort(new, max(s - margin, 0.0))


def _calibrator(name: str):
    if name == "dyadic":
        return log_dyadic_calibrator
    return lambda p, n: log_mixture_calibrator(p)


def cera_memory(ctx: RecordContext, memory: WearerMemory, A: float = 11246.0, margin: float = 0.0,
                W: int = 4, W0: int = 4, seed: int = 0, calibrator: str = "mixture",
                reset: bool = False) -> tuple[list[Boundary], int]:
    """CERA-CPD-M: conformal SR e-detector with wearer memory and margin clipping.
    No evidence is accumulated while the current mode's bag holds at most W blocks."""
    rng = np.random.default_rng(seed)
    if reset:
        memory.clear()
    cal = _calibrator(calibrator)
    mode, start = pilot_mode(ctx, W0)
    m0 = mode
    seg = ctx.segments()
    logA = math.log(A)
    out: list[Boundary] = []
    active, scores, loge, log_sr = [], [], [], -math.inf
    for j in range(start, ctx.J):
        if j > start and seg[j] != seg[j - 1]:
            if reset:
                memory.clear()
            active, scores, loge, log_sr = [], [], [], -math.inf
        s, p, n = _memory_step(memory, mode, ctx.logit[j], margin, rng)
        active.append(j)
        scores.append(s)
        if n > W:
            le = cal(p, n)
            log_sr = logaddexp(0.0, log_sr) + le
            loge.append(le)
        else:
            loge.append(None)
        if log_sr >= logA:
            # earliest maximiser of prod_{i>=k} E_i over the evidence blocks of this run
            suffix, best, khat = 0.0, -math.inf, len(loge) - 1
            for k in range(len(loge) - 1, -1, -1):
                if loge[k] is None:
                    break
                suffix += loge[k]
                if suffix >= best:
                    best, khat = suffix, k
            beat, spread = localize(ctx, active, range(khat - 1, khat + 2), mode)
            out.append(Boundary("onset" if mode == 0 else "offset", beat, float(ctx.available[j]), spread))
            _memory_switch(ctx, memory, mode, active, scores, khat, margin, reset)
            mode = 1 - mode
            active, scores, loge, log_sr = [], [], [], -math.inf
    return out, m0


def gated_memory(ctx: RecordContext, memory: WearerMemory, drift: float = 2.0, h: float = 10.0,
                 A_c: float = 10.0, W: int = 4, W0: int = 4, seed: int = 0, margin: float = 0.0,
                 calibrator: str = "mixture", reset: bool = False) -> tuple[list[Boundary], int]:
    """CERA-CPD-G with wearer memory: CUSUM proposals certified by the memory
    e-detector; the boundary and the blocks moved between bags start at the CUSUM
    change estimate (last zero of S)."""
    rng = np.random.default_rng(seed)
    if reset:
        memory.clear()
    cal = _calibrator(calibrator)
    mode, start = pilot_mode(ctx, W0)
    m0 = mode
    seg = ctx.segments()
    logA = math.log(A_c)
    out: list[Boundary] = []
    active, scores, log_sr, S, last_zero = [], [], -math.inf, 0.0, 0
    for j in range(start, ctx.J):
        if j > start and seg[j] != seg[j - 1]:
            if reset:
                memory.clear()
            active, scores, log_sr, S, last_zero = [], [], -math.inf, 0.0, 0
        l = float(ctx.logit[j])
        s, p, n = _memory_step(memory, mode, l, margin, rng)
        active.append(j)
        scores.append(s)
        if n > W:
            log_sr = logaddexp(0.0, log_sr) + cal(p, n)
        S = max(0.0, S + (1 - 2 * mode) * l - drift)
        if S == 0.0:
            last_zero = len(active)
        if S >= h and log_sr >= logA:
            lo = max(0, last_zero - 1)
            beat, spread = localize(ctx, active, range(lo, min(len(active), last_zero + 2)), mode)
            out.append(Boundary("onset" if mode == 0 else "offset", beat, float(ctx.available[j]), spread))
            _memory_switch(ctx, memory, mode, active, scores, min(last_zero, len(active) - 1), margin, reset)
            mode = 1 - mode
            active, scores, log_sr, S, last_zero = [], [], -math.inf, 0.0, 0
    return out, m0


# ---------------------------------------------------------------- comparators

def smoothing(ctx: RecordContext, k: int = 3, t_on: float = 0.7, t_off: float = 0.3,
              W0: int = 4) -> tuple[list[Boundary], int]:
    """Causal moving average of block probabilities with hysteresis."""
    mode, start = pilot_mode(ctx, W0)
    m0 = mode
    seg = ctx.segments()
    p = _sigmoid(ctx.logit.astype(np.float64))
    out, active = [], []
    for j in range(start, ctx.J):
        if j > start and seg[j] != seg[j - 1]:
            active = []
        active.append(j)
        s = p[active[-k:]].mean()
        if (mode == 0 and s >= t_on) or (mode == 1 and s <= t_off):
            lo = max(0, len(active) - k - 1)
            beat, spread = localize(ctx, active, range(lo, len(active)), mode)
            out.append(Boundary("onset" if mode == 0 else "offset", beat, float(ctx.available[j]), spread))
            mode = 1 - mode
            active = []
    return out, m0


def percentile(ctx: RecordContext, t_on: float, t_off: float, W0: int = 4):
    """Single-block population-percentile thresholds (uncalibrated variant)."""
    return smoothing(ctx, 1, t_on, t_off, W0)


def cusum(ctx: RecordContext, drift: float = 1.0, h: float = 10.0, W0: int = 4):
    """Page CUSUM on mode-signed block logits; change at last zero of S."""
    mode, start = pilot_mode(ctx, W0)
    m0 = mode
    seg = ctx.segments()
    out, active = [], []
    S, last_zero = 0.0, 0
    for j in range(start, ctx.J):
        if j > start and seg[j] != seg[j - 1]:
            active, S, last_zero = [], 0.0, 0
        active.append(j)
        z = (1 - 2 * mode) * float(ctx.logit[j]) - drift
        S = max(0.0, S + z)
        if S == 0.0:
            last_zero = len(active)
        if S >= h:
            beat, spread = localize(ctx, active, range(max(0, last_zero - 1), min(len(active), last_zero + 2)), mode)
            out.append(Boundary("onset" if mode == 0 else "offset", beat, float(ctx.available[j]), spread))
            mode = 1 - mode
            active, S, last_zero = [], 0.0, 0
    return out, m0


def bocpd(ctx: RecordContext, hazard: float = 1 / 250, threshold: float = 0.5, rmax: int = 300,
          W0: int = 4, confirm: int = 3):
    """Bayesian online CPD (Adams & MacKay) on clipped block logits with a
    Normal-Gamma model. A change is declared when the posterior mass of run
    lengths < confirm exceeds threshold; a boundary is emitted only if the
    mean probability of the post-change blocks indicates the other rhythm."""
    mode, start = pilot_mode(ctx, W0)
    m0 = mode
    seg = ctx.segments()
    x_all = np.clip(ctx.logit.astype(np.float64), -8, 8)
    mu0, k0, a0, b0 = 0.0, 0.1, 1.0, 1.0
    out, active = [], []

    def fresh():
        return (np.array([0.0]), np.array([mu0]), np.array([k0]), np.array([a0]), np.array([b0]))

    logR, mu, kap, alp, bet = fresh()
    lh, l1h = math.log(hazard), math.log1p(-hazard)
    from scipy.special import gammaln
    for j in range(start, ctx.J):
        if j > start and seg[j] != seg[j - 1]:
            active = []
            logR, mu, kap, alp, bet = fresh()
        active.append(j)
        x = x_all[j]
        scale2 = bet * (kap + 1) / (alp * kap)
        nu = 2 * alp
        lpred = (gammaln((nu + 1) / 2) - gammaln(nu / 2) - 0.5 * np.log(nu * math.pi * scale2)
                 - (nu + 1) / 2 * np.log1p((x - mu) ** 2 / (nu * scale2)))
        growth = logR + lpred + l1h
        cp = np.logaddexp.reduce(logR + lpred + lh)
        logR = np.concatenate([[cp], growth])
        logR -= np.logaddexp.reduce(logR)
        mu_n = (kap * mu + x) / (kap + 1)
        kap_n, alp_n = kap + 1, alp + 0.5
        bet_n = bet + kap * (x - mu) ** 2 / (2 * (kap + 1))
        mu = np.concatenate([[mu0], mu_n])
        kap = np.concatenate([[k0], kap_n])
        alp = np.concatenate([[a0], alp_n])
        bet = np.concatenate([[b0], bet_n])
        if len(logR) > rmax:
            logR, mu, kap, alp, bet = logR[:rmax], mu[:rmax], kap[:rmax], alp[:rmax], bet[:rmax]
            logR -= np.logaddexp.reduce(logR)
        if len(active) <= confirm + 1:
            continue
        short = float(np.exp(np.logaddexp.reduce(logR[:confirm + 1])))
        recent = _sigmoid(x_all[active[-confirm:]]).mean()
        flips = (mode == 0 and recent >= 0.5) or (mode == 1 and recent < 0.5)
        if short >= threshold and flips:
            lo = max(0, len(active) - confirm - 2)
            beat, spread = localize(ctx, active, range(lo, len(active)), mode)
            out.append(Boundary("onset" if mode == 0 else "offset", beat, float(ctx.available[j]), spread))
            mode = 1 - mode
            active = []
            logR, mu, kap, alp, bet = fresh()
    return out, m0


# ---------------------------------------------------------------- episodes

def assemble(boundaries: list[Boundary], m0: int, n_beats: int, min_beats: int = 5):
    """Pair boundaries into half-open episodes (start_beat, end_beat_excl,
    left_censored, right_censored, onset_alarm_s, spreads)."""
    eps = []
    open_ep = (0, True, None, 0.0) if m0 == 1 else None
    for bd in boundaries:
        if bd.direction == "onset":
            open_ep = (bd.beat, False, bd.alarm_s, bd.spread_s)
        else:
            if open_ep is not None:
                s, lc, al, sp = open_ep
                if bd.beat - s >= min_beats:
                    eps.append(dict(start=s, end=bd.beat, lc=lc, rc=False, onset_alarm_s=al,
                                    spread_s=sp + bd.spread_s, fallback_offset=bd.kind == "fallback"))
            open_ep = None
    if open_ep is not None:
        s, lc, al, sp = open_ep
        if n_beats - s >= min_beats:
            eps.append(dict(start=s, end=n_beats, lc=lc, rc=True, onset_alarm_s=al, spread_s=sp,
                            fallback_offset=False))
    return eps
