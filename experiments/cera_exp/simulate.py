"""Monte Carlo checks of the false-alarm guarantees (hypothesis H2, synthetic part).

For a fixed mode, scores are i.i.d. (exchangeable), AR(1)-dependent, or drifting.
Single run: nu = active blocks to first alarm, truncated at horizon H. The proof of
Proposition 1 gives A * P(nu <= H) <= E[min(nu, H)] for every H; we report both
sides. Repeated alarms with resets: K_N versus N / (A + W) (Corollary 1).

python -m experiments.cera_exp.simulate --out sim.json --reps 1000
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .detect import log_mixture_calibrator, logaddexp


def scores(kind: str, n: int, rng) -> np.ndarray:
    if kind == "iid":
        return rng.normal(size=n)
    if kind.startswith("ar"):
        rho = float(kind[2:])
        e = rng.normal(size=n) * math.sqrt(1 - rho * rho)
        x = np.empty(n)
        x[0] = rng.normal()
        for i in range(1, n):
            x[i] = rho * x[i - 1] + e[i]
        return x
    if kind.startswith("drift"):
        slope = float(kind[5:])          # standard deviations per 1000 blocks
        return rng.normal(size=n) + slope * np.arange(n) / 1000.0
    raise ValueError(kind)


def run_until_alarm(x: np.ndarray, A: float, W: int, clip: bool, rng) -> int | None:
    bag, log_sr, logA = [], -math.inf, math.log(A)
    for j, s in enumerate(x):
        s = max(s, 0.0) if clip else s
        bisect.insort(bag, s)
        n = len(bag)
        hi = bisect.bisect_right(bag, s)
        p = (n - hi + rng.random() * (hi - bisect.bisect_left(bag, s))) / n
        if j + 1 > W:
            log_sr = logaddexp(0.0, log_sr) + log_mixture_calibrator(p)
            if log_sr >= logA:
                return j + 1 - W
    return None


def single(args):
    kind, A, W, H, reps, seed, clip = args
    rng = np.random.default_rng(seed)
    nus, alarms = [], 0
    for _ in range(reps):
        nu = run_until_alarm(scores(kind, H + W, rng), A, W, clip, rng)
        if nu is None:
            nus.append(H)
        else:
            nus.append(nu)
            alarms += 1
    nus = np.array(nus, float)
    p_alarm = alarms / reps
    return dict(kind=kind, A=A, W=W, H=H, reps=reps, clip=clip, p_alarm_by_H=p_alarm,
                restricted_mean=float(nus.mean()), restricted_mean_se=float(nus.std(ddof=1) / math.sqrt(reps)),
                lhs_A_times_p=A * p_alarm, bound_holds=bool(A * p_alarm <= nus.mean() + 2 * nus.std(ddof=1) / math.sqrt(reps)))


def repeated(args):
    kind, A, W, N, reps, seed, clip = args
    rng = np.random.default_rng(seed)
    counts = []
    for _ in range(reps):
        x = scores(kind, N, rng)
        k, pos = 0, 0
        while pos < N:
            nu = run_until_alarm(x[pos:], A, W, clip, rng)
            if nu is None:
                break
            k += 1
            pos += nu + W
        counts.append(k)
    c = np.array(counts, float)
    return dict(kind=kind, A=A, W=W, N=N, reps=reps, clip=clip, mean_alarms=float(c.mean()),
                se=float(c.std(ddof=1) / math.sqrt(reps)), bound=N / (A + W))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    t0 = time.time()
    tasks_single, tasks_rep = [], []
    seed = 1
    for kind in ("iid", "ar0.5", "ar0.9", "drift0.5"):
        for A in (10, 30, 100, 300, 1000):
            for clip in (False, True):
                if clip and kind != "iid":
                    continue
                H = min(20 * A, 20000)
                tasks_single.append((kind, A, 4, H, a.reps, seed, clip))
                seed += 1
        for A in (10, 100, 1000):
            tasks_rep.append((kind, A, 4, 20000, max(50, a.reps // 10), seed, False))
            seed += 1
    with ProcessPoolExecutor(a.workers) as ex:
        res_single = list(ex.map(single, tasks_single))
        res_rep = list(ex.map(repeated, tasks_rep))
    json.dump(dict(single=res_single, repeated=res_rep, seconds=time.time() - t0), open(a.out, "w"), indent=1)
    for r in res_single:
        print(json.dumps(r))
    for r in res_rep:
        print(json.dumps(r))


if __name__ == "__main__":
    main()
