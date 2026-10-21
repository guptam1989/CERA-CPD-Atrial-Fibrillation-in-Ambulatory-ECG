"""Why conformal e-detection is slow: idealised simulation and empirical analysis.

(1) Ideal encoder: n pre-change blocks tied at the clip (score 0), then i.i.d.
    positive post-change scores. Blocks needed for the conformal SR to reach
    log A, versus n (reference contamination included: post-change blocks enter
    the bag). Compared with a fixed-reference bound in which every post-change
    block is extreme against the n pre-change blocks only.
(2) Empirical: CPSC2021 test episodes, detection and delay versus the length of
    the reference history (observed blocks since recording start or the previous
    reference offset), for CERA-CPD and CUSUM with each fold's validated settings.

python -m experiments.cera_exp.mechanism --sim-out mech_sim.json
python -m experiments.cera_exp.mechanism --prep PREP --folds F --runs RUNS --evals RESULTS --emp-out mech_emp.json
"""
from __future__ import annotations

import argparse
import bisect
import gzip
import hashlib
import json
import math
import os
import platform
from pathlib import Path

import numpy as np

from .detect import log_mixture_calibrator, logaddexp


def blocks_to_alarm(n: int, A: float, rng, W: int = 4, cap: int = 2000, contaminate: bool = True) -> int:
    bag = [0.0] * n
    log_sr, logA = -math.inf, math.log(A)
    for k in range(1, cap + 1):
        s = 1.0 + rng.random()
        if contaminate:
            bisect.insort(bag, s)
            m = len(bag)
            hi = bisect.bisect_right(bag, s)
            p = (m - hi + rng.random() * (hi - bisect.bisect_left(bag, s))) / m
        else:
            p = rng.random() / (n + 1)
        if contaminate and n + k <= W:
            continue
        log_sr = logaddexp(0.0, log_sr) + log_mixture_calibrator(p)
        if log_sr >= logA:
            return k
    return cap


def simulate(out: str, reps: int = 300, seed: int = 3):
    if reps <= 0:
        raise ValueError("replicates must be positive")
    rng = np.random.default_rng(seed)
    res = []
    for A in (1121, 11246, 300000):
        for n in (5, 10, 20, 50, 100, 200, 500, 1000, 3000):
            for cont in (True, False):
                ks = [blocks_to_alarm(n, A, rng, contaminate=cont) for _ in range(reps)]
                ks = np.array(ks)
                res.append(dict(A=A, n=n, contaminated=cont, median=float(np.median(ks)),
                                p90=float(np.percentile(ks, 90)), never=float((ks >= 2000).mean())))
                print(json.dumps(res[-1]), flush=True)
    payload = dict(schema_version=2, metadata=dict(
        replicates=reps, seed=seed, simulation_version="ideal_encoder_v1",
        horizon_blocks=2000, warmup_blocks=4,
        python_version=platform.python_version(), numpy_version=np.__version__,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), results=res)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as handle:
        json.dump(payload, handle, indent=1, allow_nan=False)
    return payload


def empirical(prep, folds_path, runs, evals, out, seeds=(0,)):
    from . import detect as D
    from . import run_eval as RE
    folds = json.load(open(folds_path))["folds"]
    index = json.load(open(os.path.join(prep, "index.json")))
    rows = []
    for f in range(len(folds)):
        roles = folds[f]["roles"]
        test = RE.load_role_records(prep, index, roles, "test")
        for s in seeds:
            RE.attach_scores(test, os.path.join(runs, f"fold{f}", f"full_s{s}", "scores_cpsc2021.npz"))
            cfg = json.load(gzip.open(os.path.join(evals, f"fold{f}_full_s{s}.json.gz"), "rt"))["configs"]
            for r in test:
                lab = r["labels"]
                x = np.diff(np.concatenate([[0], (lab == 1).astype(np.int8), [0]]))
                starts, ends = np.nonzero(x == 1)[0], np.nonzero(x == -1)[0]
                if not len(starts):
                    continue
                ctx = RE.ctx_of(r)
                det = {}
                for name, fn in (("cera", lambda c: D.cera(c, A=cfg["cera_tuned"]["A"], seed=s)),
                                 ("cusum", lambda c: D.cusum(c, cfg["cusum"]["drift"], cfg["cusum"]["h"]))):
                    bds, _ = fn(ctx)
                    det[name] = [b.alarm_s for b in bds if b.direction == "onset"]
                bs = r["block_start"]
                prev_end = 0
                for st, en in zip(starts, ends):
                    if st == 0:
                        prev_end = en
                        continue
                    hist = int(((bs >= prev_end) & (bs + r["b"] <= st)).sum())
                    t_on = float(r["rp_s"][st])
                    t_end = float(r["rp_s"][en]) if en < len(r["rp_s"]) else r["duration_s"]
                    for name, al in det.items():
                        cand = [a - t_on for a in al if t_on <= a <= t_end + 60]
                        rows.append(dict(fold=f, seed=s, record=r["record"], method=name, history_blocks=hist,
                                         episode_beats=int(en - st), delay=min(cand) if cand else None))
                    prev_end = en
    json.dump(rows, open(out, "w"))
    print("episodes x methods:", len(rows))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sim-out")
    ap.add_argument("--emp-out")
    ap.add_argument("--prep")
    ap.add_argument("--folds")
    ap.add_argument("--runs")
    ap.add_argument("--evals")
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--sim-seed", type=int, default=3)
    a = ap.parse_args()
    if a.sim_out:
        simulate(a.sim_out, a.reps, a.sim_seed)
    if a.emp_out:
        empirical(a.prep, a.folds, a.runs, a.evals, a.emp_out, seeds=(0, 1, 2, 3, 4))


if __name__ == "__main__":
    main()
