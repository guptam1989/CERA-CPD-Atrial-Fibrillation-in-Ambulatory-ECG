"""Operating curves: accuracy / delay / false-alarm workload versus the
detector threshold, on CPSC2021 test patients (each fold's own test set) and
false-alarm rates on change-free external sinus recordings (NSRDB) and
non-AF MIT-BIH records.

python -m experiments.cera_exp.fa_sweep --prep PREP --folds F --runs RUNS --evals RESULTS \
    --tasks 0:0,1:0 --nsrdb PREP/nsrdb --mitdb PREP/mitdb --out sweep
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import run_eval as RE
from .summary import sweep_summary

A_SWEEP = [30, 100, 300, 1121, 3000, 11246, 30000, 100000, 1000000]
H_SWEEP = [2, 5, 10, 20, 50, 100, 200]
T_SWEEP = [0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]
MITDB_AF = {"201", "202", "203", "210", "219", "221", "222"}


def alarm_rate(name, params, recs, seed):
    """Calibrated alarms (all boundaries) and false episodes per hour on change-free data."""
    hours = sum(r["duration_s"] for r in recs) / 3600
    obs_hours = sum(sum(e - s for s, e in r["observed"]) for r in recs) / 3600
    n_alarms = n_eps = 0
    blocks = 0
    for r in recs:
        ctx = RE.ctx_of(r)
        blocks += ctx.J
        if name == "cera":
            from .detect import cera, assemble
            bds, m0 = cera(ctx, A=params["A"], seed=seed)
        else:
            raise ValueError(name)
        n_alarms += len(bds)
        n_eps += len(assemble(bds, m0, len(r["rp_s"])))
    return dict(hours=hours, obs_hours=obs_hours, blocks=blocks, alarms=n_alarms, alarms_per_h=n_alarms / hours,
                false_eps_per_h=n_eps / hours, mean_block_s=obs_hours * 3600 / max(blocks, 1))


def task(args):
    prep, folds_path, runs, evals, fold, seed, nsrdb, mitdb, out = args
    dest = os.path.join(out, f"sweep_fold{fold}_s{seed}.json")
    if os.path.exists(dest):
        with open(dest) as handle:
            existing = json.load(handle)
        expected = ({f"cera_{a}" for a in A_SWEEP} | {f"cusum_{h}" for h in H_SWEEP}
                    | {f"smooth_{t}" for t in T_SWEEP})
        if (existing.get("schema_version") != 2 or existing.get("fold") != fold or existing.get("seed") != seed
                or set(existing.get("cpsc", {})) != expected
                or any("pooling_v1" not in s for s in existing.get("cpsc", {}).values())):
            raise ValueError(f"Legacy/incomplete sweep at {dest}; use a fresh output directory to regenerate.")
        return dest
    roles = json.load(open(folds_path))["folds"][fold]["roles"]
    index = json.load(open(os.path.join(prep, "index.json")))
    test = RE.load_role_records(prep, index, roles, "test")
    sf = os.path.join(runs, f"fold{fold}", f"full_s{seed}", "scores_cpsc2021.npz")
    RE.attach_scores(test, sf)
    chosen = json.load(gzip.open(os.path.join(evals, f"fold{fold}_full_s{seed}.json.gz"), "rt"))["configs"]
    res = dict(schema_version=2, fold=fold, seed=seed, cpsc={}, nsrdb={}, mitdb_nonaf={}, cpsc_nonaf={})
    for A in A_SWEEP:
        res["cpsc"][f"cera_{A}"] = sweep_summary(RE.evaluate_set("cera", dict(A=A), test, seed))
    for h in H_SWEEP:
        res["cpsc"][f"cusum_{h}"] = sweep_summary(RE.evaluate_set("cusum", dict(drift=chosen["cusum"]["drift"], h=h), test, seed))
    for t in T_SWEEP:
        sm = chosen["smooth"]
        res["cpsc"][f"smooth_{t}"] = sweep_summary(RE.evaluate_set("smooth", dict(k=sm["k"], t_on=t, t_off=min(sm["t_off"], 1 - t)), test, seed))
    nonaf = [r for r in test if r["rec_class"] == 0]
    for name, recs, key in ((nsrdb, None, "nsrdb"), (mitdb, None, "mitdb_nonaf")):
        if not name:
            continue
        tag = os.path.basename(os.path.normpath(name))
        ef = os.path.join(runs, f"fold{fold}", f"full_s{seed}", f"scores_{tag}.npz")
        if not os.path.exists(ef):
            continue
        eidx = json.load(open(os.path.join(name, "index.json")))
        metas = [m for m in eidx["records"] if not (key == "mitdb_nonaf" and m["af_beats"] > 0)]
        recs = [RE.load_eval_record(name, m, False) for m in metas]
        RE.attach_scores(recs, ef)
        for A in A_SWEEP:
            res[key][str(A)] = alarm_rate("cera", dict(A=A), recs, seed)
        ac, acc, fp = [], [], []
        for r in recs:
            x = r["logit"]
            if len(x) < 100:
                continue
            ac.append(float(np.corrcoef(x[:-1], x[1:])[0, 1]))
            cl = np.maximum(x if key else x, 0.0)
            fp.append(float((x > 0).mean()))
            if cl.std() > 0:
                acc.append(float(np.corrcoef(cl[:-1], cl[1:])[0, 1]))
        res.setdefault("autocorr", {})[key] = dict(logit_lag1_median=float(np.median(ac)) if ac else None,
                                                   logit_lag1_iqr=[float(v) for v in np.percentile(ac, [25, 75])] if ac else None,
                                                   clipped_lag1_median=float(np.median(acc)) if acc else None,
                                                   block_fp_median=float(np.median(fp)) if fp else None)
    for A in A_SWEEP:
        res["cpsc_nonaf"][str(A)] = alarm_rate("cera", dict(A=A), nonaf, seed)
    json.dump(res, open(dest, "w"))
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--evals", required=True)
    ap.add_argument("--tasks", required=True)
    ap.add_argument("--nsrdb", default="")
    ap.add_argument("--mitdb", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    jobs = [(a.prep, a.folds, a.runs, a.evals, int(t.split(":")[0]), int(t.split(":")[1]), a.nsrdb, a.mitdb, a.out)
            for t in a.tasks.split(",")]
    with ProcessPoolExecutor(a.workers) as ex:
        for d in ex.map(task, jobs):
            print(d, flush=True)


if __name__ == "__main__":
    main()
