"""Tune comparators on validation patients, evaluate on test patients,
calibrate burden intervals on calibration patients, and apply the locked
configuration to external databases.

python -m experiments.cera_exp.run_eval --prep PREP --folds folds.json --runs RUNS \
    --out RESULTS --tasks "0:full:0,0:full:1,0:rr:0" --external PREP/afdb,PREP/nsrdb --workers 8
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from functools import lru_cache

import numpy as np

from . import detect as D
from . import rr_baselines as RB
from .features import block_release_times
from .evaluate import (burden_interval, conformal_quantile, interval_coverage,
                       official_ranges, record_metrics, residual)

EVALUATION_SCHEMA_VERSION = 2


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=128)
def _source_manifest(path, size, mtime_ns):
    return dict(path=path, size_bytes=size, mtime_ns=mtime_ns, sha256=file_sha256(path))


def source_manifest(path):
    path = os.path.abspath(path)
    stat = os.stat(path)
    return _source_manifest(path, stat.st_size, stat.st_mtime_ns)


def load_json(path):
    with open(path) as source:
        return json.load(source)

A_NOMINAL = 11246.0  # 0.1 alarm/h, b=8 beats x 0.4 s, W=4 (Corollary 1 design setting)
A_GRID = [100, 300, 1121, 3000, 11246, 30000, 100000, 300000, 1000000]
SMOOTH_GRID = [(k, ton, toff) for k in (1, 3, 5, 9) for ton in (0.5, 0.6, 0.7, 0.8, 0.9)
               for toff in (0.5, 0.4, 0.3, 0.2, 0.1)]
CUSUM_GRID = [(c, h) for c in (0.0, 1.0, 2.0, 4.0) for h in (2, 5, 10, 20, 50, 100)]
BOCPD_GRID = [(hz, th) for hz in (1 / 100, 1 / 1000) for th in (0.3, 0.5, 0.7, 0.9)]
RR_K = (9, 33, 65)
RR_Q = (0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95)


# ---------------------------------------------------------------- loading

def load_eval_record(prep: str, meta: dict, ranges: bool) -> dict:
    z = np.load(os.path.join(prep, meta["file"]))
    fs = meta["fs"]
    rp = z["rpeaks"]
    b = z["atrial"].shape[1] if "atrial" in z.files else 8
    bs = z["block_start"].astype(np.int64)
    rp_s = rp / fs
    if "confirm_samples" in z.files:
        # end-to-end track: detector confirmation times replace the nominal latency
        conf_s = z["confirm_samples"] / fs
        available = (np.maximum(conf_s[bs + b], rp_s[bs + b - 1] + 0.5) if len(bs)
                     else np.empty(0, dtype=np.float64))
        policy = "max_detector_confirmation_last_beat_quality_v2"
    else:
        available = block_release_times(rp_s, bs, b)
        policy = "max_next_peak_confirmation_last_beat_quality_v2"
    rec = dict(meta, rpeaks=rp, rp_s=rp_s, labels=z["labels"], block_start=bs,
               ectopy_frac=float((z["sym_code"] == 1).mean()) if len(rp) else 0.0,
               available=available, b=b, quality=z["quality"].astype(np.float32),
               availability_policy=policy)
    obs = []
    for s in bs:
        t0, t1 = float(rp_s[s]), float(rp_s[s + b])
        if obs and abs(obs[-1][1] - t0) < 1e-9:
            obs[-1] = (obs[-1][0], t1)
        else:
            obs.append((t0, t1))
    rec["observed"] = obs
    if meta["dataset"] == "cpsc2021":
        ann, flag = z["ann_samples"], z["ann_flag"]
        st, en = np.where(flag == 1)[0], np.where(flag == 2)[0]
        rec["true_samples"] = [(int(ann[a]), int(ann[c])) for a, c in zip(st, en)]
        rec["_ranges"] = None
        rec["_ann"] = (ann.astype(np.int64), flag)   # official score evaluated point-wise (exact)
    z.close()
    return rec


def attach_scores(recs: list[dict], score_file: str):
    z = np.load(score_file)
    manifest = source_manifest(score_file)
    names = list(z["records"])
    off = z["offsets"]
    pos = {n: i for i, n in enumerate(names)}
    p = z["p"].astype(np.float64)
    lp = np.log(np.clip(p, 1e-7, 1 - 1e-7)) - np.log(np.clip(1 - p, 1e-7, 1))
    if "lp" in z.files:
        lp = z["lp"].astype(np.float64)
    pb = z["pb"]
    for r in recs:
        i = pos[r["record"]]
        s, e = off[i], off[i + 1]
        assert e - s == len(r["block_start"]), (r["record"], e - s, len(r["block_start"]))
        r["logit"] = lp[s:e]
        r["pb"] = pb[s:e].astype(np.float32)
        r["_score_source"] = manifest
    z.close()


def ctx_of(r: dict) -> D.RecordContext:
    return D.RecordContext(r["rp_s"], r["block_start"], r["available"], r["logit"], r["pb"], r["b"], r["duration_s"])


# ---------------------------------------------------------------- methods

def run_method(name: str, params: dict, r: dict, seed: int = 0):
    n = len(r["rp_s"])
    if name in ("rr_lstm", "unet"):
        return RB.detect_beats(r["beat_prob"], params["thr"], params["k"], 0, r["rp_s"])
    if name.startswith("rr_"):
        kind = name[3:]
        eps = RB.detect_beats(r["rr_score"][kind], params["thr"], params["k"], RB.LAGS[kind], r["rp_s"])
        return eps
    ctx = ctx_of(r)
    if name.startswith("gate"):
        bds, m0 = D.gated(ctx, params["drift"], params["h"], params["A_c"],
                          seed=seed + (zlib.crc32(r["record"].encode()) % 100000))
        return D.assemble(bds, m0, n)
    if name.startswith("cera"):
        bds, m0 = D.cera(ctx, A=params["A"], W=params.get("W", 4), W0=4, beta=params.get("beta", 1.0),
                         calibrator=params.get("calibrator", "mixture"), kappa=params.get("kappa", 0.3),
                         seed=seed + (zlib.crc32(r["record"].encode()) % 100000), fallback=params.get("fallback", False),
                         restart_at_change=params.get("restart", False),
                         beat_refine=params.get("beat_refine", True), clip=params.get("clip", True))
    elif name == "smooth":
        bds, m0 = D.smoothing(ctx, params["k"], params["t_on"], params["t_off"])
    elif name == "percentile":
        bds, m0 = D.percentile(ctx, params["t_on"], params["t_off"])
    elif name == "cusum":
        bds, m0 = D.cusum(ctx, params["drift"], params["h"])
    elif name == "bocpd":
        bds, m0 = D.bocpd(ctx, params["hazard"], params["threshold"])
    elif name == "offline":
        return offline_segmentation(r, params.get("k", 5))
    else:
        raise ValueError(name)
    return D.assemble(bds, m0, n)


def offline_segmentation(r: dict, k: int = 5):
    """Non-causal same-encoder reference: centred median filter of block
    probabilities (k blocks) at 0.5, refined to beats inside transition blocks."""
    J, b = len(r["block_start"]), r["b"]
    n = len(r["rp_s"])
    if J == 0:
        return []
    p = 1 / (1 + np.exp(-r["logit"]))
    pad = np.pad(p, (k // 2, k // 2), mode="edge")
    med = np.median(np.lib.stride_tricks.sliding_window_view(pad, k), axis=1)
    lab = np.full(n, -1, np.int8)
    for j in range(J):
        s = r["block_start"][j]
        lab[s:s + b] = int(med[j] >= 0.5)
    # beat-level refinement where beat probabilities disagree with neighbours at transitions
    pb = r["pb"]
    for j in range(J):
        s = r["block_start"][j]
        if (j > 0 and med[j - 1] >= 0.5) != (med[j] >= 0.5) or (j + 1 < J and med[j + 1] >= 0.5) != (med[j] >= 0.5):
            lab[s:s + b] = (pb[j] >= 0.5).astype(np.int8)
    # fill unobserved beats with the previous decision
    last = 0
    for i in range(n):
        if lab[i] < 0:
            lab[i] = last
        else:
            last = lab[i]
    x = np.concatenate([[0], lab, [0]])
    d = np.diff(x)
    eps = []
    for s, e in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0]):
        if e - s >= 5:
            eps.append(dict(start=int(s), end=int(e), lc=bool(s == 0), rc=bool(e == n),
                            onset_alarm_s=None, spread_s=0.0, fallback_offset=False))
    return eps


def evaluate_set(name, params, recs, seed=0, online=True):
    rows = []
    for r in recs:
        eps = run_method(name, params, r, seed)
        row = record_metrics(r, eps, online=online)
        row.update(record=r["record"], patient=r["patient"], dataset=r["dataset"], ectopy_frac=r.get("ectopy_frac"))
        rows.append(row)
    return rows


def mean_U(rows):
    return float(np.mean([x["U"] for x in rows]))


def tune(name, grid, recs, seed):
    best, best_p = -np.inf, None
    for params in grid:
        u = mean_U(evaluate_set(name, params, recs, seed))
        if u > best:
            best, best_p = u, params
    return best_p, best


# ---------------------------------------------------------------- burden

def calibrate_and_apply(cal_rows, test_rows, preselected, eps=0.01):
    selected = [x for x in cal_rows if x["record"] in preselected]
    out = dict(calibration_selected=len(selected),
               calibration_excluded_zero_observed=sum(x["T_o"] <= 0 for x in selected),
               calibration_excluded_unknown_reference=sum(x["T_o"] > 0 and x["B_o"] is None for x in selected))
    for normalized in (True, False):
        res = [residual(x, eps, normalized) for x in selected if x["T_o"] > 0 and x["B_o"] is not None]
        q = conformal_quantile(res, 0.1)
        tag = "norm" if normalized else "abs"
        out[f"qhat_{tag}"] = q
        out[f"M_{tag}"] = len(res)
        for x in test_rows:
            (lo, hi), (flo, fhi) = burden_interval(x, q, eps, normalized)
            interval_coverage(x, lo, hi, flo, fhi, tag)
    return out


# ---------------------------------------------------------------- task

def load_role_records(prep, index, roles, role, ranges=True):
    return [load_eval_record(prep, m, ranges) for m in index["records"] if roles[m["patient"]] == role]


def task(args):
    prep, folds_path, runs, out, fold, variant, seed, externals, val_only = args[:9]
    locked_dir = args[9] if len(args) > 9 else None
    t0 = time.time()
    os.makedirs(out, exist_ok=True)
    dest = os.path.join(out, f"fold{fold}_{variant}_s{seed}" + ("_val" if val_only else "") + ".json.gz")
    if os.path.exists(dest):
        with gzip.open(dest, "rt") as existing:
            if json.load(existing).get("evaluation_schema_version") != EVALUATION_SCHEMA_VERSION:
                raise ValueError(f"Stale evaluation output {dest}; use a fresh output directory to regenerate it")
        return dest
    locked, locked_path = None, None
    if locked_dir:
        locked_path = os.path.join(locked_dir, f"fold{fold}_{variant}_s{seed}.json.gz")
        with gzip.open(locked_path, "rt") as source:
            locked = json.load(source)
        if (locked.get("fold"), locked.get("variant"), locked.get("seed")) != (fold, variant, seed):
            raise ValueError(f"Locked evaluation identity mismatch: {locked_path}")
        if not locked.get("configs"):
            raise ValueError(f"Locked evaluation has no detector configurations: {locked_path}")
    folds = load_json(folds_path)
    roles = folds["folds"][fold]["roles"]
    pre = set(folds["preselected_record"].values())
    index = load_json(os.path.join(prep, "index.json"))
    val = load_role_records(prep, index, roles, "validation")
    cal = load_role_records(prep, index, roles, "calibration", ranges=False)
    test = load_role_records(prep, index, roles, "test")
    result = dict(fold=fold, variant=variant, seed=seed, methods={}, external={},
                  evaluation_schema_version=EVALUATION_SCHEMA_VERSION)
    result["provenance"] = dict(
        folds_sha256=file_sha256(folds_path),
        prepared_index_sha256=file_sha256(os.path.join(prep, "index.json")),
        availability_policy=(test or val or cal)[0]["availability_policy"] if (test or val or cal)
        else "max_next_peak_confirmation_last_beat_quality_v2",
        configuration_origin="locked_original_validation" if locked else "validation_tuning")
    result["provenance"]["source_code_sha256"] = {
        name: file_sha256(os.path.join(os.path.dirname(__file__), name))
        for name in ("evaluate.py", "run_eval.py", "features.py", "detect.py")}
    if locked:
        result["provenance"].update(locked_evaluation_path=os.path.abspath(locked_path),
                                      locked_evaluation_sha256=file_sha256(locked_path),
                                      detector_configs_sha256=hashlib.sha256(json.dumps(
                                          locked["configs"], sort_keys=True).encode()).hexdigest())
    if variant in ("rr_lstm", "unet"):
        def attach_beats(recs, path):
            z = np.load(path)
            manifest = source_manifest(path)
            pos = {n: i for i, n in enumerate(z["records"])}
            for r in recs:
                i = pos[r["record"]]
                r["beat_prob"] = z["prob"][z["offsets"][i]:z["offsets"][i + 1]]
                assert len(r["beat_prob"]) == len(r["rp_s"])
                r["_score_source"] = manifest
            z.close()
        base_dir = os.path.join(runs, f"fold{fold}", f"{variant}_s{seed}")
        for s in (val, cal, test):
            attach_beats(s, os.path.join(base_dir, "beats_cpsc2021.npz"))
        ext_sets = {}
        for e in externals:
            tag = os.path.basename(os.path.normpath(e))
            ef = os.path.join(base_dir, f"beats_{tag}.npz")
            if os.path.exists(ef):
                eidx = load_json(os.path.join(e, "index.json"))
                recs = [load_eval_record(e, m, m["dataset"] == "cpsc2021") for m in eidx["records"]]
                attach_beats(recs, ef)
                ext_sets[tag] = recs
        grid = [dict(thr=t, k=k) for t in (0.3, 0.4, 0.5, 0.6, 0.7) for k in (1, 9, 33, 65)]
        configs = locked["configs"] if locked else {variant: tune(variant, grid, val, seed)[0]}
    elif variant == "rr":
        train = [load_eval_record(prep, m, False) for m in index["records"] if roles[m["patient"]] == "train"]
        llr = np.asarray(locked["markov_llr"], dtype=float) if locked else RB.fit_markov(train)
        result["markov_llr"] = llr.tolist()

        def add_scores(recs):
            for r in recs:
                r["rr_score"] = dict(cv=RB.cv_score(r["rp_s"]), cosen=RB.cosen_score(r["rp_s"]),
                                     markov=RB.markov_score(r["rp_s"], llr))
        for s in (val, cal, test):
            add_scores(s)
        ext_sets = {}
        for e in externals:
            eidx = load_json(os.path.join(e, "index.json"))
            recs = [load_eval_record(e, m, m["dataset"] == "cpsc2021") for m in eidx["records"]]
            add_scores(recs)
            ext_sets[os.path.basename(os.path.normpath(e))] = recs
        configs = dict(locked["configs"]) if locked else {}
        for kind in (() if locked else ("cv", "cosen", "markov")):
            allv = np.concatenate([r["rr_score"][kind] for r in val])
            grid = [dict(thr=float(np.quantile(allv, q)), k=k) for q in RR_Q for k in RR_K]
            configs[f"rr_{kind}"] = tune(f"rr_{kind}", grid, val, seed)[0]
    else:
        sdir = "full" if variant == "gate" else variant
        sf = os.path.join(runs, f"fold{fold}", f"{sdir}_s{seed}", "scores_cpsc2021.npz")
        for s in (val, cal, test):
            attach_scores(s, sf)
        ext_sets = {}
        for e in externals:
            tag = os.path.basename(os.path.normpath(e))
            ef = os.path.join(runs, f"fold{fold}", f"{sdir}_s{seed}", f"scores_{tag}.npz")
            if not os.path.exists(ef):
                continue
            eidx = load_json(os.path.join(e, "index.json"))
            recs = [load_eval_record(e, m, m["dataset"] == "cpsc2021") for m in eidx["records"]]
            attach_scores(recs, ef)
            ext_sets[tag] = recs
        configs = dict(locked["configs"]) if locked else {"cera": dict(A=A_NOMINAL)}
        if variant == "gate" and not locked:
            # secondary analysis: CUSUM proposals certified by the conformal e-detector
            cus = tune("cusum", [dict(drift=c, h=h) for c, h in CUSUM_GRID], val, seed)[0]
            grid = [dict(drift=c, h=h, A_c=a) for c in (1.0, 2.0, 4.0) for h in (2, 5, 10, 20, 50)
                    for a in (3, 10, 30, 100, 1121)]
            configs = {"gate_tuned": tune("gate", grid, val, seed)[0]}
            for a in (3, 10, 30, 100, 300, 1121, 11246):
                configs[f"gate_A{a}"] = dict(drift=cus["drift"], h=cus["h"], A_c=float(a))
        elif variant != "full" and not locked:
            configs["cera_tuned"] = dict(A=tune("cera", [dict(A=a) for a in A_GRID], val, seed)[0]["A"])
        if variant == "full" and not locked:
            pa = np.concatenate([1 / (1 + np.exp(-r["logit"])) for r in val])
            ya = np.concatenate([np.array([int(np.mean(r["labels"][s:s + r["b"]]) >= 0.5) for s in r["block_start"]])
                                 for r in val])
            configs["cera_tuned"] = dict(A=tune("cera", [dict(A=a) for a in A_GRID], val, seed)[0]["A"])
            configs["cera_fallback"] = dict(A=A_NOMINAL, fallback=True)
            configs["cera_power"] = dict(A=A_NOMINAL, calibrator="power", kappa=0.3)
            configs["cera_restart"] = dict(A=A_NOMINAL, restart=True)
            configs["cera_blockloc"] = dict(A=A_NOMINAL, beat_refine=False)
            configs["cera_noclip"] = dict(A=A_NOMINAL, clip=False)
            configs["cera_1h"] = dict(A=1121.0)
            for W in (2, 8):
                configs[f"cera_W{W}"] = dict(A=A_NOMINAL, W=W)
            configs["smooth"] = tune("smooth", [dict(k=k, t_on=a, t_off=c) for k, a, c in SMOOTH_GRID], val, seed)[0]
            configs["percentile"] = dict(t_on=float(np.quantile(pa[ya == 0], 0.99)),
                                         t_off=float(np.quantile(pa[ya == 1], 0.01)))
            configs["cusum"] = tune("cusum", [dict(drift=c, h=h) for c, h in CUSUM_GRID], val, seed)[0]
            configs["bocpd"] = tune("bocpd", [dict(hazard=hz, threshold=th) for hz, th in BOCPD_GRID], val, seed)[0]
            configs["offline"] = dict(k=5)
    result["configs"] = configs
    result["provenance"]["score_sources"] = list({
        r["_score_source"]["path"]: r["_score_source"]
        for group in [val, cal, test, *ext_sets.values()] for r in group if "_score_source" in r}.values())
    result["provenance"]["external_prepared_indexes"] = [
        source_manifest(os.path.join(path, "index.json")) for path in externals]
    if val_only:
        from .summary import summarize
        for mname, params in configs.items():
            base = "cera" if mname.startswith("cera") else ("gate" if mname.startswith("gate") else mname)
            rows = evaluate_set(base, params, val, seed, base != "offline")
            result["methods"][mname] = dict(params=params, val=summarize(rows))
        with gzip.open(dest, "wt") as f:
            json.dump(result, f)
        return dest
    for mname, params in configs.items():
        base = "cera" if mname.startswith("cera") else ("gate" if mname.startswith("gate") else mname)
        online = base not in ("offline", "unet")
        val_U = (locked.get("methods", {}).get(mname, {}).get("val_U") if locked else None)
        if val_U is None:
            val_U = mean_U(evaluate_set(base, params, val, seed, online))
        cal_rows = evaluate_set(base, params, cal, seed, online)
        test_rows = evaluate_set(base, params, test, seed, online)
        for x in test_rows:
            x["preselected"] = x["record"] in pre
        burden = calibrate_and_apply(cal_rows, test_rows, pre)
        result["methods"][mname] = dict(params=params, val_U=val_U, burden=burden, test=test_rows)
        for tag, recs in ext_sets.items():
            rows = evaluate_set(base, params, recs, seed, online)
            calibrate_and_apply(cal_rows, rows, pre)   # CPSC2021 calibration patients (locked fold quantile)
            result["external"].setdefault(tag, {})[mname] = rows
    result["seconds"] = time.time() - t0
    with gzip.open(dest, "wt") as f:
        json.dump(result, f, default=lambda o: o.item() if hasattr(o, "item") else (None if o is None else float(o)))
    return dest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", required=True, help="fold:variant:seed,...")
    ap.add_argument("--external", default="")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val-only", action="store_true")
    ap.add_argument("--locked-evaluations", default="",
                    help="Reuse configurations/Markov model from original fold JSON outputs without retuning")
    a = ap.parse_args()
    ext = [e for e in a.external.split(",") if e]
    jobs = []
    for t in a.tasks.split(","):
        f, v, s = t.split(":")
        jobs.append((a.prep, a.folds, a.runs, a.out, int(f), v, int(s), ext, a.val_only,
                     a.locked_evaluations or None))
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for dest in ex.map(task, jobs):
            print(dest, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
