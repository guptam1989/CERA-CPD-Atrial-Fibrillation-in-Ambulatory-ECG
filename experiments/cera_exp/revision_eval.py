"""Secondary revision analysis: conformal certification of further proposers.

python -m experiments.cera_exp.revision_eval --prep PREP/cpsc2021 --folds folds.json --runs RUNS/main \
    --locked LOCKED_RESULTS --out RESULTS/gatex --tasks "0:0,0:1" --external PREP/afdb,PREP/nsrdb

For every locked (fold, seed) model the proposers keep the parameters that were
selected on validation patients in the primary analysis (causal smoothing,
percentile thresholds, BOCPD and, for seeds 0-2, the separately trained RR-LSTM
with its own beat probabilities). Only the certificate A_c is chosen, from
{3, 10, 30, 100, 1121} by the mean challenge score on the fold's validation
patients; a fixed A_c = 1121 version is also reported. Burden intervals use the
fold's calibration patients; test, external and noise outcomes never enter any
choice. Output rows follow run_eval.py, so the same summaries apply.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import detect as D
from . import run_eval as RE
from .evaluate import record_metrics

A_C_GRID = (3.0, 10.0, 30.0, 100.0, 1121.0)


def run_method_ext(name: str, params: dict, r: dict, seed: int = 0):
    if not name.startswith("gx_"):
        return RE.run_method(name, params, r, seed)
    ctx = RE.ctx_of(r)
    kind = params["proposer"]
    if kind in ("smooth", "percentile"):
        prop = D.SmoothingProposer(ctx, int(params.get("k", 1)), params["t_on"], params["t_off"])
    elif kind == "bocpd":
        prop = D.BocpdProposer(ctx, params["hazard"], params["threshold"])
    elif kind == "rr_lstm":
        prop = D.BeatStateProposer(ctx, r["beat_prob"], params["thr"], int(params["k"]), 0)
    else:
        raise ValueError(kind)
    bds, m0 = D.gated_generic(ctx, prop, A_c=float(params["A_c"]),
                              seed=seed + (zlib.crc32(r["record"].encode()) % 100000))
    return D.assemble(bds, m0, len(r["rp_s"]))


def evaluate_set(name, params, recs, seed=0):
    rows = []
    for r in recs:
        row = record_metrics(r, run_method_ext(name, params, r, seed), online=True)
        row.update(record=r["record"], patient=r["patient"], dataset=r["dataset"], ectopy_frac=r.get("ectopy_frac"))
        rows.append(row)
    return rows


def attach_beats(recs, path):
    z = np.load(path)
    pos = {n: i for i, n in enumerate(z["records"])}
    for r in recs:
        i = pos[r["record"]]
        r["beat_prob"] = z["prob"][z["offsets"][i]:z["offsets"][i + 1]]
        assert len(r["beat_prob"]) == len(r["rp_s"])
    z.close()


def task(args):
    prep, folds_path, runs, out, fold, seed, externals, locked_dir = args
    t0 = time.time()
    os.makedirs(out, exist_ok=True)
    dest = os.path.join(out, f"fold{fold}_gatex_s{seed}.json.gz")
    if os.path.exists(dest):
        return dest
    folds = RE.load_json(folds_path)
    roles = folds["folds"][fold]["roles"]
    pre = set(folds["preselected_record"].values())
    index = RE.load_json(os.path.join(prep, "index.json"))
    val = RE.load_role_records(prep, index, roles, "validation")
    cal = RE.load_role_records(prep, index, roles, "calibration", ranges=False)
    test = RE.load_role_records(prep, index, roles, "test")
    sdir = os.path.join(runs, f"fold{fold}", f"full_s{seed}")
    for s in (val, cal, test):
        RE.attach_scores(s, os.path.join(sdir, "scores_cpsc2021.npz"))
    with gzip.open(os.path.join(locked_dir, f"fold{fold}_full_s{seed}.json.gz"), "rt") as f:
        full_cfg = json.load(f)["configs"]
    proposers = {
        "smooth": dict(proposer="smooth", **full_cfg["smooth"]),
        "percentile": dict(proposer="percentile", k=1, **full_cfg["percentile"]),
        "bocpd": dict(proposer="bocpd", **full_cfg["bocpd"]),
    }
    rr_dir = os.path.join(runs, f"fold{fold}", f"rr_lstm_s{seed}")
    use_rr = os.path.exists(os.path.join(rr_dir, "beats_cpsc2021.npz"))
    if use_rr:
        with gzip.open(os.path.join(locked_dir, f"fold{fold}_rr_lstm_s{seed}.json.gz"), "rt") as f:
            proposers["rrlstm"] = dict(proposer="rr_lstm", **json.load(f)["configs"]["rr_lstm"])
        for s in (val, cal, test):
            attach_beats(s, os.path.join(rr_dir, "beats_cpsc2021.npz"))
    ext_sets = {}
    for e in externals:
        tag = os.path.basename(os.path.normpath(e))
        sf = os.path.join(sdir, f"scores_{tag}.npz")
        if not os.path.exists(sf):
            continue
        eidx = RE.load_json(os.path.join(e, "index.json"))
        recs = [RE.load_eval_record(e, m, m["dataset"] == "cpsc2021") for m in eidx["records"]]
        RE.attach_scores(recs, sf)
        bf = os.path.join(rr_dir, f"beats_{tag}.npz")
        has_beats = use_rr and os.path.exists(bf)
        if has_beats:
            attach_beats(recs, bf)
        ext_sets[tag] = (recs, has_beats)
    result = dict(fold=fold, variant="gatex", seed=seed, methods={}, external={}, tuning={},
                  evaluation_schema_version=RE.EVALUATION_SCHEMA_VERSION,
                  provenance=dict(folds_sha256=RE.file_sha256(folds_path),
                                  locked_full=os.path.abspath(os.path.join(locked_dir, f"fold{fold}_full_s{seed}.json.gz")),
                                  configuration_origin="locked proposer parameters; A_c tuned on validation"))
    configs = {}
    for name, base in proposers.items():
        scores = {a: RE.mean_U(evaluate_set(f"gx_{name}", dict(base, A_c=a), val, seed)) for a in A_C_GRID}
        best = max(A_C_GRID, key=lambda a: (scores[a], -a))
        result["tuning"][name] = {str(a): u for a, u in scores.items()}
        configs[f"gx_{name}"] = (dict(base, A_c=best), scores[best])
        configs[f"gx_{name}_A1121"] = (dict(base, A_c=1121.0), scores[1121.0])
    result["configs"] = {k: v[0] for k, v in configs.items()}
    for mname, (params, val_U) in configs.items():
        cal_rows = evaluate_set(mname, params, cal, seed)
        test_rows = evaluate_set(mname, params, test, seed)
        for x in test_rows:
            x["preselected"] = x["record"] in pre
        burden = RE.calibrate_and_apply(cal_rows, test_rows, pre)
        result["methods"][mname] = dict(params=params, val_U=val_U, burden=burden, test=test_rows)
        for tag, (recs, has_beats) in ext_sets.items():
            if params["proposer"] == "rr_lstm" and not has_beats:
                continue
            rows = evaluate_set(mname, params, recs, seed)
            RE.calibrate_and_apply(cal_rows, rows, pre)
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
    ap.add_argument("--locked", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", required=True, help="fold:seed,...")
    ap.add_argument("--external", default="")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    ext = [e for e in a.external.split(",") if e]
    jobs = []
    for t in a.tasks.split(","):
        f, s = t.split(":")
        jobs.append((a.prep, a.folds, a.runs, a.out, int(f), int(s), ext, a.locked))
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for dest in ex.map(task, jobs):
            print(dest, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
