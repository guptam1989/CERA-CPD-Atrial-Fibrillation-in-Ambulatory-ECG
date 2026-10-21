"""Wearer-memory extension: nested evaluation of CERA-CPD-M/-MD and the memory gate.

python -m experiments.cera_exp.memory_eval --prep PREP/cpsc2021 --folds folds.json --runs RUNS/main \
    --locked LOCKED --out RESULTS/memory --tasks "0:0,0:1" --external PREP/afdb,PREP/nsrdb,PREP/noise_em_0 \
    [--qrs-prep PREP_QRS/cpsc2021 --qrs-runs RUNS_QRS --qrs-external PREP_QRS/afdb,...]

Protocol (fixed before the corresponding test evaluations; development used
validation patients only). For every locked (fold, seed) encoder each variant is
tuned by the mean challenge score on the fold's validation patients and applied
unchanged to calibration (burden quantile), test, external, noise and QRS-track
recordings:

* cera_m / cera_md - standalone detector with margin-clipped scores and wearer
  memory, mixture / resolution-matched (dyadic) calibrator;
  margin {0,1,2,3,4} x A in run_eval.A_GRID.
* gate_m / gate_md - CUSUM proposals certified by the memory e-detector (margin 0);
  published gate grid drift {1,2,4} x h {2,5,10,20,50} x A_c {3,10,30,100,1121}.
* abl_* - ablations with the same tuning: margin only, margin with the dyadic
  calibrator (both with the published reset policy), wearer memory only
  (margin 0), memory within recordings only (standalone and gated).
* gate*_A{a} - certification curves with the proposer fixed at the model's
  validation-tuned CUSUM (published gate, + memory, + memory and dyadic).
* cera_tuned, cusum, gate_tuned - published locked configurations, replayed.

Wearer memory: CPSC2021 recordings of a patient are processed in ascending record
number with one memory; every external or noise recording is its own wearer.
Output rows follow run_eval.py.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from . import detect as D
from . import run_eval as RE
from .evaluate import record_metrics

MARGINS = (0.0, 1.0, 2.0, 3.0, 4.0)
CERA_GRID = [dict(A=float(a), margin=c) for c in MARGINS for a in RE.A_GRID]
GATE_GRID = [dict(drift=d, h=float(h), A_c=float(a)) for d in (1.0, 2.0, 4.0) for h in (2, 5, 10, 20, 50)
             for a in (3, 10, 30, 100, 1121)]
CURVE_AC = (3, 10, 30, 100, 300, 1121, 11246)
# name: (kind, scope, calibrator, grid)
VARIANTS = {
    "cera_m": ("cera", "wearer", "mixture", CERA_GRID),
    "cera_md": ("cera", "wearer", "dyadic", CERA_GRID),
    "gate_m": ("gate", "wearer", "mixture", GATE_GRID),
    "gate_md": ("gate", "wearer", "dyadic", GATE_GRID),
    "abl_margin": ("cera", "reset", "mixture", CERA_GRID),
    "abl_margin_dyadic": ("cera", "reset", "dyadic", CERA_GRID),
    "abl_wearer_c0": ("cera", "wearer", "mixture", [dict(A=float(a), margin=0.0) for a in RE.A_GRID]),
    "abl_record": ("cera", "record", "mixture", CERA_GRID),
    "abl_gate_record": ("gate", "record", "mixture", GATE_GRID),
}
TRANSFER = ("cera_m", "cera_md", "gate_m", "gate_md", "gate_m_A1121", "gate_md_A1121",
            "cera_tuned", "cusum", "gate_tuned", "gate_A1121")


def record_number(name: str) -> tuple:
    m = re.match(r".*?(\d+)_(\d+)$", name)
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def wearer_order(recs: list[dict]) -> list[dict]:
    return sorted(recs, key=lambda r: (r["patient"], record_number(r["record"])))


def run_variant(spec: tuple, params: dict, recs: list[dict], seed: int, wearer: bool = True):
    """Episodes per record, in the order of recs (wearer order for CPSC2021)."""
    kind, scope, calibrator = spec[:3]
    out, memory, last = [], None, None
    for r in recs:
        if scope != "wearer" or not wearer or r["patient"] != last:
            memory = D.WearerMemory()
        last = r["patient"]
        ctx = RE.ctx_of(r)
        rs = seed + (zlib.crc32(r["record"].encode()) % 100000)
        if kind == "cusum":
            bds, m0 = D.cusum(ctx, params["drift"], params["h"])
        elif kind == "cera":
            bds, m0 = D.cera_memory(ctx, memory, A=params["A"], margin=params["margin"], seed=rs,
                                    calibrator=calibrator, reset=scope == "reset")
        else:
            bds, m0 = D.gated_memory(ctx, memory, params["drift"], params["h"], params["A_c"], seed=rs,
                                     calibrator=calibrator, reset=scope == "reset")
        out.append(D.assemble(bds, m0, len(r["rp_s"])))
    return out


def evaluate(spec, params, recs, seed, wearer=True):
    rows = []
    for r, eps in zip(recs, run_variant(spec, params, recs, seed, wearer)):
        row = record_metrics(r, eps, online=True)
        row.update(record=r["record"], patient=r["patient"], dataset=r["dataset"], ectopy_frac=r.get("ectopy_frac"))
        rows.append(row)
    return rows


def load_sets(prep, folds_path, runs, fold, seed, externals):
    folds = RE.load_json(folds_path)
    roles = folds["folds"][fold]["roles"]
    index = RE.load_json(os.path.join(prep, "index.json"))
    sets = {role: wearer_order(RE.load_role_records(prep, index, roles, role, ranges=role != "calibration"))
            for role in ("validation", "calibration", "test")}
    sdir = os.path.join(runs, f"fold{fold}", f"full_s{seed}")
    for recs in sets.values():
        RE.attach_scores(recs, os.path.join(sdir, "scores_cpsc2021.npz"))
    ext = {}
    for e in externals:
        tag = os.path.basename(os.path.normpath(e))
        sf = os.path.join(sdir, f"scores_{tag}.npz")
        if not os.path.exists(sf):
            continue
        eidx = RE.load_json(os.path.join(e, "index.json"))
        recs = [RE.load_eval_record(e, m, m["dataset"] == "cpsc2021") for m in eidx["records"]]
        RE.attach_scores(recs, sf)
        ext[tag] = recs
    return sets, ext, set(folds["preselected_record"].values())


def task(args):
    prep, folds_path, runs, locked_dir, out, fold, seed, externals, qrs = args
    t0 = time.time()
    os.makedirs(out, exist_ok=True)
    dest = os.path.join(out, f"fold{fold}_memory_s{seed}.json.gz")
    if os.path.exists(dest):
        return dest
    sets, ext, pre = load_sets(prep, folds_path, runs, fold, seed, externals)
    with gzip.open(os.path.join(locked_dir, f"fold{fold}_full_s{seed}.json.gz"), "rt") as f:
        full_cfg = json.load(f)["configs"]
    with gzip.open(os.path.join(locked_dir, f"fold{fold}_gate_s{seed}.json.gz"), "rt") as f:
        gate_cfg = json.load(f)["configs"]["gate_tuned"]
    result = dict(fold=fold, variant="memory", seed=seed, methods={}, external={}, tuning={}, configs={},
                  evaluation_schema_version=RE.EVALUATION_SCHEMA_VERSION,
                  provenance=dict(folds_sha256=RE.file_sha256(folds_path),
                                  locked_dir=os.path.abspath(locked_dir),
                                  source_code_sha256={n: RE.file_sha256(os.path.join(os.path.dirname(__file__), n))
                                                      for n in ("detect.py", "memory_eval.py", "evaluate.py",
                                                                "run_eval.py")},
                                  configuration_origin="validation tuning (memory protocol); comparators locked"))
    chosen = {}
    for name, spec in VARIANTS.items():
        grid = spec[3]
        scores = [float(np.mean([x["U"] for x in evaluate(spec, p, sets["validation"], seed)])) for p in grid]
        k = int(np.argmax(scores))
        chosen[name] = (spec, grid[k], scores[k])
        result["tuning"][name] = dict(grid=grid, val_U=scores)
    cus = dict(drift=float(full_cfg["cusum"]["drift"]), h=float(full_cfg["cusum"]["h"]))
    chosen["cera_tuned"] = (("cera", "reset", "mixture"), dict(A=float(full_cfg["cera_tuned"]["A"]), margin=0.0), None)
    chosen["cusum"] = (("cusum", "reset", "mixture"), cus, None)
    chosen["gate_tuned"] = (("gate", "reset", "mixture"),
                            dict(drift=float(gate_cfg["drift"]), h=float(gate_cfg["h"]), A_c=float(gate_cfg["A_c"])),
                            None)
    for a in CURVE_AC:
        for tag, scope, calibrator in (("gate", "reset", "mixture"), ("gate_m", "wearer", "mixture"),
                                       ("gate_md", "wearer", "dyadic")):
            chosen[f"{tag}_A{a}"] = (("gate", scope, calibrator), dict(cus, A_c=float(a)), None)
    for name, (spec, params, val_U) in chosen.items():
        result["configs"][name] = dict(params, kind=spec[0], scope=spec[1], calibrator=spec[2])
        if val_U is None and not re.search(r"_A\d+$", name):
            val_U = float(np.mean([x["U"] for x in evaluate(spec, params, sets["validation"], seed)]))
        cal_rows = evaluate(spec, params, sets["calibration"], seed)
        test_rows = evaluate(spec, params, sets["test"], seed)
        for x in test_rows:
            x["preselected"] = x["record"] in pre
        burden = RE.calibrate_and_apply(cal_rows, test_rows, pre)
        result["methods"][name] = dict(params=params, scope=spec[1], calibrator=spec[2], val_U=val_U,
                                       burden=burden, test=test_rows)
        if name not in TRANSFER:
            continue
        for tag, recs in ext.items():
            rows = evaluate(spec, params, recs, seed, wearer=False)
            RE.calibrate_and_apply(cal_rows, rows, pre)
            result["external"].setdefault(tag, {})[name] = rows
    if qrs:
        q_prep, q_runs, q_ext = qrs
        q_sets, q_extsets, _ = load_sets(q_prep, folds_path, q_runs, fold, seed, q_ext)
        result["qrs"] = {"methods": {}, "external": {}}
        for name in TRANSFER:
            spec, params, _ = chosen[name]
            cal_rows = evaluate(spec, params, q_sets["calibration"], seed)
            test_rows = evaluate(spec, params, q_sets["test"], seed)
            for x in test_rows:
                x["preselected"] = x["record"] in pre
            burden = RE.calibrate_and_apply(cal_rows, test_rows, pre)
            result["qrs"]["methods"][name] = dict(params=params, burden=burden, test=test_rows)
            for tag, recs in q_extsets.items():
                rows = evaluate(spec, params, recs, seed, wearer=False)
                RE.calibrate_and_apply(cal_rows, rows, pre)
                result["qrs"]["external"].setdefault(tag, {})[name] = rows
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
    ap.add_argument("--qrs-prep", default="")
    ap.add_argument("--qrs-runs", default="")
    ap.add_argument("--qrs-external", default="")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    ext = [e for e in a.external.split(",") if e]
    qrs = (a.qrs_prep, a.qrs_runs, [e for e in a.qrs_external.split(",") if e]) if a.qrs_prep else None
    jobs = []
    for t in a.tasks.split(","):
        f, s = t.split(":")
        jobs.append((a.prep, a.folds, a.runs, a.locked, a.out, int(f), int(s), ext, qrs))
    t0 = time.time()
    with ProcessPoolExecutor(a.workers) as ex:
        for dest in ex.map(task, jobs):
            print(dest, f"{time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
