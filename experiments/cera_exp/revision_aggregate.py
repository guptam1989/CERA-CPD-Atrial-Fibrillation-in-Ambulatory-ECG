"""Aggregate the revision analyses into one compact summary.

python -m experiments.cera_exp.revision_aggregate --ext-ref R/ext_ref --qrs R/qrs --gatex R/gatex \
    --integrity INT/results/main --prep-qrs PREP_QRS --folds folds.json --out revision_summary.json

Result files are streamed one at a time and reduced to the fields needed here.
Conventions follow aggregate.py: CPSC2021 metrics are pooled over the five test
folds for each training seed and reported as mean/s.d. over seeds; external
cohorts are summarised per locked (fold, seed) model and averaged over models;
noise copies use only each patient's outer-test models; paired comparisons use
one value per patient (sign-flip test, bootstrap interval).

Standard AF-detector summaries (ANSI/AAMI EC57 style, rhythm-labelled time):
  dur_se  = AF time detected / reference AF time
  dur_ppv = AF time detected / predicted AF time
  dur_sp  = non-AF time not predicted as AF / reference non-AF time
  ep_se   = reference episodes overlapped by any predicted episode
  ep_ppv  = predicted episodes overlapping any reference AF
  se30    = ep_se restricted to reference episodes lasting >= 30 s
  ppv30, fa30_h: predicted episodes >= 30 s that overlap reference AF / that do
           not, per labelled hour
Burden agreement (one pre-selected recording per patient): Bland--Altman bias
and 95% limits of agreement of predicted minus reference burden, and Lin's
concordance correlation coefficient.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import math
import os
from collections import defaultdict

import numpy as np

from .aggregate import holm, mean_sd, paired_test
from .evaluate import burden_interval, interval_coverage

KEEP = ("record", "patient", "preselected", "class_true", "class_pred", "U", "hours", "T", "T_o",
        "T_label_known", "n_true_eps", "n_pred_eps", "ep_match_0.5", "false_eps", "delays",
        "n_true_on", "n_true_off", "n_pred_on", "n_pred_off", "on_match_1", "off_match_1",
        "true_af_known_s", "pred_af_known_s", "af_tp_s", "true_ep_s", "true_ep_hit",
        "pred_ep_s", "pred_ep_true", "B", "B_hat", "ectopy_frac", "int_norm",
        "cov_obs_norm", "cov_full_norm", "cov_full_possible_norm", "cov_full_certain_norm")
MITDB_NON_AF = lambda x: x["n_true_eps"] == 0


def compact(x: dict) -> dict:
    return {k: x[k] for k in KEEP if k in x}


def _safe(a, b):
    return float(a) / float(b) if b else float("nan")


def load_dir(directory: str, want=None, ext_tags=None, test=True):
    """rows[(variant, method, seed)] -> compact CPSC test rows (all folds);
    ext[(tag, variant, method, fold, seed)] -> compact rows.
    want(variant, method) filters methods; ext_tags filters external tags."""
    rows, ext = defaultdict(list), {}
    for path in sorted(glob.glob(os.path.join(directory, "fold*_s*.json.gz"))):
        if path.endswith("_val.json.gz"):
            continue
        with gzip.open(path, "rt") as f:
            r = json.load(f)
        v, fo, s = r["variant"], r["fold"], r["seed"]
        for m, d in r["methods"].items():
            if want is not None and not want(v, m):
                continue
            if test:
                for x in d["test"]:
                    c = compact(x)
                    c["fold"] = fo
                    rows[(v, m, s)].append(c)
            b = d["burden"]
            for tag, md in r.get("external", {}).items():
                if m not in md or (ext_tags is not None and tag not in ext_tags):
                    continue
                xs = []
                for x in md[m]:
                    for normalized, t2 in ((True, "norm"), (False, "abs")):
                        (lo, hi), (flo, fhi) = burden_interval(x, b[f"qhat_{t2}"], 0.01, normalized)
                        interval_coverage(x, lo, hi, flo, fhi, t2)
                    xs.append(compact(x))
                ext[(tag, v, m, fo, s)] = xs
        del r
    return rows, ext


def summarize(rows: list[dict]) -> dict:
    out = {}
    if not rows:
        return out
    if all("U" in x for x in rows):
        out["U"] = float(np.mean([x["U"] for x in rows]))
        ct = np.array([x["class_true"] for x in rows])
        cp = np.array([x["class_pred"] for x in rows])
        out["rec_se"] = _safe(((ct > 0) & (cp > 0)).sum(), (ct > 0).sum())
        out["rec_sp"] = _safe(((ct == 0) & (cp == 0)).sum(), (ct == 0).sum())
        out["class_acc"] = float(np.mean(ct == cp))
    nt = sum(x["n_true_eps"] for x in rows)
    npd = sum(x["n_pred_eps"] for x in rows)
    tp = sum(x["ep_match_0.5"] for x in rows)
    prec, rec = _safe(tp, npd), _safe(tp, nt)
    out["ep_f1_0.5"] = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    out["ep_prec_0.5"], out["ep_rec_0.5"] = prec, rec
    hours = sum(x["hours"] for x in rows)
    out["hours"] = hours
    out["fa_h"] = _safe(sum(x["false_eps"] for x in rows), hours)
    n_b = sum(x.get("n_true_on", 0) + x.get("n_true_off", 0) for x in rows)
    m_b = sum(x.get("on_match_1", 0) + x.get("off_match_1", 0) for x in rows)
    out["bnd_se_1"] = _safe(m_b, n_b)
    d = np.array([float(v) if v is not None else np.inf for x in rows for v in x.get("delays", [])], float)
    if len(d):
        fin = d[np.isfinite(d)]
        out["delay_median"] = float(np.median(np.maximum(fin, 0))) if len(fin) else float("nan")
        out["det_30s"] = float((d <= 30).mean())
    errs = [abs(x["B_hat"] - x["B"]) for x in rows if x.get("B") is not None and x.get("B_hat") is not None]
    out["burden_mae_pp"] = 100 * float(np.mean(errs)) if errs else float("nan")
    # EC57-style summaries
    if all("af_tp_s" in x for x in rows):
        tA = sum(x["true_af_known_s"] for x in rows)
        pA = sum(x["pred_af_known_s"] for x in rows)
        tpA = sum(x["af_tp_s"] for x in rows)
        known = sum(x["T_label_known"] for x in rows)
        out["dur_se"] = _safe(tpA, tA)
        out["dur_ppv"] = _safe(tpA, pA)
        out["dur_sp"] = _safe(known - tA - pA + tpA, known - tA)
        out["af_hours"] = tA / 3600.0
        th = [h for x in rows for h in x["true_ep_hit"]]
        pt = [t for x in rows for t in x["pred_ep_true"]]
        out["ep_se"] = float(np.mean(th)) if th else float("nan")
        out["ep_ppv"] = float(np.mean(pt)) if pt else float("nan")
        h30 = [h for x in rows for h, s in zip(x["true_ep_hit"], x["true_ep_s"]) if s >= 30]
        p30 = [t for x in rows for t, s in zip(x["pred_ep_true"], x["pred_ep_s"]) if s >= 30]
        out["n_true_30"] = len(h30)
        out["se30"] = float(np.mean(h30)) if h30 else float("nan")
        out["ppv30"] = float(np.mean(p30)) if p30 else float("nan")
        out["fa30_h"] = _safe(sum(1 for t in p30 if not t), hours)
    cov = [x["cov_full_norm"] for x in rows if x.get("cov_full_norm") is not None]
    out["cov_full_norm"] = float(np.mean(cov)) if cov else float("nan")
    for st in ("possible", "certain"):
        v = [x[f"cov_full_{st}_norm"] for x in rows if x.get(f"cov_full_{st}_norm") is not None]
        out[f"cov_full_{st}_norm"] = float(np.mean(v)) if v else float("nan")
    w = [x["int_norm"][3] - x["int_norm"][2] for x in rows if x.get("int_norm")]
    out["width_full_norm"] = float(np.mean(w)) if w else float("nan")
    out["records"] = len(rows)
    return out


def agreement(rows: list[dict]) -> dict:
    pairs = [(x["B_hat"], x["B"]) for x in rows if x.get("B") is not None and x.get("B_hat") is not None]
    if len(pairs) < 3:
        return {}
    p, t = (100 * np.array(v, float) for v in zip(*pairs))
    d = p - t
    bias, sd = float(d.mean()), float(d.std(ddof=1))
    sx, sy = p.var(), t.var()
    ccc = float(2 * np.mean((p - p.mean()) * (t - t.mean())) / (sx + sy + (p.mean() - t.mean()) ** 2)) \
        if sx + sy > 0 else float("nan")
    r = float(np.corrcoef(p, t)[0, 1]) if p.std() > 0 and t.std() > 0 else float("nan")
    return dict(ba_bias=bias, ba_lo=bias - 1.96 * sd, ba_hi=bias + 1.96 * sd, ccc=ccc, pearson=r, n=len(pairs))


def seed_stats(rows, variant, method, keys, filt=None, agree=False):
    seeds = sorted({s for (v, m, s) in rows if v == variant and m == method})
    per = []
    for s in seeds:
        rs = rows[(variant, method, s)]
        if filt is not None:
            rs = [x for x in rs if filt(x)]
        sm = summarize(rs)
        if agree:
            sm.update(agreement([x for x in rs if x.get("preselected")]))
        per.append(sm)
    allk = set(keys) | ({"ba_bias", "ba_lo", "ba_hi", "ccc", "pearson"} if agree else set())
    return {k: mean_sd([p.get(k) for p in per]) for k in allk}


def ext_stats(ext, tag, variant, method, keys, folds=None, agree=False):
    items = [(f, s, xs) for (t, v, m, f, s), xs in ext.items() if (t, v, m) == (tag, variant, method)]
    if not items:
        return None
    if tag.startswith("noise"):
        by_seed = defaultdict(list)
        for f, s, xs in items:
            by_seed[s].extend(x for x in xs if folds[f]["roles"][x["patient"]] == "test")
        per = [summarize(xs) for xs in by_seed.values()]
    else:
        per = []
        for f, s, xs in items:
            if tag == "mitdb":
                xs = [x for x in xs if MITDB_NON_AF(x)]
            sm = summarize(xs)
            if agree:
                sm.update(agreement(xs))
            per.append(sm)
    allk = set(keys) | ({"ba_bias", "ba_lo", "ba_hi", "ccc", "pearson"} if agree else set())
    return {k: mean_sd([p.get(k) for p in per]) for k in allk}


def patient_values(rows, variant, method, key):
    by = defaultdict(lambda: defaultdict(list))
    for (v, m, s), rs in rows.items():
        if (v, m) != (variant, method):
            continue
        for x in rs:
            by[x["patient"]][s].append(x)
    out = {}
    for p, seeds in by.items():
        vals = []
        for xs in seeds.values():
            if key == "fa_h":
                vals.append(sum(x["false_eps"] for x in xs) / max(sum(x["hours"] for x in xs), 1e-9))
            else:
                vals.append(float(np.mean([x[key] for x in xs])))
        out[p] = float(np.mean(vals))
    return out


def ext_patient_values(ext, tag, variant, method, key, folds):
    by = defaultdict(lambda: defaultdict(list))
    for (t, v, m, f, s), xs in ext.items():
        if (t, v, m) != (tag, variant, method):
            continue
        for x in xs:
            if tag == "mitdb" and not MITDB_NON_AF(x):
                continue
            if tag.startswith("noise") and folds[f]["roles"][x["patient"]] != "test":
                continue
            by[x["patient"]][(f, s)].append(x)
    out = {}
    for p, models in by.items():
        vals = []
        for xs in models.values():
            if key == "fa_h":
                h = sum(x["hours"] for x in xs)
                if h > 0:
                    vals.append(sum(x["false_eps"] for x in xs) / h)
            elif key == "U":
                us = [x["U"] for x in xs if x.get("U") is not None]
                if us:
                    vals.append(float(np.mean(us)))
            elif key == "burden_err":
                e = [abs(x["B_hat"] - x["B"]) for x in xs if x.get("B") is not None and x.get("B_hat") is not None]
                if e:
                    vals.append(float(np.mean(e)))
        if vals:
            out[p] = float(np.mean(vals))
    return out


def qrs_accuracy(prep_qrs: str) -> dict:
    out = {}
    for idx in sorted(glob.glob(os.path.join(prep_qrs, "*", "index.json"))):
        d = json.load(open(idx))
        ms = d["records"]
        tp, fn, fp = (sum(m[k] for m in ms) for k in ("qrs_tp", "qrs_fn", "qrs_fp"))
        lat = [m["qrs_latency_median_s"] for m in ms if m.get("qrs_latency_median_s") is not None]
        p95 = [m["qrs_latency_p95_s"] for m in ms if m.get("qrs_latency_p95_s") is not None]
        out[os.path.basename(os.path.dirname(idx))] = dict(
            records=len(ms), ref_beats=sum(m["ref_beats"] for m in ms), det_beats=sum(m["n_beats"] for m in ms),
            se=_safe(tp, tp + fn), ppv=_safe(tp, tp + fp), tp=tp, fn=fn, fp=fp,
            latency_median_s=float(np.median(lat)) if lat else None,
            latency_p95_s=float(np.median(p95)) if p95 else None,
            outside_span=sum(m.get("det_outside_annotated_span", 0) for m in ms),
            blocks=sum(m["n_blocks"] for m in ms), hours=sum(m["duration_s"] for m in ms) / 3600,
            af_beats=sum(m["af_beats"] for m in ms), ref_af_beats=sum(m["ref_af_beats"] for m in ms))
    return out


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, (np.floating, np.integer)):
        return jsonable(o.item())
    return o


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ext-ref", default="")
    ap.add_argument("--qrs", default="")
    ap.add_argument("--gatex", default="")
    ap.add_argument("--integrity", required=True)
    ap.add_argument("--prep-qrs", default="")
    ap.add_argument("--folds", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    folds = json.load(open(a.folds))["folds"]
    S = {}
    KEYS = ["U", "rec_se", "rec_sp", "class_acc", "ep_f1_0.5", "ep_prec_0.5", "ep_rec_0.5", "fa_h", "bnd_se_1",
            "delay_median", "det_30s", "burden_mae_pp", "dur_se", "dur_ppv", "dur_sp", "af_hours", "ep_se", "ep_ppv",
            "se30", "ppv30", "fa30_h", "n_true_30", "cov_full_norm", "cov_full_possible_norm",
            "cov_full_certain_norm", "width_full_norm", "records"]
    TAGS_EXT = ("afdb", "ltafdb", "nsrdb", "mitdb")
    NOISE = tuple(f"noise_{k}_{s}" for k in ("em", "ma", "bw") for s in (12, 6, 0)) + ("noise_clean",)
    # ---- (A) standard summaries on the reference-peak track
    if a.ext_ref:
        rows, ext = load_dir(a.ext_ref)
        S["ext_ref"] = {"cpsc": {}, "external": {}}
        for (v, m) in sorted({(v, m) for (v, m, s) in rows}):
            S["ext_ref"]["cpsc"][f"{v}:{m}"] = seed_stats(rows, v, m, KEYS, agree=True)
        for tag in TAGS_EXT:
            S["ext_ref"]["external"][tag] = {}
            for (v, m) in sorted({(v, m) for (t, v, m, f, s) in ext if t == tag}):
                S["ext_ref"]["external"][tag][f"{v}:{m}"] = ext_stats(ext, tag, v, m, KEYS, folds, agree=True)
        # replay identity check against the integrity outputs (same locked settings)
        S["ext_ref"]["replay_check"] = {}
        ref_rows, _ = load_dir(a.integrity, want=lambda v, m: (v, m) in {(vv, mm) for (vv, mm, _) in rows}, ext_tags=())
        for key in sorted(rows):
            if key not in ref_rows:
                continue
            A = {(x["record"]): x for x in rows[key]}
            B = {(x["record"]): x for x in ref_rows[key]}
            diffs = 0
            for rec, x in A.items():
                y = B.get(rec)
                if y is None or any(abs((x.get(k) or 0) - (y.get(k) or 0)) > 1e-9
                                    for k in ("U", "false_eps", "ep_match_0.5", "B_hat")):
                    diffs += 1
            S["ext_ref"]["replay_check"]["|".join(map(str, key))] = dict(records=len(A), differing=diffs)
        del ref_rows
        ref_cpsc_rows, ref_ext = rows, ext
    # ---- (B) end-to-end track with the causal QRS detector
    if a.qrs:
        qrows, qext = load_dir(a.qrs)
        S["qrs"] = {"cpsc": {}, "external": {}, "noise": {}, "stats": {}}
        for (v, m) in sorted({(v, m) for (v, m, s) in qrows}):
            S["qrs"]["cpsc"][f"{v}:{m}"] = seed_stats(qrows, v, m, KEYS, agree=True)
        for tag in TAGS_EXT:
            S["qrs"]["external"][tag] = {f"{v}:{m}": ext_stats(qext, tag, v, m, KEYS, folds, agree=True)
                                         for (v, m) in sorted({(v, m) for (t, v, m, f, s) in qext if t == tag})}
        for tag in NOISE:
            S["qrs"]["noise"][tag] = {f"{v}:{m}": ext_stats(qext, tag, v, m, KEYS, folds)
                                      for (v, m) in sorted({(v, m) for (t, v, m, f, s) in qext if t == tag})}
        st = S["qrs"]["stats"]
        for key in ("U", "fa_h"):
            g = patient_values(qrows, "gate", "gate_tuned", key)
            c = patient_values(qrows, "full", "cusum", key)
            e = patient_values(qrows, "full", "cera_tuned", key)
            sm = patient_values(qrows, "full", "smooth", key)
            st[f"cpsc|{key}|gate_vs_cusum"] = paired_test(g, c)
            st[f"cpsc|{key}|cera_vs_cusum"] = paired_test(e, c)
            st[f"cpsc|{key}|cera_vs_smooth"] = paired_test(e, sm)
            # same method, detected versus reference peaks (patient level)
            if a.ext_ref:
                for vm in (("full", "cera_tuned"), ("full", "cusum"), ("full", "smooth"), ("gate", "gate_tuned")):
                    st[f"cpsc|{key}|{vm[1]}|qrs_vs_ref"] = paired_test(patient_values(qrows, *vm, key),
                                                                         patient_values(ref_cpsc_rows, *vm, key))
        for tag in TAGS_EXT + NOISE:
            for key in ("fa_h", "U", "burden_err"):
                g = ext_patient_values(qext, tag, "gate", "gate_tuned", key, folds)
                c = ext_patient_values(qext, tag, "full", "cusum", key, folds)
                if len(g) >= 5 and len(c) >= 5:
                    st[f"{tag}|{key}|gate_vs_cusum"] = paired_test(g, c)
            if tag.startswith("noise"):
                for comp in ("cusum", "smooth"):
                    e = ext_patient_values(qext, tag, "full", "cera_tuned", "fa_h", folds)
                    c = ext_patient_values(qext, tag, "full", comp, "fa_h", folds)
                    if len(e) >= 5 and len(c) >= 5:
                        st[f"{tag}|fa_h|cera_vs_{comp}"] = paired_test(e, c)
        del qrows, qext
    if a.prep_qrs:
        S["qrs_accuracy"] = qrs_accuracy(a.prep_qrs)
    # ---- (C) certification of further proposers (secondary)
    if a.gatex:
        grows, gext = load_dir(a.gatex)
        base_want = lambda v, m: (v, m) in {("full", "smooth"), ("full", "percentile"), ("full", "bocpd"),
                                            ("rr_lstm", "rr_lstm")}
        brows, bext = load_dir(a.integrity, want=base_want, ext_tags=set(TAGS_EXT + NOISE))
        S["gatex"] = {"cpsc": {}, "external": {}, "noise": {}, "stats": {}, "tuning": {}}
        pairs = {"smooth": ("full", "smooth"), "percentile": ("full", "percentile"), "bocpd": ("full", "bocpd"),
                 "rrlstm": ("rr_lstm", "rr_lstm")}
        for (v, m) in sorted({(v, m) for (v, m, s) in grows}) + sorted(set(pairs.values())):
            src = grows if v == "gatex" else brows
            S["gatex"]["cpsc"][f"{v}:{m}"] = seed_stats(src, v, m, KEYS)
        for tag in TAGS_EXT + NOISE:
            dest = S["gatex"]["noise" if tag.startswith("noise") else "external"].setdefault(tag, {})
            for (v, m) in sorted({(v, m) for (t, v, m, f, s) in gext if t == tag}):
                dest[f"{v}:{m}"] = ext_stats(gext, tag, v, m, KEYS, folds)
            for (v, m) in sorted({(v, m) for (t, v, m, f, s) in bext if t == tag}):
                dest[f"{v}:{m}"] = ext_stats(bext, tag, v, m, KEYS, folds)
        st = S["gatex"]["stats"]
        for name, (bv, bm) in pairs.items():
            for suffix in ("", "_A1121"):
                gm = f"gx_{name}{suffix}"
                if not any(m == gm for (_, m, _) in grows):
                    continue
                seeds = sorted({s for (v, m, s) in grows if m == gm})
                base_rows = {k: v for k, v in brows.items() if k[2] in seeds}
                for key in ("U", "fa_h"):
                    st[f"cpsc|{key}|{gm}_vs_{name}"] = paired_test(patient_values(grows, "gatex", gm, key),
                                                                  patient_values(base_rows, bv, bm, key))
                for tag in TAGS_EXT + NOISE:
                    for key in ("fa_h", "U"):
                        gvals = ext_patient_values(gext, tag, "gatex", gm, key, folds)
                        bvals = ext_patient_values({k: v for k, v in bext.items() if k[4] in seeds},
                                                   tag, bv, bm, key, folds)
                        if len(gvals) >= 5 and len(bvals) >= 5:
                            st[f"{tag}|{key}|{gm}_vs_{name}"] = paired_test(gvals, bvals)
        for path in sorted(glob.glob(os.path.join(a.gatex, "fold*_gatex_s*.json.gz"))):
            with gzip.open(path, "rt") as f:
                d = json.load(f)
            S["gatex"]["tuning"][os.path.basename(path)] = dict(configs=d["configs"], tuning=d["tuning"])
    with open(a.out, "w") as f:
        json.dump(jsonable(S), f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
