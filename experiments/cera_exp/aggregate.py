"""Aggregate evaluation outputs into manuscript tables, statistics and figures.

python -m experiments.cera_exp.aggregate --results RESULTS_DIR --out paper_outputs [--sweeps SWEEP_DIR] [--sim sim.json]
Pooled metrics are computed over all test records of the five outer folds for
each training seed; tables report mean +/- SD over seeds. Paired comparisons
use one value per patient (records averaged within patient, then over seeds).
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

from .summary import summarize
from .evaluate import burden_interval, interval_coverage

DISPLAY = {
    "rr_cv": "CV of RR (Tateno--Glass type)",
    "rr_cosen": "COSEn (Lake--Moorman)",
    "rr_markov": "RR Markov model (Moody--Mark)",
    "rr_lstm": "RR-LSTM (causal)",
    "unet": "1-D U-Net segmenter (offline)",
    "offline": "Same encoder, offline smoothing",
    "smooth": "Same encoder + causal smoothing",
    "percentile": "Same encoder + percentile threshold",
    "cusum": "Same encoder + CUSUM",
    "bocpd": "Same encoder + BOCPD",
    "cera_tuned": "CERA-CPD (validation-selected $A$)",
    "cera": "CERA-CPD ($\\lambda^\\star=0.1$/h)",
    "cera_1h": "CERA-CPD ($\\lambda^\\star=1$/h)",
}
PRIMARY = "cera_tuned"
COMPARATORS = ["smooth", "cusum", "bocpd", "percentile", "rr_lstm", "rr_cosen", "rr_markov", "rr_cv"]
BURDEN_DIAGNOSTIC_KEYS = ["burden_mae_n", "label_known_frac", "partial_reference_records"] + [
    f"cov_{target}{status}_{tag}{count}"
    for target in ("obs", "full") for tag in ("norm", "abs")
    for status, count in (("_n", ""), ("_possible", ""), ("_certain", ""),
                          ("_possible", "_n"), ("_certain", "_n"))]


def load_results(directory: str) -> list[dict]:
    out = []
    for path in sorted(glob.glob(os.path.join(directory, "fold*_s*.json.gz"))):
        if path.endswith("_val.json.gz"):
            continue
        result = json.load(gzip.open(path, "rt"))
        if result.get("evaluation_schema_version") != 2:
            raise ValueError(f"Legacy evaluation at {path}; regenerate with evaluation_schema_version=2 before aggregation.")
        out.append(result)
    return out


def collect(results: list[dict]):
    """rows[(variant, method, seed)] -> list of CPSC test rows (all folds);
    ext[(tag, variant, method, fold, seed)] -> rows; configs[(variant, fold, seed)]."""
    rows, ext, configs, burden = defaultdict(list), {}, {}, defaultdict(list)
    for r in results:
        v, f, s = r["variant"], r["fold"], r["seed"]
        configs[(v, f, s)] = r.get("configs", {})
        for m, d in r["methods"].items():
            for x in d["test"]:
                x["fold"] = f
            rows[(v, m, s)].extend(d["test"])
            burden[(v, m, s)].append(dict(fold=f, **{k: d["burden"][k] for k in d["burden"]}))
        for tag, md in r.get("external", {}).items():
            for m, xs in md.items():
                # external intervals use the locked CPSC2021 quantile of the same fold/seed/method
                b = r["methods"][m]["burden"]
                for x in xs:
                    for normalized, t2 in ((True, "norm"), (False, "abs")):
                        (lo, hi), (flo, fhi) = burden_interval(x, b[f"qhat_{t2}"], 0.01, normalized)
                        interval_coverage(x, lo, hi, flo, fhi, t2)
                ext[(tag, v, m, f, s)] = xs
    return rows, ext, configs, burden


def mean_sd(values):
    v = np.array([x for x in values if x is not None and not (isinstance(x, float) and math.isnan(x))], float)
    if len(v) == 0:
        return (float("nan"), float("nan"), 0)
    return (float(v.mean()), float(v.std(ddof=1)) if len(v) > 1 else 0.0, len(v))


def per_seed_summaries(rows, variant, method, filt=None):
    seeds = sorted({s for (v, m, s) in rows if v == variant and m == method})
    out = []
    for s in seeds:
        rs = rows[(variant, method, s)]
        if filt is not None:
            rs = [x for x in rs if filt(x)]
        out.append(summarize(rs))
    return out


def table_stats(summaries, keys):
    return {k: mean_sd([s.get(k) for s in summaries]) for k in keys}


def patient_values(rows, variant, method, key="U"):
    """One value per patient: mean over the patient's records, then over seeds."""
    by = defaultdict(lambda: defaultdict(list))
    for (v, m, s), rs in rows.items():
        if v != variant or m != method:
            continue
        for x in rs:
            if key == "fa_h":
                by[x["patient"]][s].append((x["false_eps"], x["hours"]))
            else:
                by[x["patient"]][s].append(x[key])
    out = {}
    for p, seeds in by.items():
        vals = []
        for s, xs in seeds.items():
            if key == "fa_h":
                vals.append(sum(a for a, _ in xs) / max(sum(h for _, h in xs), 1e-9))
            else:
                vals.append(float(np.mean(xs)))
        out[p] = float(np.mean(vals))
    return out


def external_patient_values(ext, tag, variant, method, key, folds):
    """Paired external values: pooled exposure within patient/model, then mean over models.

    MIT-BIH uses precisely the non-AF population used for external tables.
    Noise copies use only each patient's outer-test models.
    """
    by = defaultdict(lambda: defaultdict(list))
    for (t, v, m, f, s), rows in ext.items():
        if (t, v, m) != (tag, variant, method):
            continue
        for row in rows:
            if tag == "mitdb" and row["n_true_eps"] != 0:
                continue
            if tag.startswith("noise") and folds[f]["roles"][row["patient"]] != "test":
                continue
            by[row["patient"]][(f, s)].append(row)
    out = {}
    for patient, models in by.items():
        vals = []
        for rows in models.values():
            if key == "fa_h":
                hours = sum(row["hours"] for row in rows)
                if hours > 0:
                    vals.append(sum(row["false_eps"] for row in rows) / hours)
            elif key == "burden_err":
                errs = [abs(row["B_hat"] - row["B"]) for row in rows
                        if row.get("B_hat") is not None and row.get("B") is not None]
                if errs:
                    vals.append(float(np.mean(errs)))
            elif key == "U":
                us = [row["U"] for row in rows if row.get("U") is not None]
                if us:
                    vals.append(float(np.mean(us)))
        if vals:
            out[patient] = float(np.mean(vals))
    return out


def paired_test(a: dict, b: dict, n_boot=2000, n_perm=20000, seed=0):
    pats = sorted(set(a) & set(b))
    d = np.array([a[p] - b[p] for p in pats])
    rng = np.random.default_rng(seed)
    boots = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    obs = abs(d.mean())
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    perm = np.abs((signs * d).mean(axis=1))
    p = (1 + (perm >= obs - 1e-12).sum()) / (n_perm + 1)
    return dict(n=len(d), mean=float(d.mean()), lo=float(lo), hi=float(hi), p=float(p))


def holm(pvals: dict) -> dict:
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        out[k] = running
    return out


def fmt(ms, digits=2, pct=False):
    m, s, n = ms
    if math.isnan(m):
        return "--"
    if pct:
        m, s = 100 * m, 100 * s
    if n <= 1:
        return f"{m:.{digits}f}"
    return f"{m:.{digits}f}$\\pm${s:.{digits}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sweeps", default="")
    ap.add_argument("--sim", default="")
    ap.add_argument("--folds", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    results = load_results(a.results)
    rows, ext, configs, burden = collect(results)
    summary = {"schema_version": 2, "reporting": {
        "burden_mae_target": "rhythm-labelled exposure only; missing reference values excluded with burden_mae_n",
        "cov_full_target": "exact full-record reference only; denominator cov_full_n_TAG",
        "cov_full_partial_identification": "certain <= true empirical coverage <= possible across all reference bounds",
        "external_fa_patient_pairing": "non-AF-only MITDB; pool exposure per patient/model, then average models",
    }, "main": {}, "stats": {}, "burden": {}, "external": {}, "ablation": {}, "noise": {}, "configs": {}}
    keys = ["U", "Ur", "class_acc", "on_se_1", "off_se_1", "on_ppv_1", "off_ppv_1", "bnd_se_1", "bnd_ppv_1", "bnd_se_2", "bnd_ppv_2", "ep_f1_0.5",
            "ep_f1_0.3", "ep_prec_0.5", "ep_rec_0.5", "loc_median", "delay_median", "detected", "det_30s", "fa_h",
            "burden_mae_pp", "obs_frac", "fallback_offsets"] + BURDEN_DIAGNOSTIC_KEYS
    methods_full = sorted({m for (v, m, s) in rows if v == "full"})
    for m in methods_full:
        summary["main"][m] = table_stats(per_seed_summaries(rows, "full", m), keys)
    for v in ("rr", "rr_lstm", "unet", "gate"):
        for m in sorted({m for (vv, m, s) in rows if vv == v}):
            summary["main"][m] = table_stats(per_seed_summaries(rows, v, m), keys)
    # paired statistics against the primary detector
    var_of = lambda m: "full" if m in methods_full else ("rr" if m.startswith("rr_") and m != "rr_lstm" else ("gate" if m.startswith("gate") else m))
    for key in ("U", "fa_h"):
        prim = patient_values(rows, "full", PRIMARY, key)
        res, pv = {}, {}
        for c in COMPARATORS + ["cera"]:
            if not any(mm == c for (_, mm, _) in rows):
                continue
            other = patient_values(rows, var_of(c), c, key)
            res[c] = paired_test(prim, other)
            pv[c] = res[c]["p"]
        adj = holm({k: v for k, v in pv.items() if k in COMPARATORS})
        for k in res:
            res[k]["p_holm"] = adj.get(k)
        summary["stats"][key] = res
    # burden coverage on pre-selected test recordings (one per patient)
    for m in methods_full + [x for x in summary["main"] if x not in methods_full]:
        v = var_of(m)
        per = []
        for s in sorted({s for (vv, mm, s) in rows if vv == v and mm == m}):
            pre = [x for x in rows[(v, m, s)] if x.get("preselected")]
            per.append(summarize(pre))
        summary["burden"][m] = table_stats(per, ["cov_obs_norm", "cov_full_norm", "width_obs_norm", "width_full_norm",
                                                  "cov_obs_abs", "cov_full_abs", "width_obs_abs", "width_full_abs",
                                                  "burden_mae_pp", "records"] + BURDEN_DIAGNOSTIC_KEYS)
        qs = [b for s in range(10) for b in burden.get((v, m, s), [])]
        summary["burden"][m]["qhat"] = {k: mean_sd([b[k] for b in qs if b.get(k) is not None and np.isfinite(b[k])])
                                        for k in ("qhat_norm", "qhat_abs", "M_norm")}
    # external cohorts: one summary per locked (fold, seed) model, mean +/- SD over models.
    # noise-stress sets: only each fold's own test patients, pooled over folds per seed.
    folds = json.load(open(a.folds))["folds"]
    tags = sorted({t for (t, *_rest) in ext})
    MITDB_AF = {"201", "202", "203", "210", "219", "221", "222"}
    ekeys = keys + ["cov_full_norm", "cov_full_abs", "width_full_norm", "width_full_abs", "cov_obs_norm", "cov_obs_abs"]
    for tag in tags:
        summary["external"][tag] = {}
        combos = sorted({(v, m) for (t, v, m, f, s) in ext if t == tag})
        for v, m in combos:
            items = [(f, s, xs) for (t, vv, mm, f, s), xs in ext.items() if t == tag and vv == v and mm == m]
            if tag.startswith("noise"):
                by_seed = defaultdict(list)
                for f, s, xs in items:
                    by_seed[s].extend(x for x in xs if folds[f]["roles"][x["patient"]] == "test")
                per = [summarize(xs) for xs in by_seed.values()]
            else:
                per = []
                for f, s, xs in items:
                    if tag == "mitdb":   # non-AF ectopy control: records without any AF/AFL annotation
                        xs = [x for x in xs if x["n_true_eps"] == 0]
                    per.append(summarize(xs))
            summary["external"][tag][f"{v}:{m}"] = table_stats(per, ekeys)
    # ablations: every (variant, CERA configuration) pair, plus ectopy-rich non-AF false-alarm rate
    for (v, m) in sorted({(v, m) for (v, m, s) in rows if m.startswith("cera")}):
        per, ect = [], []
        for s in sorted({s for (vv, mm, s) in rows if vv == v and mm == m}):
            rs = rows[(v, m, s)]
            per.append(summarize(rs))
            sub = [x for x in rs if x.get("class_true") == 0 and (x.get("ectopy_frac") or 0) >= 0.01]
            h = sum(x["hours"] for x in sub)
            ect.append(sum(x["false_eps"] for x in sub) / h if h else float("nan"))
        st = table_stats(per, ["U", "bnd_se_1", "ep_f1_0.5", "fa_h", "burden_mae_pp", "delay_median"])
        st["ectopy_fa_h"] = mean_sd(ect)
        summary["ablation"][f"{v}:{m}"] = st
    # secondary analysis: conformally gated CUSUM versus CUSUM and CERA-CPD (paired, patient level)
    if any(v == "gate" for (v, m, s) in rows):
        summary["stats_gate"] = {}
        for key in ("U", "fa_h"):
            cus = patient_values(rows, "full", "cusum", key)
            cer = patient_values(rows, "full", PRIMARY, key)
            for gm in sorted({m for (v, m, s) in rows if v == "gate"}):
                g = patient_values(rows, "gate", gm, key)
                summary["stats_gate"][f"{gm}|{key}|vs_cusum"] = paired_test(g, cus)
                summary["stats_gate"][f"{gm}|{key}|vs_cera"] = paired_test(g, cer)
    # noise robustness: per-patient false episodes per hour on each noisy copy (own-fold models, mean over seeds)
    summary["noise_stats"] = {}
    for tag in sorted({t for (t, *_r) in ext if t.startswith("noise_")}):
        def per_patient(v, m):
            return external_patient_values(ext, tag, v, m, "fa_h", folds)
        cera_fa = per_patient("full", PRIMARY)
        for comp in (("full", "cusum"), ("full", "smooth"), ("gate", "gate_tuned")):
            other = per_patient(*comp)
            if other and cera_fa:
                summary["noise_stats"][f"{tag}|cera_vs_{comp[1]}"] = paired_test(cera_fa, other)
    # gated variant versus CUSUM on external and noise cohorts (patient-level, mean over models/seeds)
    summary["ext_stats"] = {}
    for tag in sorted({t for (t, *_r) in ext}):
        for key in ("fa_h", "burden_err", "U"):
            g = external_patient_values(ext, tag, "gate", "gate_tuned", key, folds)
            cu = external_patient_values(ext, tag, "full", "cusum", key, folds)
            if len(g) >= 5 and len(cu) >= 5:
                summary["ext_stats"][f"{tag}|{key}|gate_vs_cusum"] = paired_test(g, cu)
    summary["configs"] = {f"{v}|{f}|{s}": c for (v, f, s), c in configs.items()}
    json.dump(summary, open(os.path.join(a.out, "summary.json"), "w"), indent=1, default=float)
    print("wrote", os.path.join(a.out, "summary.json"))


if __name__ == "__main__":
    main()
