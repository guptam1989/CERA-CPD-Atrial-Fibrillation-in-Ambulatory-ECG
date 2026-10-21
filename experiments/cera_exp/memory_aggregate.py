"""Aggregate the wearer-memory evaluation (memory_eval.py) into one compact summary.

python -m experiments.cera_exp.memory_aggregate --memory RESULTS/memory --folds folds.json --out memory_summary.json

Conventions follow revision_aggregate.py: CPSC2021 metrics are pooled over the five
test folds per training seed (mean/s.d. over seeds); external cohorts are summarised
per (fold, seed) model and averaged over models; noise copies use only each
patient's outer-test models; paired comparisons use one value per patient
(sign-flip test, bootstrap interval), with Holm adjustment within the primary family.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
from collections import Counter, defaultdict

from .aggregate import holm, paired_test
from .evaluate import burden_interval, interval_coverage
from .revision_aggregate import (compact, ext_patient_values, ext_stats, jsonable, patient_values,
                                 seed_stats)

KEYS = ["U", "rec_se", "rec_sp", "class_acc", "ep_f1_0.5", "ep_prec_0.5", "ep_rec_0.5", "fa_h", "bnd_se_1",
        "delay_median", "det_30s", "burden_mae_pp", "dur_se", "dur_ppv", "dur_sp", "ep_se", "ep_ppv",
        "se30", "ppv30", "fa30_h", "cov_full_norm", "cov_full_possible_norm", "cov_full_certain_norm",
        "width_full_norm", "records"]
TAGS_EXT = ("afdb", "ltafdb", "nsrdb", "mitdb")
NOISE = tuple(f"noise_{k}_{s}" for k in ("em", "ma", "bw") for s in (12, 6, 0)) + ("noise_clean",)
V = "memory"
PRIMARY = [("cera_m", "cera_tuned"), ("cera_md", "cera_tuned"), ("cera_m", "cusum"), ("cera_md", "cusum"),
           ("gate_m", "cusum"), ("gate_md", "cusum"), ("gate_m", "gate_tuned"), ("gate_md", "gate_tuned")]
SECONDARY = [("cera_md", "cera_m"), ("gate_md", "gate_m"), ("abl_margin", "cera_tuned"),
             ("abl_margin_dyadic", "cera_tuned"), ("abl_wearer_c0", "cera_tuned"), ("abl_record", "cera_tuned"),
             ("abl_gate_record", "gate_tuned"), ("gate_tuned", "cusum"), ("cera_m", "gate_tuned"),
             ("cera_md", "gate_tuned")]
CURVE_AC = (3, 10, 30, 100, 300, 1121, 11246)


def _with_coverage(xs, b):
    out = []
    for x in xs:
        for normalized, t2 in ((True, "norm"), (False, "abs")):
            (lo, hi), (flo, fhi) = burden_interval(x, b[f"qhat_{t2}"], 0.01, normalized)
            interval_coverage(x, lo, hi, flo, fhi, t2)
        out.append(compact(x))
    return out


def load(directory):
    rows, ext, qrows, qext = defaultdict(list), {}, defaultdict(list), {}
    configs, tuning = defaultdict(Counter), {}
    for path in sorted(glob.glob(os.path.join(directory, "fold*_memory_s*.json.gz"))):
        with gzip.open(path, "rt") as f:
            r = json.load(f)
        fo, s = r["fold"], r["seed"]
        for m, d in r["methods"].items():
            for x in d["test"]:
                c = compact(x)
                c["fold"] = fo
                rows[(V, m, s)].append(c)
            if not m.startswith(("gate_A", "gate_m_A", "gate_md_A", "cera_tuned", "cusum", "gate_tuned")):
                configs[m][json.dumps(d["params"], sort_keys=True)] += 1
            for tag, md in r.get("external", {}).items():
                if m in md:
                    ext[(tag, V, m, fo, s)] = _with_coverage(md[m], d["burden"])
        q = r.get("qrs")
        if q:
            for m, d in q["methods"].items():
                for x in d["test"]:
                    c = compact(x)
                    c["fold"] = fo
                    qrows[(V, m, s)].append(c)
                for tag, md in q["external"].items():
                    if m in md:
                        qext[(tag, V, m, fo, s)] = _with_coverage(md[m], d["burden"])
        tuning[os.path.basename(path)] = {k: v for k, v in r["configs"].items()}
        del r
    return rows, ext, qrows, qext, configs, tuning


def cpsc_tests(rows, pairs):
    out = {}
    for a, b in pairs:
        for key in ("U", "fa_h"):
            pa, pb = patient_values(rows, V, a, key), patient_values(rows, V, b, key)
            if len(pa) >= 5 and len(pb) >= 5:
                out[f"{a}_vs_{b}|{key}"] = paired_test(pa, pb)
    return out


def ext_tests(ext, tags, pairs, keys, folds):
    out = {}
    for tag in tags:
        for a, b in pairs:
            for key in keys:
                pa = ext_patient_values(ext, tag, V, a, key, folds)
                pb = ext_patient_values(ext, tag, V, b, key, folds)
                if len(pa) >= 5 and len(pb) >= 5:
                    out[f"{tag}|{a}_vs_{b}|{key}"] = paired_test(pa, pb)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--memory", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    folds = json.load(open(a.folds))["folds"]
    rows, ext, qrows, qext, configs, tuning = load(a.memory)
    S = {"cpsc": {}, "external": {}, "noise": {}, "qrs": {"cpsc": {}, "external": {}, "noise": {}},
         "stats": {}, "configs": {m: dict(c) for m, c in configs.items()}, "tuning": tuning}
    for m in sorted({m for (_, m, _) in rows}):
        S["cpsc"][m] = seed_stats(rows, V, m, KEYS, agree=True)
    st = cpsc_tests(rows, PRIMARY)
    adj = holm({k: v["p"] for k, v in st.items() if k.endswith("|U")})
    for k, p in adj.items():
        st[k]["p_holm"] = p
    st.update(cpsc_tests(rows, SECONDARY))
    curve_pairs = [(f"gate_md_A{x}", f"gate_A{x}") for x in CURVE_AC] + \
                  [(f"gate_m_A{x}", f"gate_A{x}") for x in CURVE_AC] + \
                  [(f"gate_md_A{x}", "cusum") for x in CURVE_AC]
    st.update(cpsc_tests(rows, curve_pairs))
    S["stats"]["cpsc"] = st
    transfer = sorted({m for (_, _, m, _, _) in ext})
    for tag in TAGS_EXT:
        S["external"][tag] = {m: ext_stats(ext, tag, V, m, KEYS, folds, agree=True) for m in transfer}
    for tag in NOISE:
        S["noise"][tag] = {m: ext_stats(ext, tag, V, m, KEYS, folds) for m in transfer}
    tpairs = [("cera_m", "cera_tuned"), ("cera_md", "cera_tuned"), ("cera_m", "cusum"), ("cera_md", "cusum"),
              ("gate_m", "cusum"), ("gate_md", "cusum"), ("gate_m", "gate_tuned"), ("gate_md", "gate_tuned"),
              ("gate_md_A1121", "gate_A1121"), ("gate_md_A1121", "cusum"), ("cera_md", "gate_tuned")]
    S["stats"]["external"] = ext_tests(ext, TAGS_EXT, tpairs, ("fa_h", "burden_err"), folds)
    S["stats"]["noise"] = ext_tests(ext, NOISE, tpairs, ("fa_h", "U"), folds)
    if qrows:
        for m in sorted({m for (_, m, _) in qrows}):
            S["qrs"]["cpsc"][m] = seed_stats(qrows, V, m, KEYS, agree=True)
        qtransfer = sorted({m for (_, _, m, _, _) in qext})
        for tag in TAGS_EXT:
            S["qrs"]["external"][tag] = {m: ext_stats(qext, tag, V, m, KEYS, folds, agree=True) for m in qtransfer}
        for tag in NOISE:
            S["qrs"]["noise"][tag] = {m: ext_stats(qext, tag, V, m, KEYS, folds) for m in qtransfer}
        S["stats"]["qrs_cpsc"] = cpsc_tests(qrows, tpairs)
        S["stats"]["qrs_external"] = ext_tests(qext, TAGS_EXT, tpairs, ("fa_h", "burden_err"), folds)
        S["stats"]["qrs_noise"] = ext_tests(qext, NOISE, tpairs, ("fa_h", "U"), folds)
    with open(a.out, "w") as f:
        json.dump(jsonable(S), f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
