"""Supplementary tables: encoder selection, validation-selected detector settings,
noise results and duration-stratified detection.

python -m experiments.cera_exp.supp_tables --summary agg/summary.json --selection selection.json \
    --results RESULTS --out tables
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
from collections import defaultdict

import numpy as np

from .summary import summarize


def tab_selection(sel: dict) -> str:
    names = [("base", "No augmentation"), ("aug", "Augmentation"), ("augreg", "Aug.\\ + dropout/WD"),
             ("augregema", "Aug.\\ + dropout/WD + EMA")]
    lines = ["\\begin{table}[!t]", "\\centering",
             "\\caption{Nested encoder selection: validation fused BCE of each candidate (seed 0); the selected candidate is in bold.}",
             "\\label{tab:s-selection}", "\\footnotesize", "\\begin{tabular}{@{}lccccc@{}}", "\\toprule",
             "Candidate & Fold 1 & Fold 2 & Fold 3 & Fold 4 & Fold 5\\\\", "\\midrule"]
    for key, lab in names:
        cells = []
        for f in range(5):
            v = sel[str(f)]["val_bce"].get(key)
            c = f"{v:.3f}" if v is not None else "--"
            if sel[str(f)]["selected"] == key:
                c = "\\textbf{" + c + "}"
            cells.append(c)
        lines.append(lab + " & " + " & ".join(cells) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def tab_configs(S: dict) -> str:
    cfg = S["configs"]
    rows = defaultdict(list)
    for key, c in cfg.items():
        v, f, s = key.split("|")
        if v != "full":
            continue
        rows["A"].append(c["cera_tuned"]["A"])
        rows["smooth"].append((c["smooth"]["k"], c["smooth"]["t_on"], c["smooth"]["t_off"]))
        rows["cusum"].append((c["cusum"]["drift"], c["cusum"]["h"]))
        rows["bocpd"].append((c["bocpd"]["hazard"], c["bocpd"]["threshold"]))

    def mode(xs):
        vals, counts = np.unique([json.dumps(x) for x in xs], return_counts=True)
        order = np.argsort(-counts)
        return "; ".join(f"{vals[i]} ({counts[i]})".replace("[", "(").replace("]", ")") for i in order[:3])
    lines = ["\\begin{table}[!t]", "\\centering",
             "\\caption{Validation-selected detector parameters over the 25 (fold, seed) models: most frequent settings with counts.}",
             "\\label{tab:s-configs}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{@{}lp{6.0cm}@{}}", "\\toprule", "Detector & Selected setting (count)\\\\", "\\midrule",
             f"\\method{{}} threshold $A$ & {mode(rows['A'])}\\\\",
             f"Smoothing ($k$, $t_{{\\mathrm{{on}}}}$, $t_{{\\mathrm{{off}}}}$) & {mode(rows['smooth'])}\\\\",
             f"CUSUM (drift, $h$) & {mode(rows['cusum'])}\\\\",
             f"BOCPD (hazard, threshold) & {mode(rows['bocpd'])}\\\\",
             "\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def tab_noise(S: dict) -> str:
    ext = S["external"]
    meth = [("full:cera_tuned", "\\method{}"), ("equal_gate:cera_tuned", "\\method{}, equal weights"),
            ("full:cusum", "CUSUM"), ("full:smooth", "Smoothing")]
    conds = [("noise_clean", "clean")] + [(f"noise_{k}_{s}", f"{k} {s}\\,dB") for k in ("em", "ma", "bw") for s in (12, 6, 0)]
    lines = ["\\begin{table*}[!t]", "\\centering",
             "\\caption{Noise stress on the 105 pre-selected CPSC2021 recordings: episode F1 (\\%) / false episodes per hour "
             "(mean over seeds; each recording scored only by its own fold's models).}",
             "\\label{tab:s-noise}", "\\footnotesize", "\\setlength{\\tabcolsep}{2.5pt}",
             "\\begin{tabular}{@{}l" + "c" * len(conds) + "@{}}", "\\toprule",
             "Method & " + " & ".join(c for _, c in conds) + "\\\\", "\\midrule"]
    for key, lab in meth:
        cells = []
        for tag, _ in conds:
            d = ext.get(tag, {}).get(key)
            cells.append(f"{100 * d['ep_f1_0.5'][0]:.1f}/{d['fa_h'][0]:.2f}" if d else "--")
        lines.append(lab + " & " + " & ".join(cells) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table*}"]
    return "\n".join(lines)


def tab_duration(results: str) -> str:
    meth = [("cera_tuned", "\\method{}"), ("cusum", "CUSUM"), ("smooth", "Smoothing"), ("bocpd", "BOCPD")]
    per = defaultdict(list)
    for p in sorted(glob.glob(os.path.join(results, "fold*_full_s*.json.gz"))):
        d = json.load(gzip.open(p, "rt"))
        for m, _ in meth:
            per[(m, d["seed"])].extend(d["methods"][m]["test"])
    bins = [(5, "5--19"), (20, "20--99"), (100, "100--999"), (1000, "$\\ge$1000")]
    lines = ["\\begin{table}[!t]", "\\centering",
             "\\caption{CPSC2021 reference episodes overlapped by a predicted episode (\\%) and median delay of alarm-detected "
             "episodes (s), by episode duration in beats (mean over five seeds).}",
             "\\label{tab:s-duration}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{@{}lcccc@{}}", "\\toprule",
             "Method & " + " & ".join(b for _, b in bins) + "\\\\", "\\midrule"]
    counts = None
    for m, lab in meth:
        sums = [summarize(per[(m, s)]) for s in range(5) if per.get((m, s))]
        cells = []
        for lo, _ in bins:
            det = np.mean([x[f"det_rate_{lo}"] for x in sums])
            dl = np.nanmean([x[f"delay_med_{lo}"] for x in sums])
            cells.append(f"{100 * det:.1f} / {dl:.0f}")
        counts = [sums[0][f"n_eps_{lo}"] for lo, _ in bins]
        lines.append(lab + " & " + " & ".join(cells) + "\\\\")
    lines.insert(-0, "")
    lines += ["\\midrule", "Episodes ($n$) & " + " & ".join(str(c) for c in counts) + "\\\\",
              "\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(l for l in lines if l != "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--selection", required=True)
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S = json.load(open(a.summary))
    sel = json.load(open(a.selection))
    for name, text in [("tab_s_selection", tab_selection(sel)), ("tab_s_configs", tab_configs(S)),
                       ("tab_s_noise", tab_noise(S)), ("tab_s_duration", tab_duration(a.results))]:
        open(os.path.join(a.out, name + ".tex"), "w").write(text)
        print("wrote", name)


if __name__ == "__main__":
    main()
