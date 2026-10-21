"""LaTeX tables and result macros from aggregate summary.json.

python -m experiments.cera_exp.tables --summary agg/summary.json --cost cost.json --sim sim.json --out tables_dir
Writes one .tex file per table plus results_macros.tex (\\newcommand per headline number).
"""
from __future__ import annotations

import argparse
import json
import math
import os


def f(ms, d=2, pct=False, sd=True):
    if ms is None:
        return "--"
    m, s, n = ms
    if m is None or (isinstance(m, float) and math.isnan(m)):
        return "--"
    if pct:
        m, s = 100 * m, 100 * s
    if n <= 1 or not sd:
        return f"{m:.{d}f}"
    return f"{m:.{d}f}\\,$\\pm$\\,{s:.{d}f}"


def bold_best(cells, values, higher=True):
    vals = [v for v in values if v is not None and not math.isnan(v)]
    if not vals:
        return cells
    best = max(vals) if higher else min(vals)
    return [("\\textbf{" + c + "}" if v is not None and not math.isnan(v) and abs(v - best) < 1e-12 else c)
            for c, v in zip(cells, values)]


MAIN_ROWS = [
    ("Rhythm features", [("rr_cv", "CV of RR \\cite{tateno2001}"), ("rr_cosen", "COSEn \\cite{lake2011cosen}"),
                         ("rr_markov", "RR Markov \\cite{moody1983afdb}")]),
    ("Deep sequence", [("rr_lstm", "RR-LSTM, causal \\cite{faust2018}"), ("unet", "1-D U-Net, offline$^\\dagger$ \\cite{wen2022paf}")]),
    ("Same encoder", [("offline", "Median filter, offline$^\\dagger$"), ("smooth", "Causal smoothing + hysteresis"),
                      ("percentile", "Percentile threshold"), ("cusum", "CUSUM \\cite{page1954}"),
                      ("bocpd", "BOCPD \\cite{adams2007bocpd}")]),
    ("Proposed", [("cera_1h", "\\method{}, $\\lambda^\\star=1$/h"), ("cera", "\\method{}, $\\lambda^\\star=0.1$/h"),
                  ("cera_tuned", "\\method{}, validated $A$")]),
]
MAIN_COLS = [("U", "$U$", 2, False, True), ("Ur", "$U_r$", 2, False, True), ("on_se_1", "On.\\ Se$_{\\pm1}$", 1, True, True),
             ("off_se_1", "Off.\\ Se$_{\\pm1}$", 1, True, True), ("bnd_ppv_1", "Bnd.\\ PPV$_{\\pm1}$", 1, True, True),
             ("ep_f1_0.5", "Ep.\\ F1", 1, True, True), ("loc_median", "Loc.$\\downarrow$", 1, False, False),
             ("delay_median", "Delay$\\downarrow$", 1, False, False), ("fa_h", "FA/h$\\downarrow$", 2, False, False),
             ("burden_mae_pp", "Burden$\\downarrow$", 1, False, False)]


def table_main(S):
    main = S["main"]
    lines = ["\\begin{table*}[!t]", "\\centering",
             "\\caption{Episode delineation on CPSC2021 (patient-wise five-fold cross-validation, 1425 recordings, 105 patients). "
             "Metrics are pooled over the five test folds per training seed; mean\\,$\\pm$\\,s.d.\\ over five seeds "
             "(three for the deep sequence models; classical detectors are deterministic). Se/PPV and F1 in \\%; Loc.: median absolute "
             "boundary error (beats); Delay: median alarm delay (s); FA/h: false episodes per hour; Burden: mean absolute burden "
             "error (percentage points). $^\\dagger$Offline methods use future context and have no alarm delay. "
             "Best online value per column in bold.}",
             "\\label{tab:cpsc}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}", "\\resizebox{\\textwidth}{!}{%",
             "\\begin{tabular}{@{}ll" + "c" * len(MAIN_COLS) + "@{}}", "\\toprule",
             "Group & Method & " + " & ".join(c[1] for c in MAIN_COLS) + "\\\\", "\\midrule"]
    online = [k for g, rows in MAIN_ROWS for k, _ in rows if k not in ("unet", "offline") and k in main]
    best = {}
    for key, _, d, pct, hi in MAIN_COLS:
        vals = [main[k][key][0] for k in online if main[k].get(key) and main[k][key][0] is not None]
        vals = [v for v in vals if not math.isnan(v)]
        if vals:
            best[key] = (max(vals) if hi else min(vals))
    for gi, (group, rows) in enumerate(MAIN_ROWS):
        rows = [(k, n) for k, n in rows if k in main]
        for i, (k, name) in enumerate(rows):
            cells = []
            for key, _, d, pct, hi in MAIN_COLS:
                ms = main[k].get(key)
                c = f(ms, d, pct)
                if key in ("delay_median",) and k in ("unet", "offline"):
                    c = "--"
                if k in online and key in best and ms and ms[0] is not None and not math.isnan(ms[0]) \
                        and round(ms[0] * (100 if pct else 1), d) == round(best[key] * (100 if pct else 1), d):
                    c = "\\textbf{" + c + "}"
                cells.append(c)
            lead = f"\\multirow{{{len(rows)}}}{{*}}{{{group}}}" if i == 0 else ""
            lines.append(f"{lead} & {name} & " + " & ".join(cells) + "\\\\")
        if gi < len(MAIN_ROWS) - 1:
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
    return "\n".join(lines)


def table_stats(S):
    st = S["stats"]
    names = dict(smooth="Causal smoothing", cusum="CUSUM", bocpd="BOCPD", percentile="Percentile threshold",
                 rr_lstm="RR-LSTM", rr_cosen="COSEn", rr_markov="RR Markov", rr_cv="CV of RR",
                 cera="\\method{}, $\\lambda^\\star=0.1$/h")
    lines = ["\\begin{table}[!t]", "\\centering",
             "\\caption{Patient-level paired comparisons of \\method{} (validated $A$) with each comparator on CPSC2021: "
             "mean difference (proposed minus comparator) with 95\\% patient bootstrap interval and Holm-adjusted two-sided "
             "sign-flip $p$ ($n=105$ patients; the last row is unadjusted).}",
             "\\label{tab:stats}", "\\footnotesize", "\\setlength{\\tabcolsep}{2.5pt}", "\\resizebox{\\columnwidth}{!}{%",
             "\\begin{tabular}{@{}lcccc@{}}", "\\toprule",
             " & \\multicolumn{2}{c}{$\\Delta U$} & \\multicolumn{2}{c}{$\\Delta$FA/h} \\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
             "Comparator & $\\Delta$ [95\\% CI] & $p_{\\mathrm{Holm}}$ & $\\Delta$ [95\\% CI] & $p_{\\mathrm{Holm}}$ \\\\", "\\midrule"]

    def pfmt(p):
        if p is None:
            return "--"
        return "$<$0.001" if p < 0.001 else f"{p:.3f}"
    for k in ["smooth", "cusum", "bocpd", "percentile", "rr_lstm", "rr_cosen", "rr_markov", "rr_cv", "cera"]:
        if k not in st.get("U", {}):
            continue
        u, fa = st["U"][k], st["fa_h"].get(k)
        pu = u.get("p_holm") if k != "cera" else u["p"]
        pf = (fa.get("p_holm") if k != "cera" else fa["p"]) if fa else None
        cell_fa = f"{fa['mean']:+.2f} [{fa['lo']:+.2f}, {fa['hi']:+.2f}]" if fa else "--"
        lines.append(f"{names[k]} & {u['mean']:+.2f} [{u['lo']:+.2f}, {u['hi']:+.2f}] & {pfmt(pu)} & {cell_fa} & {pfmt(pf)}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    return "\n".join(lines)


EXT_METHODS = [("rr:rr_cosen", "COSEn"), ("rr:rr_markov", "RR Markov"), ("rr_lstm:rr_lstm", "RR-LSTM"),
               ("unet:unet", "1-D U-Net$^\\dagger$"), ("full:smooth", "Causal smoothing"), ("full:cusum", "CUSUM"),
               ("full:bocpd", "BOCPD"), ("full:cera", "\\method{}, $\\lambda^\\star=0.1$/h"),
               ("full:cera_tuned", "\\method{}, validated $A$")]


def table_external(S):
    ext = S["external"]
    lines = ["\\begin{table*}[!t]", "\\centering",
             "\\caption{Zero-shot external testing of the 25 locked CPSC2021 models (5 folds $\\times$ 5 seeds; "
             "three seeds for the sequence models), mean\\,$\\pm$\\,s.d.\\ over models. No external data were used for "
             "training, tuning or calibration. Bnd.\\ Se: boundary sensitivity within $\\pm2$ beats (\\%); FA/h: false episodes "
             "per hour of rhythm-labelled exposure; Burden: mean absolute error on rhythm-labelled time (percentage points); "
             "Cov./width: full-record coverage and mean interval width (\\%) of 90\\% intervals calibrated on CPSC2021 patients. "
             "For incomplete reference annotations, coverage is bounded by the percentages whose intervals contain the entire "
             "reference burden range or intersect that range, respectively; $^\\dagger$offline.}",
             "\\label{tab:external}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}", "\\resizebox{\\textwidth}{!}{%",
             "\\begin{tabular}{@{}l" + "ccccc" * 2 + "@{}}", "\\toprule",
             " & \\multicolumn{5}{c}{AFDB (23 recordings, 234\\,h)} & \\multicolumn{5}{c}{LTAFDB (84 recordings, 1961\\,h)}\\\\",
             "\\cmidrule(lr){2-6}\\cmidrule(lr){7-11}",
             "Method & Ep.\\ F1 & Bnd.\\ Se & FA/h & Burden & Cov./width & Ep.\\ F1 & Bnd.\\ Se & FA/h & Burden & Cov./width\\\\", "\\midrule"]
    for key, name in EXT_METHODS:
        cells = []
        for tag in ("afdb", "ltafdb"):
            d = ext.get(tag, {}).get(key)
            if not d:
                cells += ["--"] * 5
                continue
            cov = (f"{100 * d['cov_full_norm'][0]:.0f}/{100 * d['width_full_norm'][0]:.0f}"
                   if d.get("cov_full_norm") and d["cov_full_norm"][0] == d["cov_full_norm"][0] else "--")
            if d.get("partial_reference_records", [0])[0] > 0:
                lower = f(d.get("cov_full_certain_norm"), 1, True, sd=False)
                upper = f(d.get("cov_full_possible_norm"), 1, True, sd=False)
                width = f(d.get("width_full_norm"), 0, True, sd=False)
                cov = f"[{lower}, {upper}]/{width}"
            cells += [f(d.get("ep_f1_0.5"), 1, True), f(d.get("bnd_se_2"), 1, True), f(d.get("fa_h"), 2),
                      f(d.get("burden_mae_pp"), 1), cov]
        lines.append(name + " & " + " & ".join(cells) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
    return "\n".join(lines)


def table_controls(S):
    ext = S["external"]
    lines = ["\\begin{table}[!t]", "\\centering",
             "\\caption{False episodes per hour on change-free and ectopy-rich recordings (mean\\,$\\pm$\\,s.d.\\ over locked models).}",
             "\\label{tab:controls}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
             "\\begin{tabular}{@{}lcc@{}}", "\\toprule",
             "Method & NSRDB (437\\,h) & MIT-BIH non-AF (20\\,h)\\\\", "\\midrule"]
    for key, name in EXT_METHODS + [("no_ectopy:cera_tuned", "\\method{} w/o ectopy sampling")]:
        a = ext.get("nsrdb", {}).get(key)
        b = ext.get("mitdb", {}).get(key)
        if not a and not b:
            continue
        lines.append(f"{name} & {f(a.get('fa_h'), 2) if a else '--'} & {f(b.get('fa_h'), 2) if b else '--'}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


ABL = [("full", "cera_tuned", "Full model (validated $A$)"), ("no_atrial", "cera_tuned", "w/o atrial stream"),
       ("no_rhythm", "cera_tuned", "w/o rhythm stream"), ("equal_gate", "cera_tuned", "Equal weights (no gate)"),
       ("no_ectopy", "cera_tuned", "w/o ectopy-nuisance sampling"), ("no_matched", "cera_tuned", "w/o patient-matched negatives"),
       ("no_contrastive", "cera_tuned", "w/o contrastive loss"),
       ("full", "cera", "Full model, $\\lambda^\\star=0.1$/h (reference)$^\\ast$"),
       ("full", "cera_noclip", "w/o decision clipping$^\\ast$"), ("full", "cera_blockloc", "Block-level boundary only$^\\ast$"),
       ("full", "cera_power", "Power calibrator, $\\kappa=0.3^\\ast$"), ("full", "cera_restart", "Bag restarted at $\\hat k^\\ast$"),
       ("full", "cera_fallback", "Short-episode fallback$^\\ast$"), ("full", "cera_W2", "Warm-up $W=2^\\ast$"),
       ("full", "cera_W8", "Warm-up $W=8^\\ast$")]


def table_ablation(S):
    ab = S.get("ablation", {})
    lines = ["\\begin{table}[!t]", "\\centering",
             "\\caption{Ablations on CPSC2021 (pooled cross-validation, mean\\,$\\pm$\\,s.d.\\ over seeds; three seeds for encoder "
             "ablations). Encoder ablations use the validated threshold; detector ablations ($^\\ast$) use the full encoder at "
             "$\\lambda^\\star=0.1$/h. Ectopy FA/h: non-AF test recordings with $\\ge$1\\% ectopic beats.}",
             "\\label{tab:ablation}", "\\footnotesize", "\\setlength{\\tabcolsep}{2.5pt}", "\\resizebox{\\columnwidth}{!}{%",
             "\\begin{tabular}{@{}lccccc@{}}", "\\toprule",
             "Variant & $U$ & Bnd.\\ Se$_{\\pm1}$ & Ep.\\ F1 & FA/h & Ectopy FA/h\\\\", "\\midrule"]
    for v, m, name in ABL:
        d = ab.get(f"{v}:{m}")
        if not d:
            continue
        lines.append(f"{name} & {f(d.get('U'), 2)} & {f(d.get('bnd_se_1'), 1, True)} & {f(d.get('ep_f1_0.5'), 1, True)} & "
                     f"{f(d.get('fa_h'), 2)} & {f(d.get('ectopy_fa_h'), 2)}\\\\")
        if name == "w/o contrastive loss":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    return "\n".join(lines)


def table_burden(S):
    b = S["burden"]
    rows = [("smooth", "Causal smoothing"), ("cusum", "CUSUM"), ("bocpd", "BOCPD"), ("rr_lstm", "RR-LSTM"),
            ("cera", "\\method{}, $\\lambda^\\star=0.1$/h"), ("cera_tuned", "\\method{}, validated $A$")]
    lines = ["\\begin{table}[!t]", "\\centering",
             "\\caption{Burden on the 105 pre-selected CPSC2021 test recordings (one per patient): mean absolute error and "
             "coverage/mean width (percentage points) of 90\\% split-conformal intervals, normalised by the localisation scale "
             "($U+\\epsilon$) or unnormalised. Mean over seeds.}",
             "\\label{tab:burden}", "\\footnotesize", "\\setlength{\\tabcolsep}{2.5pt}", "\\resizebox{\\columnwidth}{!}{%",
             "\\begin{tabular}{@{}lccccc@{}}", "\\toprule",
             " & & \\multicolumn{2}{c}{Normalised} & \\multicolumn{2}{c}{Unnormalised}\\\\",
             "\\cmidrule(lr){3-4}\\cmidrule(lr){5-6}",
             "Method & MAE & Cov. (\\%) & Width & Cov. (\\%) & Width\\\\", "\\midrule"]
    for k, name in rows:
        d = b.get(k)
        if not d:
            continue
        lines.append(f"{name} & {f(d['burden_mae_pp'], 1, sd=False)} & {f(d['cov_full_norm'], 1, True, sd=False)} & "
                     f"{f(d['width_full_norm'], 1, True, sd=False)} & {f(d['cov_full_abs'], 1, True, sd=False)} & "
                     f"{f(d['width_full_abs'], 1, True, sd=False)}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table}"]
    return "\n".join(lines)


def table_gate(S):
    main, ext, st = S["main"], S["external"], S.get("ext_stats", {})
    rows = [("full", "cusum", "Same encoder + CUSUM (best baseline)"), ("full", "cera_tuned", "\\method{} (standalone, validated $A$)"),
            ("gate", "gate_tuned", "\\method{}-G (gated, validated)"), ("gate", "gate_A1121", "\\method{}-G, certified $A_c=1121$")]
    cols = [("cpsc", "U", 2, False, True), ("cpsc", "ep_f1_0.5", 1, True, True), ("cpsc", "fa_h", 3, False, False),
            ("afdb", "ep_f1_0.5", 1, True, True), ("afdb", "fa_h", 3, False, False), ("ltafdb", "fa_h", 3, False, False),
            ("nsrdb", "fa_h", 3, False, False), ("mitdb", "fa_h", 2, False, False), ("noise_em_0", "U", 2, False, True),
            ("noise_em_0", "fa_h", 2, False, False)]

    def val(v, m, tag, key):
        if tag == "cpsc":
            d = main.get(m)
        else:
            d = ext.get(tag, {}).get(f"{v}:{m}")
        return d.get(key) if d else None
    best = {}
    for i, (tag, key, d, pct, hi) in enumerate(cols):
        vals = [val(v, m, tag, key)[0] for v, m, _ in rows if val(v, m, tag, key)]
        best[i] = max(vals) if hi else min(vals)
    lines = ["\\begin{table*}[!t]", "\\centering",
             "\\caption{Secondary analysis: CUSUM proposals certified by the conformal e-detector (\\method{}-G; Section~\\ref{sec:gate}). "
             "CPSC2021: pooled cross-validation, mean\\,$\\pm$\\,s.d.\\ over five seeds; external cohorts and noise: mean over the "
             "locked models (each noisy recording scored by its own fold's models). F1 in \\%, FA/h: false episodes per hour. "
             "Best value per column in bold; $^\\ast$: patient-level paired difference between \\method{}-G (validated) and CUSUM with "
             "$p<0.05$ (two-sided sign-flip test); the CPSC2021 score difference favours CUSUM, all other marked differences favour \\method{}-G.}",
             "\\label{tab:gate}", "\\footnotesize", "\\setlength{\\tabcolsep}{2.5pt}", "\\resizebox{\\textwidth}{!}{%",
             "\\begin{tabular}{@{}l" + "c" * len(cols) + "@{}}", "\\toprule",
             " & \\multicolumn{3}{c}{CPSC2021 (CV)} & \\multicolumn{2}{c}{AFDB} & LTAFDB & NSRDB & MIT-BIH non-AF & "
             "\\multicolumn{2}{c}{Electrode motion 0\\,dB}\\\\",
             "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\\cmidrule(lr){7-7}\\cmidrule(lr){8-8}\\cmidrule(lr){9-9}\\cmidrule(lr){10-11}",
             "Method & $U$ & Ep.\\ F1 & FA/h & Ep.\\ F1 & FA/h & FA/h & FA/h & FA/h & $U$ & FA/h\\\\", "\\midrule"]
    sig_keys = {2: "cpsc_fa", 4: "afdb|fa_h", 5: "ltafdb|fa_h", 6: "nsrdb|fa_h", 7: "mitdb|fa_h", 8: "noise_em_0|U", 9: "noise_em_0|fa_h"}
    gs = S.get("stats_gate", {})
    for v, m, name in rows:
        cells = []
        for i, (tag, key, d, pct, hi) in enumerate(cols):
            ms = val(v, m, tag, key)
            c = f(ms, d, pct, sd=(tag == "cpsc"))
            if ms and abs(ms[0] - best[i]) < 1e-12:
                c = "\\textbf{" + c + "}"
            if m == "gate_tuned":
                p = None
                if i == 0:
                    p = gs.get("gate_tuned|U|vs_cusum", {}).get("p")
                elif i == 2:
                    p = gs.get("gate_tuned|fa_h|vs_cusum", {}).get("p")
                elif i in sig_keys:
                    p = st.get(sig_keys[i] + "|gate_vs_cusum", {}).get("p")
                if p is not None and p < 0.05:
                    c += "$^\\ast$"
            cells.append(c)
        lines.append(name + " & " + " & ".join(cells) + "\\\\")
    lines += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    S = json.load(open(a.summary))
    for name, fn in [("tab_cpsc", table_main), ("tab_stats", table_stats), ("tab_external", table_external),
                     ("tab_controls", table_controls), ("tab_ablation", table_ablation), ("tab_burden", table_burden),
                     ("tab_gate", table_gate)]:
        try:
            open(os.path.join(a.out, name + ".tex"), "w").write(fn(S))
            print("wrote", name)
        except Exception as e:
            print("skip", name, type(e).__name__, e)


if __name__ == "__main__":
    main()
