"""LaTeX tables for the revision analyses (reads revision_summary.json).

python -m experiments.cera_exp.revision_tables --summary revision_summary.json --out TABLES_DIR
"""
from __future__ import annotations

import argparse
import json
import math
import os

PM = "\\,$\\pm$\\,"
FIRST = ">{\\raggedright\\arraybackslash}p{4.3cm}"   # method column without wrapped names


def g(d, key, idx=0):
    if not d or key not in d or d[key] is None:
        return float("nan")
    v = d[key][idx] if isinstance(d[key], list) else d[key]
    return float("nan") if v is None else float(v)


def num(v, digits=1, scale=1.0):
    return "--" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{scale * v:.{digits}f}"


def mpm(d, key, digits=1, scale=1.0, sd=True):
    m, s = g(d, key, 0), g(d, key, 1)
    if math.isnan(m):
        return "--"
    n = d[key][2] if isinstance(d.get(key), list) and len(d[key]) > 2 else 1
    if not sd or n <= 1 or math.isnan(s):
        return f"{scale * m:.{digits}f}"
    return f"{scale * m:.{digits}f}{PM}{scale * s:.{digits}f}"


def star(p):
    return "$^\\ast$" if p is not None and p < 0.05 else ""


def bold_min(cells, values):
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return cells
    best = min(vals)
    return ["\\textbf{" + c + "}" if not math.isnan(v) and abs(v - best) < 1e-12 else c for c, v in zip(cells, values)]


def bold_max(cells, values):
    vals = [v for v in values if not math.isnan(v)]
    if not vals:
        return cells
    best = max(vals)
    return ["\\textbf{" + c + "}" if not math.isnan(v) and abs(v - best) < 1e-12 else c for c, v in zip(cells, values)]


EC57_ROWS = [
    ("Rhythm features", [("rr:rr_cv", "CV of RR \\cite{tateno2001}"), ("rr:rr_cosen", "COSEn \\cite{lake2011cosen}"),
                         ("rr:rr_markov", "RR Markov \\cite{moody1983afdb}")]),
    ("Deep sequence", [("rr_lstm:rr_lstm", "RR-LSTM, causal \\cite{faust2018}"),
                       ("unet:unet", "1-D U-Net, offline$^\\dagger$ \\cite{wen2022paf}")]),
    ("Same encoder", [("full:smooth", "Causal smoothing"), ("full:percentile", "Percentile threshold"),
                      ("full:cusum", "CUSUM \\cite{page1954}"), ("full:bocpd", "BOCPD \\cite{adams2007bocpd}")]),
    ("Proposed", [("full:cera_tuned", "\\method{}, validated $A$"), ("gate:gate_tuned", "\\method{}-G, validated")]),
]


def ec57_panel(block, title, sd):
    cols = ["AF-time Se", "AF-time PPV", "AF-time Sp", "Ep.\\ Se", "Ep.\\ PPV", "Se$_{\\ge30\\,\\mathrm{s}}$",
            "FA$_{30}$/h"]
    lines = ["\\par\\smallskip", f"\\textbf{{{title}}}", "\\par\\smallskip", "\\fitwidth{%",
             "\\begin{tabular}{@{}" + FIRST + "c" * len(cols) + "@{}}", "\\toprule",
             "Method & " + " & ".join(cols) + "\\\\", "\\midrule"]
    for group, rows in EC57_ROWS:
        present = [(k, n) for k, n in rows if block.get(k)]
        if not present:
            continue
        lines.append(f"\\multicolumn{{{len(cols) + 1}}}{{@{{}}l}}{{\\textit{{{group}}}}}\\\\")
        for k, name in present:
            d = block[k]
            cells = [mpm(d, "dur_se", 1, 100, sd), mpm(d, "dur_ppv", 1, 100, sd), mpm(d, "dur_sp", 1, 100, sd),
                     mpm(d, "ep_se", 1, 100, sd), mpm(d, "ep_ppv", 1, 100, sd), mpm(d, "se30", 1, 100, sd),
                     mpm(d, "fa30_h", 3, 1, sd)]
            lines.append(name + " & " + " & ".join(cells) + "\\\\")
        lines.append("\\midrule" if group != EC57_ROWS[-1][0] else "")
    while lines and lines[-1] in ("", "\\midrule"):
        lines.pop()
    lines += ["\\bottomrule", "\\end{tabular}}"]
    return lines


def agreement_panel(cp, af, title):
    lines = ["\\par\\smallskip", f"\\textbf{{{title}}}", "\\par\\smallskip", "\\fitwidth{%",
             "\\begin{tabular}{@{}" + FIRST + "cccc@{}}", "\\toprule",
             " & \\multicolumn{2}{c}{CPSC2021 (pre-selected recordings)} & \\multicolumn{2}{c}{AFDB}\\\\",
             "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}",
             "Method & Bias [95\\% LoA] & CCC & Bias [95\\% LoA] & CCC\\\\", "\\midrule"]
    for group, rows in EC57_ROWS:
        for k, name in rows:
            a, b = cp.get(k), af.get(k)
            if not a and not b:
                continue
            def ba(d):
                if not d or math.isnan(g(d, "ba_bias")):
                    return "--", "--"
                txt = f"{g(d, 'ba_bias'):+.1f} [{g(d, 'ba_lo'):.1f}, {g(d, 'ba_hi'):.1f}]"
                return txt.replace("-", "$-$"), f"{g(d, 'ccc'):.2f}"
            c1, c2 = ba(a)
            c3, c4 = ba(b)
            lines.append(f"{name} & {c1} & {c2} & {c3} & {c4}\\\\")
    lines += ["\\bottomrule", "\\end{tabular}}"]
    return lines


def table_ec57(S):
    er = S["ext_ref"]
    cap = ("\\caption{Standard AF-detector metrics (ANSI/AAMI EC57 conventions) and clinically oriented measures on the "
           "reference-peak track, replaying the locked settings. CPSC2021: pooled cross-validation, mean\\,$\\pm$\\,s.d.\\ "
           "over seeds; AFDB and LTAFDB: mean over the locked models. AF-time Se/PPV/Sp: sensitivity, positive predictivity "
           "and specificity of AF time on rhythm-labelled exposure; Ep.\\ Se/PPV: episode sensitivity and positive "
           "predictivity with any temporal overlap; Se$_{\\ge30\\,\\mathrm{s}}$: sensitivity for reference episodes lasting "
           "at least 30\\,s; FA$_{30}$/h: predicted episodes of at least 30\\,s without overlap with reference AF per hour "
           "(all in \\%, except FA$_{30}$/h). Burden agreement: Bland--Altman bias and 95\\% limits of agreement (predicted minus "
           "reference, percentage points) and Lin's concordance correlation coefficient, averaged over seeds or models. "
           "$^\\dagger$Offline.}")
    out = ["\\begin{table*}[!tp]", "\\centering", cap, "\\label{tab:ec57}", "\\footnotesize",
           "\\setlength{\\tabcolsep}{3pt}"]
    out += ec57_panel(er["cpsc"], "(a) CPSC2021 cross-validation", True)
    out += agreement_panel(er["cpsc"], er["external"]["afdb"], "(b) Burden agreement")
    out += ["\\end{table*}", "", "\\begin{table*}[!tp]", "\\ContinuedFloat", "\\centering",
            "\\caption[]{Standard AF-detector metrics and clinically oriented measures (continued): zero-shot external "
            "testing. Conventions as in part (a).}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}"]
    out += ec57_panel(er["external"]["afdb"], "(c) AFDB, zero-shot (23 recordings)", False)
    out += ec57_panel(er["external"]["ltafdb"], "(d) LTAFDB, zero-shot (84 recordings)", False)
    out += ["\\end{table*}"]
    return "\n".join(out) + "\n"


QRS_DATASETS = [("cpsc2021", "CPSC2021"), ("afdb", "AFDB$^\\ddagger$"), ("ltafdb", "LTAFDB"), ("nsrdb", "NSRDB"),
                ("mitdb", "MIT-BIH Arrhythmia"), ("noise_clean", "CPSC2021 pre-selected (clean)"),
                ("noise_em_12", "\\quad electrode motion 12\\,dB"), ("noise_em_6", "\\quad electrode motion 6\\,dB"),
                ("noise_em_0", "\\quad electrode motion 0\\,dB"), ("noise_ma_0", "\\quad muscle artefact 0\\,dB"),
                ("noise_bw_0", "\\quad baseline wander 0\\,dB")]
QRS_METHODS = [("rr:rr_cosen", "COSEn"), ("full:smooth", "Causal smoothing"), ("full:percentile", "Percentile threshold"),
               ("full:cusum", "CUSUM"), ("full:bocpd", "BOCPD"), ("full:cera_tuned", "\\method{}, validated $A$"),
               ("gate:gate_tuned", "\\method{}-G, validated")]


def table_qrs(S, ref_summary):
    acc = S["qrs_accuracy"]
    q = S["qrs"]
    out = ["\\begin{table*}[!tp]", "\\centering",
           "\\caption{End-to-end track with the causal two-lead QRS detector. (a) Detector accuracy against the reference "
           "beat annotations (one-to-one matching within $\\pm150$\\,ms on the annotated span) and median confirmation "
           "latency after the R peak; $^\\ddagger$the AFDB reference beats are largely the database's automatic detections. "
           "(b) Delineation with the 25 locked encoders and detector settings applied unchanged to blocks built from the "
           "detected beats; CPSC2021: pooled cross-validation, mean over seeds (s.d.\\ at most 0.04 for $U$, 2.4 for F1 and "
           "0.17 for FA/h), with reference-peak values in parentheses; external cohorts and noise: mean over the locked models "
           "(each noisy recording scored by its own fold's models). F1 in \\%, FA/h: false episodes per hour.}",
           "\\label{tab:qrs}", "\\footnotesize", "\\setlength{\\tabcolsep}{3pt}",
           "\\par\\smallskip", "\\textbf{(a) QRS detection}", "\\par\\smallskip", "\\fitwidth{%",
           "\\begin{tabular}{@{}lrrccc@{}}", "\\toprule",
           "Recordings & Records & Reference beats & Se (\\%) & PPV (\\%) & Latency (s)\\\\", "\\midrule"]
    for key, name in QRS_DATASETS:
        a = acc.get(key)
        if not a:
            continue
        beats = f"{a['ref_beats']:,}".replace(",", "\\,")
        out.append(f"{name} & {a['records']} & {beats} & {100 * a['se']:.2f} & {100 * a['ppv']:.2f} & "
                   f"{a['latency_median_s']:.2f}\\\\")
    out += ["\\bottomrule", "\\end{tabular}}"]
    out += ["\\par\\smallskip", "\\textbf{(b) Delineation on detected beats}", "\\par\\smallskip", "\\fitwidth{%",
            "\\begin{tabular}{@{}" + FIRST + "ccccccc@{}}", "\\toprule",
            " & \\multicolumn{3}{c}{CPSC2021 (CV)} & AFDB & NSRDB & \\multicolumn{2}{c}{Electrode motion 0\\,dB}\\\\",
            "\\cmidrule(lr){2-4}\\cmidrule(lr){5-5}\\cmidrule(lr){6-6}\\cmidrule(lr){7-8}",
            "Method & $U$ & Ep.\\ F1 & FA/h & Ep.\\ F1 & FA/h & Ep.\\ F1 & FA/h\\\\", "\\midrule"]
    rmain = ref_summary.get("main", {})
    for key, name in QRS_METHODS:
        c = q["cpsc"].get(key)
        if not c:
            continue
        mname = key.split(":")[1]
        r = rmain.get(mname, {})
        af = q["external"].get("afdb", {}).get(key)
        ns = q["external"].get("nsrdb", {}).get(key)
        em = q["noise"].get("noise_em_0", {}).get(key)
        u = mpm(c, "U", 2, sd=False) + (f" ({g(r, 'U'):.2f})" if r else "")
        f1 = mpm(c, "ep_f1_0.5", 1, 100, sd=False) + (f" ({100 * g(r, 'ep_f1_0.5'):.1f})" if r else "")
        fa = mpm(c, "fa_h", 3, sd=False) + (f" ({g(r, 'fa_h'):.3f})" if r else "")
        out.append(f"{name} & {u} & {f1} & {fa} & {num(g(af, 'ep_f1_0.5'), 1, 100)} & {num(g(ns, 'fa_h'), 3)} & "
                   f"{num(g(em, 'ep_f1_0.5'), 1, 100)} & {num(g(em, 'fa_h'), 2)}\\\\")
    out += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
    return "\n".join(out) + "\n"


GX = [("smooth", "Causal smoothing", "full:smooth"), ("percentile", "Percentile threshold", "full:percentile"),
      ("bocpd", "BOCPD", "full:bocpd"), ("rrlstm", "RR-LSTM (separate model)", "rr_lstm:rr_lstm")]


def table_gatex(S):
    gx = S["gatex"]
    st = gx["stats"]
    out = ["\\begin{table*}[!tp]", "\\centering",
           "\\caption{Secondary analysis: conformal certification of further proposers. Each proposer keeps its "
           "validation-selected parameters; the certificate $A_c$ is selected on validation patients or fixed at 1121. "
           "CPSC2021: pooled cross-validation, mean over seeds (three for the RR-LSTM); external cohorts "
           "and noise: mean over the locked models. F1 in \\%, FA/h: false episodes per hour; $^\\ast$: patient-level "
           "paired difference from the ungated proposer with unadjusted exploratory $p<0.05$ (two-sided sign-flip test); all "
           "marked differences in FA/h are reductions, and marked differences in $U$ are decreases for smoothing and BOCPD on "
           "CPSC2021 and increases otherwise. No noise-stress scores exist for the RR-LSTM.}",
           "\\label{tab:gatex}", "\\footnotesize", "\\setlength{\\tabcolsep}{2.5pt}", "\\fitwidth{%",
           "\\begin{tabular}{@{}" + FIRST + "cccccccc@{}}", "\\toprule",
           " & \\multicolumn{3}{c}{CPSC2021 (CV)} & AFDB & LTAFDB & NSRDB & \\multicolumn{2}{c}{Electrode motion 0\\,dB}\\\\",
           "\\cmidrule(lr){2-4}\\cmidrule(lr){5-5}\\cmidrule(lr){6-6}\\cmidrule(lr){7-7}\\cmidrule(lr){8-9}",
           "Proposer / certificate & $U$ & Ep.\\ F1 & FA/h & FA/h & FA/h & FA/h & $U$ & FA/h\\\\", "\\midrule"]
    for name, disp, base in GX:
        variants = [(base, "ungated"), (f"gatex:gx_{name}", "gated, validated $A_c$"),
                    (f"gatex:gx_{name}_A1121", "gated, $A_c=1121$")]
        if not gx["cpsc"].get(f"gatex:gx_{name}"):
            continue
        out.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\textit{{{disp}}}}}\\\\")
        for key, label in variants:
            c = gx["cpsc"].get(key)
            if not c:
                continue
            gm = key.split(":")[1]
            sfx = f"_vs_{name}" if key.startswith("gatex") else None

            def s(tag, k):
                return star(st.get(f"{tag}|{k}|{gm}{sfx}", {}).get("p")) if sfx else ""
            ext = lambda tag: gx["external"].get(tag, {}).get(key)
            em = gx["noise"].get("noise_em_0", {}).get(key)
            cells = [mpm(c, "U", 2, sd=False) + s("cpsc", "U"), mpm(c, "ep_f1_0.5", 1, 100, sd=False),
                     mpm(c, "fa_h", 3, sd=False) + s("cpsc", "fa_h"),
                     num(g(ext("afdb"), "fa_h"), 3) + s("afdb", "fa_h"), num(g(ext("ltafdb"), "fa_h"), 3) + s("ltafdb", "fa_h"),
                     num(g(ext("nsrdb"), "fa_h"), 3) + s("nsrdb", "fa_h"),
                     num(g(em, "U"), 2) + s("noise_em_0", "U"), num(g(em, "fa_h"), 2) + s("noise_em_0", "fa_h")]
            out.append(f"\\quad {label} & " + " & ".join(cells) + "\\\\")
    out += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
    return "\n".join(out) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--ref-summary", required=True, help="paper_outputs/summary.json of the locked replay")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    S = json.load(open(a.summary))
    R = json.load(open(a.ref_summary))
    os.makedirs(a.out, exist_ok=True)
    if "ext_ref" in S:
        open(os.path.join(a.out, "tab_ec57.tex"), "w").write(table_ec57(S))
    if "qrs" in S and "qrs_accuracy" in S:
        open(os.path.join(a.out, "tab_qrs.tex"), "w").write(table_qrs(S, R))
    if "gatex" in S:
        open(os.path.join(a.out, "tab_gatex.tex"), "w").write(table_gatex(S))
    print("tables written to", a.out)


if __name__ == "__main__":
    main()
