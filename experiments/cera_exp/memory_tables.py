"""LaTeX table and figure for the wearer-memory extension (reads memory_summary.json).

python -m experiments.cera_exp.memory_tables --summary memory_summary.json --tables TABLES_DIR --figures FIG_DIR
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from .revision_tables import FIRST, g, mpm, num

# (summary key, display name, published counterpart for *, compared with CUSUM for dagger)
FIRST_M = "l"   # natural width; \fitwidth scales the table to the text width
ROWS = [
    ("Primary configurations", [
        ("cusum", "Same encoder + CUSUM", None, False),
        ("cera_tuned", "\\method{}, validated $A$", None, False),
        ("gate_tuned", "\\method{}-G, validated $A_c$", None, False),
        ("gate_A1121", "\\method{}-G, $A_c=1121$", None, False)]),
    ("Wearer memory", [
        ("cera_m", "\\method{}-M, validated $\\gamma$, $A$", "cera_tuned", True),
        ("cera_md", "\\method{}-MD, validated $\\gamma$, $A$", "cera_tuned", True),
        ("gate_md", "\\method{}-GMD, validated $A_c$", "gate_tuned", True),
        ("gate_md_A1121", "\\method{}-GMD, $A_c=1121$", "gate_A1121", True)]),
]
ABLATIONS = [("abl_margin", "Margin, no memory"), ("abl_margin_dyadic", "Margin and calibrator, no memory"),
             ("abl_wearer_c0", "Wearer memory, no margin"), ("abl_record", "Memory within recordings"),
             ("abl_gate_record", "\\method{}-G, memory within recordings")]


def _marks(st, prefix, key, base, vs_cusum, name):
    out = ""
    if base:
        t = st.get(f"{prefix}{name}_vs_{base}|{key}")
        if t and t["p"] < 0.05:
            out += "$^\\ast$"
    if vs_cusum and name != "cusum":
        t = st.get(f"{prefix}{name}_vs_cusum|{key}")
        if t and t["p"] < 0.05:
            out += "$^\\dagger$"
    return out


def table_memory(S):
    cp, st = S["cpsc"], S["stats"]
    cs, es, ns, qs = st["cpsc"], st["external"], st["noise"], st.get("qrs_cpsc", {})
    cap = ("\\caption{Secondary analysis: wearer memory (validation-selected variants). F1 and boundary Se "
           "($\\pm1$ beat) in \\%, burden MAE in percentage points, FA/h: false episodes per hour; $^\\ast$, "
           "$^\\dagger$: paired $p<0.05$ versus the primary configuration and CUSUM.}")
    out = ["\\begin{table*}[!htbp]", "\\centering", cap, "\\label{tab:memory}", "\\footnotesize",
           "\\setlength{\\tabcolsep}{2.5pt}", "\\par\\smallskip", "\\textbf{(a) CPSC2021 cross-validation}",
           "\\par\\smallskip", "\\fitwidth{%", "\\begin{tabular}{@{}" + FIRST_M + "cccccc@{}}", "\\toprule",
           "Method & $U$ & Ep.\\ F1 & FA/h & Boundary Se & Burden MAE & CCC\\\\", "\\midrule"]
    for group, rows in ROWS:
        out.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\textit{{{group}}}}}\\\\")
        for name, disp, base, vc in rows:
            c = cp[name]
            m = lambda k: _marks(cs, "", k, base, vc, name)
            cells = [mpm(c, "U", 2) + m("U"), mpm(c, "ep_f1_0.5", 1, 100), mpm(c, "fa_h", 3) + m("fa_h"),
                     mpm(c, "bnd_se_1", 1, 100), mpm(c, "burden_mae_pp", 1), mpm(c, "ccc", 2)]
            out.append(f"\\quad {disp} & " + " & ".join(cells) + "\\\\")
    out.append("\\multicolumn{7}{@{}l}{\\textit{Ablations}}\\\\")
    for name, disp in ABLATIONS:
        c = cp[name]
        base = "gate_tuned" if name.startswith("abl_gate") else "cera_tuned"
        m = lambda k: _marks(cs, "", k, base, False, name)
        cells = [mpm(c, "U", 2) + m("U"), mpm(c, "ep_f1_0.5", 1, 100), mpm(c, "fa_h", 3) + m("fa_h"),
                 mpm(c, "bnd_se_1", 1, 100), mpm(c, "burden_mae_pp", 1), mpm(c, "ccc", 2)]
        out.append(f"\\quad {disp} & " + " & ".join(cells) + "\\\\")
    out += ["\\bottomrule", "\\end{tabular}}", "\\par\\smallskip",
            "\\textbf{(b) Zero-shot cohorts, noise stress and end-to-end track}", "\\par\\smallskip", "\\fitwidth{%",
            "\\begin{tabular}{@{}" + FIRST_M + "cccccccc@{}}", "\\toprule",
            " & \\multicolumn{2}{c}{AFDB} & LTAFDB & NSRDB & MIT-BIH non-AF & \\multicolumn{2}{c}{Electrode motion 0\\,dB}"
            " & End-to-end\\\\",
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-4}\\cmidrule(lr){5-5}\\cmidrule(lr){6-6}\\cmidrule(lr){7-8}"
            "\\cmidrule(lr){9-9}",
            "Method & Ep.\\ F1 & FA/h & FA/h & FA/h & FA/h & $U$ & FA/h & $U$\\\\", "\\midrule"]
    for group, rows in ROWS:
        out.append(f"\\multicolumn{{9}}{{@{{}}l}}{{\\textit{{{group}}}}}\\\\")
        for name, disp, base, vc in rows:
            ext = lambda tag: S["external"][tag].get(name)
            em = S["noise"]["noise_em_0"].get(name)
            q = S["qrs"]["cpsc"].get(name)
            e = lambda tag, k: _marks(es, f"{tag}|", k, base, vc, name)
            n = lambda k: _marks(ns, "noise_em_0|", k, base, vc, name)
            cells = [num(g(ext("afdb"), "ep_f1_0.5"), 1, 100), num(g(ext("afdb"), "fa_h"), 3) + e("afdb", "fa_h"),
                     num(g(ext("ltafdb"), "fa_h"), 3) + e("ltafdb", "fa_h"),
                     num(g(ext("nsrdb"), "fa_h"), 3) + e("nsrdb", "fa_h"),
                     num(g(ext("mitdb"), "fa_h"), 2) + e("mitdb", "fa_h"),
                     num(g(em, "U"), 2) + n("U"), num(g(em, "fa_h"), 2) + n("fa_h"),
                     num(g(q, "U"), 2) + _marks(qs, "", "U", base, vc, name)]
            out.append(f"\\quad {disp} & " + " & ".join(cells) + "\\\\")
    out += ["\\bottomrule", "\\end{tabular}}", "\\end{table*}"]
    return "\n".join(out).replace("+ matched", "+\\,matched") + "\n"


def fig_memory(S, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from .figures import C, MARK, LS, W2, save, style
    style()
    cp = S["cpsc"]
    acs = sorted({int(k.split("_A")[-1]) for k in cp if k.startswith("gate_A")})
    series = [("gate", "primary gate (bag reset at alarms)"), ("gate_m", "gate + wearer memory"),
              ("gate_md", "gate + wearer memory + matched calibrator")]
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.2))
    for a, key, yl in zip(ax, ("U", "ep_f1_0.5", "fa_h"), ("challenge score $U$", "episode F1", "false episodes per hour")):
        for i, (pref, lab) in enumerate(series):
            m = np.array([cp[f"{pref}_A{x}"][key][0] for x in acs])
            s = np.array([cp[f"{pref}_A{x}"][key][1] for x in acs])
            a.errorbar(acs, m, yerr=s, color=C[i if i < 2 else 4], marker=MARK[i], ls=LS[i], capsize=2, label=lab)
        a.axhline(cp["cusum"][key][0], color=C[2], ls="-.", lw=1.0, label="same encoder + CUSUM (ungated)")
        a.axhline(cp["cera_md"][key][0], color=C[3], ls=":", lw=1.2, label="CERA-CPD-MD (standalone)")
        a.set_xscale("log")
        a.set_xlabel("certified ARL bound $A_c$ (blocks)")
        a.set_ylabel(yl)
    for a, t in zip(ax, "abc"):
        a.set_title(f"({t})", loc="left")
    h, l = ax[0].get_legend_handles_labels()
    order = [l.index(lab) for _, lab in series] + [i for i, lab in enumerate(l) if lab not in dict(series).values()]
    fig.legend([h[i] for i in order], [l[i] for i in order], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.13))
    fig.tight_layout()
    save(fig, out, "fig_memory")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--tables", required=True)
    ap.add_argument("--figures", required=True)
    a = ap.parse_args()
    S = json.load(open(a.summary))
    os.makedirs(a.tables, exist_ok=True)
    os.makedirs(a.figures, exist_ok=True)
    open(os.path.join(a.tables, "tab_memory.tex"), "w").write(table_memory(S))
    fig_memory(S, a.figures)
    print("wrote tab_memory.tex and fig_memory")


if __name__ == "__main__":
    main()
