"""Manuscript figures (vector PDF) from aggregated outputs.

python -m experiments.cera_exp.figures --summary agg/summary.json --sweeps SWEEPS --sim sim.json \
    --results RESULTS --out figures
Colour slots follow a validated categorical order (blue, orange, aqua, yellow,
magenta); every series also has a distinct marker and line style for print.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .summary import pool_sweep_summaries

C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"
MARK = ["o", "s", "^", "D", "v", "P"]
LS = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1))]
W1, W2 = 3.5, 7.16


def save(fig, out, name):
    fig.savefig(os.path.join(out, name + ".pdf"))
    fig.savefig(os.path.join(out, name + ".png"), dpi=170)


def style():
    plt.rcParams.update({
        "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "legend.fontsize": 6.8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
        "grid.linewidth": 0.5, "axes.spines.top": False, "axes.spines.right": False,
        "lines.linewidth": 1.4, "lines.markersize": 4, "legend.frameon": False,
        "pdf.fonttype": 42, "font.family": "DejaVu Sans", "savefig.bbox": "tight", "savefig.pad_inches": 0.02})


def fig_false_alarms(sim_path, sweep_dir, out):
    sim = json.load(open(sim_path))
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.25))
    kinds = [("iid", "i.i.d. (exchangeable)"), ("drift0.5", "drift 0.5 s.d./1000 blocks"),
             ("ar0.5", "AR(1), $\\rho=0.5$"), ("ar0.9", "AR(1), $\\rho=0.9$")]
    for i, (k, lab) in enumerate(kinds):
        rs = sorted([r for r in sim["single"] if r["kind"] == k and not r["clip"]], key=lambda r: r["A"])
        A = np.array([r["A"] for r in rs])
        ratio = np.array([r["restricted_mean"] / r["A"] for r in rs])
        se = np.array([r["restricted_mean_se"] / r["A"] for r in rs])
        ax[0].errorbar(A, ratio, yerr=1.96 * se, color=C[i], marker=MARK[i], ls=LS[i], label=lab, capsize=2)
    ax[0].axhline(1.0, color=INK, lw=0.9, ls="--")
    ax[0].text(11, 1.12, "unit reference", color=INK, fontsize=6.5)
    ax[0].set_xscale("log")
    ax[0].set_yscale("log")
    ax[0].set_xlabel("threshold $A$ (active blocks)")
    ax[0].set_ylabel("$E[\\min(\\nu,20A)]\\,/\\,A$")
    ax[0].set_title("(a) Simulated fixed-mode run length", loc="left")
    ax[0].legend(loc="lower left")
    # (b) empirical alarm rates on change-free ECG
    rows = defaultdict(lambda: defaultdict(list))
    for p in glob.glob(os.path.join(sweep_dir, "sweep_fold*_s*.json")):
        d = json.load(open(p))
        for key in ("nsrdb", "mitdb_nonaf", "cpsc_nonaf"):
            for A, v in d.get(key, {}).items():
                rows[key][float(A)].append(v)
    labels = {"nsrdb": "NSRDB (sinus, 437 h)", "mitdb_nonaf": "MIT-BIH non-AF (ectopy, 20 h)",
              "cpsc_nonaf": "CPSC2021 non-AF test records"}
    blk = []
    for i, key in enumerate(("nsrdb", "mitdb_nonaf", "cpsc_nonaf")):
        if not rows[key]:
            continue
        A = np.array(sorted(rows[key]))
        med = np.array([np.median([v["alarms_per_h"] for v in rows[key][a]]) for a in A])
        lo = np.array([np.min([v["alarms_per_h"] for v in rows[key][a]]) for a in A])
        hi = np.array([np.max([v["alarms_per_h"] for v in rows[key][a]]) for a in A])
        floor = 1e-3
        ax[1].plot(A, np.maximum(med, floor), color=C[i], marker=MARK[i], ls=LS[i], label=labels[key])
        if key == "nsrdb":
            ax[1].fill_between(A, np.maximum(lo, floor), np.maximum(hi, floor), color=C[i], alpha=0.15, lw=0)
        blk += [v["mean_block_s"] for a in A for v in rows[key][a]]
    A = np.logspace(np.log10(30), 6, 50)
    ax[1].plot(A, 3600 / ((A + 4) * 3.2), color=INK, ls="--", lw=0.9, label="Nominal bound, $d_{\\min}=3.2$ s")
    if blk:
        d = float(np.median(blk))
        ax[1].plot(A, 3600 / ((A + 4) * d), color=MUTED, ls=":", lw=0.9, label=f"same, mean block {d:.1f} s")
    ax[1].set_xscale("log")
    ax[1].set_yscale("log")
    ax[1].set_ylim(1e-3, 200)
    ax[1].set_xlabel("threshold $A$")
    ax[1].set_ylabel("calibrated alarms per hour")
    ax[1].set_title("(b) Alarm rate on non-AF ECG (median of 25 models)", loc="left")
    ax[1].legend(loc="upper right", fontsize=6.2)
    fig.tight_layout()
    save(fig, out, "fig_false_alarms")
    plt.close(fig)


def fig_operating(sweep_dir, out):
    pts = defaultdict(lambda: defaultdict(list))
    for p in glob.glob(os.path.join(sweep_dir, "sweep_fold*_s*.json")):
        d = json.load(open(p))
        for k, v in d["cpsc"].items():
            pts[(int(p.split("_s")[-1].split(".")[0]))][k].append(v)
    fams = [("cera", "\\method{}".replace("\\method{}", "CERA-CPD (sweep $A$)")), ("cusum", "Same encoder + CUSUM (sweep $h$)"),
            ("smooth", "Same encoder + smoothing (sweep $t_{on}$)")]
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.2))
    for i, (fam, lab) in enumerate(fams):
        keys = sorted({k for s in pts for k in pts[s] if k.startswith(fam + "_")}, key=lambda k: float(k.split("_")[1]))
        xs, f1, dl, U = [], [], [], []
        for k in keys:
            per_seed = []
            for s in pts:
                if not pts[s].get(k):
                    continue
                if len(pts[s][k]) != 5:
                    raise ValueError(f"Operating curve {k}, seed {s}: expected all five outer folds")
                pooled = pool_sweep_summaries(pts[s][k])
                per_seed.append(tuple(pooled[key] for key in ("fa_h", "ep_f1_0.5", "delay_median", "U")))
            arr = np.array(per_seed)
            xs.append(arr[:, 0].mean()); f1.append(arr[:, 1].mean()); U.append(arr[:, 3].mean())
            # No detected alarm in a seed contributes no median, never a zero delay.
            finite_delays = arr[np.isfinite(arr[:, 2]), 2]
            dl.append(float(finite_delays.mean()) if len(finite_delays) else np.nan)
        ax[0].plot(xs, U, color=C[i], marker=MARK[i], ls=LS[i], label=lab)
        ax[1].plot(xs, f1, color=C[i], marker=MARK[i], ls=LS[i])
        ax[2].plot(xs, dl, color=C[i], marker=MARK[i], ls=LS[i])
    for a, yl, t in zip(ax, ["challenge score $U$", "episode F1 (IoU$\\geq$0.5)", "median overlap delay (s)"],
                        ["(a)", "(b)", "(c)"]):
        a.set_xscale("log")
        a.set_xlabel("false episodes per hour")
        a.set_ylabel(yl)
        a.set_title(t, loc="left")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.06))
    fig.tight_layout()
    save(fig, out, "fig_operating")
    plt.close(fig)


def fig_noise(summary, out, method="cera_tuned"):
    ext = summary["external"]
    kinds = [("em", "electrode motion"), ("ma", "muscle artefact"), ("bw", "baseline wander")]
    series = [("full", method, "CERA-CPD, learned gate"), ("equal_gate", method, "CERA-CPD, equal weights"),
              ("full", "cusum", "same encoder + CUSUM"), ("full", "smooth", "same encoder + smoothing")]
    fig, ax = plt.subplots(2, 3, figsize=(W2, 3.2), sharex=True)
    snrs = [12, 6, 0]
    for j, (k, name) in enumerate(kinds):
        for i, (v, m, lab) in enumerate(series):
            key = f"{v}:{m}"
            f1 = [ext.get(f"noise_{k}_{s}", {}).get(key, {}).get("ep_f1_0.5", [np.nan])[0] for s in snrs]
            fa = [ext.get(f"noise_{k}_{s}", {}).get(key, {}).get("fa_h", [np.nan])[0] for s in snrs]
            c0 = ext.get("noise_clean", {}).get(key, {})
            xs = ["clean"] + [f"{s} dB" for s in snrs]
            ax[0, j].plot(xs, [c0.get("ep_f1_0.5", [np.nan])[0]] + f1, color=C[i], marker=MARK[i], ls=LS[i], label=lab)
            ax[1, j].plot(xs, [c0.get("fa_h", [np.nan])[0]] + fa, color=C[i], marker=MARK[i], ls=LS[i], label=lab)
        ax[0, j].set_title(f"({'abc'[j]}) {name}", loc="left")
        ax[1, j].set_xlabel("noise level (SNR)")
    ax[0, 0].set_ylabel("episode F1 (IoU$\\geq$0.5)")
    ax[1, 0].set_ylabel("false episodes per hour")
    top = max(0.05, max(l.get_ydata().max() for a in ax[1] for l in a.get_lines()) * 1.15)
    for j in range(3):
        ax[0, j].set_ylim(0, 1)
        ax[1, j].set_ylim(0, top)
    fig.legend(*ax[0, 0].get_legend_handles_labels(), loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.04))
    fig.tight_layout()
    save(fig, out, "fig_noise")
    plt.close(fig)


def fig_burden(results_dir, out, methods=(("cera_tuned", "CERA-CPD, validated $A$"), ("cusum", "same encoder + CUSUM")), seed=0):
    panels = []
    files = sorted(glob.glob(os.path.join(results_dir, f"fold*_full_s{seed}.json.gz")))
    loaded = [json.load(gzip.open(p, "rt")) for p in files]
    for i, (m, lab) in enumerate(methods):
        pre = [x for d in loaded for x in d["methods"][m]["test"] if x.get("preselected")]
        panels.append((pre, f"({'ab'[i]}) {lab}"))
    fig, ax = plt.subplots(1, 2, figsize=(W2, 2.5))
    for a, (rows, title) in zip(ax, panels):
        if not rows:
            continue
        rows = sorted(rows, key=lambda x: x["B"])
        B = np.array([x["B"] for x in rows]) * 100
        Bh = np.array([x["B_hat"] for x in rows]) * 100
        lo = np.array([x["int_norm"][2] for x in rows]) * 100
        hi = np.array([x["int_norm"][3] for x in rows]) * 100
        cov = (B >= lo - 1e-9) & (B <= hi + 1e-9)
        a.vlines(B[cov], lo[cov], hi[cov], color=C[0], alpha=0.35, lw=1.0)
        a.vlines(B[~cov], lo[~cov], hi[~cov], color=C[1], alpha=0.8, lw=1.0)
        a.scatter(B[cov], Bh[cov], s=9, color=C[0], marker="o", label=f"covered ({cov.mean() * 100:.0f}%)", zorder=3)
        a.scatter(B[~cov], Bh[~cov], s=12, color=C[1], marker="x", label="not covered", zorder=3)
        a.plot([0, 100], [0, 100], color=MUTED, lw=0.8, ls="--")
        a.set_xlabel("reference AF burden (%)")
        a.set_ylabel("estimated burden and 90% interval (%)")
        a.set_title(title, loc="left")
        a.legend(loc="upper left")
        a.set_xlim(-3, 103)
        a.set_ylim(-3, 103)
    fig.tight_layout()
    save(fig, out, "fig_burden")
    plt.close(fig)


def fig_traces(trace_path, out):
    tr = json.load(open(trace_path))
    names = {0: "non-AF", 1: "persistent AF", 2: "paroxysmal AF"}
    order = sorted(tr["records"], key=lambda r: [2, 0, 1].index(r["rec_class"]))
    fig, ax = plt.subplots(3, len(order), figsize=(W2, 3.6), gridspec_kw=dict(height_ratios=[1, 1, 1.1]))
    logA = np.log(tr["A"])
    for c, r in enumerate(order):
        T = r["duration_s"] / 60
        e = np.array(r["ecg"])
        t = r["ecg_t0"] + np.arange(len(e)) / r["fs"]
        a0 = ax[0, c]
        a0.plot(t, e, color=INK, lw=0.5)
        for s, f in r["true"]:
            a0.axvspan(max(s, t[0]), min(f, t[-1]), color=C[1], alpha=0.12, lw=0) if f > t[0] and s < t[-1] else None
        a0.set_xlim(t[0], t[-1])
        a0.set_title(f"({'abc'[c]}) {r['record']}, {names[r['rec_class']]}, U={r['U']:.1f}", loc="left", fontsize=7)
        a0.set_yticks([])
        a0.set_xlabel("time (s), lead II", labelpad=1)
        a1 = ax[1, c]
        bt = np.array(r["block_t"]) / 60
        a1.plot(bt, r["block_p"], color=C[0], lw=0.8, label="block $\\hat\\pi_j$")
        for s, f in r["true"]:
            a1.axvspan(s / 60, f / 60, color=C[1], alpha=0.15, lw=0)
        for s, f in r["pred"]:
            a1.plot([s / 60, f / 60], [1.08, 1.08], color=C[2], lw=3, solid_capstyle="butt")
        a1.set_ylim(-0.05, 1.15)
        a1.set_xlim(0, T)
        a1.set_xlabel("time (min)", labelpad=1)
        a2 = ax[2, c]
        pts = [(x / 60, y) for x, y, m in r["log_sr"] if y is not None]
        if pts:
            x, y = zip(*pts)
            a2.plot(x, y, color=C[0], lw=0.8)
        a2.axhline(logA, color=INK, ls="--", lw=0.8)
        for al in r["alarms"]:
            a2.plot(al["alarm_s"] / 60, logA, marker="v" if al["direction"] == "onset" else "^",
                    color=C[1] if al["direction"] == "onset" else C[2], ms=5, ls="none")
            a1.axvline(al["boundary_s"] / 60, color=MUTED, lw=0.7, ls=":")
        a2.set_xlim(0, T)
        a2.set_ylim(min(-2, np.nanmin([p[1] for p in pts]) if pts else -2), logA + 3)
        a2.set_xlabel("time (min)", labelpad=1)
        if c == 0:
            a1.set_ylabel("AF probability")
            a2.set_ylabel("$\\log R_j$")
    fig.tight_layout(h_pad=0.4, w_pad=0.6)
    save(fig, out, "fig_traces")
    plt.close(fig)


def fig_mechanism(sim_path, emp_path, out):
    payload = json.load(open(sim_path))
    sim = payload["results"] if isinstance(payload, dict) else payload
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.3))
    i = 0
    for A in (11246, 300000):
        for cont, ls_lab in ((True, "growing bag"), (False, "fixed reference")):
            rs = sorted([r for r in sim if r["A"] == A and r["contaminated"] == cont], key=lambda r: r["n"])
            n = np.array([r["n"] for r in rs])
            med = np.array([r["median"] for r in rs], float)
            med[med >= 2000] = np.nan
            ax[0].plot(n, med, color=C[i], marker=MARK[i], ls=LS[i], label=f"$A$={A:g}, {ls_lab}")
            ax[1].plot(n, [100 * r["never"] for r in rs], color=C[i], marker=MARK[i], ls=LS[i])
            i += 1
    for a in ax[:2]:
        a.set_xscale("log")
        a.set_xlabel("reference blocks before change $n$")
    ax[0].set_yscale("log")
    ax[0].set_ylim(1, 100)
    ax[0].set_ylabel("median blocks to alarm")
    ax[0].set_title("(a) Ideal encoder, delay", loc="left")
    ax[1].set_ylim(-3, 103)
    ax[1].set_ylabel("runs never alarming (%)")
    ax[1].set_title("(b) Ideal encoder, misses", loc="left")
    handles, labels = ax[0].get_legend_handles_labels()
    if emp_path and os.path.exists(emp_path):
        emp = json.load(open(emp_path))
        bins = [(0, 10), (10, 30), (30, 10 ** 9)]
        names = ["<10", "10 to 29", "$\\geq$30"]
        for j, (m, lab) in enumerate((("cera", "CERA-CPD, CPSC2021 test"), ("cusum", "CUSUM, CPSC2021 test"))):
            rate, ns = [], []
            for lo, hi in bins:
                xs = [r for r in emp if r["method"] == m and lo <= r["history_blocks"] < hi]
                rate.append(100 * np.mean([r["delay"] is not None and r["delay"] <= 60 for r in xs]) if xs else np.nan)
                ns.append(len(xs))
            ln, = ax[2].plot(range(len(bins)), rate, color=C[4 + j], marker=MARK[4 + j], ls=LS[j])
            handles.append(ln)
            labels.append(lab)
        ax[2].set_xticks(range(len(bins)))
        ax[2].set_xticklabels([f"{l}\nn={n // 5}" for l, n in zip(names, ns)], fontsize=5.8)
        ax[2].set_ylim(-3, 103)
        ax[2].set_xlabel("observed blocks before onset")
        ax[2].set_ylabel("onsets alarmed within 60 s (%)")
        ax[2].set_title("(c) Real episodes (five seeds)", loc="left")
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.12), fontsize=6.3)
    fig.tight_layout()
    save(fig, out, "fig_mechanism")
    plt.close(fig)

def fig_certification(summary, out):
    main = summary["main"]
    keys = sorted([k for k in main if k.startswith("gate_A")], key=lambda k: float(k[6:]))
    if not keys:
        return
    A = np.array([float(k[6:]) for k in keys])
    fig, ax = plt.subplots(1, 3, figsize=(W2, 2.1))
    for a, key, yl in zip(ax, ("U", "ep_f1_0.5", "fa_h"), ("challenge score $U$", "episode F1", "false episodes per hour")):
        m = np.array([main[k][key][0] for k in keys])
        s = np.array([main[k][key][1] for k in keys])
        a.errorbar(A, m, yerr=s, color=C[0], marker="o", capsize=2, label="CUSUM gated by conformal e-detector")
        a.axhline(main["cusum"][key][0], color=C[2], ls="-.", lw=1.0, label="same encoder + CUSUM (ungated)")
        a.axhline(main["cera_tuned"][key][0], color=C[1], ls="--", lw=1.0, label="CERA-CPD (validated $A$)")
        a.set_xscale("log")
        a.set_xlabel("certified ARL bound $A_c$ (blocks)")
        a.set_ylabel(yl)
    for a, t in zip(ax, "abc"):
        a.set_title(f"({t})", loc="left")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    save(fig, out, "fig_certification")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--sweeps", default="")
    ap.add_argument("--sim", default="")
    ap.add_argument("--results", default="")
    ap.add_argument("--traces", default="")
    ap.add_argument("--mech-sim", default="")
    ap.add_argument("--mech-emp", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    style()
    if a.traces:
        fig_traces(a.traces, a.out)
    if a.mech_sim:
        fig_mechanism(a.mech_sim, a.mech_emp, a.out)
    summary = json.load(open(a.summary))
    fig_certification(summary, a.out)
    if a.sim and a.sweeps:
        fig_false_alarms(a.sim, a.sweeps, a.out)
    if a.sweeps:
        fig_operating(a.sweeps, a.out)
    if any(k.startswith("noise") for k in summary["external"]):
        fig_noise(summary, a.out)
    if a.results:
        fig_burden(a.results, a.out)


if __name__ == "__main__":
    main()
