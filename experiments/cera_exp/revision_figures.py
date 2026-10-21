"""Revision figure: noise stress on the end-to-end track (detected beats).

python -m experiments.cera_exp.revision_figures --summary revision_summary.json --out figures
Same layout and visual conventions as figures.fig_noise.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .figures import C, LS, MARK, W2, save, style


def fig_noise_qrs(S, out):
    noise = S["qrs"]["noise"]
    kinds = [("em", "electrode motion"), ("ma", "muscle artefact"), ("bw", "baseline wander")]
    series = [("full:cera_tuned", "CERA-CPD, validated $A$"), ("gate:gate_tuned", "CERA-CPD-G, validated"),
              ("full:cusum", "same encoder + CUSUM"), ("full:smooth", "same encoder + smoothing"),
              ("rr:rr_cosen", "COSEn (RR only)")]
    fig, ax = plt.subplots(2, 3, figsize=(W2, 3.2), sharex=True)
    snrs = [12, 6, 0]
    xs = ["clean"] + [f"{s} dB" for s in snrs]

    def val(tag, key, metric):
        d = (noise.get(tag) or {}).get(key) or {}
        v = d.get(metric)
        return np.nan if not v or v[0] is None else float(v[0])
    for j, (k, name) in enumerate(kinds):
        for i, (key, lab) in enumerate(series):
            f1 = [val("noise_clean", key, "ep_f1_0.5")] + [val(f"noise_{k}_{s}", key, "ep_f1_0.5") for s in snrs]
            fa = [val("noise_clean", key, "fa_h")] + [val(f"noise_{k}_{s}", key, "fa_h") for s in snrs]
            ax[0, j].plot(xs, f1, color=C[i], marker=MARK[i], ls=LS[i], label=lab)
            ax[1, j].plot(xs, fa, color=C[i], marker=MARK[i], ls=LS[i], label=lab)
        ax[0, j].set_title(f"({'abc'[j]}) {name}", loc="left")
        ax[1, j].set_xlabel("noise level (SNR)")
    ax[0, 0].set_ylabel("episode F1")
    ax[1, 0].set_ylabel("false episodes/h")
    vals = [y for a in ax[1] for l in a.get_lines() for y in l.get_ydata() if np.isfinite(y) and y > 0]
    lo = 10 ** np.floor(np.log10(min(vals))) if vals else 0.01
    hi = 10 ** np.ceil(np.log10(max(vals))) if vals else 10
    for j in range(3):
        ax[0, j].set_ylim(0, 1)
        ax[1, j].set_yscale("log")
        ax[1, j].set_ylim(lo, hi)
    fig.legend(*ax[0, 0].get_legend_handles_labels(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
    fig.tight_layout()
    save(fig, out, "fig_noise_qrs")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    style()
    S = json.load(open(a.summary))
    os.makedirs(a.out, exist_ok=True)
    if "qrs" in S:
        fig_noise_qrs(S, a.out)
    print("figures written to", a.out)


if __name__ == "__main__":
    main()
