"""Computational cost: parameters, multiply-accumulates per block, single-thread
CPU latency of the encoder per block (batch 1) and of the online detector update.

python -m experiments.cera_exp.cost --out cost.json
"""
from __future__ import annotations

import argparse
import json
import platform
import time

import numpy as np
import torch
from torch import nn

from .model import DualStream
from . import detect as D


def count_macs(model, inputs):
    macs = [0]

    def conv_hook(m, i, o):
        macs[0] += o.numel() * (m.in_channels // m.groups) * m.kernel_size[0]

    def lin_hook(m, i, o):
        macs[0] += o.numel() * m.in_features

    hooks = []
    for mod in model.modules():
        if isinstance(mod, nn.Conv1d):
            hooks.append(mod.register_forward_hook(conv_hook))
        elif isinstance(mod, nn.Linear):
            hooks.append(mod.register_forward_hook(lin_hook))
    with torch.no_grad():
        model(*inputs)
    for h in hooks:
        h.remove()
    return macs[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--reps", type=int, default=2000)
    a = ap.parse_args()
    torch.set_num_threads(1)
    model = DualStream(beats=8, d=64).eval()
    b, L = 8, 64
    x = (torch.randn(1, b - 1).abs() + 0.5, torch.randn(1, 5), torch.randn(1, b, 4, L),
         torch.ones(1, b, 4, L, dtype=torch.bool), torch.ones(1))
    params = sum(p.numel() for p in model.parameters())
    # projector is used only for training (contrastive loss)
    params_inf = params - sum(p.numel() for p in model.projector.parameters())
    macs = count_macs(model, x)
    with torch.no_grad():
        for _ in range(50):
            model(*x)
        t = []
        for _ in range(a.reps):
            t0 = time.perf_counter()
            model(*x)
            t.append(time.perf_counter() - t0)
    t = np.array(t) * 1e3
    # detector update cost on a synthetic stream of 20000 blocks (one segment, no alarms)
    rng = np.random.default_rng(0)
    J = 20000
    rp = np.cumsum(np.full(J * 8 + 9, 0.8))
    ctx = D.RecordContext(rp, np.arange(J) * 8, rp[np.arange(J) * 8 + 8] + 0.15,
                          rng.normal(-4, 1, J), np.full((J, 8), 0.02, np.float32), 8, float(rp[-1]))
    t0 = time.perf_counter()
    D.cera(ctx, A=1e12)
    det_us = (time.perf_counter() - t0) / J * 1e6
    out = dict(params_total=params, params_inference=params_inf, macs_per_block=macs,
               cpu=platform.processor(), threads=1, torch=torch.__version__,
               latency_ms_median=float(np.median(t)), latency_ms_p95=float(np.percentile(t, 95)),
               detector_update_us_per_block=det_us)
    json.dump(out, open(a.out, "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
