"""Export scores of the frozen encoders for additional prepared datasets.

python -m experiments.cera_exp.score_export --runs RUNS/main --out-runs RUNS_QRS \
    --preps PREP_QRS/cpsc2021,PREP_QRS/afdb --variant full [--verify PREP/afdb]

No parameter is changed: each locked (fold, seed) checkpoint is reloaded and
applied to the new blocks exactly as in train.export_scores. Output files are
named scores_<basename of the prepared directory>.npz inside
OUT_RUNS/fold<f>/<variant>_s<seed>/, the layout read by run_eval.py. With
--verify, fold-0/seed-0 scores of an already scored dataset are recomputed and
compared with the stored export.
"""
from __future__ import annotations

import argparse
import json
import os
import tempfile
import time

import numpy as np
import torch

from .model import DualStream
from .train import export_scores


def load_model(run_dir: str, variant: str, device):
    hist = json.load(open(os.path.join(run_dir, "history.json")))
    cfg = hist["cfg"]
    model = DualStream(beats=8, d=cfg["d"], dropout=cfg["dropout"], variant=variant).to(device)
    model.load_state_dict(torch.load(os.path.join(run_dir, "model.pt"), map_location=device))
    model.eval()
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--out-runs", required=True)
    ap.add_argument("--preps", required=True)
    ap.add_argument("--variant", default="full")
    ap.add_argument("--folds", default="0,1,2,3,4")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--verify", default="")
    a = ap.parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    keepalive = torch.ones(8, device=device)  # hold the GPU handle for the whole job
    t0 = time.time()
    models = {}
    for f in map(int, a.folds.split(",")):
        for s in map(int, a.seeds.split(",")):
            models[(f, s)] = load_model(os.path.join(a.runs, f"fold{f}", f"{a.variant}_s{s}"), a.variant, device)
    print(json.dumps(dict(event="loaded", models=len(models), device=str(device),
                          gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None)), flush=True)
    if a.verify:
        tag = os.path.basename(os.path.normpath(a.verify))
        ref = np.load(os.path.join(a.runs, "fold0", f"{a.variant}_s0", f"scores_{tag}.npz"))
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "check.npz")
            export_scores({(0, 0): models[(0, 0)]}, a.verify, {(0, 0): path}, device)
            new = np.load(path)
            diff = float(np.max(np.abs(new["lp"].astype(np.float64) - ref["lp"].astype(np.float64)))) if len(ref["lp"]) else 0.0
            same_offsets = bool(np.array_equal(new["offsets"], ref["offsets"]))
        print(json.dumps(dict(event="verify", dataset=tag, max_abs_logit_diff=diff, same_offsets=same_offsets)), flush=True)
    for prep in [p for p in a.preps.split(",") if p]:
        tag = os.path.basename(os.path.normpath(prep))
        paths = {}
        for (f, s) in models:
            od = os.path.join(a.out_runs, f"fold{f}", f"{a.variant}_s{s}")
            os.makedirs(od, exist_ok=True)
            paths[(f, s)] = os.path.join(od, f"scores_{tag}.npz")
        todo = {k: m for k, m in models.items() if not os.path.exists(paths[k])}
        if todo:
            export_scores(todo, prep, {k: paths[k] for k in todo}, device)
        print(json.dumps(dict(event="exported", dataset=tag, models=len(todo), elapsed=time.time() - t0)), flush=True)
    keepalive.add_(0)
    print(json.dumps(dict(event="finished", elapsed=time.time() - t0)), flush=True)


if __name__ == "__main__":
    main()
