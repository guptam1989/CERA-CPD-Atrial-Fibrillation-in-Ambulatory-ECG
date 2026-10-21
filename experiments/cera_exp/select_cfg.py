"""Nested encoder selection: per outer fold, the candidate configuration with
the lowest validation fused BCE (seed 0). Prints the selected JSON config.

python -m experiments.cera_exp.select_cfg --runs RUNS_ROOT --fold 0 [--report]
"""
from __future__ import annotations

import argparse
import json
import os

CANDIDATES = {
    "base": {},
    "aug": {"augment": True},
    "augreg": {"augment": True, "dropout": 0.3, "wd": 0.01},
    "augregema": {"augment": True, "dropout": 0.3, "wd": 0.01, "ema": 0.999},
}


def select(runs_root: str, fold: int):
    scores = {}
    for name in CANDIDATES:
        h = os.path.join(runs_root, f"cand_{name}", f"fold{fold}", "full_s0", "history.json")
        if os.path.exists(h):
            scores[name] = json.load(open(h))["best_val_bce"]
    best = min(scores, key=scores.get)
    return best, scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--fold", type=int)
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        out = {}
        for f in range(5):
            best, scores = select(a.runs, f)
            out[f] = dict(selected=best, val_bce=scores)
        print(json.dumps(out, indent=1))
        return
    best, _ = select(a.runs, a.fold)
    print(json.dumps(CANDIDATES[best]))


if __name__ == "__main__":
    main()
