"""Block-size sensitivity summary (b = 4, 8, 16 beats; seed 0, nested validation tuning).

python -m experiments.cera_exp.revision_blocks --b8 RESULTS/main --b4 RESULTS/b4 --b16 RESULTS/b16 --out blocks.json

b = 8 uses the seed-0 models of the primary analysis; b = 4 and 16 retrain the full
encoder with the fold's selected configuration and re-tune every detector on the
fold's validation patients. CPSC2021 metrics are pooled over the five test folds;
external cohorts are averaged over the five fold models.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os

import numpy as np

from .summary import summarize

METHODS = ("cera_tuned", "cusum", "smooth")
KEYS = ("U", "ep_f1_0.5", "fa_h", "burden_mae_pp", "delay_median", "det_30s", "bnd_se_1",
        "det_rate_5", "det_rate_20", "det_rate_100", "det_rate_1000", "n_eps_5")


def load(directory: str, seed: int = 0):
    test, ext, cfg = {m: [] for m in METHODS}, {}, {}
    for path in sorted(glob.glob(os.path.join(directory, f"fold*_full_s{seed}.json.gz"))):
        with gzip.open(path, "rt") as f:
            r = json.load(f)
        cfg[r["fold"]] = {m: r["configs"].get(m) for m in METHODS}
        for m in METHODS:
            test[m].extend(r["methods"][m]["test"])
            for tag, md in r.get("external", {}).items():
                if m in md:
                    ext.setdefault(tag, {}).setdefault(m, []).append(summarize(md[m]))
    return test, ext, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--b8", required=True)
    ap.add_argument("--b4", required=True)
    ap.add_argument("--b16", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = {}
    for b, d in (("4", a.b4), ("8", a.b8), ("16", a.b16)):
        test, ext, cfg = load(d)
        out[b] = {"cpsc": {m: {k: summarize(test[m]).get(k) for k in KEYS} for m in METHODS},
                  "external": {tag: {m: {k: float(np.nanmean([s.get(k, np.nan) for s in v]))
                                          for k in ("ep_f1_0.5", "fa_h", "burden_mae_pp")}
                                     for m, v in md.items()} for tag, md in ext.items()},
                  "configs": cfg, "folds": len(cfg)}
    json.dump(out, open(a.out, "w"), indent=1, default=float)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
