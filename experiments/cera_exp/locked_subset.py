"""Write locked-configuration files restricted to selected methods.

python -m experiments.cera_exp.locked_subset --src RESULTS/main --dst LOCKED \
    --keep "full=cera_tuned,cera,cusum;gate=gate_tuned;rr=*"

The output keeps fold, variant, seed, the validation-selected configurations,
their validation scores and the Markov transition model, which is everything
run_eval.py --locked-evaluations reads. Test and external rows are dropped, so a
replay evaluates only the requested methods with their original settings.
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--keep", required=True)
    a = ap.parse_args()
    keep = {}
    for item in a.keep.split(";"):
        variant, methods = item.split("=")
        keep[variant] = None if methods == "*" else set(methods.split(","))
    os.makedirs(a.dst, exist_ok=True)
    n = 0
    for path in sorted(glob.glob(os.path.join(a.src, "fold*_s*.json.gz"))):
        if path.endswith("_val.json.gz"):
            continue
        with gzip.open(path, "rt") as f:
            d = json.load(f)
        if d["variant"] not in keep:
            continue
        wanted = keep[d["variant"]]
        configs = {m: c for m, c in d["configs"].items() if wanted is None or m in wanted}
        out = dict(fold=d["fold"], variant=d["variant"], seed=d["seed"], configs=configs,
                   methods={m: dict(val_U=d["methods"][m].get("val_U")) for m in configs if m in d["methods"]},
                   source=os.path.abspath(path))
        if "markov_llr" in d:
            out["markov_llr"] = d["markov_llr"]
        with gzip.open(os.path.join(a.dst, os.path.basename(path)), "wt") as f:
            json.dump(out, f)
        n += 1
    print(json.dumps(dict(written=n, dst=a.dst)))


if __name__ == "__main__":
    main()
