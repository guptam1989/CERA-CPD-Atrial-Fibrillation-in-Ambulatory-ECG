"""Patient-level outer folds with disjoint train/validation/calibration roles.

python -m experiments.cera_exp.folds --index PREP/index.json --out folds.json
Patients are stratified by type (any paroxysmal record / persistent only /
non-AF only). Each patient is a test patient in exactly one outer fold. The
pre-selected recording for burden calibration and coverage is the longest
recording of each patient (a frozen rule fixed before scoring).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np


def patient_types(records: list[dict]) -> dict[str, str]:
    classes = defaultdict(set)
    for r in records:
        classes[r["patient"]].add(r["rec_class"])
    out = {}
    for p, c in classes.items():
        out[p] = "paroxysmal" if 2 in c else ("persistent" if 1 in c else "non_af")
    return out


def make_folds(records: list[dict], k: int = 5, seed: int = 2026) -> dict:
    types = patient_types(records)
    rng = np.random.default_rng(seed)
    outer = {}
    for t in sorted(set(types.values())):
        pats = sorted(p for p, v in types.items() if v == t)
        rng.shuffle(pats)
        offset = int(rng.integers(k))
        for i, p in enumerate(pats):
            outer[p] = (i + offset) % k
    longest = {}
    for r in records:
        p = r["patient"]
        if p not in longest or r["duration_s"] > longest[p][1]:
            longest[p] = (r["record"], r["duration_s"])
    folds = []
    for f in range(k):
        roles = {}
        for t in sorted(set(types.values())):
            dev = sorted(p for p, v in types.items() if v == t and outer[p] != f)
            rng.shuffle(dev)
            for i, p in enumerate(dev):
                slot = i % 20
                roles[p] = "validation" if slot < 3 else ("calibration" if slot < 7 else "train")
            for p in sorted(p for p, v in types.items() if v == t and outer[p] == f):
                roles[p] = "test"
        folds.append(dict(fold=f, roles=roles))
    return dict(seed=seed, k=k, patient_type=types,
                preselected_record={p: v[0] for p, v in longest.items()}, folds=folds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--seed", type=int, default=2026)
    a = ap.parse_args()
    index = json.load(open(a.index))
    result = make_folds(index["records"], a.k, a.seed)
    json.dump(result, open(a.out, "w"), indent=1)
    for f in result["folds"]:
        counts = defaultdict(lambda: defaultdict(int))
        for p, role in f["roles"].items():
            counts[role][result["patient_type"][p]] += 1
        print(f["fold"], {r: dict(c) for r, c in counts.items()})


if __name__ == "__main__":
    main()
