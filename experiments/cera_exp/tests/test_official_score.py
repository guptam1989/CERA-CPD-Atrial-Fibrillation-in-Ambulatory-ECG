"""Cross-check evaluate.official_score against the organisers' score_2021.py.

python -m experiments.cera_exp.tests.test_official_score DATA_DIR OFFICIAL_SCORE_PY
Predictions: truth, truth shifted by one beat, a spurious episode, and none.
"""
import importlib.util
import json
import os
import sys
import tempfile

import numpy as np

from experiments.cera_exp.records import read_cpsc
from experiments.cera_exp.evaluate import official_ranges, official_score, runs


def load_official(path):
    np.float = float  # the organisers' script uses the removed np.float alias
    spec = importlib.util.spec_from_file_location("score_2021", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main(data_dir, official_py):
    off = load_official(official_py)
    names = sorted(f[:-4] for f in os.listdir(data_dir) if f.endswith(".hea"))
    worst = 0.0
    for name in names:
        path = os.path.join(data_dir, name)
        rec = read_cpsc(path)
        meta = dict(n_samples=rec["n_samples"], rec_class=rec["rec_class"])
        on, offr, ma = official_ranges(rec["ann_samples"], rec["ann_flag"], rec["n_samples"], rec["rec_class"])
        meta["_ranges"] = (on, offr, ma)
        rp = rec["rpeaks"]
        N = len(rp)
        truth = runs(rec["labels"])

        def to_samples(eps, shift=0):
            out = []
            for s, e in eps:
                s2, e2 = min(max(s + shift, 0), N - 1), min(max(e + shift, 1), N)
                a = 0 if s == 0 else int(rp[s2])
                b = rec["n_samples"] - 1 if e == N else int(rp[e2 - 1])
                out.append((a, b))
            return out
        cases = {"truth": to_samples(truth), "shift1": to_samples(truth, 1), "shift3": to_samples(truth, 3),
                 "none": [], "spurious": to_samples(truth) + [(int(rp[N // 2]), int(rp[N // 2 + 20]))]}
        ref = off.RefInfo(path)
        for case, pred in cases.items():
            mine = official_score(meta, pred)[0]
            arr = np.array(pred) if pred else np.zeros((0, 2))
            if len(arr) == 0:
                cp = 0
            elif len(arr) == 1 and np.diff(arr)[-1] == ref.len_sig - 1:
                cp = 1
            else:
                cp = 2
            theirs = off.ur_calculate(ref.class_true, cp)
            if ref.class_true in (1, 2):
                theirs += off.ue_calculate(arr, ref.endpoints_true, ref.onset_score_range, ref.offset_score_range)
            worst = max(worst, abs(mine - theirs))
            print(f"{name:12s} class {rec['rec_class']} {case:9s} mine {mine:7.3f} official {theirs:7.3f}")
    print("max abs difference", worst)
    assert worst < 1e-9


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
