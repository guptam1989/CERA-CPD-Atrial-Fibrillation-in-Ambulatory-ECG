"""Export example traces for the qualitative figure.

Pre-specified selection rule (independent of performance): in outer fold 0,
for each recording class (non-AF, persistent, paroxysmal), the test recording
whose duration is closest to 600 s. Detector: fold-0 seed-0 full model with the
validation-selected threshold.
python -m experiments.cera_exp.trace_export --prep PREP --folds F --runs RUNS --evals RESULTS --out traces.json
"""
from __future__ import annotations

import argparse
import gzip
import json
import os

import numpy as np
import wfdb

from . import detect as D
from . import run_eval as RE
from .features import bandpass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--evals", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    roles = json.load(open(a.folds))["folds"][0]["roles"]
    index = json.load(open(os.path.join(a.prep, "index.json")))
    test = [m for m in index["records"] if roles[m["patient"]] == "test"]
    chosen = []
    for cls in (0, 1, 2):
        cand = [m for m in test if m["rec_class"] == cls]
        chosen.append(min(cand, key=lambda m: abs(m["duration_s"] - 600)))
    recs = [RE.load_eval_record(a.prep, m, True) for m in chosen]
    RE.attach_scores(recs, os.path.join(a.runs, "fold0", "full_s0", "scores_cpsc2021.npz"))
    cfg = json.load(gzip.open(os.path.join(a.evals, "fold0_full_s0.json.gz"), "rt"))["configs"]["cera_tuned"]
    out = dict(rule="fold-0 test recording closest to 600 s per class", A=cfg["A"], records=[])
    for m, r in zip(chosen, recs):
        trace = []
        ctx = RE.ctx_of(r)
        bds, m0 = D.cera(ctx, A=cfg["A"], seed=0, trace=trace)
        eps = D.assemble(bds, m0, len(r["rp_s"]))
        row = RE.record_metrics(r, eps)
        path = os.path.join(index["root"], "Training_set_I" if int(m["patient"].split("_")[1]) < 54 else "Training_set_II", m["record"])
        sig = wfdb.rdrecord(path).p_signal[:, 1].astype(np.float32)
        filt = bandpass(sig[:, None], m["fs"])[:, 0]
        true = r["true_samples"]
        onsets = [s / m["fs"] for s, e in true if s > 0]
        offsets = [e / m["fs"] for s, e in true if e < m["n_samples"] - 1]
        centre = onsets[0] if onsets else (offsets[0] if offsets else m["duration_s"] / 2)
        s0 = max(0, int((centre - 6) * m["fs"]))
        s1 = min(len(filt), int((centre + 6) * m["fs"]))
        bt = (r["rp_s"][r["block_start"]] + r["rp_s"][r["block_start"] + r["b"]]) / 2
        tr = {j: (mode, p, lsr) for j, mode, p, lsr in trace}
        out["records"].append(dict(
            record=m["record"], rec_class=m["rec_class"], duration_s=m["duration_s"], fs=m["fs"],
            true=[(s / m["fs"], e / m["fs"]) for s, e in true],
            pred=[(0.0 if e["lc"] else float(r["rp_s"][e["start"]]),
                   m["duration_s"] if e["rc"] else float(r["rp_s"][min(e["end"], len(r["rp_s"]) - 1)])) for e in eps],
            alarms=[dict(direction=b.direction, alarm_s=b.alarm_s, boundary_s=float(r["rp_s"][b.beat])) for b in bds],
            block_t=bt.tolist(), block_p=(1 / (1 + np.exp(-r["logit"]))).tolist(),
            log_sr=[(float(bt[j]), (tr[j][2] if tr[j][2] is not None and np.isfinite(tr[j][2]) else None), tr[j][0])
                    for j in sorted(tr)],
            ecg_t0=s0 / m["fs"], ecg=filt[s0:s1].tolist(), U=row.get("U"), m0=m0))
    json.dump(out, open(a.out, "w"))
    print(json.dumps([(x["record"], x["rec_class"], x["duration_s"], x["U"]) for x in out["records"]]))


if __name__ == "__main__":
    main()
