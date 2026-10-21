"""Prepare a dataset on the end-to-end track: R peaks from the causal detector.

python -m experiments.cera_exp.prep_qrs --dataset afdb --root DATA/afdb --out PREP_QRS/afdb --workers 8

Identical to prep.py except that the reference beat annotations are replaced by
the causal two-lead Pan--Tompkins-type detector of qrs.py. Rhythm labels are
re-evaluated at the detected beat times from the reference rhythm annotations
(CPSC2021: official episode sample intervals), symbols are copied from matched
reference beats (unmatched detections: 'Q'), and every block is released only
after the detector has confirmed the following beat and the final beat's
quality window has elapsed. Detector accuracy against the reference beats
(+/-150 ms, one-to-one, within the annotated span) is stored per record in
index.json.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import wfdb

from .features import PrepConfig, block_features, config_dict
from .prep import list_records
from .qrs import detect_qrs, match_beats
from .records import READERS, _clean, _labels_from_rhythm


def detected_labels(dataset: str, path: str, rec: dict, det: np.ndarray) -> np.ndarray:
    if dataset == "cpsc2021":
        lab = np.zeros(len(det), dtype=np.int8)
        for a, b in rec["ref_episodes"]:
            lab[(det >= a) & (det < b)] = 1
        return lab
    if dataset == "nsrdb":
        return np.zeros(len(det), dtype=np.int8)
    ann = wfdb.rdann(path, "atr")
    notes = [_clean(n) for n in ann.aux_note]
    idx = [i for i, n in enumerate(notes) if n.startswith("(")]
    return _labels_from_rhythm(det, np.asarray(ann.sample)[idx], [notes[i] for i in idx])


def process(args):
    dataset, path, out, cfg_kwargs, noise = args
    cfg = PrepConfig(**cfg_kwargs)
    rec = READERS[dataset](path)
    if noise is not None:
        from .noise import add_noise
        rec["sig"] = add_noise(rec["sig"], rec["fs"], **noise)
    fs = rec["fs"]
    ref, ref_sym = rec["rpeaks"], rec["symbols"]
    det, conf = detect_qrs(rec["sig"], fs)
    tol = int(round(0.15 * fs))
    tp, fn, _, m = match_beats(ref, det, tol)
    # accuracy over the annotated span only: detections in unannotated stretches
    # (e.g. after the last reference beat) cannot be judged against the reference
    in_span = (det >= ref[0] - tol) & (det <= ref[-1] + tol) if len(ref) else np.zeros(len(det), bool)
    fp = int(in_span.sum()) - tp
    outside = int((~in_span).sum())
    sym = np.full(len(det), "Q", dtype=ref_sym.dtype if len(ref_sym) else "<U1")
    ok = m >= 0
    sym[m[ok]] = ref_sym[ok]
    lab = detected_labels(dataset, path, rec, det)
    drec = dict(rec, rpeaks=det, symbols=sym, labels=lab)
    feats = block_features(drec, cfg)
    bs = feats["block_start"].astype(np.int64)
    b = cfg.block
    rp_s = det / fs
    if len(bs):
        feats["available"] = np.maximum(conf[bs + b] / fs, rp_s[bs + b - 1] + cfg.quality_post_window)
    extra = dict(confirm_samples=conf.astype(np.int64))
    if dataset == "cpsc2021":
        extra.update(ann_samples=rec["ann_samples"].astype(np.int64), ann_flag=rec["ann_flag"])
    fname = rec["record"] + ".npz"
    np.savez_compressed(os.path.join(out, fname), **feats, **extra)
    lat = (conf - det) / fs
    meta = dict(record=rec["record"], patient=rec["patient"], dataset=dataset, file=fname,
                rec_class=rec["rec_class"], fs=fs, n_samples=int(rec["n_samples"]),
                duration_s=float(rec["n_samples"] / fs), n_beats=int(len(lab)),
                af_beats=int((lab == 1).sum()), unknown_beats=int((lab < 0).sum()),
                n_blocks=int(len(bs)), lead_names=rec["lead_names"],
                rejected_beats=int((feats["quality"].astype(np.float32) < cfg.q_min).sum()),
                beat_source="pan_tompkins_causal", ref_beats=int(len(ref)), qrs_tp=tp, qrs_fn=fn, qrs_fp=fp,
                det_outside_annotated_span=outside,
                ref_af_beats=int((rec["labels"] == 1).sum()),
                qrs_latency_median_s=float(np.median(lat)) if len(lat) else None,
                qrs_latency_p95_s=float(np.percentile(lat, 95)) if len(lat) else None,
                qrs_latency_max_s=float(lat.max()) if len(lat) else None)
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(READERS))
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--records", nargs="*", help="optional record basenames to restrict")
    ap.add_argument("--noise", help="JSON dict for noise.add_noise (stress test)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cfg = dict(block=8, q_min=0.5)
    paths = list_records(args.dataset, args.root)
    if args.records:
        keep = set(args.records)
        paths = [p for p in paths if os.path.basename(p) in keep]
    noise = json.loads(args.noise) if args.noise else None
    t0 = time.time()
    jobs = [(args.dataset, p, args.out, cfg, noise) for p in paths]
    metas = []
    with ProcessPoolExecutor(args.workers) as ex:
        for i, m in enumerate(ex.map(process, jobs, chunksize=1)):
            metas.append(m)
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(jobs)} records, {time.time() - t0:.0f}s", flush=True)
    index = dict(dataset=args.dataset, root=os.path.abspath(args.root), config=config_dict(PrepConfig(**cfg)),
                 noise=noise, records=metas, beat_source="pan_tompkins_causal",
                 created=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(os.path.join(args.out, "index.json"), "w") as f:
        json.dump(index, f, indent=1)
    tp = sum(m["qrs_tp"] for m in metas)
    fn = sum(m["qrs_fn"] for m in metas)
    fp = sum(m["qrs_fp"] for m in metas)
    print(json.dumps(dict(records=len(metas), blocks=sum(m["n_blocks"] for m in metas),
                          hours=sum(m["duration_s"] for m in metas) / 3600, qrs_se=tp / max(1, tp + fn),
                          qrs_ppv=tp / max(1, tp + fp), seconds=time.time() - t0)))


if __name__ == "__main__":
    main()
