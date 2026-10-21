"""Prepare a dataset: one compressed NPZ per record plus index.json.

python -m experiments.cera_exp.prep --dataset cpsc2021 --root DATA --out PREP --workers 8
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

from .features import PrepConfig, block_features, config_dict
from .records import READERS


def list_records(dataset: str, root: str) -> list[str]:
    if dataset == "cpsc2021":
        heads = sorted(glob.glob(os.path.join(root, "Training_set_I", "*.hea")) +
                       glob.glob(os.path.join(root, "Training_set_II", "*.hea")))
        return [h[:-4] for h in heads]
    names = open(os.path.join(root, "RECORDS")).read().split()
    return [os.path.join(root, n) for n in names if os.path.exists(os.path.join(root, n + ".dat"))]


def process(args):
    dataset, path, out, cfg_kwargs, noise = args
    cfg = PrepConfig(**cfg_kwargs)
    rec = READERS[dataset](path)
    if noise is not None:
        from .noise import add_noise
        rec["sig"] = add_noise(rec["sig"], rec["fs"], **noise)
    feats = block_features(rec, cfg)
    extra = {}
    if dataset == "cpsc2021":
        extra = dict(ann_samples=rec["ann_samples"].astype(np.int64), ann_flag=rec["ann_flag"])
    fname = rec["record"] + ".npz"
    np.savez_compressed(os.path.join(out, fname), **feats, **extra)
    lab = rec["labels"]
    meta = dict(record=rec["record"], patient=rec["patient"], dataset=dataset, file=fname,
                rec_class=rec["rec_class"], fs=rec["fs"], n_samples=int(rec["n_samples"]),
                duration_s=float(rec["n_samples"] / rec["fs"]), n_beats=int(len(lab)),
                af_beats=int((lab == 1).sum()), unknown_beats=int((lab < 0).sum()),
                n_blocks=int(len(feats["block_start"])), lead_names=rec["lead_names"],
                rejected_beats=int((feats["quality"].astype(np.float32) < cfg.q_min).sum()),
                beat_source=rec.get("beat_source", "atr"))
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=sorted(READERS))
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--block", type=int, default=8)
    ap.add_argument("--q-min", type=float, default=0.5)
    ap.add_argument("--records", nargs="*", help="optional record basenames to restrict")
    ap.add_argument("--noise", help="JSON dict for noise.add_noise (stress test)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cfg = dict(block=args.block, q_min=args.q_min)
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
                 noise=noise, records=metas, created=time.strftime("%Y-%m-%d %H:%M:%S"))
    with open(os.path.join(args.out, "index.json"), "w") as f:
        json.dump(index, f, indent=1)
    print(json.dumps(dict(records=len(metas), blocks=sum(m["n_blocks"] for m in metas),
                          hours=sum(m["duration_s"] for m in metas) / 3600, seconds=time.time() - t0)))


if __name__ == "__main__":
    main()
