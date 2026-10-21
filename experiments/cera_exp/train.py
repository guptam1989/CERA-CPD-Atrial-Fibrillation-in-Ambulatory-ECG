"""Train dual-stream encoders per outer fold and export frozen block/beat scores.

python -m experiments.cera_exp.train --prep PREP/cpsc2021 --folds folds.json --fold 0 \
    --runs RUNS --plan "full:0,1,2,3,4;no_atrial:0" --external PREP/afdb,PREP/ltafdb

Fitting uses only training-patient blocks; checkpoint selection uses only
validation-patient blocks. Scores are exported for every CPSC2021 record (roles
are attached later from the fold file) and for every external dataset.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
import zlib

import numpy as np
import torch

from .model import DualStream, objective

FIELDS = ("rr", "desc", "atrial", "mask", "qmean")


def load_records(prep: str, names: set[str] | None = None, labels: bool = True, counts: list | None = None):
    index = json.load(open(os.path.join(prep, "index.json")))
    recs = [r for r in index["records"] if names is None or r["record"] in names]
    parts = {k: [] for k in FIELDS + ("y", "yb", "ectopy", "patient", "record")}
    for i, r in enumerate(recs):
        z = np.load(os.path.join(prep, r["file"]))
        J = len(z["block_start"])
        if counts is not None:
            counts.append(J)
        if J == 0:
            continue
        for k in FIELDS:
            parts[k].append(z[k])
        if labels:
            b = z["atrial"].shape[1]
            bi = z["block_start"][:, None] + np.arange(b)[None, :]
            parts["y"].append(z["block_label"])
            parts["yb"].append(z["labels"][bi])
            parts["ectopy"].append(z["ectopy"])
        parts["patient"].append(np.full(J, zlib.crc32(r["patient"].encode()), np.int64))
        parts["record"].append(np.full(J, i, np.int64))
    return {k: np.concatenate(v) for k, v in parts.items() if v}, recs


def to_device(arrays: dict, device):
    out = {}
    for k, v in arrays.items():
        t = torch.from_numpy(np.ascontiguousarray(v))
        out[k] = t.to(device)
    return out


def batch_inputs(d: dict, idx: torch.Tensor):
    return (d["rr"][idx].float(), d["desc"][idx].float(), d["atrial"][idx].float(),
            d["mask"][idx], d["qmean"][idx].float())


class Sampler:
    """Half uniform, a quarter ectopy-nuisance non-AF blocks, a quarter
    same-patient opposite-rhythm partners of uniform anchors."""

    def __init__(self, d: dict, variant: str, batch: int, gen: torch.Generator):
        self.n = len(d["y"])
        self.batch, self.gen, self.variant = batch, gen, variant
        dev = d["y"].device
        self.ect = torch.nonzero(d["ectopy"] & (d["y"] == 0)).squeeze(1)
        pat = d["patient"]
        _, pid = torch.unique(pat, return_inverse=True)
        self.pid = pid
        lab = d["y"].long()
        key = pid * 2 + lab
        self.order = torch.argsort(key)
        counts = torch.bincount(key, minlength=int(pid.max() + 1) * 2)
        self.counts = counts
        self.starts = torch.cumsum(counts, 0) - counts
        self.lab = lab
        self.device = dev

    def _rand(self, n, high):
        return (torch.rand(n, generator=self.gen, device=self.device) * high).long().clamp_max(high - 1)

    def __call__(self):
        B = self.batch
        nb, ne = B // 2, B // 4
        nm = B - nb - ne
        base = self._rand(nb, self.n)
        if self.variant != "no_ectopy" and len(self.ect) > 0:
            ect = self.ect[self._rand(ne, len(self.ect))]
        else:
            ect = self._rand(ne, self.n)
        if self.variant != "no_matched":
            anchors = base[:nm]
            key = self.pid[anchors] * 2 + (1 - self.lab[anchors])
            c = self.counts[key]
            r = (torch.rand(nm, generator=self.gen, device=self.device) * c.clamp_min(1)).long()
            match = self.order[(self.starts[key] + r).clamp_max(self.n - 1)]
            match = torch.where(c > 0, match, self._rand(nm, self.n))
        else:
            match = self._rand(nm, self.n)
        return torch.cat([base, ect, match])


@torch.no_grad()
def predict(model, d: dict, batch: int = 4096):
    model.eval()
    n = len(d["rr"])
    out = {k: [] for k in ("p", "pr", "pa", "g", "pb", "lp")}
    for s in range(0, n, batch):
        idx = torch.arange(s, min(n, s + batch), device=d["rr"].device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=d["rr"].is_cuda):
            o = model(*batch_inputs(d, idx))
        out["p"].append(torch.sigmoid(o["fused"].float()))
        out["lp"].append(o["fused"].float())
        out["pr"].append(torch.sigmoid(o["rhythm"].float()))
        out["pa"].append(torch.sigmoid(o["atrial"].float()))
        out["g"].append(o["gate"].float())
        out["pb"].append(torch.sigmoid(o["beat"].float()))
    return {k: torch.cat(v).cpu().numpy() for k, v in out.items()}


def augment(inputs, gen: torch.Generator, cfg: dict):
    """Training-only waveform augmentation: per-channel log-normal amplitude
    scaling, additive Gaussian noise, random lead swap and a common time jitter.
    RR inputs receive small multiplicative jitter. Masks are preserved."""
    rr, desc, atrial, mask, qmean = inputs
    B = atrial.shape[0]
    dev = atrial.device
    scale = torch.exp(torch.randn((B, 1, 4, 1), generator=gen, device=dev) * cfg["aug_scale"])
    atrial = atrial * scale + torch.randn(atrial.shape, generator=gen, device=dev) * cfg["aug_noise"]
    swap = torch.rand(B, generator=gen, device=dev) < 0.5
    perm = torch.tensor([1, 0, 3, 2], device=dev)
    atrial = torch.where(swap[:, None, None, None], atrial[:, :, perm], atrial)
    mask = torch.where(swap[:, None, None, None], mask[:, :, perm], mask)
    shift = int(torch.randint(-cfg["aug_shift"], cfg["aug_shift"] + 1, (1,), generator=gen, device=dev))
    if shift:
        atrial = torch.roll(atrial, shift, dims=-1)
        mask = torch.roll(mask, shift, dims=-1)
    atrial = atrial * mask.to(atrial.dtype)
    rr = rr * torch.exp(torch.randn(rr.shape, generator=gen, device=dev) * cfg["aug_rr"])
    return rr, desc, atrial, mask, qmean


def fit(train: dict, val: dict, variant: str, seed: int, cfg: dict, log=print):
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = train["rr"].device
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    model = DualStream(beats=train["atrial"].shape[1], d=cfg["d"], dropout=cfg["dropout"], variant=variant).to(device)
    ema = torch.optim.swa_utils.AveragedModel(
        model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(cfg["ema"])) if cfg["ema"] else None
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["wd"])
    steps = min(cfg["max_steps_per_epoch"], max(1, len(train["y"]) // cfg["batch"]))
    total = steps * cfg["epochs"]
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    sampler = Sampler(train, variant, cfg["batch"], gen)
    lc = 0.0 if variant == "no_contrastive" else cfg["lc"]
    best, best_state, wait, history = math.inf, None, 0, []
    for epoch in range(cfg["epochs"]):
        model.train()
        t0 = time.time()
        agg = {}
        for _ in range(steps):
            idx = sampler()
            y = train["y"][idx].float()
            yb = train["yb"][idx].long()
            inputs = batch_inputs(train, idx)
            if cfg["augment"]:
                inputs = augment(inputs, gen, cfg)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=train["rr"].is_cuda):
                out = model(*inputs)
            out = {k: (v.float() if torch.is_floating_point(v) else v) for k, v in out.items()}
            loss, parts = objective(out, y, yb, variant, cfg["ls"], cfg["lb"], lc, cfg["temperature"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            sched.step()
            if ema is not None:
                ema.update_parameters(model)
            for k, v in parts.items():
                agg[k] = agg.get(k, 0.0) + v / steps
        eval_model = ema.module if ema is not None else model
        pv = predict(eval_model, val)["p"]
        yv = val["y"].float().cpu().numpy()
        pv = np.clip(pv, 1e-6, 1 - 1e-6)
        vbce = float(-np.mean(yv * np.log(pv) + (1 - yv) * np.log(1 - pv)))
        vacc = float(np.mean((pv >= 0.5) == (yv == 1)))
        history.append(dict(epoch=epoch, train=agg, val_bce=vbce, val_acc=vacc, seconds=time.time() - t0))
        log(json.dumps(history[-1]))
        if vbce < best - 1e-4:
            best, wait = vbce, 0
            best_state = {k: v.detach().cpu().clone() for k, v in eval_model.state_dict().items()}
        else:
            wait += 1
            if wait >= cfg["patience"]:
                break
    model.load_state_dict(best_state)
    return model, history, best


def export_scores(models: dict, prep: str, out_paths: dict, device, chunk_records: int = 64,
                  max_blocks: int = 150000):
    """Infer every record of a prepared dataset with several frozen models."""
    index = json.load(open(os.path.join(prep, "index.json")))
    recs = index["records"]
    collected = {name: {k: [] for k in ("p", "pr", "pa", "g", "pb", "lp")} for name in models}
    counts = []
    groups, cur, cur_blocks = [], [], 0
    for r in recs:                      # chunks bounded by records and by blocks (RAM)
        cur.append(r["record"])
        cur_blocks += r["n_blocks"]
        if len(cur) >= chunk_records or cur_blocks >= max_blocks:
            groups.append(cur)
            cur, cur_blocks = [], 0
    if cur:
        groups.append(cur)
    for group in groups:
        names = set(group)
        arrays, _ = load_records(prep, names, labels=False, counts=counts)
        if not arrays:
            continue
        d = to_device({k: arrays[k] for k in FIELDS}, device)
        for name, model in models.items():
            pred = predict(model, d)
            for k, v in pred.items():
                collected[name][k].append(v.astype(np.float32 if k == "lp" else np.float16))
        del d
    offsets = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    rec_names = np.array([r["record"] for r in recs])
    for name, path in out_paths.items():
        arrays = {k: (np.concatenate(v) if v else np.zeros((0,), np.float16)) for k, v in collected[name].items()}
        np.savez(path, records=rec_names, offsets=offsets, **arrays)


DEFAULT_CFG = dict(d=64, dropout=0.1, lr=1e-3, wd=1e-4, batch=256, epochs=40, patience=8,
                   max_steps_per_epoch=800, ls=0.5, lb=0.5, lc=0.2, temperature=0.1,
                   augment=False, aug_scale=0.25, aug_noise=0.1, aug_shift=2, aug_rr=0.02, ema=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--plan", default="full:0,1,2,3,4")
    ap.add_argument("--external", default="")
    ap.add_argument("--external-variants", default="full")
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--cfg", default="{}", help="JSON overrides of DEFAULT_CFG")
    a = ap.parse_args()
    device = torch.device(a.device if torch.cuda.is_available() else "cpu")
    keepalive = torch.ones(8, device=device)  # hold the GPU handle for the whole job
    cfg = dict(DEFAULT_CFG)
    cfg.update(json.loads(a.cfg))
    if a.epochs:
        cfg["epochs"] = a.epochs
    folds = json.load(open(a.folds))
    roles = folds["folds"][a.fold]["roles"]
    index = json.load(open(os.path.join(a.prep, "index.json")))
    by_role = {}
    for r in index["records"]:
        by_role.setdefault(roles[r["patient"]], set()).add(r["record"])
    fold_dir = os.path.join(a.runs, f"fold{a.fold}")
    os.makedirs(fold_dir, exist_ok=True)
    logf = open(os.path.join(fold_dir, "train_log.jsonl"), "a")

    def log(msg):
        print(msg, flush=True)
        logf.write(msg + "\n")
        logf.flush()

    t0 = time.time()
    tr_np, _ = load_records(a.prep, by_role["train"])
    va_np, _ = load_records(a.prep, by_role["validation"])
    keep = tr_np["y"] >= 0
    tr_np = {k: v[keep] for k, v in tr_np.items()}
    keepv = va_np["y"] >= 0
    va_np = {k: v[keepv] for k, v in va_np.items()}
    train = to_device(tr_np, device)
    val = to_device(va_np, device)
    log(json.dumps(dict(event="loaded", fold=a.fold, train_blocks=len(tr_np["y"]), val_blocks=len(va_np["y"]),
                        train_af=float(tr_np["y"].mean()), ectopy_nonaf=int((tr_np["ectopy"] & (tr_np["y"] == 0)).sum()),
                        seconds=time.time() - t0, device=str(device),
                        gpu=torch.cuda.get_device_name(0) if device.type == "cuda" else None)))
    plan = []
    for item in a.plan.split(";"):
        variant, seeds = item.split(":")
        plan += [(variant, int(s)) for s in seeds.split(",")]
    ext_variants = set(a.external_variants.split(","))
    ext_models = {}
    for variant, seed in plan:
        run_dir = os.path.join(fold_dir, f"{variant}_s{seed}")
        os.makedirs(run_dir, exist_ok=True)
        ckpt = os.path.join(run_dir, "model.pt")
        if os.path.exists(ckpt):
            model = DualStream(beats=tr_np["atrial"].shape[1], d=cfg["d"], dropout=cfg["dropout"], variant=variant).to(device)
            model.load_state_dict(torch.load(ckpt, map_location=device))
            log(json.dumps(dict(event="reuse", run=run_dir)))
        else:
            t1 = time.time()
            model, history, best = fit(train, val, variant, seed, cfg, log)
            torch.save(model.state_dict(), ckpt)
            json.dump(dict(variant=variant, seed=seed, fold=a.fold, cfg=cfg, history=history, best_val_bce=best,
                           train_records=sorted(by_role["train"]), validation_records=sorted(by_role["validation"]),
                           seconds=time.time() - t1, params=sum(p.numel() for p in model.parameters())),
                      open(os.path.join(run_dir, "history.json"), "w"), indent=1)
        sc = os.path.join(run_dir, "scores_cpsc2021.npz")
        if not os.path.exists(sc):
            export_scores({"m": model}, a.prep, {"m": sc}, device)
        if variant in ext_variants:
            ext_models[f"{variant}_s{seed}"] = model
        log(json.dumps(dict(event="done", run=run_dir, elapsed=time.time() - t0)))
    del train, val
    torch.cuda.empty_cache() if device.type == "cuda" else None
    for ext in [e for e in a.external.split(",") if e]:
        name = json.load(open(os.path.join(ext, "index.json")))["dataset"]
        tag = os.path.basename(os.path.normpath(ext))
        todo = {k: m for k, m in ext_models.items()
                if not os.path.exists(os.path.join(fold_dir, k, f"scores_{tag}.npz"))}
        if todo:
            export_scores(todo, ext, {k: os.path.join(fold_dir, k, f"scores_{tag}.npz") for k in todo}, device)
        log(json.dumps(dict(event="external", dataset=name, tag=tag, models=len(todo), elapsed=time.time() - t0)))
    keepalive.add_(0)
    log(json.dumps(dict(event="finished", elapsed=time.time() - t0)))


if __name__ == "__main__":
    main()
