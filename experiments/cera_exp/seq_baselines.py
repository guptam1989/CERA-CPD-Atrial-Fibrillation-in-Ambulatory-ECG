"""Deep beat-sequence comparators trained on the same patient folds.

  rr_lstm   causal two-layer LSTM over RR features (window classifier family,
            cf. Faust et al. 2018); streamed over the whole record at inference.
  unet      offline 1-D U-Net beat segmenter over RR features (direct event
            segmentation family, cf. Wen & Kang 2022); non-causal, sliding
            512-beat windows with 50% overlap.

Per-beat features: log RR_prev, log RR_next (U-Net only; next interval is
future information), RR_prev / causal median of 16 previous RR, |dRR|.
python -m experiments.cera_exp.seq_baselines --prep PREP --folds F --fold 0 --runs RUNS --model rr_lstm
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def beat_features(rp_s: np.ndarray, causal: bool) -> np.ndarray:
    rr = np.diff(rp_s)
    rr_prev = np.concatenate([[rr[0] if len(rr) else 0.8], rr])
    rr_next = np.concatenate([rr, [rr[-1] if len(rr) else 0.8]])
    rr_prev = np.clip(rr_prev, 0.2, 3.0)
    rr_next = np.clip(rr_next, 0.2, 3.0)
    n = len(rr_prev)
    from numpy.lib.stride_tricks import sliding_window_view
    pad = np.concatenate([np.full(15, rr_prev[0]), rr_prev])
    med = np.median(sliding_window_view(pad, 16), axis=1)[:n]
    drr = np.abs(np.diff(rr_prev, prepend=rr_prev[0]))
    cols = [np.log(rr_prev), rr_prev / med - 1, drr / med]
    if not causal:
        cols.append(np.log(rr_next))
        cols.append(rr_next / med - 1)
    return np.stack(cols, axis=1).astype(np.float32)


class RRLSTM(nn.Module):
    def __init__(self, nin=3, h=64):
        super().__init__()
        self.lstm = nn.LSTM(nin, h, num_layers=2, batch_first=True, dropout=0.1)
        self.head = nn.Linear(h, 1)

    def forward(self, x, state=None):
        y, state = self.lstm(x, state)
        return self.head(y).squeeze(-1), state


class UNet1D(nn.Module):
    def __init__(self, nin=5, w=32):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(nn.Conv1d(i, o, 7, padding=3), nn.GELU(), nn.Conv1d(o, o, 7, padding=3), nn.GELU())
        self.e1, self.e2, self.e3 = blk(nin, w), blk(w, 2 * w), blk(2 * w, 4 * w)
        self.b = blk(4 * w, 4 * w)
        self.d3, self.d2, self.d1 = blk(8 * w, 2 * w), blk(4 * w, w), blk(2 * w, w)
        self.out = nn.Conv1d(w, 1, 1)

    def forward(self, x):                       # x [B, T, C], T divisible by 8
        x = x.transpose(1, 2)
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool1d(e1, 2))
        e3 = self.e3(F.max_pool1d(e2, 2))
        b = self.b(F.max_pool1d(e3, 2))
        d3 = self.d3(torch.cat([F.interpolate(b, scale_factor=2), e3], 1))
        d2 = self.d2(torch.cat([F.interpolate(d3, scale_factor=2), e2], 1))
        d1 = self.d1(torch.cat([F.interpolate(d2, scale_factor=2), e1], 1))
        return self.out(d1).squeeze(1)


def load_beats(prep: str, metas: list[dict], causal: bool):
    xs, ys = [], []
    for m in metas:
        z = np.load(os.path.join(prep, m["file"]))
        rp_s = z["rpeaks"] / m["fs"]
        xs.append(beat_features(rp_s, causal))
        ys.append(z["labels"].astype(np.int8))
    return xs, ys


def chunks(xs, ys, T, rng, n):
    lens = np.array([len(x) for x in xs])
    w = np.maximum(lens - T, 1).astype(float)
    pick = rng.choice(len(xs), size=n, p=w / w.sum())
    X = np.zeros((n, T, xs[0].shape[1]), np.float32)
    Y = np.full((n, T), -1, np.int8)
    for i, r in enumerate(pick):
        L = lens[r]
        s = int(rng.integers(0, max(1, L - T + 1)))
        seg = slice(s, min(L, s + T))
        X[i, :seg.stop - seg.start] = xs[r][seg]
        Y[i, :seg.stop - seg.start] = ys[r][seg]
    return X, Y


def masked_bce(logit, y):
    m = y >= 0
    return F.binary_cross_entropy_with_logits(logit[m], y[m].float()) if m.any() else logit.sum() * 0


@torch.no_grad()
def infer(model, x: np.ndarray, kind: str, device) -> np.ndarray:
    model.eval()
    if kind == "rr_lstm":
        # stream in chunks carrying the recurrent state (identical causal output;
        # avoids cuDNN limits on day-long sequences)
        state, outs = None, []
        for s in range(0, len(x), 8192):
            chunk = torch.from_numpy(np.ascontiguousarray(x[s:s + 8192]))[None].to(device)
            logit, state = model(chunk, state)
            outs.append(torch.sigmoid(logit[0]).cpu().numpy())
        return np.concatenate(outs) if outs else np.zeros(0, np.float32)
    T, hop = 512, 256
    n = len(x)
    L = max(T, int(math.ceil((n - T) / hop)) * hop + T) if n > T else T
    xp = np.zeros((L, x.shape[1]), np.float32)
    xp[:n] = x
    acc = np.zeros(L)
    cnt = np.zeros(L)
    w = np.hanning(T) + 1e-3
    starts = list(range(0, L - T + 1, hop))
    batch = torch.from_numpy(np.stack([xp[s:s + T] for s in starts])).to(device)
    prob = torch.sigmoid(model(batch)).cpu().numpy()
    for k, s in enumerate(starts):
        acc[s:s + T] += prob[k] * w
        cnt[s:s + T] += w
    return (acc / cnt)[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prep", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--runs", required=True)
    ap.add_argument("--model", choices=("rr_lstm", "unet"), required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--external", default="")
    ap.add_argument("--epochs", type=int, default=40)
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    keep = torch.ones(4, device=device)
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    causal = a.model == "rr_lstm"
    roles = json.load(open(a.folds))["folds"][a.fold]["roles"]
    index = json.load(open(os.path.join(a.prep, "index.json")))
    tr = [m for m in index["records"] if roles[m["patient"]] == "train"]
    va = [m for m in index["records"] if roles[m["patient"]] == "validation"]
    xs, ys = load_beats(a.prep, tr, causal)
    vx, vy = load_beats(a.prep, va, causal)
    out_dir = os.path.join(a.runs, f"fold{a.fold}", f"{a.model}_s{a.seed}")
    os.makedirs(out_dir, exist_ok=True)
    model = (RRLSTM(xs[0].shape[1]) if causal else UNet1D(xs[0].shape[1])).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    T = 256 if causal else 512
    best, best_state, wait, hist = math.inf, None, 0, []
    ckpt = os.path.join(out_dir, "model.pt")
    if os.path.exists(ckpt) and os.path.exists(os.path.join(out_dir, "history.json")):
        best_state = torch.load(ckpt, map_location="cpu")      # reuse a completed fit
        a.epochs = 0
        reused = True
    else:
        reused = False
    for ep in range(a.epochs):
        t0 = time.time()
        model.train()
        X, Y = chunks(xs, ys, T, rng, 128 * 100)
        tot = 0.0
        for s in range(0, len(X), 128):
            xb = torch.from_numpy(X[s:s + 128]).to(device)
            yb = torch.from_numpy(Y[s:s + 128]).to(device)
            logit = model(xb)[0] if causal else model(xb)
            loss = masked_bce(logit, yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += float(loss) / (len(X) / 128)
        ll, nn_ = 0.0, 0
        for x, y in zip(vx, vy):
            p = np.clip(infer(model, x, a.model, device), 1e-6, 1 - 1e-6)
            m = y >= 0
            ll += float(-(y[m] * np.log(p[m]) + (1 - y[m]) * np.log(1 - p[m])).sum())
            nn_ += int(m.sum())
        vb = ll / max(nn_, 1)
        hist.append(dict(epoch=ep, train=tot, val_bce=vb, seconds=time.time() - t0))
        print(json.dumps(hist[-1]), flush=True)
        if vb < best - 1e-4:
            best, wait = vb, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            # RR-LSTM fits often plateau at the class prior for several epochs before
            # learning; a longer patience and a 10-epoch minimum avoid premature stops.
            if wait >= (15 if causal else 6) and ep >= 10:
                break
    model.load_state_dict(best_state)
    if not reused:
        torch.save(best_state, os.path.join(out_dir, "model.pt"))
        json.dump(dict(history=hist, best_val_bce=best, params=sum(p.numel() for p in model.parameters())),
                  open(os.path.join(out_dir, "history.json"), "w"), indent=1)
    for prep in [a.prep] + [e for e in a.external.split(",") if e]:
        tag = "cpsc2021" if prep == a.prep else os.path.basename(os.path.normpath(prep))
        idx = json.load(open(os.path.join(prep, "index.json")))
        probs, counts = [], []
        for m in idx["records"]:
            z = np.load(os.path.join(prep, m["file"]))
            x = beat_features(z["rpeaks"] / m["fs"], causal)
            probs.append(infer(model, x, a.model, device).astype(np.float32))
            counts.append(len(x))
        np.savez(os.path.join(out_dir, f"beats_{tag}.npz"), records=np.array([m["record"] for m in idx["records"]]),
                 offsets=np.concatenate([[0], np.cumsum(counts)]), prob=np.concatenate(probs))
        print("exported", tag, flush=True)
    keep.add_(0)


if __name__ == "__main__":
    main()
