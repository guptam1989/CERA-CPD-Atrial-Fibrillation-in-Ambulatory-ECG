"""Dual-stream block encoder (Phase II) and training objective.

Variants used in the ablation study:
  full            rhythm + atrial streams, learned quality/rate gate
  no_atrial       rhythm stream only (gate fixed to the rhythm stream)
  no_rhythm       atrial stream only
  equal_gate      both streams, gate fixed at 0.5
  no_ectopy       full model, sampler without ectopy-nuisance enrichment
  no_matched      full model, sampler without patient-matched opposites
  no_contrastive  full model, lambda_c = 0
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

VARIANTS = ("full", "no_atrial", "no_rhythm", "equal_gate", "no_ectopy", "no_matched", "no_contrastive")


class TemporalEncoder(nn.Module):
    def __init__(self, channels: int, width: int, dropout: float):
        super().__init__()
        self.convs = nn.ModuleList()
        for kernel in (7, 5, 3):
            self.convs.append(nn.Conv1d(channels, width, kernel, padding=kernel // 2))
            channels = width
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if mask is None:
            for conv in self.convs:
                x = self.dropout(F.gelu(conv(x)))
            return x.mean(dim=-1)
        valid = mask.any(dim=1, keepdim=True).to(x.dtype)
        x = x * mask.to(x.dtype)
        for conv in self.convs:
            x = self.dropout(F.gelu(conv(x))) * valid
        return x.sum(dim=-1) / valid.sum(dim=-1).clamp_min(1)


class DualStream(nn.Module):
    def __init__(self, beats: int = 8, d: int = 64, dropout: float = 0.1, variant: str = "full"):
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant, self.d, self.beats = variant, d, beats
        self.rhythm = TemporalEncoder(1, d, dropout)
        self.atrial = TemporalEncoder(4, d, dropout)
        self.rhythm_embedding = nn.Sequential(nn.Linear(d + 5, d), nn.GELU())
        self.atrial_embedding = nn.Sequential(nn.Linear(d, d), nn.GELU())
        self.rhythm_head = nn.Linear(d, 1)
        self.atrial_head = nn.Linear(d, 1)
        self.gate = nn.Sequential(nn.Linear(2, 16), nn.GELU(), nn.Linear(16, 1))
        self.beat_head = nn.Linear(2 * d, 1)
        self.projector = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))

    def forward(self, rr, desc, atrial, mask, qmean):
        n, b, c, length = atrial.shape
        use_r = self.variant != "no_rhythm"
        use_a = self.variant != "no_atrial"
        hr = self.rhythm_embedding(torch.cat((self.rhythm(rr[:, None, :]), desc), dim=1))
        ha = self.atrial_embedding(self.atrial(atrial.reshape(n * b, c, length), mask.reshape(n * b, c, length)))
        ha = ha.reshape(n, b, self.d)
        beat_valid = mask.reshape(n, b, -1).any(dim=-1)
        w = beat_valid[..., None].to(ha.dtype)
        ha_block = (ha * w).sum(dim=1) / w.sum(dim=1).clamp_min(1)
        if not use_r:
            hr = torch.zeros_like(hr)
        if not use_a:
            ha = torch.zeros_like(ha)
            ha_block = torch.zeros_like(ha_block)
        lr = self.rhythm_head(hr).squeeze(-1)
        la = self.atrial_head(ha_block).squeeze(-1)
        if self.variant == "no_atrial":
            g = torch.ones_like(lr)
        elif self.variant == "no_rhythm":
            g = torch.zeros_like(lr)
        elif self.variant == "equal_gate":
            g = torch.full_like(lr, 0.5)
        else:
            g = torch.sigmoid(self.gate(torch.stack((qmean, desc[:, 0]), dim=1))).squeeze(-1)
        fused = g * lr + (1 - g) * la
        beat = self.beat_head(torch.cat((ha, hr[:, None, :].expand(-1, b, -1)), dim=-1)).squeeze(-1)
        proj = F.normalize(self.projector(torch.cat((hr, ha_block), dim=1)), dim=-1)
        return dict(fused=fused, rhythm=lr, atrial=la, gate=g, beat=beat, beat_valid=beat_valid, proj=proj)


def supcon(proj: torch.Tensor, labels: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
    n = len(labels)
    if n < 2:
        return proj.sum() * 0
    logits = proj @ proj.T / temperature
    eye = torch.eye(n, device=proj.device, dtype=torch.bool)
    pos = (labels[:, None] == labels[None, :]) & ~eye
    anchors = pos.any(dim=1)
    if not anchors.any():
        return proj.sum() * 0
    denom = torch.logsumexp(logits.masked_fill(eye, -torch.inf), dim=1)
    logp = logits - denom[:, None]
    mean_pos = logp.masked_fill(~pos, 0).sum(dim=1) / pos.sum(dim=1).clamp_min(1)
    return -mean_pos[anchors].mean()


def objective(out, y, yb, variant: str, ls=0.5, lb=0.5, lc=0.2, temperature=0.1):
    fused = F.binary_cross_entropy_with_logits(out["fused"], y)
    streams = torch.zeros((), device=y.device)
    if variant != "no_atrial":
        streams = streams + F.binary_cross_entropy_with_logits(out["atrial"], y)
    if variant != "no_rhythm":
        streams = streams + F.binary_cross_entropy_with_logits(out["rhythm"], y)
    sup = out["beat_valid"] & (yb >= 0)
    beat = (F.binary_cross_entropy_with_logits(out["beat"][sup], yb[sup].float())
            if sup.any() else out["beat"].sum() * 0)
    con = supcon(out["proj"].float(), y, temperature) if (variant != "no_contrastive" and lc > 0) else fused * 0
    total = fused + ls * streams + lb * beat + lc * con
    return total, dict(fused=float(fused.detach()), streams=float(streams.detach()),
                       beat=float(beat.detach()), con=float(con.detach()))
