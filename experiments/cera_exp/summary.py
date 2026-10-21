"""Pooled metric summaries over record rows produced by evaluate.record_metrics."""
from __future__ import annotations

import numpy as np


def _safe(a, b):
    return float(a) / float(b) if b else float("nan")


def sweep_summary(rows: list[dict]) -> dict:
    """A displayed summary plus sufficient statistics for exact outer-fold pooling."""
    out = summarize(rows)
    finite = [float(d) for r in rows for d in r.get("delays", [])
              if d is not None and np.isfinite(d)]
    out["pooling_v1"] = {
        "records": len(rows),
        "U_sum": float(sum(r["U"] for r in rows)),
        "hours": float(sum(r["hours"] for r in rows)),
        "false_eps": int(sum(r["false_eps"] for r in rows)),
        "n_true_eps": int(sum(r["n_true_eps"] for r in rows)),
        "n_pred_eps": int(sum(r["n_pred_eps"] for r in rows)),
        "ep_match_0.5": int(sum(r["ep_match_0.5"] for r in rows)),
        "finite_delays_s": finite,
    }
    return out


def pool_sweep_summaries(summaries: list[dict]) -> dict:
    """Pool counts and raw finite delays; legacy fold means cannot recover these."""
    if not summaries or any("pooling_v1" not in s for s in summaries):
        raise ValueError("Operating curves require pooling_v1 sufficient statistics; regenerate legacy sweeps.")
    stats = [s["pooling_v1"] for s in summaries]
    nt = sum(s["n_true_eps"] for s in stats)
    npred = sum(s["n_pred_eps"] for s in stats)
    tp = sum(s["ep_match_0.5"] for s in stats)
    delays = [max(float(d), 0.0) for s in stats for d in s["finite_delays_s"]]
    if not all(np.isfinite(d) for d in delays):
        raise ValueError("pooling_v1 finite_delays_s contains a non-finite delay")
    return {
        "U": _safe(sum(s["U_sum"] for s in stats), sum(s["records"] for s in stats)),
        "fa_h": _safe(sum(s["false_eps"] for s in stats), sum(s["hours"] for s in stats)),
        "ep_f1_0.5": _safe(2 * tp, nt + npred) if nt + npred else 0.0,
        "delay_median": float(np.median(delays)) if delays else float("nan"),
    }


def summarize(rows: list[dict]) -> dict:
    out = {}
    if not rows:
        return out
    if all("U" in r for r in rows):
        out["U"] = float(np.mean([r["U"] for r in rows]))
        out["Ur"] = float(np.mean([r["Ur"] for r in rows]))
        out["Ue"] = float(np.mean([r["Ue"] for r in rows]))
        out["class_acc"] = float(np.mean([r["class_pred"] == r["class_true"] for r in rows]))
    n_on = sum(r["n_true_on"] for r in rows)
    n_off = sum(r["n_true_off"] for r in rows)
    p_on = sum(r["n_pred_on"] for r in rows)
    p_off = sum(r["n_pred_off"] for r in rows)
    for tol in (1, 2, 5):
        m_on = sum(r[f"on_match_{tol}"] for r in rows)
        m_off = sum(r[f"off_match_{tol}"] for r in rows)
        out[f"on_se_{tol}"] = _safe(m_on, n_on)
        out[f"off_se_{tol}"] = _safe(m_off, n_off)
        out[f"on_ppv_{tol}"] = _safe(m_on, p_on)
        out[f"off_ppv_{tol}"] = _safe(m_off, p_off)
        out[f"bnd_se_{tol}"] = _safe(m_on + m_off, n_on + n_off)
        out[f"bnd_ppv_{tol}"] = _safe(m_on + m_off, p_on + p_off)
    nt = sum(r["n_true_eps"] for r in rows)
    npd = sum(r["n_pred_eps"] for r in rows)
    for thr in (0.1, 0.3, 0.5):
        tp = sum(r[f"ep_match_{thr}"] for r in rows)
        prec, rec = _safe(tp, npd), _safe(tp, nt)
        out[f"ep_f1_{thr}"] = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
        out[f"ep_prec_{thr}"] = prec
        out[f"ep_rec_{thr}"] = rec
    hours = sum(r["hours"] for r in rows)
    out["hours"] = hours
    out["fa_h"] = _safe(sum(r["false_eps"] for r in rows), hours)
    loc = [e for r in rows for e in r["loc_errors"]]
    out["loc_median"] = float(np.median(loc)) if loc else float("nan")
    delays = [float(d) if d is not None else float("inf") for r in rows for d in r.get("delays", [])]
    if delays:
        d = np.array(delays, dtype=float)
        fin = d[np.isfinite(d)]
        out["delay_median"] = float(np.median(np.maximum(fin, 0))) if len(fin) else float("nan")
        out["detected"] = float(np.isfinite(d).mean())
        for s in (10, 30, 60):
            out[f"det_{s}s"] = float((d <= s).mean())
    bins = [(5, 20), (20, 100), (100, 1000), (1000, 10 ** 9)]
    for lo, hi in bins:
        hits, dls = [], []
        for r in rows:
            for k, nb in enumerate(r.get("run_beats", [])):
                if lo <= nb < hi:
                    hits.append(r["run_hit"][k])
                    if r.get("run_delay") is not None and r["run_delay"][k] is not None:
                        dls.append(r["run_delay"][k])
        out[f"det_rate_{lo}"] = float(np.mean(hits)) if hits else float("nan")
        out[f"n_eps_{lo}"] = len(hits)
        out[f"delay_med_{lo}"] = float(np.median(np.maximum(dls, 0))) if dls else float("nan")
    burden_errors = [abs(r["B_hat"] - r["B"]) for r in rows
                     if r.get("B_hat") is not None and r.get("B") is not None]
    out["burden_mae_pp"] = 100 * float(np.mean(burden_errors)) if burden_errors else float("nan")
    out["burden_mae_n"] = len(burden_errors)
    out["label_known_frac"] = _safe(sum(r.get("T_label_known", r["T"]) for r in rows), sum(r["T"] for r in rows))
    out["partial_reference_records"] = sum(r.get("T_label_unknown", 0.0) > 0 for r in rows)
    out["obs_frac"] = _safe(sum(r["T_o"] for r in rows), sum(r["T"] for r in rows))
    out["records"] = len(rows)
    out["fallback_offsets"] = int(sum(r.get("fallback_offsets", 0) for r in rows))
    for tag in ("norm", "abs"):
        cov_o = [r[f"cov_obs_{tag}"] for r in rows if r.get(f"cov_obs_{tag}") is not None]
        cov_f = [r[f"cov_full_{tag}"] for r in rows if r.get(f"cov_full_{tag}") is not None]
        out[f"cov_obs_{tag}"] = float(np.mean(cov_o)) if cov_o else float("nan")
        out[f"cov_full_{tag}"] = float(np.mean(cov_f)) if cov_f else float("nan")
        out[f"cov_obs_n_{tag}"] = len(cov_o)
        out[f"cov_full_n_{tag}"] = len(cov_f)
        for target in ("obs", "full"):
            for status in ("possible", "certain"):
                key = f"cov_{target}_{status}_{tag}"
                vals = [r[key] for r in rows if r.get(key) is not None]
                out[key] = float(np.mean(vals)) if vals else float("nan")
                out[f"{key}_n"] = len(vals)
        w_o = [r[f"int_{tag}"][1] - r[f"int_{tag}"][0] for r in rows if r.get(f"int_{tag}")]
        w_f = [r[f"int_{tag}"][3] - r[f"int_{tag}"][2] for r in rows if r.get(f"int_{tag}")]
        out[f"width_obs_{tag}"] = float(np.mean(w_o)) if w_o else float("nan")
        out[f"width_full_{tag}"] = float(np.mean(w_f)) if w_f else float("nan")
    return out
