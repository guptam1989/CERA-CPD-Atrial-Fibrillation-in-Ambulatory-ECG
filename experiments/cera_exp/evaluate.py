"""Record-level evaluation: official CPSC2021 score, boundary, episode,
delay, false-alarm and burden metrics.

The CPSC2021 score reproduces score_2021.py of the organisers
(github.com/CPSC-Committee/cpsc2021-python-entry): U = Ur + Ue per record,
averaged over records.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

R_MATRIX = np.array([[1, -1, -.5], [-2, 1, 0], [-1, 0, 1]])


# ---------------------------------------------------------------- official score

def official_ranges(ann_samples: np.ndarray, ann_flag: np.ndarray, n_samples: int, rec_class: int):
    """onset/offset score ranges exactly as RefInfo._gen_endpoint_score_range."""
    beat_loc = ann_samples
    starts = np.where(ann_flag == 1)[0]
    ends = np.where(ann_flag == 2)[0]
    on = np.zeros(n_samples)
    off = np.zeros(n_samples)
    n_loc = len(beat_loc)
    for s in starts:
        if rec_class == 2:
            if max(s - 1, 0) == 0:
                on[: beat_loc[s + 2]] += 1
            elif max(s - 2, 0) == 0:
                on[beat_loc[s - 1]: beat_loc[s + 2]] += 1
                on[: beat_loc[s - 1]] += .5
            else:
                on[beat_loc[s - 1]: beat_loc[s + 2]] += 1
                on[beat_loc[s - 2]: beat_loc[s - 1]] += .5
            on[beat_loc[s + 2]: beat_loc[s + 3]] += .5
        elif rec_class == 1:
            on[: beat_loc[s + 2]] += 1
            on[beat_loc[s + 2]: beat_loc[s + 3]] += .5
    for e in ends:
        if rec_class == 2:
            if min(e + 1, n_loc - 1) == n_loc - 1:
                off[beat_loc[e - 2]:] += 1
            elif min(e + 2, n_loc - 1) == n_loc - 1:
                off[beat_loc[e - 2]: beat_loc[e + 1]] += 1
                off[beat_loc[e + 1]:] += 0.5
            else:
                off[beat_loc[e - 2]: beat_loc[e + 1]] += 1
                off[beat_loc[e + 1]: min(beat_loc[e + 2], n_samples - 1)] += .5
            off[beat_loc[e - 3]: beat_loc[e - 2]] += .5
        elif rec_class == 1:
            off[beat_loc[e - 2]:] += 1
            off[beat_loc[e - 3]: beat_loc[e - 2]] += .5
    return on, off, len(starts)


def _in(x, a, b, n):
    """x in python slice arr[a:b] of an n-long array (a, b already resolved sample indices)."""
    a = 0 if a is None else max(0, min(a, n))
    b = n if b is None else max(0, min(b, n))
    return a <= x < b


def official_point(x: int, beat_loc: np.ndarray, starts, ends, n: int, rec_class: int, kind: str) -> float:
    """Value of onset_score_range[x] (kind='on') or offset_score_range[x] (kind='off')
    computed exactly as the organisers' slicing, without materialising the arrays."""
    bl = beat_loc
    L = len(bl)
    v = 0.0
    if kind == "on":
        for s in starts:
            if rec_class == 2:
                if max(s - 1, 0) == 0:
                    v += _in(x, None, bl[s + 2], n)
                elif max(s - 2, 0) == 0:
                    v += _in(x, bl[s - 1], bl[s + 2], n)
                    v += .5 * _in(x, None, bl[s - 1], n)
                else:
                    v += _in(x, bl[s - 1], bl[s + 2], n)
                    v += .5 * _in(x, bl[s - 2], bl[s - 1], n)
                v += .5 * _in(x, bl[s + 2], bl[s + 3], n)
            elif rec_class == 1:
                v += _in(x, None, bl[s + 2], n)
                v += .5 * _in(x, bl[s + 2], bl[s + 3], n)
    else:
        for e in ends:
            if rec_class == 2:
                if min(e + 1, L - 1) == L - 1:
                    v += _in(x, bl[e - 2], None, n)
                elif min(e + 2, L - 1) == L - 1:
                    v += _in(x, bl[e - 2], bl[e + 1], n)
                    v += .5 * _in(x, bl[e + 1], None, n)
                else:
                    v += _in(x, bl[e - 2], bl[e + 1], n)
                    v += .5 * _in(x, bl[e + 1], min(bl[e + 2], n - 1), n)
                v += .5 * _in(x, bl[e - 3], bl[e - 2], n)
            elif rec_class == 1:
                v += _in(x, bl[e - 2], None, n)
                v += .5 * _in(x, bl[e - 3], bl[e - 2], n)
    return v


def official_score(rec: dict, pred_samples: list[tuple[int, int]]):
    """pred_samples: [(start_sample, end_sample)] inclusive endpoints as in
    the challenge answer files. Returns (U, Ur, Ue, class_pred)."""
    n = rec["n_samples"]
    if len(pred_samples) == 0:
        cls = 0
    elif len(pred_samples) == 1 and pred_samples[0][1] - pred_samples[0][0] == n - 1:
        cls = 1
    else:
        cls = 2
    ur = float(R_MATRIX[rec["rec_class"], cls])
    ue = 0.0
    if rec["rec_class"] in (1, 2):
        mr = len(pred_samples)
        if rec.get("_ann") is not None:      # exact point-wise evaluation (no arrays)
            ann, flag = rec["_ann"]
            st, en = np.where(flag == 1)[0], np.where(flag == 2)[0]
            ma = len(st)
            for s, e in pred_samples:
                ue += official_point(int(s), ann, st, en, n, rec["rec_class"], "on")
                ue += official_point(int(e), ann, st, en, n, rec["rec_class"], "off")
        else:
            on, off, ma = rec["_ranges"]
            if mr:
                for s, e in pred_samples:
                    ue += on[int(s)] + off[int(e)]
        ue *= ma / max(ma, mr)
    return ur + ue, ur, ue, cls


# ---------------------------------------------------------------- helpers

def runs(labels: np.ndarray) -> list[tuple[int, int]]:
    """Runs of label==1 as half-open beat ranges [start, end)."""
    x = np.concatenate([[0], (labels == 1).astype(np.int8), [0]])
    d = np.diff(x)
    return list(zip(np.nonzero(d == 1)[0].tolist(), np.nonzero(d == -1)[0].tolist()))


def match_points(true: list[int], pred: list[int], tol: int):
    """Maximum one-to-one matching of sorted points within +/- tol (greedy is optimal in 1-D)."""
    t, p = sorted(true), sorted(pred)
    i = j = 0
    errs = []
    while i < len(t) and j < len(p):
        if p[j] < t[i] - tol:
            j += 1
        elif p[j] > t[i] + tol:
            i += 1
        else:
            errs.append(p[j] - t[i])
            i += 1
            j += 1
    return len(errs), errs


def interval_overlap(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> float:
    i = j = 0
    tot = 0.0
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            tot += hi - lo
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return tot


def intersect_intervals(a, b):
    """Intersection of sorted, internally disjoint half-open interval lists."""
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            out.append((float(lo), float(hi)))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def normalize_intervals(intervals, duration):
    """Clip and merge support intervals; an explicitly empty support stays empty."""
    out = []
    for lo, hi in sorted((max(0.0, float(a)), min(duration, float(b))) for a, b in intervals):
        if hi <= lo:
            continue
        if out and lo <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], hi))
        else:
            out.append((lo, hi))
    return out


def overlap_pairs(a: list[tuple[float, float]], b: list[tuple[float, float]]):
    """All overlapping (i, j, intersection) for sorted, internally disjoint interval lists (linear sweep)."""
    i = j = 0
    out = []
    while i < len(a) and j < len(b):
        lo, hi = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if hi > lo:
            out.append((i, j, hi - lo))
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def match_episodes(true_iv, pred_iv, thr, pairs=None):
    """Maximum-cardinality one-to-one matching at IoU >= thr (ties: largest total IoU),
    solved exactly on each connected component of the overlap graph."""
    if not true_iv or not pred_iv:
        return 0
    if pairs is None:
        pairs = overlap_pairs(true_iv, pred_iv)
    edges = []
    for i, j, inter in pairs:
        union = max(true_iv[i][1], pred_iv[j][1]) - min(true_iv[i][0], pred_iv[j][0])
        iou = inter / union
        if iou >= thr:
            edges.append((i, j, iou))
    if not edges:
        return 0
    parent = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for i, j, _ in edges:
        parent[find(("t", i))] = find(("p", j))
    comps = {}
    for e in edges:
        comps.setdefault(find(("t", e[0])), []).append(e)
    total = 0
    for es in comps.values():
        if len(es) == 1:
            total += 1
            continue
        ti = sorted({e[0] for e in es})
        pj = sorted({e[1] for e in es})
        cost = np.zeros((len(ti), len(pj)))
        ok = np.zeros_like(cost, dtype=bool)
        for i, j, iou in es:
            cost[ti.index(i), pj.index(j)] = -(1.0 + iou)
            ok[ti.index(i), pj.index(j)] = True
        r, cc = linear_sum_assignment(cost)
        total += int(ok[r, cc].sum())
    return total

# ---------------------------------------------------------------- record metrics

def record_metrics(rec: dict, eps: list[dict], online: bool = True) -> dict:
    rp = rec["rp_s"]
    N = len(rp)
    T = rec["duration_s"]
    fs = rec["fs"]
    if T <= 0:
        raise ValueError("Recording duration must be positive")
    labels = np.asarray(rec["labels"])
    if len(labels) != N:
        raise ValueError("Each R peak must have a rhythm label")

    def beat_time(i, end=False):
        return T if i >= N else float(rp[i])

    known_runs = runs((labels >= 0).astype(np.int8))
    known_iv = [(0.0 if s == 0 else beat_time(s), beat_time(e)) for s, e in known_runs]
    known_time = sum(e - s for s, e in known_iv)

    def boundary_known(i):
        return 0 < i < N and labels[i - 1] >= 0 and labels[i] >= 0

    def supported_match(truth, prediction, tol):
        # Never match boundaries on opposite sides of an unlabelled gap.
        errors, count = [], 0
        for s, e in known_runs:
            n, err = match_points([i for i in truth if s < i < e],
                                  [i for i in prediction if s < i < e], tol)
            count += n
            errors.extend(err)
        return count, errors

    # truth
    if rec.get("true_samples") is not None:      # CPSC: official sample intervals
        true_iv = [(a / fs, b / fs) for a, b in rec["true_samples"]]
    else:
        true_iv = [(0.0 if s == 0 else beat_time(s), beat_time(e)) for s, e in runs(rec["labels"])]
    true_iv = intersect_intervals(true_iv, known_iv)
    true_runs = runs(labels)
    run_iv = [(0.0 if s == 0 else beat_time(s), beat_time(e)) for s, e in true_runs]
    t_on = [s for s, e in true_runs if boundary_known(s)]
    t_off = [e for s, e in true_runs if boundary_known(e)]
    # prediction
    pred_record_iv = [(0.0 if ep["lc"] else beat_time(ep["start"]), T if ep["rc"] else beat_time(ep["end"])) for ep in eps]
    # Fragment episodes only at gaps in reference knowledge, retaining their
    # source identity for delay evaluation. Unknown-only predictions are unscored.
    pred_iv, pred_source = [], []
    for j, interval in enumerate(pred_record_iv):
        fragments = intersect_intervals([interval], known_iv)
        pred_iv.extend(fragments)
        pred_source.extend([j] * len(fragments))
    p_on = [ep["start"] for ep in eps if not ep["lc"] and boundary_known(ep["start"])]
    p_off = [ep["end"] for ep in eps if not ep["rc"] and boundary_known(ep["end"])]
    row = dict(n_true_eps=len(true_iv), n_pred_eps=len(pred_iv), n_true_on=len(t_on), n_true_off=len(t_off),
               n_pred_on=len(p_on), n_pred_off=len(p_off), hours=known_time / 3600.0,
               hours_record=T / 3600.0, T_label_known=known_time, T_label_unknown=max(0.0, T - known_time),
               n_pred_eps_record=len(eps), n_unknown_beats=int((labels < 0).sum()),
               evaluation_policy="known_label_episode_fragments_v2")
    # official score (CPSC only)
    if rec.get("rec_class") is not None and (rec["rec_class"] == 0 or rec.get("_ranges") is not None
                                             or rec.get("_ann") is not None):
        ns = rec["n_samples"]
        rps = rec["rpeaks"]
        ps = []
        for ep in eps:
            s = 0 if ep["lc"] else int(rps[ep["start"]])
            e = ns - 1 if ep["rc"] else int(rps[max(ep["end"] - 1, ep["start"])])
            ps.append((s, e))
        U, Ur, Ue, cls = official_score(rec, ps)
        row.update(U=U, Ur=Ur, Ue=Ue, class_pred=cls, class_true=rec["rec_class"])
    for tol in (1, 2, 5):
        m_on, e_on = supported_match(t_on, p_on, tol)
        m_off, e_off = supported_match(t_off, p_off, tol)
        row[f"on_match_{tol}"] = m_on
        row[f"off_match_{tol}"] = m_off
    m30_on, e30_on = supported_match(t_on, p_on, 30)
    m30_off, e30_off = supported_match(t_off, p_off, 30)
    row["loc_errors"] = [abs(x) for x in e30_on + e30_off]
    pairs = overlap_pairs(true_iv, pred_iv)
    for thr in (0.1, 0.3, 0.5):
        row[f"ep_match_{thr}"] = match_episodes(true_iv, pred_iv, thr, pairs)
    # false episodes (no temporal overlap with any true AF interval)
    row["false_eps"] = len(pred_iv) - len({j for _, j, _ in pairs})
    # alarm delay for each non-censored true episode, first overlapping predicted onset alarm
    delays = []
    if online:
        first = {}
        for i, j, _ in overlap_pairs(run_iv, pred_iv):
            ep = eps[pred_source[j]]
            if not ep["lc"] and boundary_known(ep["start"]) and ep.get("onset_alarm_s") is not None:
                first[i] = min(first.get(i, np.inf), ep["onset_alarm_s"])
        for i, ((ts, te), (s, e)) in enumerate(zip(run_iv, true_runs)):
            if not boundary_known(s):
                continue
            delays.append(first[i] - ts if i in first else np.inf)
    row["delays"] = delays
    # per reference episode (run of AF beats): duration in beats, overlap-detected, delay (None if censored/offline)
    hit = {i for i, _, _ in overlap_pairs(run_iv, pred_iv)}
    row["run_beats"] = [int(e - s) for s, e in true_runs]
    row["run_hit"] = [i in hit for i in range(len(true_runs))]
    if online:
        row["run_delay"] = [None if not boundary_known(s) else (first[i] - run_iv[i][0] if i in first else None)
                            for i, (s, e) in enumerate(true_runs)]
    # Burden accuracy is assessable on known-label time only. Reference gaps
    # produce a partially identified full target, never an implicit negative.
    true_af = sum(e - s for s, e in true_iv)
    pred_af = sum(e - s for s, e in pred_iv)
    row["B"] = true_af / known_time if known_time > 0 else None
    row["B_hat"] = pred_af / known_time if known_time > 0 else None
    row["B_target"] = "known_label_time"
    row["true_af_known_s"] = true_af
    row["pred_af_known_s"] = pred_af
    row["B_hat_record"] = sum(e - s for s, e in pred_record_iv) / T
    row["B_full_bounds"] = [true_af / T, min(1.0, (true_af + T - known_time) / T)]
    row["B_full"] = true_af / T if T - known_time <= 1e-8 else None
    # observed time: union of accepted block supports
    obs = rec.get("observed")
    obs = normalize_intervals([(0.0, T)] if obs is None else obs, T)
    To = float(sum(e - s for s, e in obs))
    obs_known = intersect_intervals(obs, known_iv)
    To_known = sum(e - s for s, e in obs_known)
    row["T_o"] = To
    row["T_observed_ecg"] = To
    row["T_o_known"] = To_known
    row["T"] = T
    if To > 0:
        af_o = interval_overlap(obs, true_iv)
        row["B_o_bounds"] = [af_o / To, min(1.0, (af_o + To - To_known) / To)]
        row["B_o"] = af_o / To if To - To_known <= 1e-8 else None
        row["B_hat_o"] = interval_overlap(obs, pred_record_iv) / To
        row["U_scale"] = sum(ep.get("spread_s", 0.0) for ep in eps) / To
    else:
        row["B_o"] = row["B_hat_o"] = row["U_scale"] = None
        row["B_o_bounds"] = None
    row["B_o_known"] = interval_overlap(obs_known, true_iv) / To_known if To_known > 0 else None
    row["B_hat_o_known"] = interval_overlap(obs_known, pred_record_iv) / To_known if To_known > 0 else None
    row["fallback_offsets"] = sum(1 for ep in eps if ep.get("fallback_offset"))
    # Standard AF-detector summaries (ANSI/AAMI EC57-style), all on rhythm-labelled time:
    # duration overlap for AF-time sensitivity/specificity/PPV, and per-episode
    # durations with any-overlap flags for episode sensitivity/PPV and the
    # clinically motivated >=30-s analyses.
    row["af_tp_s"] = interval_overlap(true_iv, pred_iv)
    hit_true = {i for i, _, _ in pairs}
    hit_pred = {j for _, j, _ in pairs}
    row["true_ep_s"] = [float(e - s) for s, e in true_iv]
    row["true_ep_hit"] = [i in hit_true for i in range(len(true_iv))]
    row["pred_ep_s"] = [float(e - s) for s, e in pred_iv]
    row["pred_ep_true"] = [j in hit_pred for j in range(len(pred_iv))]
    return row


# ---------------------------------------------------------------- conformal burden

def conformal_quantile(residuals: list[float], delta: float = 0.1) -> float:
    M = len(residuals)
    k = int(np.ceil((1 - delta) * (M + 1)))
    if k > M:
        return np.inf
    return float(np.sort(residuals)[k - 1])


def burden_interval(row: dict, qhat: float, eps: float = 0.01, normalized: bool = True):
    T, To = row["T"], row["T_o"]
    if not To or To <= 0 or row["B_hat_o"] is None:
        return (0.0, 1.0), (0.0, 1.0)
    w = qhat * ((row["U_scale"] + eps) if normalized else 1.0)
    lo, hi = max(0.0, row["B_hat_o"] - w), min(1.0, row["B_hat_o"] + w)
    if not np.isfinite(w):
        lo, hi = 0.0, 1.0
    Tu = T - To
    return (lo, hi), (To * lo / T, (To * hi + Tu) / T)


def residual(row: dict, eps: float = 0.01, normalized: bool = True) -> float:
    err = abs(row["B_hat_o"] - row["B_o"])
    return err / (row["U_scale"] + eps) if normalized else err


def interval_coverage(row: dict, lo: float, hi: float, flo: float, fhi: float, tag: str) -> None:
    """Attach exact coverage and honest coverage bounds for incomplete truth.

    'possible' means the prediction interval intersects the reference's
    identified set; 'certain' means it contains that entire set. They are
    upper/lower bounds on unknowable coverage, not extra observed outcomes.
    """
    row[f"int_{tag}"] = [lo, hi, flo, fhi]
    for target, left, right, value, bounds in (
            ("obs", lo, hi, row.get("B_o"), row.get("B_o_bounds")),
            ("full", flo, fhi, row.get("B_full", row.get("B")), row.get("B_full_bounds"))):
        if target == "obs" and row.get("T_o", 0) <= 0:
            value, bounds = None, None
        if bounds is None and value is not None:
            bounds = [value, value]
        row[f"cov_{target}_{tag}"] = (None if value is None else
                                            bool(left - 1e-12 <= value <= right + 1e-12))
        row[f"cov_{target}_possible_{tag}"] = (None if bounds is None else
                                                     bool(left <= bounds[1] + 1e-12 and right >= bounds[0] - 1e-12))
        row[f"cov_{target}_certain_{tag}"] = (None if bounds is None else
                                                    bool(left - 1e-12 <= bounds[0] and right + 1e-12 >= bounds[1]))
