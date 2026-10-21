"""Regression checks for the revision analyses (generic gate, causal QRS detector)."""
import unittest

import numpy as np

from experiments.cera_exp import detect as D
from experiments.cera_exp.qrs import detect_qrs, match_beats


def _context(rng, J=120, b=8):
    N = J * b + 1
    rp = np.cumsum(rng.uniform(0.4, 1.2, N))
    bs = np.arange(J) * b
    bs = bs[rng.random(J) > 0.05]
    J = len(bs)
    lab = np.zeros(J)
    t = 0
    while t < J:
        n = int(rng.integers(3, 40))
        lab[t:t + n] = rng.integers(0, 2)
        t += n
    logit = np.where(lab == 1, 3.0, -3.0) + rng.normal(0, 2.0, J)
    pb = 1 / (1 + np.exp(-(logit[:, None] + rng.normal(0, 1, (J, b)))))
    return D.RecordContext(rp, bs, rp[bs + b] + 0.15, logit, pb.astype(np.float32), b, float(rp[-1] + 1)), N


def _key(bds):
    return [(x.direction, x.beat, x.alarm_s) for x in bds[0]], bds[1]


class GenericGate(unittest.TestCase):
    def test_disabled_certificate_reproduces_ungated_detectors(self):
        rng = np.random.default_rng(1)
        for _ in range(40):
            ctx, _ = _context(rng)
            for k, ton, toff in ((1, 0.8, 0.2), (3, 0.7, 0.3)):
                self.assertEqual(_key(D.smoothing(ctx, k, ton, toff)),
                                 _key(D.gated_generic(ctx, D.SmoothingProposer(ctx, k, ton, toff), A_c=1e-300, W=0)))
            self.assertEqual(_key(D.bocpd(ctx, 1 / 100, 0.5)),
                             _key(D.gated_generic(ctx, D.BocpdProposer(ctx, 1 / 100, 0.5), A_c=1e-300, W=0)))

    def test_gated_alarms_follow_warm_up_and_episodes_are_valid(self):
        rng = np.random.default_rng(2)
        for _ in range(20):
            ctx, N = _context(rng)
            states = (rng.random(N) < 0.5).astype(float)
            bds, m0 = D.gated_generic(ctx, D.BeatStateProposer(ctx, states, 0.5, 9), A_c=10.0, seed=3)
            for e in D.assemble(bds, m0, N):
                self.assertGreater(e["end"], e["start"])
            starts = set(ctx.available[:4 + 1].tolist())
            self.assertTrue(all(b.alarm_s not in starts for b in bds))


class WearerMemory(unittest.TestCase):
    def test_reset_policy_reproduces_published_detectors(self):
        rng = np.random.default_rng(4)
        for _ in range(30):
            ctx, _ = _context(rng)
            for A in (30.0, 1121.0):
                self.assertEqual(_key(D.cera(ctx, A=A, seed=5)),
                                 _key(D.cera_memory(ctx, D.WearerMemory(), A=A, seed=5, reset=True)))
            self.assertEqual(_key(D.gated(ctx, 2.0, 5.0, 10.0, seed=6)),
                             _key(D.gated_memory(ctx, D.WearerMemory(), 2.0, 5.0, 10.0, seed=6, reset=True)))

    def test_dyadic_calibrator_integrates_to_one(self):
        for n in (1, 2, 5, 10, 37, 100, 1000):
            K = max(0, int(np.ceil(np.log2(n))))
            # exact integral of the step mixture: breakpoints at 2^-k
            edges = [0.0] + [2.0 ** -k for k in range(K, -1, -1)]
            total = sum((b - a) * np.exp(D.log_dyadic_calibrator((a + b) / 2, n)) for a, b in zip(edges, edges[1:]))
            self.assertAlmostEqual(total, 1.0, places=12)

    def test_memory_conserves_blocks_and_carries_over(self):
        rng = np.random.default_rng(7)
        mem = D.WearerMemory()
        total = 0
        for _ in range(5):
            ctx, N = _context(rng)
            bds, m0 = D.cera_memory(ctx, mem, A=30.0, margin=1.0, seed=8)
            total += ctx.J - min(4, ctx.J)                # pilot blocks are never stored
            self.assertEqual(len(mem.bags[0]) + len(mem.bags[1]), total)
            for e in D.assemble(bds, m0, N):
                self.assertGreater(e["end"], e["start"])


class CausalQRS(unittest.TestCase):
    def test_synthetic_ecg_with_offset(self):
        fs = 250.0
        rng = np.random.default_rng(0)
        r_true = np.cumsum(rng.uniform(0.55, 1.1, 300) * fs).astype(int) + int(fs)
        n = int(r_true[-1] + 2 * fs)
        t = np.arange(-int(0.05 * fs), int(0.05 * fs) + 1)
        qrs = np.exp(-0.5 * (t / (0.012 * fs)) ** 2)
        x = np.zeros(n)
        for r in r_true:
            x[r + t] += 1.2 * qrs
        x += 0.05 * rng.normal(size=n) + 5.0          # DC offset as in some CPSC2021 records
        sig = np.stack([x, 0.6 * x], axis=1).astype(np.float32)
        r, c = detect_qrs(sig, fs)
        tp, fn, fp, _ = match_beats(r_true, r, int(0.15 * fs))
        self.assertGreaterEqual(tp / len(r_true), 0.99)
        self.assertLessEqual(fp, 3)
        self.assertTrue(np.all(c >= r))                 # confirmation never precedes the beat
        self.assertTrue(np.all(np.diff(r) > 0))

    def test_lead_off_noise_is_not_detected(self):
        fs = 128.0
        rng = np.random.default_rng(1)
        sig = (0.007 * rng.normal(size=(int(600 * fs), 2))).astype(np.float32)   # ~7 uV electrode noise
        r, _ = detect_qrs(sig, fs)
        self.assertLess(len(r), 10)


if __name__ == "__main__":
    unittest.main()
