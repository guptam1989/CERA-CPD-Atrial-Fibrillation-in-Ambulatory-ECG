import unittest
import numpy as np
from experiments.cera_exp.features import block_release_times


class ReleaseIntegrityTests(unittest.TestCase):
    def test_fast_final_rr_waits_for_all_quality_samples(self):
        rp = np.arange(17, dtype=float) * 0.2
        starts = np.array([0, 8])
        available = block_release_times(rp, starts, 8)
        np.testing.assert_allclose(available, rp[starts + 7] + 0.5)
        self.assertTrue(np.all(available >= rp[starts + 8] + 0.15))

    def test_ordinary_rr_keeps_existing_next_peak_release(self):
        rp = np.arange(9, dtype=float) * 0.8
        np.testing.assert_allclose(block_release_times(rp, np.array([0]), 8), [6.55])

    def test_release_is_monotone_and_conservative(self):
        rp = np.cumsum(np.resize([0.2, 0.3, 0.4, 0.9], 41))
        starts = np.arange(0, 40, 8)
        release = block_release_times(rp, starts, 8)
        self.assertTrue(np.all(np.diff(release) > 0))
        for start, time in zip(starts, release):
            self.assertGreaterEqual(time, max(rp[start:start + 8] + 0.5))

    def test_empty_blocks_and_missing_next_peak(self):
        self.assertEqual(block_release_times(np.array([]), np.array([], int), 8).size, 0)
        with self.assertRaises(ValueError):
            block_release_times(np.arange(8), np.array([0]), 8)


if __name__ == '__main__':
    unittest.main()
