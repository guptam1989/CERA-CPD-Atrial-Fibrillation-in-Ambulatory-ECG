"""Regression checks for missing reference labels, observation and uncertainty.

Run: python -m unittest experiments.cera_exp.tests.test_evaluation_integrity -v
"""
import tempfile
import unittest
import gzip
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from experiments.cera_exp.evaluate import burden_interval, interval_coverage, record_metrics
from experiments.cera_exp.run_eval import calibrate_and_apply, load_eval_record, task


def record(labels, observed=None, include_observed=True):
    n = len(labels)
    out = dict(rp_s=np.arange(n, dtype=float), rpeaks=np.arange(n), fs=1,
               duration_s=float(n), n_samples=n, labels=np.asarray(labels, dtype=np.int8))
    if include_observed:
        out["observed"] = [(0.0, float(n))] if observed is None else observed
    return out


def episode(s, e, n, **overrides):
    out = dict(start=s, end=e, lc=s == 0, rc=e == n,
               onset_alarm_s=float(s + 1), spread_s=0.0, fallback_offset=False)
    out.update(overrides)
    return out


class EvaluationIntegrityTests(unittest.TestCase):
    def test_empty_support_stays_unassessable_but_absent_support_defaults_full(self):
        empty = record_metrics(record([0] * 6, []), [])
        self.assertEqual(empty["T_o"], 0)
        self.assertIsNone(empty["B_o"])
        self.assertIsNone(empty["B_hat_o"])
        self.assertEqual(burden_interval(empty, 0), ((0.0, 1.0), (0.0, 1.0)))
        full = record_metrics(record([0] * 6, include_observed=False), [])
        self.assertEqual(full["T_o"], 6)
        self.assertEqual(full["B_o"], 0)

    def test_observation_union_cannot_double_count_time(self):
        row = record_metrics(record([0] * 6, [(-1, 3), (2, 10)]), [])
        self.assertEqual(row["T_o"], 6)

    def test_unknown_only_predictions_are_unscored(self):
        row = record_metrics(record([-1] * 6), [episode(0, 6, 6)])
        self.assertEqual(row["hours"], 0)
        self.assertEqual(row["hours_record"], 6 / 3600)
        self.assertEqual(row["n_pred_eps"], 0)
        self.assertEqual(row["false_eps"], 0)
        self.assertIsNone(row["B"])
        self.assertIsNone(row["B_full"])
        self.assertEqual(row["B_full_bounds"], [0, 1])
        self.assertEqual(row["B_hat_record"], 1)

    def test_unknown_prefix_is_not_negative_or_a_reference_onset(self):
        row = record_metrics(record([-1, -1, 1, 1, 0, 0]),
                             [episode(0, 2, 6), episode(2, 4, 6)])
        self.assertEqual(row["n_true_on"], 0)
        self.assertEqual(row["n_pred_on"], 0)
        self.assertEqual(row["n_true_off"], 1)
        self.assertEqual(row["off_match_1"], 1)
        self.assertEqual(row["n_pred_eps"], 1)
        self.assertEqual(row["false_eps"], 0)
        self.assertEqual(row["delays"], [])
        self.assertEqual(row["run_delay"], [None])
        self.assertAlmostEqual(row["B"], 0.5)
        self.assertEqual(row["T_label_known"], 4)
        self.assertEqual(row["T_o"], 6)
        self.assertEqual(row["T_o_known"], 4)
        self.assertEqual(row["B_full_bounds"], [2 / 6, 4 / 6])

    def test_internal_unknown_gap_splits_episode_units_and_censors_edges(self):
        row = record_metrics(record([0, 1, -1, 1, 0]), [episode(1, 4, 5)])
        self.assertEqual(row["n_true_eps"], 2)
        self.assertEqual(row["n_pred_eps"], 2)
        self.assertEqual(row["ep_match_0.5"], 2)
        self.assertEqual(row["n_true_on"], 1)
        self.assertEqual(row["n_true_off"], 1)
        self.assertEqual(row["on_match_1"], 1)
        self.assertEqual(row["off_match_1"], 1)
        self.assertIsNone(row["run_delay"][1])

    def test_boundary_matching_never_crosses_unknown_gap(self):
        row = record_metrics(record([0, 1, -1, 0, 0, 0]), [episode(4, 5, 6)])
        self.assertEqual(row["n_true_on"], 1)
        self.assertEqual(row["n_pred_on"], 1)
        self.assertEqual(row["on_match_5"], 0)
        self.assertEqual(row["loc_errors"], [])

    def test_unknown_time_cannot_reduce_known_time_episode_iou(self):
        row = record_metrics(record([-1, -1, -1, 1, 1, 0]), [episode(0, 5, 6)])
        self.assertEqual(row["ep_match_0.5"], 1)
        self.assertAlmostEqual(row["B"], 2 / 3)
        self.assertAlmostEqual(row["B_hat"], 2 / 3)

    def test_reference_availability_does_not_change_deployed_prediction_or_interval(self):
        eps = [episode(1, 4, 6, spread_s=0.2)]
        a = record_metrics(record([0, 1, 1, 1, 0, 0], [(0, 5)]), eps)
        b = record_metrics(record([-1, 1, 1, 1, 0, 0], [(0, 5)]), eps)
        self.assertEqual(a["B_hat_o"], b["B_hat_o"])
        self.assertEqual(a["U_scale"], b["U_scale"])
        self.assertEqual(burden_interval(a, 2), burden_interval(b, 2))
        self.assertIsNone(b["B_o"])
        self.assertEqual(b["B_o_bounds"], [3 / 5, 4 / 5])

    def test_partial_identification_gives_coverage_bounds_without_fabricated_truth(self):
        row = dict(T_o=5, B_o=None, B_o_bounds=[0.2, 0.5], B_full=None,
                   B_full_bounds=[0.1, 0.6])
        interval_coverage(row, 0.3, 0.7, 0, 0.8, "norm")
        self.assertIsNone(row["cov_obs_norm"])
        self.assertTrue(row["cov_obs_possible_norm"])
        self.assertFalse(row["cov_obs_certain_norm"])
        self.assertIsNone(row["cov_full_norm"])
        self.assertTrue(row["cov_full_certain_norm"])
        interval_coverage(row, 0.7, 0.8, 0.7, 0.8, "abs")
        self.assertFalse(row["cov_full_possible_abs"])

    def test_calibration_excludes_undefined_targets_and_reports_counts(self):
        cal = []
        for i in range(9):
            row = record_metrics(record([0] * 6), [])
            row["record"] = str(i)
            cal.append(row)
        unknown = record_metrics(record([-1] + [0] * 5), [])
        unknown["record"] = "unknown"
        empty = record_metrics(record([0] * 6, []), [])
        empty["record"] = "empty"
        cal += [unknown, empty]
        out = calibrate_and_apply(cal, [unknown, empty], {x["record"] for x in cal})
        self.assertEqual(out["M_norm"], 9)
        self.assertEqual(out["qhat_norm"], 0)
        self.assertEqual(out["calibration_excluded_zero_observed"], 1)
        self.assertEqual(out["calibration_excluded_unknown_reference"], 1)
        self.assertIsNone(unknown["cov_full_norm"])
        self.assertTrue(empty["cov_full_norm"])
        self.assertIsNone(empty["cov_obs_norm"])

    def test_cpsc_official_classification_remains_unchanged(self):
        rec = record([0] * 6, [])
        rec.update(rec_class=0, true_samples=[])
        self.assertEqual(record_metrics(rec, [])["U"], 1)
        self.assertEqual(record_metrics(rec, [episode(1, 4, 6)])["U"], -0.5)

    def test_loader_recomputes_truthful_availability_for_legacy_prepared_files(self):
        with tempfile.TemporaryDirectory() as temp:
            rp = np.arange(10) * 20
            np.savez(Path(temp) / "record.npz", rpeaks=rp, labels=np.zeros(10),
                     block_start=np.array([0]), atrial=np.zeros((1, 8, 4, 64)),
                     sym_code=np.zeros(10), available=np.array([0.0]), quality=np.ones(10))
            meta = dict(file="record.npz", fs=100, dataset="nsrdb", duration_s=2)
            row = load_eval_record(temp, meta, False)
            self.assertAlmostEqual(row["available"][0], 1.9)
            self.assertEqual(row["observed"], [(0, 1.6)])

    def test_locked_configuration_replay_never_retunes_and_rejects_stale_cache(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            prep, locked, out = root / "prep", root / "locked", root / "out"
            prep.mkdir()
            locked.mkdir()
            (prep / "index.json").write_text(json.dumps(dict(records=[])))
            folds = root / "folds.json"
            folds.write_text(json.dumps(dict(folds=[dict(roles={})], preselected_record={})))
            configs = {"cera_tuned": {"A": 300}}
            original = dict(fold=0, variant="full", seed=0, configs=configs,
                            methods={"cera_tuned": {"val_U": 1.23}})
            with gzip.open(locked / "fold0_full_s0.json.gz", "wt") as destination:
                json.dump(original, destination)
            args = (str(prep), str(folds), str(root / "runs"), str(out), 0, "full", 0, [], False, str(locked))
            with patch("experiments.cera_exp.run_eval.attach_scores"), \
                    patch("experiments.cera_exp.run_eval.evaluate_set", return_value=[]) as evaluate, \
                    patch("experiments.cera_exp.run_eval.tune", side_effect=AssertionError("retuned")):
                path = task(args)
            with gzip.open(path, "rt") as source:
                revised = json.load(source)
            self.assertEqual(revised["configs"], configs)
            self.assertEqual(revised["methods"]["cera_tuned"]["val_U"], 1.23)
            self.assertEqual(evaluate.call_count, 2)  # calibration and test, no validation refit
            self.assertEqual(revised["evaluation_schema_version"], 2)
            self.assertEqual(len(revised["provenance"]["locked_evaluation_sha256"]), 64)
            with gzip.open(path, "wt") as destination:
                json.dump({}, destination)
            with self.assertRaisesRegex(ValueError, "Stale evaluation output"):
                task(args)


if __name__ == "__main__":
    unittest.main()
