"""Regression tests for population alignment, exact pooling and provenance."""
import contextlib
import io
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from experiments.cera_exp.aggregate import external_patient_values, load_results
from experiments.cera_exp.mechanism import simulate
from experiments.cera_exp.summary import pool_sweep_summaries, summarize, sweep_summary
from experiments.cera_exp.tables import table_external


def record(**updates):
    row = dict(U=1.0, Ur=0.5, Ue=0.5, class_pred=0, class_true=0, n_true_on=1, n_true_off=1,
               n_pred_on=1, n_pred_off=1, n_true_eps=1, n_pred_eps=1,
               hours=1.0, false_eps=0, loc_errors=[], delays=[],
               B=0.0, B_hat=0.0, T_o=3600.0, T=3600.0)
    for tol in (1, 2, 5):
        row[f"on_match_{tol}"] = row[f"off_match_{tol}"] = 0
    for threshold in (0.1, 0.3, 0.5):
        row[f"ep_match_{threshold}"] = 0
    row.update(updates)
    return row


class PoolingTests(unittest.TestCase):
    def test_non_linear_f1_and_median_use_raw_observations(self):
        rows = [record(n_true_eps=1, n_pred_eps=1, **{"ep_match_0.5": 1}, delays=[1.0]),
                record(n_true_eps=9, n_pred_eps=1, delays=[100.0, 101.0, None])]
        pooled = pool_sweep_summaries([sweep_summary([r]) for r in rows])
        direct = summarize(rows)
        self.assertAlmostEqual(pooled["ep_f1_0.5"], 1 / 6)
        self.assertEqual(pooled["delay_median"], 100.0)
        for key in pooled:
            self.assertAlmostEqual(pooled[key], direct[key])

    def test_no_alarm_fold_does_not_impute_zero(self):
        folds = [sweep_summary([record(delays=[50.0])]), sweep_summary([record(delays=[None])])]
        self.assertEqual(pool_sweep_summaries(folds)["delay_median"], 50.0)
        self.assertTrue(math.isnan(pool_sweep_summaries([folds[1]])["delay_median"]))

    def test_legacy_aggregates_and_invalid_delay_rejected(self):
        with self.assertRaisesRegex(ValueError, "regenerate"):
            pool_sweep_summaries([{"ep_f1_0.5": 0.8, "delay_median": 20}])
        row = sweep_summary([record()])
        row["pooling_v1"]["finite_delays_s"] = [float("inf")]
        with self.assertRaisesRegex(ValueError, "non-finite"):
            pool_sweep_summaries([row])

    def test_unknown_burden_omitted_with_denominator(self):
        result = summarize([record(B=None, B_hat=None, T_label_unknown=3600.0, T_label_known=0.0)])
        self.assertTrue(math.isnan(result["burden_mae_pp"]))
        self.assertEqual(result["burden_mae_n"], 0)
        self.assertEqual(result["partial_reference_records"], 1)
        self.assertEqual(result["label_known_frac"], 0.0)

    def test_partial_identification_not_misreported_as_exact_coverage(self):
        row = record(cov_full_norm=None, cov_full_certain_norm=False, cov_full_possible_norm=True,
                     int_norm=[0.1, 0.3, 0.1, 0.5])
        result = summarize([row])
        self.assertTrue(math.isnan(result["cov_full_norm"]))
        self.assertEqual(result["cov_full_n_norm"], 0)
        self.assertEqual(result["cov_full_certain_norm"], 0)
        self.assertEqual(result["cov_full_possible_norm"], 1)

    def test_partial_reference_table_displays_coverage_bounds(self):
        stat = lambda value: [value, 0, 1]
        data = {"partial_reference_records": stat(20), "cov_full_norm": stat(0.99),
                "cov_full_certain_norm": stat(0.90), "cov_full_possible_norm": stat(0.95),
                "width_full_norm": stat(0.40)}
        latex = table_external({"external": {"ltafdb": {"full:cusum": data}}})
        self.assertIn("[90.0, 95.0]/40", latex)
        self.assertNotIn("99/40", latex)

    def test_legacy_evaluation_archive_rejected(self):
        import gzip
        with tempfile.TemporaryDirectory() as directory:
            with gzip.open(Path(directory) / "fold0_full_s0.json.gz", "wt") as handle:
                json.dump({"variant": "full", "fold": 0, "seed": 0}, handle)
            with self.assertRaisesRegex(ValueError, "regenerate"):
                load_results(directory)


class ExternalPopulationTests(unittest.TestCase):
    def test_nonaf_filter_and_exposure_pooling_within_model(self):
        first = [dict(patient="p", n_true_eps=0, hours=1, false_eps=1),
                 dict(patient="p", n_true_eps=0, hours=9, false_eps=0),
                 dict(patient="af", n_true_eps=1, hours=2, false_eps=99)]
        second = [dict(patient="p", n_true_eps=0, hours=2, false_eps=1)]
        ext = {("mitdb", "gate", "gate_tuned", 0, 0): first,
               ("mitdb", "gate", "gate_tuned", 1, 0): second}
        got = external_patient_values(ext, "mitdb", "gate", "gate_tuned", "fa_h", [])
        self.assertEqual(set(got), {"p"})
        self.assertAlmostEqual(got["p"], (1 / 10 + 1 / 2) / 2)

    def test_noise_uses_own_test_fold(self):
        ext = {("noise_em_0", "full", "cusum", f, 0):
               [dict(patient="p", n_true_eps=0, hours=1, false_eps=10 * f + 1)] for f in (0, 1)}
        folds = [{"roles": {"p": "test"}}, {"roles": {"p": "train"}}]
        got = external_patient_values(ext, "noise_em_0", "full", "cusum", "fa_h", folds)
        self.assertEqual(got, {"p": 1.0})


class ProvenanceTests(unittest.TestCase):
    def test_simulation_records_actual_arguments_and_reproducibility(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("experiments.cera_exp.mechanism.blocks_to_alarm",
                       side_effect=lambda n, A, rng, **kw: int(rng.integers(1, 2000))):
                with contextlib.redirect_stdout(io.StringIO()):
                    a = simulate(str(Path(directory) / "a.json"), reps=7, seed=19)
                    b = simulate(str(Path(directory) / "b.json"), reps=7, seed=19)
            self.assertEqual(a, b)
            self.assertEqual(a["metadata"]["replicates"], 7)
            self.assertEqual(a["metadata"]["seed"], 19)
            self.assertEqual(len(a["metadata"]["source_sha256"]), 64)
            self.assertEqual(a["schema_version"], 2)
            self.assertEqual(len(a["results"]), 54)
            self.assertEqual(json.loads((Path(directory) / "a.json").read_text()), a)


if __name__ == "__main__":
    unittest.main()
