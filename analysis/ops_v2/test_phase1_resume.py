"""Focused unit coverage for restoring a resumable Phase-1 search incumbent."""

import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

import phase1_host


class SearchResumeTests(unittest.TestCase):
    def test_restores_best_trial_when_all_artifacts_match_completed_log(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config = {"trial": 7, "seed": 49, "max_depth": 4}
            summary = {
                "config": config,
                "best_iteration": 23,
                "best_validation": {"pr_auc": 0.31, "roc_auc": 0.67},
            }
            (root / "xgboost_best.json").write_text(json.dumps(summary), encoding="utf-8")
            np.save(root / "xgboost_validation_scores.npy", np.asarray([0.1, 0.8, 0.4]))
            with (root / "xgboost_best_model.pkl").open("wb") as handle:
                pickle.dump({"model": "persisted"}, handle)

            restored = phase1_host.restore_search_incumbent(
                "xgboost", {7: {"pr_auc": "0.31", "best_iteration": "23"}},
                expected_score_rows=3, results_dir=root)

            self.assertIsNotNone(restored)
            self.assertAlmostEqual(restored[0], 0.31)
            self.assertEqual(restored[1], config)
            self.assertEqual(restored[2], {"model": "persisted"})
            np.testing.assert_array_equal(restored[3], np.asarray([0.1, 0.8, 0.4], dtype=np.float32))
            self.assertEqual(restored[4], 23)

    def test_refuses_resume_when_incumbent_artifacts_are_missing(self):
        with tempfile.TemporaryDirectory() as temp:
            restored = phase1_host.restore_search_incumbent(
                "xgboost", {2, 7}, expected_score_rows=3, results_dir=Path(temp))
            self.assertIsNone(restored)

    def test_refuses_resume_if_best_trial_is_not_marked_successful(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            summary = {
                "config": {"trial": 7},
                "best_iteration": 1,
                "best_validation": {"pr_auc": 0.31},
            }
            (root / "xgboost_best.json").write_text(json.dumps(summary), encoding="utf-8")
            np.save(root / "xgboost_validation_scores.npy", np.asarray([0.1, 0.8, 0.4]))
            with (root / "xgboost_best_model.pkl").open("wb") as handle:
                pickle.dump({"model": "persisted"}, handle)

            restored = phase1_host.restore_search_incumbent(
                "xgboost", {2: {"pr_auc": "0.20", "best_iteration": "1"}},
                expected_score_rows=3, results_dir=root)
            self.assertIsNone(restored)

    def test_refuses_resume_when_search_log_disagrees_with_saved_best(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            summary = {
                "config": {"trial": 7},
                "best_iteration": 23,
                "best_validation": {"pr_auc": 0.31},
            }
            (root / "xgboost_best.json").write_text(json.dumps(summary), encoding="utf-8")
            np.save(root / "xgboost_validation_scores.npy", np.asarray([0.1, 0.8, 0.4]))
            with (root / "xgboost_best_model.pkl").open("wb") as handle:
                pickle.dump({"model": "persisted"}, handle)

            restored = phase1_host.restore_search_incumbent(
                "xgboost", {7: {"pr_auc": "0.29", "best_iteration": "23"}},
                expected_score_rows=3, results_dir=root)
            self.assertIsNone(restored)


if __name__ == "__main__":
    unittest.main()
