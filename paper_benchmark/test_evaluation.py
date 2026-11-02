"""Independent hand-worked and randomized checks of the metric definitions."""
import unittest

import numpy as np
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             matthews_corrcoef, f1_score, balanced_accuracy_score,
                             brier_score_loss, log_loss, auc, precision_recall_curve)
from scipy.special import expit

from evaluation import (independent_ap, independent_auc, topk_expected, threshold_metrics,
                        ranking_metrics, uncalibrated_score_diagnostics)


class MetricTests(unittest.TestCase):
    def test_hand_worked_rankings(self):
        y = np.array([1, 1, 0, 0])
        self.assertEqual(independent_auc(y, [4, 3, 2, 1]), 1)
        self.assertEqual(independent_ap(y, [4, 3, 2, 1]), 1)
        self.assertEqual(independent_auc(y, [1, 2, 3, 4]), 0)
        self.assertAlmostEqual(independent_ap(y, [1, 2, 3, 4]), (1/3+2/4)/2)
        self.assertEqual(independent_auc(y, [1, 1, 1, 1]), 0.5)
        self.assertEqual(independent_ap(y, [1, 1, 1, 1]), 0.5)

    def test_topk_ties_ignore_input_order(self):
        y, s = np.array([1, 0, 0, 1]), np.array([1, 1, 1, 0])
        self.assertAlmostEqual(topk_expected(y, s, 2)["precision"], 1/3)
        rng = np.random.default_rng(73)
        for _ in range(30):
            p = rng.permutation(4)
            self.assertAlmostEqual(topk_expected(y[p], s[p], 2)["precision"], 1/3)

    def test_independent_formulas_against_sklearn(self):
        rng = np.random.default_rng(37)
        for _ in range(40):
            y = np.r_[np.ones(7), np.zeros(93)]
            score = rng.integers(0, 12, size=100) / 12
            self.assertAlmostEqual(independent_auc(y, score), roc_auc_score(y, score), places=13)
            self.assertAlmostEqual(independent_ap(y, score), average_precision_score(y, score), places=13)
            m = threshold_metrics(y, score)
            pred = score >= 0.5
            self.assertAlmostEqual(m["mcc"], matthews_corrcoef(y, pred), places=13)
            self.assertAlmostEqual(m["f1"], f1_score(y, pred), places=13)
            self.assertAlmostEqual(m["balanced_accuracy"], balanced_accuracy_score(y, pred), places=13)

    def test_invalid_data_fails_explicitly(self):
        for y, s in [([1, 1], [0, 1]), ([0, 2], [0, 1]), ([0, 1], [0, np.nan])]:
            with self.assertRaises(ValueError):
                independent_auc(y, s)

    def test_diagnostic_losses_and_pr_conventions(self):
        y, logits = np.array([1, 1, 0, 0]), np.array([-2., 1., 0., -1.])
        scores = expit(logits)
        d = uncalibrated_score_diagnostics(y, logits, scores)
        self.assertAlmostEqual(d["brier_uncalibrated_score"], brier_score_loss(y, scores), places=13)
        self.assertAlmostEqual(d["log_loss_uncalibrated_score"], log_loss(y, scores), places=13)
        p, r, _ = precision_recall_curve(y, logits)
        m = ranking_metrics(y, logits)
        self.assertAlmostEqual(m["pr_auc_trapezoid"], auc(r, p), places=13)
        self.assertNotAlmostEqual(m["pr_auc_trapezoid"], m["average_precision"])


if __name__ == "__main__":
    unittest.main()
