"""Unit tests: returns recomputation from fake tables (no Harbor)."""

from __future__ import annotations

import unittest

from research1.returns import (
    StepScores,
    cumulative_regression_parent_step,
    label_branch,
    parent_step_deltas,
    recompute_from_tables,
    terminal_gain,
)


class ReturnsTests(unittest.TestCase):
    def test_parent_step_and_u(self) -> None:
        path = [
            StepScores("H0", 0.50),
            StepScores("A1", 0.40),
            StepScores("A2", 0.70),
        ]
        deltas = parent_step_deltas(path)
        self.assertEqual(len(deltas), 2)
        self.assertAlmostEqual(deltas[0], -0.10)
        self.assertAlmostEqual(deltas[1], 0.30)
        cr = cumulative_regression_parent_step(deltas)
        self.assertAlmostEqual(cr, 0.10)
        tg = terminal_gain(path)
        self.assertAlmostEqual(tg, 0.20)
        lab = label_branch(label_id="t", branch_id="A", path=path, lambda_value=0.5)
        self.assertAlmostEqual(lab.terminal_gain, 0.20)
        self.assertAlmostEqual(lab.cumulative_regression, 0.10)
        self.assertAlmostEqual(lab.u_lambda, 0.20 - 0.5 * 0.10)
        self.assertEqual(lab.regression_primary, "parent-step")

    def test_recompute_independent(self) -> None:
        scores = {
            "H0": {"t1": 1.0, "t2": 0.0},
            "A1": {"t1": 1.0, "t2": 1.0},
        }
        lab = recompute_from_tables(
            branch_id="A",
            version_order=["H0", "A1"],
            score_tables=scores,
            lambda_value=0.5,
        )
        self.assertAlmostEqual(lab.terminal_gain, 0.5)
        self.assertAlmostEqual(lab.cumulative_regression, 0.0)


if __name__ == "__main__":
    unittest.main()
