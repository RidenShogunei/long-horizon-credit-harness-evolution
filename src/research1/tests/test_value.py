"""Research-value regressions; no model, Harbor, or remote calls."""

from __future__ import annotations

import copy
import math
import tempfile
import unittest
from pathlib import Path

from research1.contracts import MCPath, MCValueLabel, ReturnLabel, TaskRun, validate_task_run
from research1.evaluation import aggregate_task_evaluation
from research1.returns import IncompleteReturnError, aggregate_mc_paths, recompute_from_tables
from research1.rollout import label_path_from_score_tables


class ValueTests(unittest.TestCase):
    def tables(self):
        return {"H0": {"t1": 1.0, "t2": 0.0}, "A1": {"t1": 0.0, "t2": 1.0}}

    def label(self, **overrides):
        args = dict(branch_id="A", version_order=["H0", "A1"], score_tables=self.tables(),
                    expected_tasks=["t1", "t2"], required_repetitions=2,
                    trial_counts={v: {"t1": 2, "t2": 2} for v in ("H0", "A1")})
        args.update(overrides)
        return recompute_from_tables(**args)

    def path(self, seed=1):
        leaf = f"A2_{seed}"
        tables = self.tables()
        tables[leaf] = {"t1": 1.0, "t2": 1.0} if seed == 1 else {"t1": 0.0, "t2": 0.0}
        return MCPath(path_id=f"path_{seed}", candidate_id="plan_A", policy_id="frozen_policy_v1",
                      horizon=1, first_implementation_hash="a" * 64, continuation_seed=seed,
                      version_order=["H0", "A1", leaf], score_tables=tables,
                      trial_counts={v: {"t1": 2, "t2": 2} for v in tables},
                      expected_tasks=["t1", "t2"], required_repetitions=2)

    def test_task_exchange_does_not_cancel_regression(self):
        label = self.label()
        self.assertTrue(label.is_complete)
        self.assertEqual(label.metric_definition, "taskwise-parent-step-v2")
        self.assertEqual(label.terminal_gain, 0.0)
        self.assertEqual(label.cumulative_regression, 0.5)
        self.assertEqual(label.u_lambda, -0.25)
        self.assertEqual(label.per_task_deltas, [{"t1": -1.0, "t2": 1.0}])
        self.assertEqual(ReturnLabel.from_dict(label.to_dict()).to_dict(), label.to_dict())

    def test_fixed_task_weights_apply_before_summing(self):
        label = self.label(task_weights={"t1": 3.0, "t2": 1.0})
        self.assertEqual(label.task_weights, {"t1": 0.75, "t2": 0.25})
        self.assertEqual(label.terminal_gain, -0.5)
        self.assertEqual(label.cumulative_regression, 0.75)
        self.assertEqual(label.u_lambda, -0.875)

    def test_missing_task_is_not_apparent_improvement(self):
        tables = self.tables()
        del tables["A1"]["t1"]
        with self.assertRaises(IncompleteReturnError):
            self.label(score_tables=tables)

    def test_partial_or_extra_r_is_not_complete(self):
        for count in (0, 1, 3):
            counts = {v: {"t1": 2, "t2": 2} for v in ("H0", "A1")}
            counts["A1"]["t1"] = count
            with self.subTest(count=count), self.assertRaises(IncompleteReturnError):
                self.label(trial_counts=counts)

    def test_missing_count_metadata_cannot_claim_completeness(self):
        self.assertFalse(self.label(trial_counts=None).is_complete)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(IncompleteReturnError):
                label_path_from_score_tables(Path(directory), branch_id="A", version_order=["H0", "A1"],
                                             score_tables=self.tables())
            self.assertFalse((Path(directory) / "returns").exists())

    def test_nonfinite_and_invalid_weights_rejected(self):
        for weights in ({"t1": 1.0}, {"t1": 0.0, "t2": 1.0}, {"t1": math.nan, "t2": 1.0}):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                self.label(task_weights=weights)
        tables = self.tables()
        tables["A1"]["t1"] = math.inf
        with self.assertRaises(ValueError):
            self.label(score_tables=tables)

    def test_two_independent_real_paths_aggregate_g_d_and_dispersion(self):
        paths = [self.path(1), self.path(2)]
        paths = [MCPath.from_dict(path.to_dict()) for path in paths]
        result = aggregate_mc_paths(paths)
        self.assertEqual(result.n_paths, 2)
        self.assertEqual(result.mean_terminal_gain, 0.0)
        self.assertAlmostEqual(result.std_terminal_gain, math.sqrt(0.5))
        self.assertEqual(result.mean_cumulative_regression, 0.75)
        self.assertAlmostEqual(result.std_cumulative_regression, math.sqrt(0.125))
        self.assertEqual(result.mean_u_lambda, -0.375)
        self.assertEqual(result.policy_id, "frozen_policy_v1")
        self.assertEqual(result.horizon, 1)
        self.assertEqual(MCValueLabel.from_dict(result.to_dict()).to_dict(), result.to_dict())

    def test_exact_fixed_m_no_selective_subset(self):
        for paths in ([self.path(1)], [self.path(1), self.path(2), self.path(3)]):
            with self.assertRaises(IncompleteReturnError):
                aggregate_mc_paths(paths, expected_paths=2)

    def test_duplicate_path_or_seed_not_independent_m(self):
        for field, value in (("path_id", "path_1"), ("continuation_seed", 1)):
            second = self.path(2)
            setattr(second, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                aggregate_mc_paths([self.path(1), second])

    def test_mixed_conditions_and_marker_controls_rejected(self):
        for field, value in (("policy_id", "other"), ("first_implementation_hash", "b" * 64),
                             ("candidate_id", "plan_B"), ("path_kind", "comment_marker_control"),
                             ("task_weights", {"t1": 3.0, "t2": 1.0}), ("lambda_value", 0.1)):
            second = self.path(2)
            setattr(second, field, value)
            with self.subTest(field=field), self.assertRaises(ValueError):
                aggregate_mc_paths([self.path(1), second])

    def test_incomplete_path_r_or_horizon_rejected(self):
        second = self.path(2)
        second.trial_counts["A2_2"]["t1"] = 1
        with self.assertRaises(IncompleteReturnError):
            aggregate_mc_paths([self.path(1), second])
        second = self.path(2)
        second.horizon = 2
        with self.assertRaises(IncompleteReturnError):
            aggregate_mc_paths([self.path(1), second])

    def run_record(self, task, rep, reward):
        return TaskRun("run", "H0", task, rep, reward, "model" if reward is not None else "unknown",
                       "PASS" if reward is not None else "INCOMPLETE", "unused")

    def test_evaluation_does_not_drop_null_or_duplicate_r(self):
        runs = [self.run_record("t1", 1, 1.0), self.run_record("t1", 2, None),
                self.run_record("t2", 1, 0.0), self.run_record("t2", 2, 1.0)]
        result = aggregate_task_evaluation(runs, expected_tasks=["t1", "t2"], required_repetitions=2)
        self.assertFalse(result.is_complete)
        self.assertEqual(result.trial_counts, {"t1": 1, "t2": 2})
        runs[1] = self.run_record("t1", 2, 1.0)
        self.assertTrue(aggregate_task_evaluation(runs, expected_tasks=["t1", "t2"], required_repetitions=2).is_complete)
        runs.append(copy.deepcopy(runs[0]))
        self.assertFalse(aggregate_task_evaluation(runs, expected_tasks=["t1", "t2"], required_repetitions=2).is_complete)

    def test_unknown_null_is_valid_incomplete_record(self):
        self.assertEqual(validate_task_run(self.run_record("t1", 1, None).to_dict()), [])


if __name__ == "__main__":
    unittest.main()
