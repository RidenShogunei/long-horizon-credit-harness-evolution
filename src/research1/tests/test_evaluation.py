"""Failure classifications use actual errors, never routine configuration text."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from research1.evaluation import classify_failure, collect_log_text, summarize_trial


class FailureClassificationTests(unittest.TestCase):
    def _trial(self, *, reward, exception=None):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        (root / "result.json").write_text(json.dumps({
            "config": {"environment": {"type": "docker"}, "proxy": "configured", "verifier": "pytest test.sh"},
            "verifier_result": {"rewards": {"reward": reward}},
            "exception_info": exception,
        }))
        return root, summarize_trial(run_id="test", version_id="H0", task_name="task", trial_index=1, output_dir=root)

    def test_scored_zero_with_routine_config_is_valid_task_failure(self):
        root, trial = self._trial(reward=0.0)
        self.assertEqual(collect_log_text(root), "")
        self.assertEqual(trial.status, "PASS")
        self.assertEqual(trial.reward, 0.0)
        self.assertEqual(trial.failure_layer, "model")
        self.assertEqual(classify_failure("docker proxy verifier pytest test.sh composer", 0.0), "model")

    def test_agent_budget_timeout_with_verifier_zero_is_model_outcome(self):
        root, trial = self._trial(reward=0.0, exception={
            "exception_type": "AgentTimeoutError", "exception_message": "Agent execution timed out after 3600.0 seconds",
        })
        self.assertEqual(trial.failure_layer, "model")
        self.assertEqual(trial.status, "PASS")
        self.assertNotIn("docker", collect_log_text(root))

    def test_real_network_and_authentication_failures_remain_infrastructure(self):
        for error in ("ConnectionError: connection refused", "HTTP Error 403: Forbidden", "AuthenticationError: invalid credentials", "ProxyError: tunnel failed"):
            with self.subTest(error=error):
                _, trial = self._trial(reward=None, exception={"exception_type": "RequestError", "exception_message": error})
                self.assertEqual(trial.failure_layer, "infrastructure")
                self.assertEqual(trial.status, "INCOMPLETE")
                self.assertIsNone(trial.reward)

    def test_real_verifier_failures_remain_scorer(self):
        for error in ("RewardFileNotFoundError", "VerifierTimeoutError", "pytest: not found", "test.sh: No such file"):
            with self.subTest(error=error):
                _, trial = self._trial(reward=None, exception={"exception_type": error, "exception_message": "evaluation could not finish"})
                self.assertEqual(trial.failure_layer, "scorer")
                self.assertEqual(trial.status, "INCOMPLETE")

    def test_unknown_unscored_result_is_not_coerced_to_zero(self):
        _, trial = self._trial(reward=None)
        self.assertIsNone(trial.reward)
        self.assertEqual(trial.status, "INCOMPLETE")
        self.assertEqual(trial.failure_layer, "unknown")


if __name__ == "__main__":
    unittest.main()
