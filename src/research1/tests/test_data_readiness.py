import datetime
import json
from pathlib import Path
import tempfile
import unittest

from research1.data_pipeline.readiness import build_readiness, DEFAULT_SOLVER


class DataReadinessTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.dataset = self.root / "dataset"
        self.tasks = ["zero", "one", "slow", "unknown", "other", "network", "missing"]
        for task in self.tasks[:-1]:
            env = self.dataset / "tasks" / task / "environment"
            env.mkdir(parents=True)
            (env / "Dockerfile").write_text("FROM python:3.11-slim\n")

    def wrapper(self, task, name="run", **extra):
        path = self.root / "runs" / name / task / "task_run.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        row = {"task_name": task, "status": "PASS", "reward": 0.0, "failure_layer": "model",
               "model": DEFAULT_SOLVER["model"], "temperature": 0.7, "provider_id": "tongji_agi",
               "duration_seconds": 100,
               "evaluation_conditions": {"max_turns": 200, "wall_seconds": 900}}
        row.update(extra)
        path.write_text(json.dumps(row))
        return path, row

    def concrete(self, task, name, elapsed=100, reward=0, start="2026-01-01T00:00:00Z"):
        path = self.root / "runs" / name / (task + "__trial") / "result.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        began = datetime.datetime.fromisoformat(start.replace("Z", "+00:00"))
        end = began + datetime.timedelta(seconds=elapsed)
        row = {"task_name": task, "trial_name": path.parent.name,
               "started_at": start, "finished_at": end.isoformat(),
               "agent_execution": {"started_at": start, "finished_at": end.isoformat()},
               "config": {"task": {"path": str(self.dataset / "tasks" / task)}, "agent": {"model_name": DEFAULT_SOLVER["model"], "kwargs": {
                   "api_base": DEFAULT_SOLVER["api_base"], "temperature": 0.7, "max_turns": 200}}},
               "verifier_result": {"rewards": {"reward": reward}}}
        path.write_text(json.dumps(row))
        return path

    def test_valid_zero_and_one_have_identical_resource_eligibility(self):
        self.wrapper("zero", reward=0)
        self.wrapper("one", reward=1)
        readiness = build_readiness(self.root, self.dataset, self.tasks)
        for task in ("zero", "one"):
            self.assertEqual(readiness[task]["status"], "completed_compatible")
            self.assertTrue(readiness[task]["noise_eligible"])
            self.assertTrue(readiness[task]["observations"][0]["scored"])
        self.assertNotIn('"reward":', json.dumps(readiness))
        self.assertEqual(readiness["unknown"]["status"], "unknown")
        self.assertFalse(readiness["unknown"]["runtime_certified"])
        self.assertEqual(readiness["missing"]["status"], "env_unavailable")

    def test_same_budget_timeout_is_not_network_and_different_config_does_not_certify(self):
        self.wrapper("slow", status="INCOMPLETE", reward=None, duration_seconds=906,
                     wall_limit_reached=True, stopped_for_budget=True, retryable=False)
        self.wrapper("other", model="another-model")
        self.wrapper("network", status="INCOMPLETE", reward=None, duration_seconds=50, failure_layer="infrastructure")
        readiness = build_readiness(self.root, self.dataset, self.tasks)
        self.assertEqual(readiness["slow"]["status"], "known_budget_incompatible")
        self.assertFalse(readiness["slow"]["initialization_eligible"])
        self.assertEqual(readiness["slow"]["transient_infrastructure_count"], 0)
        self.assertEqual(readiness["other"]["status"], "unknown")
        self.assertEqual(readiness["network"]["status"], "transient_infra")
        self.assertFalse(readiness["network"]["noise_eligible"])

    def test_shorter_limit_or_different_turn_cap_does_not_prove_current_budget_incompatible(self):
        self.wrapper("slow", status="INCOMPLETE", reward=None, duration_seconds=301,
                     wall_limit_reached=True, evaluation_conditions={"max_turns": 200, "wall_seconds": 300})
        self.wrapper("other", status="INCOMPLETE", reward=None, duration_seconds=906,
                     wall_limit_reached=True, evaluation_conditions={"max_turns": 30, "wall_seconds": 900})
        readiness = build_readiness(self.root, self.dataset, self.tasks)
        self.assertEqual(readiness["slow"]["status"], "unknown")
        self.assertEqual(readiness["other"]["status"], "unknown")

    def test_coarse_wrapper_uses_unique_exact_trial_elapsed_and_deduplicates(self):
        concrete = self.concrete("zero", "actual", elapsed=220)
        wrapper, row = self.wrapper("zero", duration_seconds=None, output_dir=str(concrete.parent.parent))
        # Omit model settings from the coarse wrapper: actual config is binding.
        for key in ("model", "temperature", "provider_id"):
            row.pop(key)
        wrapper.write_text(json.dumps(row))
        readiness = build_readiness(self.root, self.dataset, ["zero"])["zero"]
        self.assertEqual(readiness["observation_count"], 1)
        self.assertEqual(readiness["observations"][0]["elapsed_seconds"], 220)
        self.assertTrue(readiness["noise_eligible"])

    def test_all_concrete_trials_are_kept_without_favorable_latest_selection(self):
        self.concrete("zero", "trial_a", elapsed=100, reward=1)
        self.concrete("zero", "trial_b", elapsed=800, reward=0)
        self.wrapper("zero", duration_seconds=None, output_dir=str(self.root / "runs"))
        readiness = build_readiness(self.root, self.dataset, ["zero"])["zero"]
        self.assertEqual(readiness["observation_count"], 3)
        self.assertEqual(readiness["compatible_completion_count"], 2)
        self.assertTrue(any(row["ambiguous_result_binding"] for row in readiness["observations"]))

    def test_duplicate_wrapper_does_not_hide_explicit_timeout(self):
        result = self.concrete("slow", "raw", elapsed=906, reward=None)
        self.wrapper("slow", "a", status="INCOMPLETE", reward=None, duration_seconds=None,
                     trial_result_path=str(result))
        self.wrapper("slow", "b", status="INCOMPLETE", reward=None, duration_seconds=906,
                     trial_result_path=str(result), wall_limit_reached=True)
        readiness = build_readiness(self.root, self.dataset, ["slow"])["slow"]
        self.assertEqual(readiness["observation_count"], 1)
        self.assertEqual(readiness["budget_limit_count"], 1)
        self.assertEqual(readiness["status"], "known_budget_incompatible")

    def test_ambiguous_coarse_duration_and_foreign_trial_task_do_not_certify(self):
        for name in ("trial_a", "trial_b"):
            path = self.concrete("zero", name, elapsed=100)
            row = json.loads(path.read_text())
            row["config"]["agent"]["model_name"] = "other-model"
            path.write_text(json.dumps(row))
        self.wrapper("zero", duration_seconds=10, output_dir=str(self.root / "runs"))
        self.assertEqual(build_readiness(self.root, self.dataset, ["zero"])["zero"]["status"], "unknown")
        path = self.concrete("one", "foreign", elapsed=10)
        row = json.loads(path.read_text());row["config"]["task"]["path"] = str(self.dataset / "tasks/wrong-task")
        path.write_text(json.dumps(row))
        self.assertEqual(build_readiness(self.root, self.dataset, ["one"])["one"]["observation_count"], 0)

    def test_later_completion_retains_mixed_latency_risk_and_older_fits_do_not_erase_timeout(self):
        self.wrapper("slow", "old", status="INCOMPLETE", reward=None, duration_seconds=906,
                     wall_limit_reached=True, finished_at="2026-01-01T00:00:00Z")
        self.wrapper("slow", "new", duration_seconds=100, finished_at="2026-01-02T00:00:00Z")
        readiness = build_readiness(self.root, self.dataset, ["slow"])["slow"]
        self.assertEqual(readiness["status"], "mixed_latency")
        self.assertFalse(readiness["noise_eligible"])
        self.assertTrue(readiness["later_compatible_completion"])
        self.wrapper("slow", "latest", status="INCOMPLETE", reward=None, duration_seconds=906,
                     wall_limit_reached=True, finished_at="2026-01-03T00:00:00Z")
        self.assertEqual(build_readiness(self.root, self.dataset, ["slow"])["slow"]["status"], "known_budget_incompatible")

    def test_agent_time_alone_and_negative_elapsed_do_not_certify_total_wall(self):
        concrete = self.concrete("zero", "agent_only", elapsed=10)
        row = json.loads(concrete.read_text())
        row.pop("started_at")
        concrete.write_text(json.dumps(row))
        self.wrapper("other", duration_seconds=-1)
        readiness = build_readiness(self.root, self.dataset, ["zero", "other"])
        self.assertEqual(readiness["zero"]["status"], "unknown")
        self.assertEqual(readiness["zero"]["observations"][0]["agent_execution_seconds"], 10)
        self.assertEqual(readiness["other"]["status"], "unknown")
