"""Offline trial scheduling, provenance, null-score and at-most-once tests."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    from research1.data_pipeline import edit_trial as trial
except ImportError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "server_fix" / "src"))
    spec = importlib.util.spec_from_file_location("offline_edit_trial", Path(__file__).with_name("edit_trial.py"))
    trial = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(trial)

from research1.data_pipeline.execution import tree_hash
from research1.data_pipeline.ledger import Ledger
from research1.data_pipeline.util import digest


class EditTrialTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.modules = self.root / "modules"
        for module in ("agent_loop", "observation", "tools", "context_mgmt", "verification"):
            path = self.modules / module / "baseline.py"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("x = 1\n")
        self.campaign = self.root / "campaign"
        self.campaign.mkdir()
        self.protocol = {"protocol_id": "offline-trial", "protocol_hash": "p" * 64,
            "execution_mode": "initialization", "trial_driver": trial.TRIAL_DRIVER,
            "grounded_edit_trial": {"arms": copy.deepcopy(trial.ARMS), "stop_after_qualified_edit": 1},
            "initialization": {"task_order": ["bloom"], "max_parents": 1},
            "modules_root": str(self.modules), "modules_hash": tree_hash(self.modules), "policies": {},
            "budget": {"regular": 2, "retries": 0, "total": 2}, "stage_budget": {"proposal": 2},
            "task_sets": {"proposal": ["bloom"], "proposal_train": ["bloom"], "proposal_validation": [],
                "reward": ["reward"], "dev": ["dev"]},
            "source_parents": {"source_campaign": "/original/campaign", "source_protocol_hash": "s" * 64,
                "snapshot_sha256": "a" * 64}}
        (self.campaign / "protocol.json").write_text(json.dumps(self.protocol))
        artifact = self.root / "original_result.json"
        artifact.write_text("original immutable score fixture\n")
        trajectory = self.root / "original_trajectory.json"
        trajectory.write_text("original immutable trajectory fixture\n")
        self.parent = {"status": "PASS", "reward": 0.0, "failure_layer": "model", "evaluation_id": "old-eval",
            "trial_result_path": str(artifact), "result_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
            "trajectory": str(trajectory), "trajectory_sha256": hashlib.sha256(trajectory.read_bytes()).hexdigest(),
            "evaluation_spec": {"role": "proposal", "task": "bloom", "modules_root": str(self.modules),
                "code_hash": tree_hash(self.modules)}}
        self.source = {"task": "bloom", "root_lineage_id": "original-root", "split": "train",
            "source_state_id": "original-state", "parent_result": self.parent}
        self.trace = []
        self.actions = {"standard": "edit", "reasoned": "edit"}
        self.contract = {"checks_id": "checks-fixture", "selected": [], "parent_hashes": {}, "preprobed": ["completion_requirement_recheck"]}
        mocked = patch("research1.data_pipeline.protocol.load_protocol", return_value=self.protocol)
        mocked.start()
        self.addCleanup(mocked.stop)

    def facts(self, state, directory):
        self.trace.append(("facts", None))
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "fact_output.json").write_text(json.dumps({"raw": "offline no-card fixture"}))
        return None

    def make_plan(self, state, action):
        return {"action": action, "target_module": "agent_loop" if action == "edit" else None,
            "locations": ["agent_loop/baseline.py"] if action == "edit" else [],
            "evidence_refs": [state["evidence"][0]["id"]], "problem_hypothesis": "Offline fixture observed requirement",
            "intended_change": "Offline fixture bounded change" if action == "edit" else "Keep bounded observation",
            "preserve_behaviors": ["Retain transport"], "verification_plan": ["Frozen fixture checks"],
            "verification_kind": "completion_requirement_recheck" if action == "edit" else None}

    def propose(self, state, directory):
        arm = state["teacher_arm_id"]
        self.trace.append(("teacher", arm))
        self.assertEqual(state["root_lineage_id"], "original-root")
        self.assertEqual(state["split"], "train")
        self.assertEqual(state["max_candidates"], 1)
        self.assertEqual(state["teacher_reasoning_effort"], "none" if arm == "standard" else "low")
        self.assertNotIn("expert", json.dumps(state).lower())
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "teacher.json").write_text(json.dumps({"raw": "offline plan fixture", "prompt_sha256": arm}))
        return [self.make_plan(state, self.actions[arm])]

    def review(self, plan, state):
        self.trace.append(("review", state["teacher_arm_id"]))
        self.assertEqual(len(Ledger(self.campaign).records("edittrial_teacher_outcomes")), 2)
        return {"status": "PASS", "independent": True, "reviewer_id": "offline-fact-review",
            "reason": "Bounded observed fixture", "evidence_refs": plan["evidence_refs"]}

    def applicability(self, state, plan, review, directory):
        self.trace.append(("binding", state["teacher_arm_id"]))
        self.assertEqual(state["plan_factual_review"], review)
        return {"status": "PASS", "binding_id": "offline-binding-" + state["teacher_arm_id"]}

    def implement(self, plan, state, directory):
        arm = state["teacher_arm_id"]
        self.trace.append(("implement", arm))
        self.assertEqual(sum(kind == "review" for kind, _ in self.trace), len(Ledger(self.campaign).records("samples")))
        self.assertEqual(state["plan_contract_binding"]["status"], "PASS")
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "editor_output.json").write_text(json.dumps({"raw": "offline implementation"}))
        child = directory / "child"
        shutil.copytree(self.modules, child)
        (child / "agent_loop/baseline.py").write_text("x = 2\n")
        patch_path = directory / "edit.diff"
        patch_path.write_text("offline saved exact fixture patch\n")
        return {"status": "applied", "modules_root": str(child), "code_hash": tree_hash(child),
            "parent_code_hash": state["code_hash"], "patch_path": str(patch_path),
            "patch_sha256": hashlib.sha256(patch_path.read_bytes()).hexdigest()}

    def verify(self, plan, state, attempt, contract):
        self.trace.append(("verify", state["teacher_arm_id"]))
        return {"status": "PASS", "covers_plan": True, "executable": True,
            "checks_id": contract["checks_id"], "mechanism_id": plan["verification_kind"]}

    def evaluate(self, campaign, protocol, ledger, code, task, role, scope, repeat):
        self.trace.append(("evaluate", scope))
        self.assertEqual(role, "proposal")
        self.assertTrue(scope.startswith("edittrial:child:"))
        self.assertEqual(task, "bloom")
        spec = {"role": role, "task": task, "scope": scope, "code_hash": tree_hash(code)}
        key = digest(spec)
        ledger.reserve_job(key, spec)
        ledger.start_job(key, "offline-test")
        result = {"status": "PASS", "reward": 0.0, "failure_layer": "model", "evaluation_id": key,
            "evaluation_spec": {"role": role, "task": task, "modules_root": str(code), "code_hash": tree_hash(code)}}
        ledger.complete_job(key, result)
        return result

    def run_trial(self, **overrides):
        kwargs = {"evaluator": self.evaluate, "proposer": self.propose, "implementer": self.implement,
            "verifier": self.verify, "reviewer": self.review, "fact_collector": self.facts,
            "interface_builder": lambda state: None, "applicability_reviewer": self.applicability,
            "contract_builder": lambda state: copy.deepcopy(self.contract),
            "source_validator": lambda campaign, protocol: [copy.deepcopy(self.source)],
            "evidence_selector": lambda parent: [{"id": "trajectory", "text": "Executed fixture returned the bounded expected body", "partition": "proposal"}]}
        kwargs.update(overrides)
        return trial.run_grounded_edit_trial(self.campaign, **kwargs)

    def test_both_arms_and_reviews_precede_one_zero_score_qualified_edit(self):
        result = self.run_trial()
        self.assertEqual(result["status"], "GROUNDED_EDIT_TRIAL_COMPLETE", result)
        self.assertEqual(result["counts"]["raw_plans"], 2)
        self.assertEqual(result["counts"]["qualified_edit"], 1)
        self.assertEqual(result["counts"]["pending"], 1)
        self.assertFalse(result["data_collection_complete"])
        self.assertEqual(Ledger(self.campaign).status()["budget"]["used"], {"regular": 1})
        rows = Ledger(self.campaign).records("samples")
        self.assertEqual(len({row["state_id"] for row in rows}), 2)
        self.assertEqual({row["root_lineage_id"] for row in rows}, {"original-root"})
        self.assertTrue(all(row["qualification"]["proposal_task_result"] == self.parent for row in rows))
        self.assertTrue(all(row["source_kind"] == "fresh_teacher_plan_from_reused_parent" for row in rows))
        self.assertEqual(len({row["teacher_record_path"] for row in rows}), 2)
        pending = next(row for row in rows if row["qualification"]["status"] == "pending")
        self.assertIn("remaining fixed arm", pending["qualification"]["pending_reason"])

    def test_completed_resume_reuses_all_calls_results_and_budget(self):
        first = self.run_trial()
        before = list(self.trace)
        second = self.run_trial()
        self.assertEqual(second["status"], first["status"])
        self.assertEqual(self.trace, before)
        self.assertEqual(second["counts"], first["counts"])

    def test_unknown_binding_preserves_raw_pending_without_child_jobs(self):
        result = self.run_trial(applicability_reviewer=lambda *args: {"status": "pending", "reason": "Unknown mechanism"})
        self.assertEqual(result["counts"]["raw_plans"], 2)
        self.assertEqual(result["counts"]["pending"], 2)
        self.assertFalse(any(kind == "implement" for kind, _ in self.trace))
        self.assertEqual(Ledger(self.campaign).status()["budget"]["used"], {})

    def test_failed_factual_arm_is_retained_while_fixed_second_arm_runs(self):
        def review(plan, state):
            result = self.review(plan, state)
            if state["teacher_arm_id"] == "standard":
                result["status"] = "FAIL"
            return result
        result = self.run_trial(reviewer=review)
        self.assertEqual(result["counts"]["rejected"], 1)
        self.assertEqual(result["counts"]["qualified_edit"], 1)
        self.assertEqual([arm for kind, arm in self.trace if kind == "implement"], ["reasoned"])

    def test_null_child_scores_stay_null_and_consume_only_two_fixed_jobs(self):
        def evaluate(*args):
            result = self.evaluate(*args)
            result.update(status="INCOMPLETE", reward=None, failure_layer="infrastructure")
            return result
        result = self.run_trial(evaluator=evaluate)
        self.assertEqual(result["counts"]["qualified_edit"], 0)
        self.assertEqual(result["counts"]["pending"], 2)
        self.assertEqual(Ledger(self.campaign).status()["budget"]["used"], {"regular": 2})
        rows = Ledger(self.campaign).records("samples")
        self.assertTrue(all(row["qualification"]["implementation_task_result"]["reward"] is None for row in rows))
        before = list(self.trace)
        self.run_trial(evaluator=evaluate)
        self.assertEqual(self.trace, before)

    def test_reasonable_keep_can_be_reviewed_without_forcing_edit(self):
        self.actions = {"standard": "keep", "reasoned": "keep"}
        result = self.run_trial()
        self.assertEqual(result["counts"]["qualified_keep"], 2)
        self.assertEqual(result["counts"]["qualified_edit"], 0)
        self.assertFalse(result["trial_goal_met"])
        self.assertFalse(any(kind == "evaluate" for kind, _ in self.trace))

    def test_interrupted_teacher_without_raw_is_not_reissued(self):
        standard_calls = []
        def interrupted(state, directory):
            if state["teacher_arm_id"] == "standard":
                standard_calls.append(1)
                raise KeyboardInterrupt("Ambiguous external call")
            return self.propose(state, directory)
        with self.assertRaises(KeyboardInterrupt):
            self.run_trial(proposer=interrupted)
        result = self.run_trial(proposer=interrupted)
        self.assertEqual(standard_calls, [1])
        self.assertEqual(result["counts"]["teacher_errors"], 1)
        self.assertEqual(result["counts"]["qualified_edit"], 1)
        self.assertEqual(len(Ledger(self.campaign).records("edittrial_teacher_outcomes")), 2)

    def test_tampered_child_revokes_prior_qualification_without_new_job(self):
        self.run_trial()
        row = next(row for row in Ledger(self.campaign).records("samples") if row["qualification"]["status"] == "verified")
        (Path(row["implementation"]["modules_root"]) / "agent_loop/baseline.py").write_text("x = 999\n")
        before = list(self.trace)
        result = self.run_trial()
        self.assertEqual(result["status"], "BLOCKED_ARTIFACT_CHANGED")
        self.assertEqual(self.trace, before)
        self.assertTrue(all(row["revoked"] for row in Ledger(self.campaign).records("samples")))

    def test_partition_and_arm_budget_changes_fail_before_external_calls(self):
        self.protocol["grounded_edit_trial"]["arms"][1]["reasoning_effort"] = "high"
        with self.assertRaisesRegex(ValueError, "preregistered"):
            self.run_trial()
        self.assertEqual(self.trace, [])

    def test_tampered_teacher_raw_is_not_overwritten_or_reissued(self):
        self.run_trial()
        path = self.campaign / "decisions" / "edittrial" / "standard" / "teacher.json"
        path.write_text(json.dumps({"raw": "modified reply"}))
        before = list(self.trace)
        result = self.run_trial()
        self.assertEqual(result["status"], "BLOCKED_ARTIFACT_CHANGED")
        self.assertEqual(self.trace, before)
        self.assertEqual(json.loads(path.read_text())["raw"], "modified reply")
        self.assertTrue(all(row["revoked"] for row in Ledger(self.campaign).records("samples")))

    def test_preteacher_contract_tamper_is_detected_not_silently_replaced(self):
        self.run_trial()
        path = self.campaign / "edittrial_preteacher_contract.json"
        path.write_text(json.dumps({"checks_id": "mutated"}))
        before = list(self.trace)
        result = self.run_trial()
        self.assertEqual(result["status"], "BLOCKED_ARTIFACT_CHANGED")
        self.assertEqual(self.trace, before)
        self.assertEqual(json.loads(path.read_text())["checks_id"], "mutated")

    def test_invalid_fact_quotes_keep_raw_and_do_not_block_original_evidence(self):
        calls = []
        def invalid(state, directory):
            calls.append(1)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "fact_output.json").write_text(json.dumps({"raw": "invalid unchanged quote fixture"}))
            raise ValueError("Selected quote must occur exactly once in its supplied original segment")
        result = self.run_trial(fact_collector=invalid)
        self.assertEqual(result["counts"]["qualified_edit"], 1)
        self.assertEqual(result["counts"]["fact_index_errors"], 1)
        rows = Ledger(self.campaign).records("samples")
        self.assertTrue(all(row["state"]["fact_index"] == [] for row in rows))
        self.assertTrue(all(row["state"]["evidence"][0]["text"] == "Executed fixture returned the bounded expected body" for row in rows))
        self.run_trial(fact_collector=invalid)
        self.assertEqual(calls, [1])

    def test_fact_call_without_raw_stops_and_is_not_reissued(self):
        calls = []
        def interrupted(state, directory):
            calls.append(1)
            raise KeyboardInterrupt("Uncertain fact request")
        with self.assertRaises(KeyboardInterrupt):
            self.run_trial(fact_collector=interrupted)
        result = self.run_trial(fact_collector=interrupted)
        self.assertEqual(result["status"], "AMBIGUOUS_CALL_NOT_RETRIED")
        self.assertEqual(calls, [1])
        self.assertFalse(any(kind == "teacher" for kind, _ in self.trace))

    def test_authentication_error_in_fact_selection_stops_planning(self):
        def forbidden(state, directory):
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "fact_output.json").write_text(json.dumps({"raw": "auth failure fixture"}))
            raise RuntimeError("HTTP 403 authentication")
        result = self.run_trial(fact_collector=forbidden)
        self.assertEqual(result["status"], "BLOCKED_CONFIGURATION")
        self.assertEqual(result["counts"]["fact_index_errors"], 0)
        self.assertFalse(any(kind == "teacher" for kind, _ in self.trace))

    def test_framework_context_uses_the_original_parent_execution_spec(self):
        self.protocol["framework_context"] = {"declared": True}
        self.protocol["upstream_root"] = "/frozen/runtime"
        with patch("research1.data_pipeline.framework_context.build_framework_context", side_effect=RuntimeError("test binding reached")) as build:
            result = self.run_trial()
        self.assertEqual(result["status"], "INCOMPLETE")
        build.assert_called_once_with(Path("/frozen/runtime"), self.protocol, evaluation_spec=self.parent["evaluation_spec"])
        self.assertFalse(any(kind == "teacher" for kind, _ in self.trace))


if __name__ == "__main__":
    unittest.main()
