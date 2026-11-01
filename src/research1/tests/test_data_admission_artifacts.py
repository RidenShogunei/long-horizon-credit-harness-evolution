"""V6 positives use real saved review/editor/result fixtures, never asserted PASS."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from research1.data_pipeline import execution, modules
from research1.data_pipeline.policy import collect_candidates, digest
from research1.tests import test_data_plan_contract_binding as binding_fixture
from research1.tests.test_data_modules import VERIFIER

_ROOT = Path(__file__).resolve().parents[1] / "data_pipeline"
_SPEC = importlib.util.spec_from_file_location("research1.data_pipeline.admission_artifacts", _ROOT / "admission_artifacts.py")
artifacts = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = artifacts
_SPEC.loader.exec_module(artifacts)
_SFT_SPEC = importlib.util.spec_from_file_location("artifact_bound_sft", _ROOT / "sft.py")
sft = importlib.util.module_from_spec(_SFT_SPEC)
_SFT_SPEC.loader.exec_module(sft)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


class AdmissionArtifactTests(unittest.TestCase):
    def setUp(self):
        self.fixture = binding_fixture.PlanContractBindingTests("test_saved_independent_pass_binds_only_existing_tests_and_scope")
        self.fixture.setUp(); self.addCleanup(self.fixture.doCleanups)
        self.root = self.fixture.directory
        state = self.fixture.state
        state.update(code_hash=execution.tree_hash(self.fixture.parent), proposal_task_ids=["task"],
            max_candidates=1, teacher_reasoning_effort="none", allow_teacher_http_fallback=False)
        self.fixture.freeze(); self.fixture.facts()
        binding, _ = self.fixture.binding()
        state["plan_contract_binding"] = binding
        teacher = self.root / "teacher"
        teacher_meta = {**self.fixture.meta, "reasoning_effort": "none"}
        with patch("research1.distill._tongji_chat", return_value=(json.dumps({"candidates": [self.fixture.plan]}), teacher_meta)):
            collect_candidates(state, teacher)
        state["teacher_record_path"] = str(teacher / "teacher.json")
        fixed = VERIFIER.replace("        if state.consecutive_complete_signals",
            '        if getattr(getattr(state, "last_tool_result", None), "success", True) is False:\n            return False, "tool failed"\n        if state.consecutive_complete_signals')
        raw = json.dumps({"files": [{"path": "verification/baseline.py", "edits": [{"old": VERIFIER, "new": fixed}]}]})
        with patch("research1.distill._tongji_chat", return_value=(raw, self.fixture.meta)):
            attempt = modules.implement(self.fixture.plan, state, self.root / "implementation")
        self.assertEqual(attempt["status"], "applied")
        parent_probe = state["contract"]["preprobed"][0]["parent_probe"]
        # Pick the exact mechanism's already frozen probe, never run a new one.
        parent_probe = next(row["parent_probe"] for row in state["contract"]["preprobed"] if row["mechanism_id"] == self.fixture.plan["verification_kind"])
        child_probe = copy.deepcopy(parent_probe)
        child_probe["checks"] = {name: True for name in child_probe["checks"]}
        pack = state["contract"]["registry"]["packs"][self.fixture.plan["verification_kind"]]
        scope = modules._scope(self.fixture.parent, Path(attempt["modules_root"]), self.fixture.plan, state["contract"], pack)
        behavior = {"status": "PASS", "covers_plan": True, "executable": True, "checks_id": state["contract"]["checks_id"],
            "mechanism_id": self.fixture.plan["verification_kind"], "parent_probe": parent_probe,
            "child_probe": child_probe, "scope": scope}
        self.record = {"sample_id": "actual_saved_fixture", "source_kind": "fresh_teacher_plan", "protocol_id": "v6-test",
            "state_id": state["state_id"], "root_lineage_id": state["root_lineage_id"], "state": state,
            "plan": self.fixture.plan, "implementation": attempt, "mechanism_id": self.fixture.plan["verification_kind"],
            "teacher_path": state["teacher_record_path"], "teacher_record_path": state["teacher_record_path"],
            "teacher_prompt_sha256": json.loads((teacher / "teacher.json").read_text())["prompt_sha256"],
            "qualification": {"status": "verified", "factual_review": state["plan_factual_review"], "behavior": behavior,
                "proposal_task_result": self.trial("parent", self.fixture.parent),
                "implementation_task_result": self.trial("child", Path(attempt["modules_root"]))}}
        self.record["admission_artifacts"] = artifacts.seal_positive_artifacts(self.record)

    def trial(self, label, modules_root):
        output = self.root / (label + "_output")
        job, trial = output / "job", output / "job/trial"
        task = self.root / "task"; task.mkdir(exist_ok=True); (task / "task.toml").write_text("task = 'fixture'\n")
        spec = {"task": "task", "role": "proposal", "task_path": str(task), "task_hash": execution.tree_hash(task),
            "modules_root": str(modules_root), "code_hash": execution.tree_hash(modules_root), "provider_id": "fixed",
            "model": "fixed", "temperature": .7, "max_turns": 200, "reasoning_effort": "none"}
        config = {"task": {"path": str(task)}, "trial_name": "trial", "trials_dir": str(job),
            "agent": {"model_name": "fixed", "kwargs": {"modules_root": str(modules_root), "temperature": .7,
                "max_turns": 200, "composer_name": "static", "composer_scope": "all", "reasoning_effort": "none"}}}
        write(job / "config.json", {"tasks": [{"path": str(task)}], "jobs_dir": str(output)})
        write(trial / "config.json", config)
        write(trial / "result.json", {"task_name": "task", "trial_name": "trial", "config": config,
            "finished_at": "2026-11-01T00:00:00Z", "verifier_result": {"rewards": {"reward": 0}}, "exception_info": None})
        write(trial / "agent/trajectory.json", {"steps": []})
        result = execution.bind_trial(output, spec)
        result["evaluation_spec"] = spec
        return result

    def test_real_saved_fixture_passes_without_model_probe_or_task_execution(self):
        with patch("research1.distill._tongji_chat") as chat, patch.object(modules, "probe_module") as probe:
            self.assertEqual(artifacts.validate_positive_artifacts(self.record), (True, "verified_v6_positive_artifact_binding"))
            self.assertTrue(sft._qualification(self.record)[0])
        chat.assert_not_called(); probe.assert_not_called()
        self.assertEqual(self.record["qualification"]["implementation_task_result"]["reward"], 0)

    def test_new_positive_missing_seal_cannot_be_exported_or_silently_resealed(self):
        value = copy.deepcopy(self.record); value.pop("admission_artifacts")
        self.assertFalse(sft._qualification(value)[0])

    def test_unchanged_export_is_idempotent_and_byte_mutation_retracts_current(self):
        campaign = self.root / "export"
        one = sft.export_dataset(campaign, [self.record]); two = sft.export_dataset(campaign, [self.record])
        self.assertEqual(one["dataset_revision"], two["dataset_revision"])
        self.assertEqual(one["counts"]["train"]["edit"], 1)
        teacher = Path(self.record["teacher_path"])
        teacher.write_bytes(teacher.read_bytes() + b" ")
        changed = sft.export_dataset(campaign, [self.record]); repeated = sft.export_dataset(campaign, [self.record])
        self.assertEqual(changed["counts"]["train"]["edit"], 0)
        self.assertEqual(changed["dataset_revision"], repeated["dataset_revision"])
        pointer = json.loads((campaign / "exports/sft/CURRENT.json").read_text())
        self.assertEqual(pointer["dataset_revision"], changed["dataset_revision"])
        self.assertNotEqual(pointer["dataset_revision"], one["dataset_revision"])

    def test_changed_teacher_reply_content_and_effort_cannot_pass(self):
        teacher = Path(self.record["teacher_path"]); original = teacher.read_bytes()
        for mutation in ("raw", "effort", "prompt"):
            value = json.loads(original)
            if mutation == "raw": value["raw"] = json.dumps({"candidates": [dict(self.fixture.plan, intended_change="Another edit")]})
            elif mutation == "effort": value["metadata"]["reasoning_effort"] = "low"
            else: value["prompt"] += "changed"
            write(teacher, value)
            self.assertFalse(artifacts.validate_positive_artifacts(self.record)[0], mutation)
            with self.assertRaises(ValueError): artifacts.seal_positive_artifacts(self.record)
        teacher.write_bytes(original)

    def test_changed_review_applicability_editor_patch_and_result_bytes_fail(self):
        files = self.record["admission_artifacts"]["files"]
        for name in ("factual_raw", "factual_result", "applicability_output", "applicability_result", "editor_output",
                     "patch", "parent_trial_result_path", "child_trial_result_path", "child_trajectory", "child_job_config"):
            path = Path(files[name]["path"]); original = path.read_bytes()
            path.write_bytes(original + b" ")
            self.assertFalse(artifacts.validate_positive_artifacts(self.record)[0], name)
            path.write_bytes(original)
        self.assertTrue(artifacts.validate_positive_artifacts(self.record)[0])

    def test_q_review_or_probe_claim_mutation_cannot_pass(self):
        for change in ("review", "goal", "preserve", "checks", "score"):
            value = copy.deepcopy(self.record)
            q = value["qualification"]
            if change == "review": q["factual_review"] = dict(q["factual_review"], reason="Other review")
            elif change == "checks": q["behavior"]["checks_id"] = "another"
            elif change == "score": q["implementation_task_result"]["reward"] = 1
            else:
                pack = value["state"]["contract"]["registry"]["packs"][value["plan"]["verification_kind"]]
                key = pack["goal_checks" if change == "goal" else "preserve_checks"][0]
                q["behavior"]["child_probe"]["checks"][key] = False
            self.assertFalse(artifacts.validate_positive_artifacts(value)[0], change)
            value.pop("admission_artifacts")
            with self.assertRaises(ValueError, msg=change): artifacts.seal_positive_artifacts(value)

    def test_real_child_tree_drift_is_rejected(self):
        child = Path(self.record["implementation"]["modules_root"]) / "verification/baseline.py"
        child.write_text(child.read_text() + "\n# drift\n")
        self.assertFalse(artifacts.validate_positive_artifacts(self.record)[0])

    def test_historical_and_no_new_contract_fixtures_keep_the_previous_gate(self):
        for version in ("proposal_single_plan_v2", "proposal_single_plan_v3", "proposal_single_plan_v4", "proposal_single_plan_v5"):
            value = copy.deepcopy(self.record); value["state"]["plan_policy_version"] = version; value.pop("admission_artifacts")
            self.assertFalse(artifacts.artifact_gate_applies(value))
            self.assertTrue(artifacts.validate_positive_artifacts(value)[0])
        value = copy.deepcopy(self.record); value["state"]["contract"].pop("binding_mode")
        self.assertFalse(artifacts.artifact_gate_applies(value))


if __name__ == "__main__": unittest.main()
