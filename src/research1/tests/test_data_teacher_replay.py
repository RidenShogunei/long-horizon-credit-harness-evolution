"""Offline exact-byte teacher replay; no transport, indexer or probe is called."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

try:
    from research1.data_pipeline import teacher_replay as replay
except ImportError:
    for parent in Path(__file__).resolve().parents:
        if (parent / "server_fix" / "src").is_dir():
            sys.path.insert(0, str(parent / "server_fix" / "src"))
            break
    spec = importlib.util.spec_from_file_location("offline_teacher_replay", Path(__file__).with_name("teacher_replay.py"))
    replay = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(replay)

from research1.data_pipeline.engine import ArtifactChangedError
from research1.data_pipeline.execution import tree_hash
from research1.data_pipeline.policy import public_view
from research1.data_pipeline.util import digest


class TeacherReplayTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        original_protocol = {'fixture': 'original teacher generation protocol'}
        original_protocol['protocol_hash'] = digest(original_protocol)
        (self.source / 'protocol.json').write_text(json.dumps(original_protocol), encoding='utf-8')
        self.modules = self.source / "modules"
        module_code = {}
        for name in ("agent_loop", "observation", "tools", "context_mgmt", "verification"):
            path = self.modules / name / "baseline.py"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("x = 1\n", encoding="utf-8")
            module_code[name + "/baseline.py"] = "x = 1\n"
        self.row = {"task": "bloom", "root_lineage_id": "original-root", "split": "train",
            "parent_result": {"evaluation_id": "old-eval", "result_sha256": "1" * 64, "trajectory_sha256": "2" * 64}}
        self.contract = {"registry": {}, "selected": [], "parent_hashes": {}}
        self.contract["checks_id"] = "checks_" + digest(self.contract)[:24]
        self.contract_path = self.source / "edittrial_preteacher_contract.json"
        self.contract_path.write_bytes(json.dumps(self.contract, indent=4).encode() + b"\n")
        arms = {}
        self.states = {}
        for arm, effort in replay.ARMS.items():
            teacher_path = self.source / (arm + "_teacher.json")
            state = {"state_id": "old-state-" + arm, "root_lineage_id": "original-root", "split": "train",
                "code_hash": tree_hash(self.modules), "modules_root": str(self.modules), "module_code": module_code,
                "proposal_task_ids": ["bloom"], "permitted_modules": list(name.split("/")[0] for name in module_code),
                "evidence": [{"id": "bloom:original-observation", "text": "Observed bounded original fixture", "partition": "proposal"}],
                "history": [], "verification_catalog": [], "plan_policy_version": "proposal_single_plan_v2",
                "contract": self.contract, "teacher_model": "fixture-model", "teacher_provider": "fixture-provider",
                "teacher_arm_id": arm, "teacher_reasoning_effort": effort, "teacher_max_tokens": 16000,
                "max_candidates": 1, "allow_teacher_http_fallback": False, "teacher_record_path": str(teacher_path),
                "parent_execution_provenance": {"original_evaluation_id": "old-eval", "original_result_sha256": "1" * 64,
                    "original_trajectory_sha256": "2" * 64}}
            self.states[arm] = copy.deepcopy(state)
            state_path = self.source / (arm + "_state.json")
            state_path.write_text(json.dumps(state), encoding="utf-8")
            plan = {"action": "keep", "target_module": None, "locations": [],
                "evidence_refs": ["bloom:original-observation"], "problem_hypothesis": "Bounded observed fixture",
                "intended_change": "Keep unchanged", "preserve_behaviors": [],
                "verification_plan": ["Independently check the original observation"], "verification_kind": None}
            prompt = replay._prompt(state)
            record = {"policy_id": state["plan_policy_version"], "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "visible_view_sha256": digest(public_view(state)),
                "request_settings": {"reasoning_effort": effort, "max_tokens": 16000, "temperature": 0, "allow_http_fallback": False},
                "metadata": {"provider": "fixture-provider", "model": "fixture-model", "temperature": 0,
                    "reasoning_effort": effort, "raw_field": "content", "finish_reason": "stop"},
                "raw": json.dumps({"candidates": [plan]})}
            teacher_path.write_text(json.dumps(record, indent=4) + "\n", encoding="utf-8")
            arms[arm] = {"teacher_path": str(teacher_path), "state_path": str(state_path)}
        artifacts = {"contract": self.artifact(self.contract_path)}
        for arm, values in arms.items():
            for key, value in values.items():
                artifacts[arm + "_" + key] = self.artifact(Path(value))
        self.protocol = {"protocol_hash": "target-admission-protocol", "modules_hash": tree_hash(self.modules),
            "sft": {"plan_policy_version": "proposal_single_plan_v2"},
            "policies": {"teacher": {"provider": "fixture-provider", "model": "fixture-model"},
                "factual_reviewer": {"version": "typed-factual-review-new"}},
            "grounded_edit_trial": {"teacher_max_tokens": 16000, "teacher_source": {
                "campaign": str(self.source), "arms": arms, "contract_path": str(self.contract_path), "artifacts": artifacts}}}

    @staticmethod
    def artifact(path):
        return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    def update_artifact(self, path):
        artifacts = self.protocol["grounded_edit_trial"]["teacher_source"]["artifacts"]
        for label, value in list(artifacts.items()):
            if Path(value["path"]) == path:
                artifacts[label] = self.artifact(path)

    def test_replay_preserves_exact_view_prompt_plan_contract_and_original_bytes(self):
        before = {path: path.read_bytes() for path in self.source.glob("*.json")}
        with patch("research1.distill._tongji_chat", side_effect=AssertionError("No generation allowed")):
            loaded = replay.load_teacher_replay(self.protocol, self.row)
        self.assertEqual(loaded["contract"], self.contract)
        self.assertEqual(loaded["contract_bytes"], before[self.contract_path])
        for arm in replay.ARMS:
            state = loaded["arms"][arm]["state"]
            self.assertEqual(public_view(state), public_view(self.states[arm]))
            self.assertNotEqual(state["state_id"], self.states[arm]["state_id"])
            self.assertEqual(state["original_state_id"], self.states[arm]["state_id"])
            self.assertEqual(state["modules_root"], self.states[arm]["modules_root"])
            self.assertEqual(state["parent_execution_provenance"], self.states[arm]["parent_execution_provenance"])
            self.assertEqual(state["factual_review_policy"], "typed-factual-review-new")
            self.assertEqual(state["admission_protocol_hash"], self.protocol["protocol_hash"])
            self.assertEqual(replay._prompt(state), replay._prompt(self.states[arm]))
        self.assertEqual(before, {path: path.read_bytes() for path in self.source.glob("*.json")})

    def test_any_source_byte_change_is_rejected(self):
        path = self.source / "reasoned_teacher.json"
        path.write_bytes(path.read_bytes() + b" ")
        with self.assertRaisesRegex(ArtifactChangedError, "bytes changed"):
            replay.load_teacher_replay(self.protocol, self.row)

    def test_missing_byte_manifest_entry_is_rejected(self):
        del self.protocol["grounded_edit_trial"]["teacher_source"]["artifacts"]["standard_state_path"]
        with self.assertRaisesRegex(ArtifactChangedError, "missing from the byte manifest"):
            replay.load_teacher_replay(self.protocol, self.row)

    def test_rehashed_prompt_rewrite_still_fails_exact_frozen_view_binding(self):
        path = self.source / "standard_teacher.json"
        record = json.loads(path.read_text())
        record["prompt"] += "\nDifferent instruction"
        record["prompt_sha256"] = hashlib.sha256(record["prompt"].encode()).hexdigest()
        path.write_text(json.dumps(record))
        self.update_artifact(path)
        with self.assertRaisesRegex(ArtifactChangedError, "exact original teacher prompt"):
            replay.load_teacher_replay(self.protocol, self.row)

    def test_replay_does_not_accept_a_newly_substituted_preteacher_contract(self):
        changed = copy.deepcopy(self.contract)
        changed["selected"] = [{"mechanism_id": "new-after-teacher"}]
        changed["checks_id"] = "checks_" + digest({key: value for key, value in changed.items() if key != "checks_id"})[:24]
        self.contract_path.write_text(json.dumps(changed))
        self.update_artifact(self.contract_path)
        with self.assertRaisesRegex(ArtifactChangedError, "state contract"):
            replay.load_teacher_replay(self.protocol, self.row)

    def test_changed_target_generation_settings_are_rejected_without_calls(self):
        self.protocol["grounded_edit_trial"]["teacher_max_tokens"] = 8000
        with self.assertRaisesRegex(ArtifactChangedError, "request settings"):
            replay.load_teacher_replay(self.protocol, self.row)

    def test_target_cannot_upgrade_saved_plans_to_v3(self):
        for value in ('v3', 'research1.module-contracts.v3'):
            with self.subTest(value=value):
                self.protocol['verification_registry_version'] = value
                with self.assertRaisesRegex(ArtifactChangedError, 'only original registry v2'):
                    replay.load_teacher_replay(self.protocol, self.row)

    def test_original_source_protocol_registry_must_match_legacy_target(self):
        path = self.source / 'protocol.json'
        source = {'verification_registry_version': 'research1.module-contracts.v3'}
        source['protocol_hash'] = digest(source)
        path.write_text(json.dumps(source), encoding='utf-8')
        with self.assertRaisesRegex(ArtifactChangedError, 'original teacher protocol and target'):
            replay.load_teacher_replay(self.protocol, self.row)

    def test_original_state_cannot_smuggle_v3_into_legacy_replay(self):
        path = self.source / 'reasoned_state.json'
        state = json.loads(path.read_text(encoding='utf-8'))
        state['verification_registry_version'] = 'research1.module-contracts.v3'
        path.write_text(json.dumps(state), encoding='utf-8')
        self.update_artifact(path)
        with self.assertRaisesRegex(ArtifactChangedError, 'original teacher state and target'):
            replay.load_teacher_replay(self.protocol, self.row)

    def test_explicit_v2_aliases_keep_the_original_legacy_public_view(self):
        self.protocol['verification_registry_version'] = 'v2'
        loaded = replay.load_teacher_replay(self.protocol, self.row)
        for arm in replay.ARMS:
            state = loaded['arms'][arm]['state']
            self.assertNotIn('verification_registry_version', state)
            self.assertEqual(public_view(state), public_view(self.states[arm]))


if __name__ == "__main__":
    unittest.main()
