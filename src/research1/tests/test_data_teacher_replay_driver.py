"""Offline old-plan admission through the complete driver, without generation."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

for parent in Path(__file__).resolve().parents:
    if (parent / "server_fix" / "src").is_dir():
        sys.path.insert(0, str(parent / "server_fix" / "src"))
        break


def _load(name, path):
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


_HERE = Path(__file__).resolve().parent
if (_HERE / "teacher_replay.py").is_file():
    _replay = _load("offline_replay_driver_helper", _HERE / "teacher_replay.py")
    _driver = _load("offline_replay_driver", _HERE / "edit_trial.py")
else:
    from research1.data_pipeline import teacher_replay as _replay, edit_trial as _driver
try:
    from research1.tests import test_data_edit_trial as _fixtures
except ImportError:
    _fixtures = _load("offline_replay_driver_fixtures", _HERE.parent / "trial_patch" / "test_data_edit_trial.py")

from research1.data_pipeline.ledger import Ledger
from research1.data_pipeline.policy import public_view
from research1.data_pipeline.util import digest


class TeacherReplayDriverTests(unittest.TestCase):
    def setUp(self):
        original_driver = patch.object(_fixtures, "trial", _driver)
        original_driver.start()
        self.addCleanup(original_driver.stop)
        self.fixture = _fixtures.EditTrialTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        shim = patch.dict(sys.modules, {"research1.data_pipeline.teacher_replay": _replay})
        shim.start()
        self.addCleanup(shim.stop)
        fixture = self.fixture
        fixture.protocol["sft"] = {"plan_policy_version": "proposal_single_plan_v2"}
        fixture.protocol["policies"] = {"teacher": {"provider": "fixture-provider", "model": "fixture-model"},
            "factual_reviewer": {"version": "original-review"}}
        fixture.protocol["grounded_edit_trial"]["teacher_max_tokens"] = 16000
        fixture.contract["checks_id"] = "checks_" + digest({key: value for key, value in fixture.contract.items() if key != "checks_id"})[:24]
        fixture.protocol['protocol_hash'] = digest({key:value for key,value in fixture.protocol.items() if key!='protocol_hash'})
        (fixture.campaign/'protocol.json').write_text(json.dumps(fixture.protocol),encoding='utf-8')
        fixture.actions = {"standard": "keep", "reasoned": "edit"}

        def original_teacher(state, directory):
            plans = fixture.propose(state, directory)
            prompt = _replay._prompt(state)
            record = {"policy_id": state["plan_policy_version"], "prompt": prompt,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(), "visible_view_sha256": digest(public_view(state)),
                "request_settings": {"reasoning_effort": state["teacher_reasoning_effort"], "max_tokens": 16000,
                    "temperature": 0, "allow_http_fallback": False},
                "metadata": {"provider": "fixture-provider", "model": "fixture-model", "temperature": 0,
                    "reasoning_effort": state["teacher_reasoning_effort"], "raw_field": "content", "finish_reason": "stop"},
                "raw": json.dumps({"candidates": plans})}
            (directory / "teacher.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
            return plans

        def old_review(plan, state):
            result = fixture.review(plan, state)
            if state["teacher_arm_id"] == "reasoned":
                result["status"] = "pending_review"
            return result

        result = fixture.run_trial(proposer=original_teacher, reviewer=old_review)
        self.assertEqual(result["counts"]["qualified_edit"], 0)
        self.original_campaign = fixture.campaign
        states = {state["teacher_arm_id"]: state for state in Ledger(fixture.campaign).records("states")}
        source = {"campaign": str(fixture.campaign), "arms": {},
            "contract_path": str(fixture.campaign / "edittrial_preteacher_contract.json"), "artifacts": {}}
        for arm, state in states.items():
            source["arms"][arm] = {"state_path": str(fixture.campaign / "states" / state["state_id"] / "state.json"),
                                   "teacher_path": state["teacher_record_path"]}
        for label, path in [("contract", source["contract_path"])] + [
                (arm + "_" + name, value) for arm, paths in source["arms"].items() for name, value in paths.items()]:
            source["artifacts"][label] = {"path": path, "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}
        self.original_bytes = {item["path"]: Path(item["path"]).read_bytes() for item in source["artifacts"].values()}
        fixture.campaign = fixture.root / "target"
        fixture.campaign.mkdir()
        fixture.protocol["protocol_hash"] = "new-admission-protocol"
        fixture.protocol["policies"]["factual_reviewer"]["version"] = "typed-factual-review-new"
        fixture.protocol["grounded_edit_trial"]["teacher_source"] = source
        (fixture.campaign / "protocol.json").write_text(json.dumps(fixture.protocol))
        fixture.trace.clear()

    @staticmethod
    def forbidden(*args, **kwargs):
        raise AssertionError("Replay must not call generation, indexing or a new preteacher probe builder")

    def run_replay(self):
        return self.fixture.run_trial(proposer=self.forbidden, fact_collector=self.forbidden,
            interface_builder=self.forbidden, contract_builder=self.forbidden, evidence_selector=self.forbidden)

    def test_saved_keep_and_edit_pass_through_new_admission_without_generation(self):
        result = self.run_replay()
        self.assertEqual(result["status"], "GROUNDED_EDIT_TRIAL_COMPLETE", result)
        self.assertEqual(result["counts"]["qualified_edit"], 1)
        self.assertEqual(result["counts"]["qualified_keep"], 1)
        rows = Ledger(self.fixture.campaign).records("samples")
        self.assertEqual({row["source_kind"] for row in rows}, {"replayed_teacher_plan_admission"})
        self.assertTrue(all(row["original_state_id"] != row["state_id"] for row in rows))
        self.assertTrue(all(Path(row["teacher_record_path"]).is_relative_to(self.original_campaign) for row in rows))
        self.assertTrue(all(row["state"]["admission_protocol_hash"] == "new-admission-protocol" for row in rows))
        self.assertEqual(self.original_bytes, {path: Path(path).read_bytes() for path in self.original_bytes})
        self.assertEqual((self.fixture.campaign / "edittrial_preteacher_contract.json").read_bytes(),
            (self.original_campaign / "edittrial_preteacher_contract.json").read_bytes())
        self.assertFalse((self.fixture.campaign / "decisions").exists())
        self.assertEqual(Ledger(self.fixture.campaign).records("edittrial_proposals"), [])
        self.assertEqual(len(Ledger(self.fixture.campaign).records("edittrial_replayed_proposals")), 2)

    def test_completed_replay_resume_does_not_duplicate_admission_calls_or_jobs(self):
        first = self.run_replay()
        before = copy.deepcopy(self.fixture.trace)
        second = self.run_replay()
        self.assertEqual(first["counts"], second["counts"])
        self.assertEqual(self.fixture.trace, before)

    def test_source_mutation_blocks_replay_and_revokes_existing_admission(self):
        self.run_replay()
        source_path = Path(next(iter(self.original_bytes)))
        source_path.write_bytes(source_path.read_bytes() + b" ")
        before = copy.deepcopy(self.fixture.trace)
        result = self.run_replay()
        self.assertEqual(result["status"], "BLOCKED_ARTIFACT_CHANGED", result)
        self.assertEqual(self.fixture.trace, before)
        self.assertTrue(all(row["revoked"] for row in Ledger(self.fixture.campaign).records("samples")))


if __name__ == "__main__":
    unittest.main()
