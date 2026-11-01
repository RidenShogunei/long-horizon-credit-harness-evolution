"""Offline tests for pre-teacher Harness API probe I/O publication."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research1.data_pipeline import applicability as app
from research1.data_pipeline import module_adapters
from research1.data_pipeline import modules_v3 as mod
from research1.data_pipeline.predecision_probe_io import (
    SCHEMA,
    publish_parent_probe_io,
    validate_probe_io_state,
)
from research1.plan_init import frozen_false_concat_checks
from research1.tests.test_data_modules import VERIFIER


def _tools_results(*, include_unknown: bool = True) -> dict[str, dict]:
    rows = {}
    for index, check in enumerate(frozen_false_concat_checks()):
        rows[check["id"]] = {
            "warning": f"Command {index + 1} should end with newline when selected checks apply",
            "error": None if index % 2 == 0 else "",
            "is_task_complete": False,
            "commands": [[f"cmd-{index}", float(index) + 0.5]],
        }
    if include_unknown:
        rows["synthetic_unlisted_probe_case"] = {
            "warning": "",
            "error": None,
            "is_task_complete": False,
            "commands": [["orphan", 1.0]],
        }
    return rows


def _probe(root, mechanism, **kwargs):
    pack = mod.frozen_registry()["packs"][mechanism]
    value = {
        "mechanism_id": mechanism,
        "checks": {name: False for name in pack["goal_checks"]}
        | {name: True for name in pack["preserve_checks"]},
    }
    if mechanism == "false_concat_preservation":
        value["results"] = _tools_results()
    return value


class PredecisionProbeIOTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name).resolve() / "predecision"
        self.parent = Path(self.temp.name).resolve() / "parent"
        (self.parent / "verification").mkdir(parents=True)
        (self.parent / "verification/baseline.py").write_text(VERIFIER, encoding="utf-8")
        self.witness = "completion confirmed with [tool error] execution failed"
        self.state = {
            "state_id": "parent1",
            "root_lineage_id": "r",
            "modules_root": str(self.parent),
            "module_code": {"verification/baseline.py": VERIFIER},
            "permitted_modules": list(mod.MODULES),
            "evidence": [{"id": "proposal1", "text": self.witness, "partition": "proposal"}],
            "history": [],
            "diagnostics": [],
            "plan_policy_version": app.PLAN_POLICY,
            "verification_registry_version": module_adapters.STRICT_VERSION,
            "decision_protocol_hash": "protocol_fixture_sha256",
        }
        self.protocol_hash = "protocol_fixture_sha256"

    def freeze_contract(self, state=None):
        state = state or self.state
        with patch.object(mod, "probe_module", side_effect=_probe) as probe:
            contract = mod.build_contract(state)
        self.original_contract = copy.deepcopy(contract)
        return contract, probe

    def publish(self, contract=None, state=None, directory=None):
        contract = contract or self.original_contract
        state = state or self.state
        directory = directory or self.directory
        return publish_parent_probe_io(
            state, contract, directory, protocol_hash=self.protocol_hash
        )

    def test_projects_observations_without_semantic_repair_hints(self):
        contract, probe = self.freeze_contract()
        before_state = copy.deepcopy(self.state)
        new_state, final_contract = self.publish(contract)
        probe.assert_called()
        self.assertEqual(self.state, before_state)
        added = new_state["evidence"][len(before_state["evidence"]) :]
        self.assertEqual(len(added), 1)
        payload = json.loads(added[0]["text"])
        self.assertEqual(payload["invoked_api"], "BaselineTools.parse_llm_response")
        self.assertNotIn("mechanism_id", payload)
        self.assertNotIn("local_functions", payload)
        self.assertEqual([row["case"] for row in payload["observations"]], [f"case_{i}" for i in range(7)])
        unknown = payload["observations"][-1]
        self.assertIsNone(unknown["fixture_defined_input"])
        self.assertEqual(unknown["input_provenance"], "not_available")
        self.assertEqual(unknown["parsed_commands"], [["orphan", 1.0]])
        receipt = json.loads((self.directory / "predecision_harness_api_io_receipt.json").read_text(encoding="utf-8"))
        self.assertEqual(receipt["original_evidence_order"], ["proposal1"])
        self.assertIn("empty_wait_mid_batch", receipt["io_projection_trace"][0]["observation_case_map"][0]["results_key"])
        self.assertNotIn("false_concat", added[0]["text"])
        self.assertNotIn("empty_wait", added[0]["text"])
        self.assertIn("selected checks apply", added[0]["text"])
        self.assertEqual(final_contract["checks_id"], new_state["contract"]["checks_id"])

    def test_other_producer_with_colliding_case_key_has_no_fixture_input(self):
        colliding_key = frozen_false_concat_checks()[0]["id"]

        def probe(root, mechanism, **kwargs):
            value = _probe(root, mechanism, **kwargs)
            if mechanism == "verification_error_gate":
                value["results"] = {
                    colliding_key: {
                        "warning": "unrelated harness output",
                        "error": None,
                        "is_task_complete": False,
                        "commands": [["other-producer", 2.0]],
                    }
                }
            return value

        with patch.object(mod, "probe_module", side_effect=probe):
            contract = mod.build_contract(self.state)
        self.original_contract = copy.deepcopy(contract)
        published, _ = self.publish(contract, directory=self.directory.parent / "colliding")
        added = published["evidence"][1:]
        self.assertEqual(len(added), 2)
        payloads = [json.loads(item["text"]) for item in added]
        other = next(body for body in payloads if body["invoked_api"] is None)
        self.assertEqual(len(other["observations"]), 1)
        row = other["observations"][0]
        self.assertEqual(row["case"], "case_0")
        self.assertIsNone(row["fixture_defined_input"])
        self.assertEqual(row["input_provenance"], "not_available")
        self.assertEqual(row["parsed_commands"], [["other-producer", 2.0]])
        parser = next(body for body in payloads if body["invoked_api"] == "BaselineTools.parse_llm_response")
        self.assertTrue(any(obs["input_provenance"] == "fixture_definition" for obs in parser["observations"]))
        self.assertTrue(validate_probe_io_state(published)["verified"])

    def test_multi_evidence_order_survives_canonical_mapping_reorder(self):
        state = copy.deepcopy(self.state)
        state["evidence"] = [
            {"id": "z_trajectory", "text": "trajectory bytes", "partition": "proposal"},
            {"id": "a_feedback", "text": "feedback bytes", "partition": "proposal"},
        ]
        contract, _ = self.freeze_contract(state)
        reordered = json.loads(json.dumps(contract, sort_keys=True))
        self.assertNotEqual(
            list(reordered["evidence_hashes"].keys()),
            [item["id"] for item in state["evidence"]],
        )
        published, _ = self.publish(reordered, state=state, directory=self.directory.parent / "order_case")
        self.assertEqual(
            [item["id"] for item in published["evidence"]],
            ["z_trajectory", "a_feedback", published["evidence"][-1]["id"]],
        )
        proof = validate_probe_io_state(published)
        self.assertEqual(proof["original_evidence_order"], ["z_trajectory", "a_feedback"])
        self.assertEqual(len(published["evidence"]), 3)

    def test_validate_and_idempotent_resume_without_reprobe(self):
        contract, _ = self.freeze_contract()
        published, _ = self.publish(contract)
        proof = validate_probe_io_state(published)
        self.assertTrue(proof["verified"])
        self.assertEqual(proof["artifacts"], published["predecision_probe_artifacts"])
        with patch.object(mod, "probe_module", side_effect=AssertionError("must not reprobe")):
            again, final_again = publish_parent_probe_io(
                published, contract, self.directory, protocol_hash=self.protocol_hash
            )
        self.assertEqual(again, published)
        self.assertEqual(final_again, published["contract"])

    def test_rejects_collision_tamper_protocol_and_fixture_source(self):
        contract, _ = self.freeze_contract()
        published, _ = self.publish(contract)
        collision = copy.deepcopy(self.state)
        with patch(
            "research1.data_pipeline.predecision_probe_io._neutral_evidence_id",
            return_value="proposal1",
        ):
            with self.assertRaisesRegex(ValueError, "collision"):
                publish_parent_probe_io(
                    collision, contract, self.directory.parent / "collision", protocol_hash=self.protocol_hash
                )
        bad_protocol = copy.deepcopy(self.state)
        bad_protocol["decision_protocol_hash"] = "other"
        with self.assertRaisesRegex(ValueError, "decision_protocol_hash"):
            publish_parent_probe_io(bad_protocol, contract, self.directory.parent / "proto", protocol_hash=self.protocol_hash)
        raw = self.directory / "predecision_harness_api_io_raw_contract.json"
        raw.write_text(raw.read_text(encoding="utf-8") + " ", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "sha256|bytes changed"):
            publish_parent_probe_io(published, contract, self.directory, protocol_hash=self.protocol_hash)
        source_dir = self.directory.parent / "source_case"
        published, _ = publish_parent_probe_io(self.state, contract, source_dir, protocol_hash=self.protocol_hash)
        receipt = json.loads((source_dir / "predecision_harness_api_io_receipt.json").read_text(encoding="utf-8"))
        receipt["fixture_input_source"]["sha256"] = "0" * 64
        (source_dir / "predecision_harness_api_io_receipt.json").write_text(
            json.dumps(receipt, indent=2) + "\n", encoding="utf-8"
        )
        with self.assertRaisesRegex(ValueError, "sha256|fixture input"):
            validate_probe_io_state(published)

    def test_validator_still_valid_after_unrelated_teacher_metadata(self):
        contract, _ = self.freeze_contract()
        published, _ = self.publish(contract)
        published["teacher_arm_id"] = "arm-9"
        published["state_id"] = "different-public-id"
        published["fact_index"] = {"proposal1": 0}
        self.assertTrue(validate_probe_io_state(published)["verified"])

    def test_rejects_extra_evidence_and_symlink_artifact_parent(self):
        contract, _ = self.freeze_contract()
        published, _ = self.publish(contract)
        published["evidence"].append({"id": "extra", "text": "hidden"})
        with self.assertRaisesRegex(ValueError, "final evidence length"):
            validate_probe_io_state(published)
        if hasattr(Path, "is_symlink"):
            link_root = Path(self.temp.name).resolve() / "link_parent"
            target = Path(self.temp.name).resolve() / "target_dir"
            target.mkdir()
            try:
                link_root.symlink_to(target, target_is_directory=True)
                with self.assertRaisesRegex(ValueError, "symlink"):
                    publish_parent_probe_io(
                        self.state,
                        contract,
                        link_root / "nested",
                        protocol_hash=self.protocol_hash,
                    )
            except OSError:
                self.skipTest("symlink creation unavailable")


if __name__ == "__main__":
    unittest.main()
