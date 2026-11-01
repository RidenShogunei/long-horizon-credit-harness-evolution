"""Formal quota expansion uses bounded immutable children, never live tasks."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research1.data_pipeline.ledger import Ledger
from research1.data_pipeline.protocol import load_protocol
from research1.data_pipeline.scale import prepare_formal, run_formal
from research1.data_pipeline.util import atomic_json, digest, tree_hash


class FormalScaleTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.calibration = self.root / "calibration"
        self.calibration.mkdir()
        modules = self.calibration / "baseline/modules/tools"
        modules.mkdir(parents=True)
        (modules / "baseline.py").write_text("def parse():\n    return 1\n")
        registry = {"version": "fixture", "packs": {}}
        atomic_json(self.calibration / "contracts.json", registry)
        self.base = {"protocol_id": "research1-data-v1", "seed": 42,
            "modules_root": str(modules.parent), "modules_hash": tree_hash(modules.parent),
            "registry_sha256": digest(registry), "policies": {}, "source_sha256": {},
            "budget": {"regular": 96, "retries": 32, "total": 128},
            "stage_budget": {"noise": 24, "reward": 44, "proposal": 18, "dev": 10},
            "task_sets": {"proposal": ["p" + str(i) for i in range(24)],
                          "proposal_train": ["p" + str(i) for i in range(19)],
                          "proposal_validation": ["p" + str(i) for i in range(19, 24)],
                          "reward": ["r" + str(i) for i in range(24)],
                          "dev": ["d" + str(i) for i in range(24)], "debug": ["z" + str(i) for i in range(8)]},
            "sampling": {"K": 2, "M": 2, "L_after": 2, "R": 1, "reward_n": 4,
                         "dev_n": 2, "proposal_per_decision": 2, "lambda": .5},
            "sft": {"target_profile": None}, "formal_scale": None}
        self.base["protocol_hash"] = digest(self.base)
        atomic_json(self.calibration / "protocol.json", self.base)
        atomic_json(self.calibration / "status.json", {"status": "NEEDS_SCALE_LOCK", "calibration_complete": True})
        self.campaign = self.root / "formal"
        self.config = {"targets": {"origins": 2, "verified_edits": 2, "complete_mc_labels": 2,
                                   "mechanisms": 1, "covered_modules": 1, "validation_edits": 1},
                       "budget": {"regular": 20, "retries": 5, "total": 25},
                       "sampling": {"R": 1, "reward_n": 4, "dev_n": 2}, "max_groups": 5}
        self.calls = []

    def prepare(self, config=None):
        return prepare_formal(self.campaign, self.calibration, config or self.config)

    def fake_group(self, child):
        protocol = load_protocol(child)
        index = protocol["formal_group_index"]
        self.calls.append((str(child), protocol["seed"], protocol["collection_split"], deepcopy(protocol["budget"])))
        ledger = Ledger(child)
        for key, kind in (("job", "regular"), *(([("retry", "retry")] if index == 1 else []))):
            if not ledger.get_job(key):
                ledger.reserve_job(key, {"role": "proposal"}, kind)
                ledger.start_job(key, "test")
                ledger.complete_job(key, {"status": "PASS", "reward": 1.0})
        split = protocol["collection_split"]
        task = protocol["task_sets"]["proposal_validation" if split == "validation" else "proposal_train"][0]
        state = {"state_id": f"state-{index}", "root_lineage_id": f"lineage-{index}", "split": split,
                 "proposal_task_ids": [task], "permitted_modules": ["tools"], "history": [],
                 "module_code": {"tools/baseline.py": "def parse():\n    return 1\n"},
                 "evidence": [{"id": "e1", "text": "Actual proposal execution feedback"}]}
        plan = {"action": "edit", "target_module": "tools", "locations": ["tools/baseline.py"],
                "evidence_refs": ["e1"], "problem_hypothesis": "Observed parser mismatch", "intended_change": "Fix only observed behavior",
                "preserve_behaviors": ["Unrelated fields"], "verification_plan": ["Frozen check"], "verification_kind": "fixture"}
        sample = {"sample_id": f"sample-{index}", "state": state, "plan": plan,
                  "qualification": {"status": "verified", "factual_review": {"status": "PASS", "reason": "Independent exact evidence review",
                      "independent": True, "reviewer_id": "fixture-review", "evidence_refs": ["e1"]},
                      "behavior": {"status": "PASS", "covers_plan": True, "executable": True},
                      "task_result": {"status": "PASS", "reward": 0, "failure_layer": "model"}},
                  "source_kind": "fresh_teacher_plan", "protocol_id": protocol["protocol_id"],
                  "mechanism_id": "fixture", "root_lineage_id": state["root_lineage_id"], "revoked": False}
        ledger.put_record("samples", sample["sample_id"], sample)
        ledger.put_record("values", f"value-{index}", {"candidate_id": f"candidate-{index}", "condition_id": f"condition-{index}",
            "root_lineage_id": state["root_lineage_id"], "status": "COMPLETE", "is_complete": True,
            "source_refs": [sample["sample_id"]], "revoked": False})
        return {"status": "NEEDS_SCALE_LOCK", "calibration_complete": True}

    def test_no_formal_preparation_without_completed_calibration_or_explicit_counts(self):
        atomic_json(self.calibration / "status.json", {"calibration_complete": False})
        with self.assertRaisesRegex(ValueError, "completed calibration"):
            self.prepare()
        atomic_json(self.calibration / "status.json", {"calibration_complete": True})
        config = deepcopy(self.config)
        del config["targets"]["mechanisms"]
        with self.assertRaisesRegex(ValueError, "targets"):
            self.prepare(config)
        self.assertFalse(self.campaign.exists())

    def test_protocol_derives_immutably_and_cannot_overwrite_calibration_or_approval(self):
        before = (self.calibration / "protocol.json").read_bytes()
        protocol = self.prepare()
        self.assertEqual(self.prepare(), protocol)
        self.assertEqual((self.calibration / "protocol.json").read_bytes(), before)
        self.assertEqual(protocol["sampling"]["L_after"], 2)
        changed = deepcopy(self.config)
        changed["budget"].update(regular=21, total=26)
        with self.assertRaisesRegex(ValueError, "approval changed"):
            self.prepare(changed)
        with self.assertRaisesRegex(ValueError, "overwrite"):
            prepare_formal(self.calibration, self.calibration, self.config)

    def test_serial_groups_use_actual_aggregate_spend_fixed_seeds_and_validation_fold(self):
        self.prepare()
        result = run_formal(self.campaign, self.fake_group)
        self.assertEqual(result["status"], "NEEDS_STUDENT_PROFILE", result)
        self.assertEqual(result["counts"]["verified_edits"], 5)
        self.assertEqual(result["counts"]["validation_edits"], 1)
        self.assertEqual(result["aggregate_job_budget"]["used"], {"regular": 5, "retries": 1, "total": 6})
        self.assertEqual([row[2] for row in self.calls], ["train"] * 4 + ["validation"])
        self.assertEqual(len({row[1] for row in self.calls}), 5)
        self.assertEqual(self.calls[0][3]["regular"], 20)
        self.assertEqual(self.calls[-1][3]["regular"], 16)
        old = deepcopy(self.calls)
        again = run_formal(self.campaign, self.fake_group)
        self.assertEqual(self.calls, old)
        self.assertEqual(again["aggregate_job_budget"]["used"], result["aggregate_job_budget"]["used"])
        self.assertFalse(again["data_collection_complete"])

    def test_completion_requires_quota_and_tokenizer_template_validation(self):
        config = deepcopy(self.config)
        config["target_profile"] = {"model_id": "student", "max_context_tokens": 4}
        self.prepare(config)
        validator = lambda messages: {"valid": True, "input_ids": [1, 2, 3, 4], "labels": [-100, -100, 3, 4], "template_sha256": "validated-template"}
        with patch("research1.data_pipeline.sft._load_validator", return_value=(validator, None)):
            result = run_formal(self.campaign, self.fake_group)
        self.assertEqual(result["status"], "DATA_COLLECTION_COMPLETE", result)
        self.assertTrue(result["data_collection_complete"])
        self.assertFalse(any(result["deficits"].values()))

    def test_global_budget_halts_before_next_group_and_never_reports_complete(self):
        config = deepcopy(self.config)
        config["budget"] = {"regular": 1, "retries": 0, "total": 1}
        self.prepare(config)
        result = run_formal(self.campaign, self.fake_group)
        self.assertEqual(result["status"], "BUDGET_EXHAUSTED", result)
        self.assertEqual(len(self.calls), 1)
        self.assertEqual(result["aggregate_job_budget"]["used"]["total"], 1)
        self.assertFalse(result["data_collection_complete"])

    def test_infrastructure_interruption_reuses_same_child_not_a_new_favourable_seed(self):
        self.prepare()
        first = []
        def interrupted(child):
            first.append(str(child))
            return {"status": "INCOMPLETE", "reason": "No official result"}
        result = run_formal(self.campaign, interrupted)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(len(first), 1)
        result = run_formal(self.campaign, self.fake_group)
        self.assertEqual(self.calls[0][0], first[0])
        self.assertEqual(result["status"], "NEEDS_STUDENT_PROFILE")

    def test_root_revocation_survives_child_reimport_and_reduces_active_exports(self):
        self.prepare()
        run_formal(self.campaign, self.fake_group)
        root = Ledger(self.campaign)
        root.revoke("sample-0", "Factual review withdrawn")
        result = run_formal(self.campaign, self.fake_group)
        self.assertEqual(result["counts"]["verified_edits"], 4)
        self.assertTrue(dict(root.record_items("samples"))["sample-0"]["revoked"])
        self.assertTrue(dict(root.record_items("values"))["value-0"]["revoked"])
        self.assertEqual(len(self.calls), 5)

    def test_no_keep_or_positive_score_substitute_for_missing_edit_evidence(self):
        config = deepcopy(self.config)
        config["max_groups"] = 2
        self.prepare(config)
        def no_evidence(child):
            self.calls.append(str(child))
            return {"status": "INSUFFICIENT_EVIDENCE"}
        result = run_formal(self.campaign, no_evidence)
        self.assertEqual(result["status"], "EXHAUSTED_EVIDENCE")
        self.assertEqual(result["counts"]["verified_edits"], 0)
        self.assertEqual(len(self.calls), 2)
        self.assertFalse(result["data_collection_complete"])

    def test_foreign_child_protocol_cannot_be_imported_as_formal_quota_evidence(self):
        self.prepare()
        run_formal(self.campaign, self.fake_group)
        group = self.campaign / "groups/group_00001"
        spec = json.loads((group / "protocol.json").read_text())
        spec["formal_group_of"] = "different-approved-campaign"
        spec.pop("protocol_hash")
        spec["protocol_hash"] = digest(spec)
        atomic_json(group / "protocol.json", spec)
        old_calls = deepcopy(self.calls)
        result = run_formal(self.campaign, self.fake_group)
        self.assertEqual(result["status"], "INCOMPLETE")
        self.assertEqual(self.calls, old_calls)
        self.assertIn("Unregistered", result["reason"])
        self.assertIsNone(result["aggregate_job_budget"]["used"])


if __name__ == "__main__":
    unittest.main()
