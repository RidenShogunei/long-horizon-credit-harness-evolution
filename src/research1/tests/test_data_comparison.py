import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from research1.data_pipeline.policy import canonical, digest

_FILE = Path(__file__).resolve().parents[1] / "data_pipeline" / "comparison.py"
spec = importlib.util.spec_from_file_location("comparison_under_test", _FILE)
comparison = importlib.util.module_from_spec(spec)
spec.loader.exec_module(comparison)


def state():
    return {"state_id": "parent", "root_lineage_id": "root", "split": "train",
            "module_code": {"agent_loop/baseline.py": "def f():\n    return 1\n"},
            "permitted_modules": ["agent_loop"], "code_hash": "abc", "proposal_task_ids": ["task"],
            "evidence": [{"id": "obs", "text": "visible terminal output"}], "history": [],
            "verification_catalog": [], "plan_policy_version": "proposal_single_plan_v4",
            "teacher_provider": "tongji_agi", "teacher_model": "DeepSeek-V4.1-Flash", "max_candidates": 2,
            "allow_teacher_http_fallback": False}


def plan():
    return {"action": "keep", "target_module": None, "locations": [], "evidence_refs": ["obs"],
            "problem_hypothesis": "The bounded observation does not ground a change.",
            "intended_change": "Keep the source unchanged.", "preserve_behaviors": [],
            "verification_plan": ["Check the referenced output against its unchanged bytes."], "verification_kind": None}


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source, self.condition = self.root / "source.json", self.root / "condition.json"
        self.source.write_text(canonical(state()), encoding="utf-8")
        self.condition.write_text(canonical(state()), encoding="utf-8")
        self.directory = self.root / "comparison"

    def prepare(self, **changes):
        case = {"case_id": "fixed_0", "parent_id": "parent", "condition": "old_old",
                "source_state_path": str(self.source), "state_path": str(self.condition), "input_variant": "original"}
        case.update(changes)
        return comparison.prepare_comparison(self.directory, [case])

    def proposer(self, state_value, directory):
        prompt = comparison._collection_prompt(state_value)
        reply = {"candidates": [plan()]}
        comparison.atomic_json(directory / "teacher.json", {
            "prompt": prompt, "prompt_sha256": __import__("hashlib").sha256(prompt.encode()).hexdigest(),
            "raw": canonical(reply), "metadata": {"provider": "tongji_agi", "model": "DeepSeek-V4.1-Flash",
                "temperature": 0, "raw_field": "content", "finish_reason": "stop"}})
        return reply["candidates"]

    def test_fixed_prompts_and_idempotent_resume(self):
        manifest = self.prepare()
        self.assertEqual(manifest["fixed_order"], ["fixed_0"])
        self.assertEqual(manifest["evaluation_jobs"], 0)
        self.assertFalse(manifest["automatic_sft_admission"])
        self.assertIs(manifest["cases"][0]["allow_teacher_http_fallback"], False)
        frozen_state = comparison._read(self.directory / manifest["cases"][0]["frozen_state_path"])
        self.assertIs(frozen_state["allow_teacher_http_fallback"], False)
        calls = []
        def invoke(s, d):
            calls.append(1); return self.proposer(s, d)
        first = comparison.run_comparison(self.directory, proposer=invoke)
        second = comparison.run_comparison(self.directory, proposer=invoke)
        self.assertEqual(first, second)
        self.assertEqual(len(calls), 1)
        self.assertEqual(first["by_condition"]["old_old"]["keep"], 1)
        self.assertIsNone(first["qualified_keep"])

    def test_original_view_cannot_add_evidence(self):
        altered = state(); altered["evidence"].append({"id": "future", "text": "new data"})
        self.condition.write_text(canonical(altered), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Original-input"):
            self.prepare()

    def test_enriched_view_preserves_original_evidence(self):
        altered = state(); altered["evidence"][0]["text"] = "changed"
        self.condition.write_text(canonical(altered), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Enrichment changed"):
            self.prepare(input_variant="enriched")

    def test_source_mutation_blocks_remaining_calls(self):
        self.prepare()
        self.source.write_text(canonical({**state(), "hidden": "mutated"}), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "source bytes changed"):
            comparison.run_comparison(self.directory, proposer=lambda *_: self.fail("Must not call"))

    def test_started_without_raw_is_not_retried(self):
        self.prepare()
        row = comparison._read(self.directory / "manifest.json")["cases"][0]
        comparison.atomic_json(self.directory / "cases/fixed_0/started.json", {"prompt_sha256": row["prompt_sha256"]})
        result = comparison.run_comparison(self.directory, proposer=lambda *_: self.fail("Must not retry"))
        self.assertEqual(result["cases"][0]["status"], "AMBIGUOUS_CALL_NOT_RETRIED")
        self.assertEqual(result["by_condition"]["old_old"]["uncertain"], 1)

    def test_raw_saved_before_interrupt_is_parsed_without_new_call(self):
        self.prepare()
        row = comparison._read(self.directory / "manifest.json")["cases"][0]
        case_dir = self.directory / "cases/fixed_0"
        comparison.atomic_json(case_dir / "started.json", {"prompt_sha256": row["prompt_sha256"]})
        self.proposer(state(), case_dir)
        result = comparison.run_comparison(self.directory)
        self.assertEqual(result["cases"][0]["status"], "COMPLETE")

    def test_malformed_raw_retained_and_not_recalled(self):
        self.prepare()
        calls = []
        def malformed(s, d):
            calls.append(1); self.proposer(s, d)
            raw = comparison._read(d / "teacher.json"); raw["raw"] = "{"; comparison.atomic_json(d / "teacher.json", raw)
            raise ValueError("Malformed JSON")
        first = comparison.run_comparison(self.directory, proposer=malformed)
        second = comparison.run_comparison(self.directory, proposer=malformed)
        self.assertEqual(first, second)
        self.assertEqual(len(calls), 1)
        self.assertEqual(first["by_condition"]["old_old"]["parse_error"], 1)
        self.assertEqual(comparison._read(self.directory / "cases/fixed_0/teacher.json")["raw"], "{")

    def test_model_drift_is_error_not_success(self):
        self.prepare()
        def drift(s, d):
            result = self.proposer(s, d)
            raw = comparison._read(d / "teacher.json"); raw["metadata"]["model"] = "another"; comparison.atomic_json(d / "teacher.json", raw)
            return result
        result = comparison.run_comparison(self.directory, proposer=drift)
        self.assertEqual(result["cases"][0]["status"], "SIGNATURE_ERROR")

    def test_changed_completed_raw_invalidates_summary(self):
        self.prepare(); comparison.run_comparison(self.directory, proposer=self.proposer)
        path = self.directory / "cases/fixed_0/teacher.json"
        raw = comparison._read(path); raw["raw"] += " "; comparison.atomic_json(path, raw)
        with self.assertRaisesRegex(ValueError, "Raw teacher record changed"):
            comparison.comparison_status(self.directory)

    def test_existing_preregistration_cannot_be_replaced(self):
        self.prepare()
        with self.assertRaisesRegex(ValueError, "already prepared"):
            self.prepare()

    def test_http_fallback_must_be_explicitly_disabled(self):
        altered = state(); altered.pop("allow_teacher_http_fallback")
        self.condition.write_text(canonical(altered), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "explicitly disable"):
            self.prepare()

    def test_same_signature_required_across_conditions(self):
        altered = state(); altered["teacher_model"] = "different"
        second = self.root / "second.json"; second.write_text(canonical(altered), encoding="utf-8")
        cases = [{"case_id": "a", "parent_id": "parent", "condition": "a", "source_state_path": str(self.source),
                  "state_path": str(self.condition), "input_variant": "original"},
                 {"case_id": "b", "parent_id": "parent", "condition": "b", "source_state_path": str(self.source),
                  "state_path": str(second), "input_variant": "original"}]
        with self.assertRaisesRegex(ValueError, "identical"):
            comparison.prepare_comparison(self.directory, cases)

    def test_enriched_catalog_allowed_with_unchanged_evidence(self):
        from research1.data_pipeline.modules import planning_interfaces
        altered = state()
        altered["verification_catalog"] = [{"mechanism_id": "completion_requirement_recheck", "module": "agent_loop"}]
        altered["verification_interfaces"] = planning_interfaces(["agent_loop"])
        self.condition.write_text(canonical(altered), encoding="utf-8")
        manifest = self.prepare(input_variant="enriched")
        self.assertEqual(manifest["cases"][0]["input_variant"], "enriched")

    def test_invented_interface_claim_cannot_be_preregistered(self):
        from research1.data_pipeline.modules import planning_interfaces
        altered = state(); altered["verification_interfaces"] = planning_interfaces(["agent_loop"])
        altered["verification_interfaces"][0]["coverage"] = "Automatically proves all future tasks."
        self.condition.write_text(canonical(altered), encoding="utf-8")
        with self.assertRaises(ValueError):
            self.prepare(input_variant="enriched")

    def test_enriched_code_cannot_change(self):
        altered = state(); altered["module_code"]["agent_loop/baseline.py"] = "return future\n"
        self.condition.write_text(canonical(altered), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "Enrichment changed"):
            self.prepare(input_variant="enriched")

    def prepare_guarded(self):
        cases = []
        states = {}
        for parent in ("one", "two"):
            current = {**state(), "state_id": "state_" + parent, "root_lineage_id": "root_" + parent}
            path = self.root / (parent + ".json"); path.write_text(canonical(current), encoding="utf-8")
            states[parent] = current
            cases.append({"case_id": parent + "_D", "parent_id": parent, "condition": "D", "input_variant": "original",
                          "source_state_path": str(path), "state_path": str(path)})
        manifest = comparison.prepare_comparison(self.directory, cases,
            metadata={"required_preteacher_checks": "preteacher_checks.json"})
        checks = {"comparison_manifest_sha256": manifest["manifest_sha256"], "frozen_before_teacher_calls": True, "parents": {}}
        for parent, current in states.items():
            contract = {"selected": [], "evidence_hashes": {row["id"]: digest(row) for row in current["evidence"]}}
            contract["checks_id"] = "checks_" + digest(contract)[:24]
            path = self.directory / "preteacher_contracts" / (parent + ".json")
            comparison.atomic_json(path, {"state": {**current, "diagnostics": []}, "contract": contract})
            checks["parents"][parent] = {"path": str(path), "sha256": comparison._sha(path),
                                         "checks_id": contract["checks_id"], "selected_mechanisms": []}
        comparison.atomic_json(self.directory / "preteacher_checks.json", checks)
        return manifest, checks

    def test_required_preteacher_checks_must_exist_before_calls(self):
        self.prepare_guarded()
        (self.directory / "preteacher_checks.json").unlink()
        with self.assertRaises(FileNotFoundError):
            comparison.run_comparison(self.directory, proposer=lambda *_: self.fail("No teacher call allowed"))

    def test_valid_preteacher_artifacts_bind_before_first_call(self):
        self.prepare_guarded()
        def invoke(s, d):
            self.assertTrue((self.directory / "preteacher_binding.json").is_file())
            return self.proposer(s, d)
        result = comparison.run_comparison(self.directory, proposer=invoke)
        self.assertTrue(result["complete"])
        self.assertEqual(len(result["cases"]), 2)

    def test_preteacher_parent_set_must_be_exact(self):
        _, checks = self.prepare_guarded()
        checks["parents"].pop("two")
        comparison.atomic_json(self.directory / "preteacher_checks.json", checks)
        with self.assertRaisesRegex(ValueError, "exact preregistered parents"):
            comparison.run_comparison(self.directory, proposer=lambda *_: self.fail("No teacher call allowed"))

    def test_preteacher_contract_cannot_escape_comparison_directory(self):
        _, checks = self.prepare_guarded()
        outside = self.root / "outside.json"
        outside.write_bytes(Path(checks["parents"]["one"]["path"]).read_bytes())
        checks["parents"]["one"]["path"] = str(outside)
        comparison.atomic_json(self.directory / "preteacher_checks.json", checks)
        with self.assertRaisesRegex(ValueError, "inside the comparison"):
            comparison.run_comparison(self.directory, proposer=lambda *_: self.fail("No teacher call allowed"))

    def test_preteacher_contract_cannot_change_frozen_view(self):
        _, checks = self.prepare_guarded()
        path = Path(checks["parents"]["one"]["path"])
        artifact = comparison._read(path)
        artifact["state"]["evidence"][0]["text"] = "Different trajectory"
        comparison.atomic_json(path, artifact)
        checks["parents"]["one"]["sha256"] = comparison._sha(path)
        comparison.atomic_json(self.directory / "preteacher_checks.json", checks)
        with self.assertRaisesRegex(ValueError, "condition D inputs"):
            comparison.run_comparison(self.directory, proposer=lambda *_: self.fail("No teacher call allowed"))

    def test_preteacher_contract_content_binds_checks_id(self):
        _, checks = self.prepare_guarded()
        path = Path(checks["parents"]["one"]["path"])
        artifact = comparison._read(path); artifact["contract"]["selected"] = [{"mechanism_id": "new"}]
        comparison.atomic_json(path, artifact)
        checks["parents"]["one"]["sha256"] = comparison._sha(path)
        comparison.atomic_json(self.directory / "preteacher_checks.json", checks)
        with self.assertRaisesRegex(ValueError, "checks ID"):
            comparison.run_comparison(self.directory, proposer=lambda *_: self.fail("No teacher call allowed"))

    def test_poststart_preteacher_mutation_is_rejected(self):
        _, checks = self.prepare_guarded()
        calls = []
        def invoke(s, d):
            calls.append(1)
            result = self.proposer(s, d)
            checks["note"] = "Changed after a teacher call started"
            comparison.atomic_json(self.directory / "preteacher_checks.json", checks)
            return result
        with self.assertRaisesRegex(ValueError, "after their first execution binding"):
            comparison.run_comparison(self.directory, proposer=invoke)
        self.assertEqual(len(calls), 1)

    def test_started_case_cannot_acquire_checks_retroactively(self):
        manifest, _ = self.prepare_guarded()
        row = manifest["cases"][0]
        comparison.atomic_json(self.directory / "cases" / row["case_id"] / "started.json", {"prompt_sha256": row["prompt_sha256"]})
        with self.assertRaisesRegex(ValueError, "before preteacher checks were bound"):
            comparison.run_comparison(self.directory, proposer=lambda *_: self.fail("No teacher call allowed"))


if __name__ == "__main__":
    unittest.main()
