"""Policy/SFT boundary, admission and immutable export regression checks."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research1.data_pipeline.policy import collect_candidates, parse_candidates, render_messages, review_plan
from research1.data_pipeline.sft import export_dataset


def record(index=0, action="edit", split="train"):
    state = {"state_id": f"state-{index}", "root_lineage_id": f"root-{index}", "split": split,
             "proposal_task_ids": [split + "-task"], "permitted_modules": ["tools", "verification"],
             "module_code": {"tools/baseline.py": "def parse():\n    return 1\n",
                             "verification/baseline.py": "def done():\n    return False\n"},
             "evidence": [{"id": "evidence-1", "text": "Proposal-visible parser feedback", "partition": "proposal"}],
             "history": []}
    plan = {"action": action, "target_module": "tools" if action == "edit" else None,
            "locations": ["tools/baseline.py"] if action == "edit" else [], "evidence_refs": ["evidence-1"],
            "problem_hypothesis": "Observed behavior supports this bounded decision",
            "intended_change": "Apply the justified behavior change" if action == "edit" else "Keep current behavior",
            "preserve_behaviors": ["Preserve unrelated commands"], "verification_plan": ["Frozen parser contract"],
            "verification_kind": "parser-contract" if action == "edit" else None}
    return {"sample_id": f"sample-{index}", "state": state, "plan": plan,
            "qualification": {"status": "verified", "factual_review": {
                "status": "PASS", "reason": "The selected evidence supports the claim", "independent": True,
                "reviewer_id": "independent-review", "evidence_refs": ["evidence-1"]},
                "behavior": {"status": "PASS", "covers_plan": True, "executable": True},
                "task_result": {"status": "PASS", "reward": 0.0, "failure_layer": "model"}},
            "source_kind": "fresh_teacher_plan", "protocol_id": "research1-data-v1",
            "mechanism_id": "parser-contract" if action == "edit" else None}


def rows(manifest, name):
    return [json.loads(line) for line in Path(manifest["files"][name]).read_text(encoding="utf-8").splitlines()]


class PolicyContractTests(unittest.TestCase):
    def test_verification_catalog_contains_only_predeclared_interfaces(self):
        state = record()["state"]
        state["verification_catalog"] = [{"module": "tools", "mechanism_id": "parser-contract"}]
        visible = json.loads(render_messages(state)[1]["content"])
        self.assertEqual(visible["verification_catalog"], state["verification_catalog"])
        state["verification_catalog"][0]["fix_recipe"] = "The desired repair is ..."
        with self.assertRaisesRegex(ValueError, "catalog"):
            render_messages(state)

    def test_model_messages_exclude_identity_outcomes_and_scores_and_emit_one_plan(self):
        sample = record()
        state = dict(sample["state"], future_results={"reward": 1}, private_metadata={"api_key": "NOT_INPUT"})
        messages = render_messages(state, sample["plan"])
        text = json.dumps(messages)
        self.assertNotIn("NOT_INPUT", text)
        self.assertNotIn("future_results", text)
        self.assertNotIn("state-0", text)
        self.assertEqual(json.loads(messages[-1]["content"]), sample["plan"])
        self.assertNotIn("candidates", json.loads(messages[-1]["content"]))
        self.assertIn("ONE JSON Plan", messages[0]["content"])

    def test_nested_history_leakage_and_nonproposal_evidence_are_rejected(self):
        for field in ("reward", "scores", "child_diff", "hidden_tests", "long_horizon_label"):
            with self.subTest(field=field):
                state = record()["state"]
                state["history"] = [{"verification_feedback": {field: 1}}]
                with self.assertRaisesRegex(ValueError, "Non-policy"):
                    render_messages(state)
        state = record()["state"]
        state["evidence"][0]["partition"] = "dev"
        with self.assertRaisesRegex(ValueError, "proposal"):
            render_messages(state)
        state = record()["state"]
        state["module_code"]["task/tests.py"] = "assert False"
        with self.assertRaises(ValueError):
            render_messages(state)

    def test_cross_module_edits_unknown_refs_and_extra_labels_cannot_parse(self):
        sample = record()
        raw = {"candidates": [sample["plan"], deepcopy(sample["plan"])]}
        self.assertEqual(len(parse_candidates(raw, sample["state"])), 1)
        for changes in ({"locations": ["verification/baseline.py"]}, {"evidence_refs": ["unknown"]}, {"predicted_score": 1}):
            with self.subTest(changes=changes):
                plan = dict(sample["plan"], **changes)
                with self.assertRaises(ValueError):
                    parse_candidates({"candidates": [plan]}, sample["state"])

    def test_collection_saves_exact_raw_output_and_reuses_it_without_a_model_call(self):
        sample = record()
        raw = json.dumps({"candidates": [sample["plan"]]}, indent=3)
        meta = {"raw_field": "content", "finish_reason": "stop", "temperature": 0, "model": "fixed"}
        with tempfile.TemporaryDirectory() as td, patch("research1.distill._tongji_chat", return_value=(raw, meta)) as chat:
            directory = Path(td)
            first = collect_candidates(sample["state"], directory)
            saved = (directory / "teacher.json").read_bytes()
            self.assertEqual(collect_candidates(sample["state"], directory), first)
            self.assertEqual((directory / "teacher.json").read_bytes(), saved)
            self.assertEqual(json.loads(saved)["raw"], raw)
            chat.assert_called_once()
            self.assertTrue(chat.call_args.kwargs["json_output"])
            self.assertEqual(chat.call_args.kwargs["reasoning_effort"], "none")
            changed = deepcopy(sample["state"])
            changed["evidence"][0]["text"] += " extra observation"
            with self.assertRaisesRegex(ValueError, "immutable"):
                collect_candidates(changed, directory)

    def test_reasoning_and_truncated_replies_remain_evidence_not_labels(self):
        sample = record()
        raw = json.dumps({"candidates": [sample["plan"]]})
        for field, finish in (("reasoning", "stop"), ("content", "length")):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as td:
                meta = {"raw_field": field, "finish_reason": finish, "temperature": 0}
                with patch("research1.distill._tongji_chat", return_value=(raw, meta)) as chat:
                    for _ in range(2):
                        with self.assertRaisesRegex(ValueError, "complete final"):
                            collect_candidates(sample["state"], Path(td))
                    chat.assert_called_once()
                self.assertEqual(json.loads((Path(td) / "teacher.json").read_text())["raw"], raw)

    def test_first_teacher_provider_mismatch_is_preserved_and_not_admitted(self):
        sample = record()
        sample["state"].update(teacher_provider="frozen-provider", teacher_model="fixed")
        raw = json.dumps({"candidates": [sample["plan"]]})
        meta = {"raw_field": "content", "finish_reason": "stop", "temperature": 0, "model": "fixed", "provider": "other"}
        with tempfile.TemporaryDirectory() as td, patch("research1.distill._tongji_chat", return_value=(raw, meta)):
            with self.assertRaisesRegex(ValueError, "Teacher provider"):
                collect_candidates(sample["state"], Path(td))
            self.assertEqual(json.loads((Path(td) / "teacher.json").read_text())["raw"], raw)


class FactualReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.sample = record(action="keep")
        self.sample["state"]["evidence"][0]["text"] = "Ran actual proposal commands. FINAL CHECK PASSED: GET 200, POST 201."
        self.metadata = {"raw_field": "content", "finish_reason": "stop", "temperature": 0, "model": "frozen-review"}
        self.result = {"status": "PASS", "reason": "The bounded abstention is supported by endpoint results",
                       "evidence_refs": ["evidence-1"], "witnesses": [{"evidence_ref": "evidence-1",
                       "text": "FINAL CHECK PASSED: GET 200, POST 201.", "kind": "positive_execution"}]}

    def review(self, value=None, metadata=None):
        raw = json.dumps(self.result if value is None else value)
        with patch("research1.distill._tongji_chat", return_value=(raw, metadata or self.metadata)) as chat:
            result = review_plan(self.sample["state"], self.sample["plan"], self.directory)
        return result, chat

    def test_factual_review_uses_only_view_and_candidate_preserves_raw_and_is_cached(self):
        self.sample["state"]["later_result"] = {"reward": 1234, "secret": "FORBIDDEN_OUTCOME"}
        result, chat = self.review()
        self.assertEqual(result["status"], "PASS")
        self.assertTrue(result["independent"])
        self.assertEqual(result["review_type"], "model_fact_review")
        saved = (self.directory / "factual_review.json").read_bytes()
        self.assertNotIn("FORBIDDEN_OUTCOME", json.loads(saved)["prompt"])
        with patch("research1.distill._tongji_chat") as another_call:
            repeated = review_plan(self.sample["state"], self.sample["plan"], self.directory)
        another_call.assert_not_called()
        self.assertEqual(repeated, result)
        self.assertEqual((self.directory / "factual_review.json").read_bytes(), saved)
        self.assertEqual(chat.call_args.kwargs["role"], "research1_data_factual_review")

    def test_verdict_only_unknown_refs_and_invented_witness_do_not_pass(self):
        cases = [dict(self.result, witnesses=[]), dict(self.result, evidence_refs=["hidden"]),
                 dict(self.result, witnesses=[{"evidence_ref": "evidence-1", "text": "INVENTED FINAL CHECK PASSED", "kind": "positive_execution"}])]
        for i, value in enumerate(cases):
            with self.subTest(case=i):
                self.directory = Path(self.temp.name) / str(i)
                result, _ = self.review(value)
                self.assertEqual(result["status"], "pending_review")

    def test_reasoning_truncated_and_malformed_reviews_remain_pending(self):
        for i, changed in enumerate(({"raw_field": "reasoning"}, {"finish_reason": "length"})):
            with self.subTest(changed=changed):
                self.directory = Path(self.temp.name) / str(i)
                result, _ = self.review(metadata=dict(self.metadata, **changed))
                self.assertEqual(result["status"], "pending_review")
        self.directory = Path(self.temp.name) / "malformed"
        with patch("research1.distill._tongji_chat", return_value=("{broken", self.metadata)):
            result = review_plan(self.sample["state"], self.sample["plan"], self.directory)
        self.assertEqual(result["status"], "pending_review")

    def test_global_absence_and_score_only_witness_cannot_be_passes(self):
        self.sample["plan"]["problem_hypothesis"] = "No defects exist anywhere"
        result, _ = self.review()
        self.assertEqual(result["status"], "pending_review")
        self.directory = Path(self.temp.name) / "score-only"
        self.sample["plan"] = record(action="keep")["plan"]
        self.sample["state"]["evidence"][0]["text"] = 'The observed task result was reward=1.0'
        self.result["witnesses"][0]["text"] = self.sample["state"]["evidence"][0]["text"]
        result, _ = self.review()
        self.assertEqual(result["status"], "pending_review")

    def test_echoed_success_command_is_not_positive_execution_observation(self):
        self.sample["state"]["evidence"][0]["text"] = json.dumps({"steps": [{
            "message_excerpt": {"segments": [{"text": "FINAL CHECK PASSED: GET 200, POST 201."}]},
            "observation_excerpt": {"segments": [{"text": "Still waiting for command output"}]}}]})
        result, _ = self.review()
        self.assertEqual(result["status"], "pending_review")

    def test_first_review_provider_mismatch_retains_raw_without_pass(self):
        self.sample["state"]["review_provider"] = "frozen-provider"
        result, _ = self.review(metadata=dict(self.metadata, provider="other"))
        self.assertEqual(result["status"], "pending_review")
        self.assertIn("Review provider", result["reason"])
        self.assertEqual(json.loads((self.directory / "factual_review.json").read_text())["metadata"]["provider"], "other")


class SFTExportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.campaign = Path(self.temp.name)

    def test_valid_zero_is_eligible_but_no_selected_tokenizer_is_not_training_ready(self):
        report = export_dataset(self.campaign, [record()])
        self.assertEqual(report["counts"]["train"]["edit"], 1)
        self.assertFalse(report["training_ready"])
        self.assertIn("student_model_and_template_not_selected", report["readiness_reasons"])
        example = rows(report, "edit.train.jsonl")[0]
        self.assertEqual(set(example), {"messages"})
        self.assertNotIn("reward", json.dumps(example))
        self.assertEqual(rows(report, "metadata.jsonl")[0]["qualification"]["task_result"]["reward"], 0)

    def test_keep_requires_independent_fact_review_and_cannot_fill_a_noedit_training_set(self):
        good, bad = record(1, "keep"), record(2, "keep")
        bad["qualification"]["factual_review"]["independent"] = False
        report = export_dataset(self.campaign, [good, bad])
        self.assertEqual(report["counts"]["train"]["keep"], 1)
        self.assertEqual(report["counts"]["train"]["mixed"], 0)
        self.assertEqual(rows(report, "excluded.jsonl")[0]["reason"], "independent_factual_review_required")

    def test_keep_cap_uses_counts_and_sidecar_order_matches_messages(self):
        samples = [record(i, "keep" if i < 8 else "edit") for i in range(12)]
        report = export_dataset(self.campaign, samples)
        self.assertEqual(report["counts"]["train"], {"edit": 4, "keep": 8, "mixed": 5, "mixed_keep": 1})
        examples = rows(report, "mixed.train.jsonl")
        sidecar = rows(report, "mixed.train.metadata.jsonl")
        self.assertEqual([json.loads(e["messages"][-1]["content"])["action"] for e in examples], [s["action"] for s in sidecar])

    def test_textual_verdict_infra_failure_pending_and_legacy_are_not_sft(self):
        samples = [record(i) for i in range(5)]
        samples[0]["qualification"]["behavior"]["executable"] = False
        samples[1]["qualification"]["task_result"].update(reward=None, failure_layer="infrastructure")
        samples[2]["qualification"]["status"] = "pending"
        samples[3]["source_kind"] = "archived_teacher_plan_reimplementation"
        samples[4]["qualification"]["factual_review"]["evidence_refs"] = ["hidden-result"]
        report = export_dataset(self.campaign, samples)
        self.assertEqual(report["qualified_records"], 0)
        self.assertEqual(report["excluded_records"], 5)

    def test_edit_can_use_hidden_child_reward_as_metadata_but_requires_matching_implementation(self):
        sample = record()
        qualification = sample["qualification"]
        qualification["proposal_task_result"] = qualification.pop("task_result")
        qualification["implementation_task_result"] = {
            "status": "PASS", "reward": 0, "failure_layer": "model",
            "role": "reward", "evaluation_spec": {"code_hash": "actual-child"}}
        sample["implementation"] = {"code_hash": "actual-child"}
        report = export_dataset(self.campaign, [sample])
        self.assertEqual(report["qualified_records"], 1)
        self.assertNotIn("actual-child", json.dumps(rows(report, "edit.train.jsonl")))
        qualification["implementation_task_result"]["evaluation_spec"]["code_hash"] = "different-child"
        report = export_dataset(self.campaign, [sample])
        self.assertEqual(report["qualified_records"], 0)
        self.assertEqual(rows(report, "excluded.jsonl")[0]["reason"], "implementation_execution_binding_mismatch")

    def test_sibling_lineages_and_source_tasks_cannot_leak_across_folds(self):
        first, second = record(), record(1, split="validation")
        second["state"]["root_lineage_id"] = first["state"]["root_lineage_id"]
        with self.assertRaisesRegex(ValueError, "lineage"):
            export_dataset(self.campaign, [first, second])
        second["state"]["root_lineage_id"] = "other-root"
        second["state"]["proposal_task_ids"] = first["state"]["proposal_task_ids"]
        with self.assertRaisesRegex(ValueError, "task leaked"):
            export_dataset(self.campaign, [first, second])
        self.assertFalse((self.campaign / "exports/sft/CURRENT.json").exists())

    def test_idempotent_snapshots_and_revocation_replace_active_view_preserving_history(self):
        sample = record()
        first = export_dataset(self.campaign, [sample])
        self.assertEqual(export_dataset(self.campaign, [sample]), first)
        old_file = Path(first["files"]["edit.train.jsonl"])
        old_bytes = old_file.read_bytes()
        revoked = dict(sample, revoked=True)
        second = export_dataset(self.campaign, [sample, revoked])
        self.assertNotEqual(second["dataset_revision"], first["dataset_revision"])
        self.assertEqual(second["qualified_records"], 0)
        self.assertEqual(old_file.read_bytes(), old_bytes)
        pointer = json.loads((self.campaign / "exports/sft/CURRENT.json").read_text())
        self.assertEqual(pointer["dataset_revision"], second["dataset_revision"])

    def test_tokenizer_precheck_requires_real_token_ids_assistant_mask_and_context_fit(self):
        samples = [record(), record(1, split="validation")]
        def tokenizer(messages):
            self.assertEqual(messages[-1]["role"], "assistant")
            return {"valid": True, "input_ids": [1, 2, 3, 4], "labels": [-100, -100, 3, 4], "template_sha256": "checked-template"}
        profile = {"model_id": "local-student", "max_context_tokens": 4, "validator": tokenizer}
        report = export_dataset(self.campaign, samples, profile)
        self.assertTrue(report["training_ready"])
        self.assertEqual(rows(report, "mixed.train.tokenized.jsonl")[0]["labels"], [-100, -100, 3, 4])
        too_short = export_dataset(self.campaign, samples, dict(profile, max_context_tokens=3))
        self.assertFalse(too_short["training_ready"])
        self.assertEqual(too_short["qualified_records"], 0)
        self.assertEqual({r["reason"] for r in rows(too_short, "excluded.jsonl")}, {"student_context_overflow"})
        bogus = dict(profile, validator=lambda messages: {"valid": True, "token_count": 1})
        self.assertEqual(export_dataset(self.campaign, samples, bogus)["qualified_records"], 0)

    def test_dedup_uses_origin_and_plan_and_revision_conflicts_do_not_publish(self):
        first, duplicate = record(), record(1)
        duplicate["state"] = deepcopy(first["state"])
        report = export_dataset(self.campaign, [first, duplicate])
        self.assertEqual(report["qualified_records"], 1)
        self.assertEqual(report["duplicate_records"], 1)
        versioned = dict(first, revision=2)
        conflicting = dict(first, revision=2, revoked=True)
        with self.assertRaisesRegex(ValueError, "Conflicting"):
            export_dataset(self.campaign, [versioned, conflicting])


if __name__ == "__main__":
    unittest.main()
