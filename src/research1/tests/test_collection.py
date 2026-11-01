"""Collector state/admission regression tests; no model, network, or benchmark."""
from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from research1.collection import (
    check_edit_scope, collect_parent, prepare_parent, rebuild_verified_plans, freeze_parent,
    sample_id_for_row, set_admission,
)


class CollectionTests(unittest.TestCase):
    def test_default_teacher_uses_final_json_transport(self):
        from research1.collection import _default_teacher
        answer = ('{"candidates": []}', {'raw_field':'content','finish_reason':'stop','temperature':0})
        with patch('research1.distill._tongji_chat',return_value=answer) as chat:
            self.assertEqual(_default_teacher('diagnose'),answer)
        self.assertEqual(chat.call_args.kwargs['json_output'],True)
        self.assertEqual(chat.call_args.kwargs['reasoning_effort'],'none')

    def test_default_teacher_preserves_but_rejects_reasoning_and_truncated_outputs(self):
        for field,finish in [('reasoning','stop'),('content','length')]:
            with self.subTest(field=field,finish=finish):
                metadata={'raw_field':field,'finish_reason':finish,'temperature':0}
                with patch('research1.collection._default_teacher',return_value=(json.dumps({'candidates':[self.plan]}),metadata)):
                    with self.assertRaisesRegex(ValueError,'complete final JSON'):
                        self.collect(teacher=None,public_config={'case':field})
        self.assertEqual(self.calls['editor'],0)
        self.assertFalse((self.root/'datasets/research1/raw/decisions.jsonl').exists())

    def test_revised_checks_freeze_separately_without_multiplying_origin(self):
        common = dict(root=self.root, modules_root=self.modules, task_name='task',
                      evidence=[{'id':'e1','text':'same observation'}])
        first = freeze_parent(**common, checks={'id':'contract-v1'})
        second = freeze_parent(**common, checks={'id':'contract-v2'})
        self.assertEqual(first['origin_id'],second['origin_id'])
        self.assertNotEqual(first['state_id'],second['state_id'])
        self.assertEqual(freeze_parent(**common, checks={'id':'contract-v1'}),first)
        self.assertEqual(json.loads((Path(first['modules_root']).parent/'state.json').read_text())['checks'],{'id':'contract-v1'})

    def test_new_view_does_not_multiply_one_origin_in_training_export(self):
        for sample, state in (('s1', 'view1'), ('s2', 'view2')):
            set_admission(self.root, sample, 'verified', 'reviewed',
                          sft_row={'sample_id': sample, 'origin_id': 'same-origin', 'state_id': state, 'prompt': state, 'completion': 'keep'})
        rows = [json.loads(s) for s in (self.root / 'datasets/research1/sft/verified_plans.jsonl').read_text().splitlines()]
        self.assertEqual([row['sample_id'] for row in rows], ['s2'])
        self.assertEqual(len((self.root / 'datasets/research1/admission_registry.jsonl').read_text().splitlines()), 2)

    def test_revised_excerpt_shares_origin_but_has_distinct_state(self):
        source = self.root / 'trajectory.json'
        source.write_text('{"steps": []}', encoding='utf-8')
        common = dict(root=self.root, modules_root=self.modules, task_name='task',
                      evidence=[{'id': 'e1', 'text': 'first excerpt', 'path': str(source)}])
        first = prepare_parent(**common)
        common['evidence'][0]['text'] = 'revised excerpt with middle warning'
        second = prepare_parent(**common, public_config={'evidence_selector_id': 'signals-v2'})
        self.assertEqual(first['origin_id'], second['origin_id'])
        self.assertNotEqual(first['state_id'], second['state_id'])

    def test_default_real_configuration_rejects_pool_task_before_any_call(self):
        dataset = self.root / 'dataset'
        dataset.mkdir()
        (dataset / 'manifest.json').write_text(json.dumps({'tasks': [{'name': 'allowed', 'split': 'train'}, {'name': 'outside', 'split': 'pool'}]}))
        config = self.root / 'pilot.yaml'
        config.write_text('dataset_tb: ' + dataset.as_posix() + '\n')
        with self.assertRaisesRegex(ValueError, 'manifest train'):
            prepare_parent(root=self.root, modules_root=self.modules, task_name='outside',
                           evidence=[{'id': 'e1', 'text': 'public but outside train'}],
                           public_config={'config_path': str(config)})

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.modules = self.root / "source"
        (self.modules / "tools").mkdir(parents=True)
        (self.modules / "tools/baseline.py").write_text("def parse():\n    return 1\n\ndef unrelated():\n    return 3\n")
        self.calls = {"teacher": 0, "editor": 0, "task": 0}
        self.bundle = {"id": "parser_contract_v1", "kind": "parser_contract", "mechanism_id": "parser_contract",
                       "checks": [{"id": "configured"}], "allowed_targets": ["tools/baseline.py"],
                       "coverage_statement": "The parse return-value contract and preservation of unrelated behavior.",
                       "allowed_changed_functions": ["parse"]}
        self.plan = {"action": "edit", "target_module": "tools", "locations": ["tools/baseline.py"],
                     "evidence_refs": ["failure1"], "problem_hypothesis": "Parser output violates observed contract",
                     "intended_change": "Return the configured parsed value", "preserve_behaviors": ["unrelated parser behavior"],
                     "verification_plan": ["configured contract"], "verification_kind": "parser_contract"}

    def teacher(self, prompt):
        self.calls["teacher"] += 1
        self.assertNotIn("A correct edit must stop the false warning", prompt)
        return {"candidates": [self.plan]}, {"model": "test-fixed", "temperature": 0}

    def editor(self, plan, state, dest):
        self.calls["editor"] += 1
        target = dest / "child"
        shutil.copytree(state["modules_root"], target)
        path = target / "tools/baseline.py"
        path.write_text(path.read_text().replace("return 1", "return 2"))
        return {"status": "applied", "modules_root": str(target), "child_version_id": "test-child"}

    def verifier(self, plan, state, attempt):
        scope = check_edit_scope(plan, state, attempt)
        self.assertEqual(scope["status"], "PASS")
        return {"status": "PASS", "covers_plan": True, "checks_id": state["checks_id"]}

    def evaluator(self, plan, state, attempt, dest):
        self.calls["task"] += 1
        return {"status": "PASS", "reward": 0.0, "failure_layer": "model"}

    def collect(self, **extra):
        args = {"root": self.root, "modules_root": self.modules, "task_name": "manifest-train-task",
                "evidence": [{"id": "failure1", "text": "Observed parser contract mismatch"}],
                "checks": self.bundle, "run_task": True, "teacher": self.teacher,
                "editor": self.editor, "verifier": self.verifier, "evaluator": self.evaluator}
        args.update(extra)
        return collect_parent(**args)

    def test_end_to_end_valid_zero_is_admitted_and_inputs_frozen(self):
        result = self.collect()
        self.assertEqual(result["verified"], 1)
        self.assertEqual(self.calls, {"teacher": 1, "editor": 1, "task": 1})
        state = json.loads((self.root / "datasets/research1/states" / result["state_id"] / "state.json").read_text())
        self.assertEqual(state["view"]["module_code"]["tools/baseline.py"], (self.modules / "tools/baseline.py").read_text())
        self.assertIsNone(result["long_horizon_label"])
        rows = (self.root / "datasets/research1/sft/verified_plans.jsonl").read_text().splitlines()
        self.assertEqual(len(rows), 1)
        self.assertEqual(json.loads(json.loads(rows[0])["completion"])["candidates"], [self.plan])

    def test_keep_with_valid_referenced_parent_feedback_is_admitted(self):
        self.plan = dict(self.plan, action="keep", locations=[], target_module=None,
                         intended_change="Keep because observed failure is outside the locked module")
        result = self.collect(evidence=[{"id": "failure1", "text": "Task finished with no module defect",
                                          "task_result": {"status": "PASS", "reward": 0.0, "failure_layer": "model"}}],
                              keep_reviewer=lambda plan, state: {"status": "PASS", "reason": "Independent evidence consistency review passed",
                                                                  "evidence_refs": ["failure1"], "reviewer_id": "trusted-test-review"})
        self.assertEqual(result["verified"], 1)
        self.assertEqual(self.calls["editor"], 0)
        self.assertEqual(self.calls["task"], 0)
        review_path = Path(result["candidates"][0]["keep_review_path"])
        self.assertEqual(review_path.name, "review.json")
        self.assertEqual(json.loads(review_path.read_text())["review"]["status"], "PASS")
        row = json.loads((self.root / "datasets/research1/sft/verified_plans.jsonl").read_text())
        self.assertNotIn("trusted-test-review", row["prompt"])
        self.assertNotIn("trusted-test-review", row["completion"])
        self.assertEqual(json.loads(row["completion"])["candidates"], [self.plan])

    def test_keep_with_valid_parent_defaults_to_pending_review(self):
        self.plan = dict(self.plan, action="keep", locations=[], target_module=None)
        result = self.collect(evidence=[{"id": "failure1", "text": "Known warning needs factual review",
                                         "task_result": {"status": "PASS", "reward": 1.0, "failure_layer": "model"}}])
        self.assertEqual(result["verified"], 0)
        self.assertEqual(result["pending_review"], 1)
        self.assertEqual(result["candidates"][0]["status"], "pending_review")
        self.assertEqual(rebuild_verified_plans(self.root)["verified_count"], 0)
        self.assertTrue((self.root / "datasets/research1/raw/decisions.jsonl").is_file())

    def test_keep_review_cannot_replace_missing_parent_or_reference_hidden_evidence(self):
        self.plan = dict(self.plan, action="keep", locations=[], target_module=None)
        valid_review = lambda plan, state: {"status": "PASS", "reason": "reviewed", "evidence_refs": ["failure1"]}
        result = self.collect(keep_reviewer=valid_review)
        self.assertEqual(result["verified"], 0)
        # A distinct parent input makes a separate immutable review, not a re-edit.
        result = self.collect(evidence=[{"id": "failure1", "text": "valid parent",
                                         "task_result": {"status": "PASS", "reward": 0.0, "failure_layer": "model"}}],
                              keep_reviewer=lambda p, s: {"status": "PASS", "reason": "reviewed", "evidence_refs": ["hidden_future"]})
        self.assertEqual(result["verified"], 0)
        self.assertEqual(result["pending_review"], 1)

    def test_keep_without_valid_parent_feedback_is_pending(self):
        self.plan = dict(self.plan, action="keep", locations=[], target_module=None)
        self.assertEqual(self.collect()["pending"], 1)

    def test_harmful_candidate_rejected_before_task(self):
        result = self.collect(verifier=lambda p, s, a: {"status": "FAIL", "covers_plan": True, "checks_id": s["checks_id"]})
        self.assertEqual(result["candidates"][0]["status"], "rejected")
        self.assertEqual(self.calls["task"], 0)
        self.assertEqual(rebuild_verified_plans(self.root)["verified_count"], 0)

    def test_missing_checks_stays_pending_without_edit(self):
        self.assertEqual(self.collect(checks=None)["pending"], 1)
        self.assertEqual(self.calls["editor"], 0)

    def test_infrastructure_and_missing_contract_cannot_promote(self):
        result = self.collect(evaluator=lambda *args: {"status": "INCOMPLETE", "reward": None, "failure_layer": "scorer"})
        self.assertEqual(result["pending"], 1)
        self.assertEqual(rebuild_verified_plans(self.root)["verified_count"], 0)

    def test_rerun_does_not_call_models_or_duplicate_exports(self):
        first = self.collect()
        second = self.collect()
        self.assertEqual(first, second)
        self.assertEqual(self.calls, {"teacher": 1, "editor": 1, "task": 1})
        for name in ("decisions", "attempts", "task_runs"):
            self.assertEqual(len((self.root / f"datasets/research1/raw/{name}.jsonl").read_text().splitlines()), 1)
        self.assertEqual(len((self.root / "datasets/research1/admission_registry.jsonl").read_text().splitlines()), 1)

    def test_revocation_preserves_raw_and_is_idempotent(self):
        result = self.collect()
        sample_id = result["candidates"][0]["sample_id"]
        event = set_admission(self.root, sample_id, "revoked", "mixed warning preservation failed")
        set_admission(self.root, sample_id, "revoked", "mixed warning preservation failed")
        self.assertEqual(rebuild_verified_plans(self.root)["verified_count"], 0)
        self.assertTrue((self.root / "datasets/research1/raw/decisions.jsonl").is_file())
        self.assertEqual(sample_id_for_row(event["sft_row"]), sample_id)
        self.assertEqual(len((self.root / "datasets/research1/admission_registry.jsonl").read_text().splitlines()), 2)

    def test_future_information_and_private_fields_rejected(self):
        for key in ("child_reward", "api_key"):
            with self.assertRaises(ValueError):
                self.collect(public_config={key: "forbidden"})
        self.assertEqual(self.calls["teacher"], 0)

    def test_scope_requires_declaration_and_rejects_unrelated_changes(self):
        state = prepare_parent(root=self.root, modules_root=self.modules, task_name="train",
                               evidence=[{"id": "failure1", "text": "observed"}], checks=self.bundle)
        dest = self.root / "attempt"
        dest.mkdir()
        attempt = self.editor(self.plan, state, dest)
        self.assertEqual(check_edit_scope(self.plan, state, attempt)["status"], "PASS")
        child = Path(attempt["modules_root"]) / "tools/baseline.py"
        child.write_text(child.read_text().replace("return 3", "return 4"))
        self.assertEqual(check_edit_scope(self.plan, state, attempt)["status"], "FAIL")
        state["checks"] = {"kind": "parser_contract"}
        self.assertEqual(check_edit_scope(self.plan, state, attempt)["status"], "pending")

    def test_evidence_source_is_hashed_and_recoverable(self):
        original = self.root / "trace.json"
        original.write_text('{"visible":"original evidence"}')
        state = prepare_parent(root=self.root, modules_root=self.modules, task_name="train",
                               evidence=[{"id": "failure1", "text": "bounded", "path": str(original)}])
        digest = state["view"]["evidence"][0]["source_sha256"]
        copied = Path(state["modules_root"]).parent / "evidence" / digest
        self.assertEqual(copied.read_bytes(), original.read_bytes())


if __name__ == "__main__":
    unittest.main()
