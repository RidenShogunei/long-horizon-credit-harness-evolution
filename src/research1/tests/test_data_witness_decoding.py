"""Exact decoded selector witnesses, without promoting unproven JSON success."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research1.data_pipeline.policy import FACTUAL_WITNESS_VALIDATOR, _review_witnesses, review_plan
from research1.tests.test_data_sft import record


class DecodedWitnessTests(unittest.TestCase):
    def state(self, observation, *, message="", command="python -m pytest\n"):
        state = record(action="keep")["state"]
        step = {"step_id": 4, "source_index": 3, "source": "agent", "visible_step_sha256": "a" * 64,
                "observation_excerpt": {"segments": [{"start": 100, "end": 100 + len(observation), "text": observation}]},
                "message_excerpt": {"segments": [{"text": message}]},
                "tool_calls": [{"tool_call_id": "call_4_1", "position": 1, "function_name": "bash_command", "arguments": {
                    "keystrokes": command, "keystrokes_excerpt": {"segments": [{"text": command}]}}}]}
        state["evidence"][0]["text"] = json.dumps({"steps": [step]}, ensure_ascii=False)
        return state

    def witness(self, text, kind="context"):
        return {"evidence_ref": "evidence-1", "text": text, "kind": kind}

    def test_decoded_quotes_preserve_serialized_hash_and_exact_source_offsets(self):
        quoted = '{"pre_order": [32, 4, 2, 1, 3, 11.33, 7]}'
        state = self.state("prefix\n" + quoted + "\nsuffix")
        original = state["evidence"][0]["text"]
        self.assertNotIn(quoted, original)
        result = _review_witnesses([self.witness(quoted)], state, "edit")[0]
        self.assertEqual(result["text"], quoted)
        validation = result["validation"]
        self.assertEqual(validation["validator"], FACTUAL_WITNESS_VALIDATOR)
        self.assertEqual(validation["serialized_evidence_sha256"], hashlib.sha256(original.encode()).hexdigest())
        location = validation["locations"][0]
        self.assertEqual(location["field"], "observation_excerpt")
        self.assertEqual(location["source_start"], 107)
        self.assertEqual(location["source_end"], 107 + len(quoted))
        self.assertEqual(state["evidence"][0]["text"], original)

    def test_newlines_and_backslashes_are_exact_not_manually_unescaped(self):
        quoted = 'actual line one\nactual line two at C:\\logs'
        state = self.state(quoted)
        self.assertEqual(_review_witnesses([self.witness(quoted)], state, "edit")[0]["text"], quoted)
        with self.assertRaisesRegex(ValueError, "exact supplied"):
            _review_witnesses([self.witness(quoted.replace("\n", "\\n"))], state, "edit")

    def test_decoded_positive_observation_with_quotes_keeps_existing_status_rule(self):
        quoted = 'Checks passed: case "quoted" completed'
        state = self.state(quoted)
        result = _review_witnesses([self.witness(quoted, "positive_execution")], state, "keep")
        self.assertEqual(result[0]["validation"]["locations"][0]["field"], "observation_excerpt")

    def test_messages_and_unexecuted_commands_are_not_positive_observations(self):
        text = 'FINAL CHECK PASSED: GET 200, POST 201.'
        for state in (self.state("Still waiting for output", message=text),
                      self.state("Still waiting for output", command="echo '" + text + "'"),
                      self.state("Still waiting for output", command="curl -i http://localhost; # " + text)):
            with self.subTest(state=state), self.assertRaisesRegex(ValueError, "positive_execution_unproven"):
                _review_witnesses([self.witness(text, "positive_execution")], state, "keep")

    def test_real_endpoint_json_without_success_status_remains_pending(self):
        body = '{"pre_order": [3, 1, 2]}'
        command = "curl -s -X POST http://localhost:8000/api/post-to-pre -d '{\"post_order\": [2, 1, 3]}'\n"
        state = self.state("root@container:/app# " + command + body + "\nroot@container:/app# ", command=command)
        with self.assertRaisesRegex(ValueError, "positive_execution_unproven"):
            _review_witnesses([self.witness(body, "positive_execution")], state, "keep")

    def test_arbitrary_or_error_json_does_not_supply_success_status(self):
        for body in ('{"arbitrary": "payload"}', '{"error": "invalid request"}'):
            with self.subTest(body=body), self.assertRaisesRegex(ValueError, "positive_execution_unproven"):
                _review_witnesses([self.witness(body, "positive_execution")], self.state(body), "keep")

    def test_review_result_versions_validation_and_preserves_raw(self):
        body = '{"pre_order": [3, 1, 2]}'
        state = self.state(body)
        plan = record(action="keep")["plan"]
        raw = json.dumps({"status": "PASS", "reason": "Bounded review", "evidence_refs": ["evidence-1"],
                          "witnesses": [self.witness(body, "positive_execution")]}, indent=2)
        metadata = {"raw_field": "content", "finish_reason": "stop", "temperature": 0, "model": "review"}
        with tempfile.TemporaryDirectory() as td, patch("research1.distill._tongji_chat", return_value=(raw, metadata)):
            path = Path(td)
            result = review_plan(state, plan, path)
            self.assertEqual(result["status"], "pending_review")
            self.assertIn("positive_execution_unproven", result["reason"])
            self.assertEqual(result["witness_validator"], FACTUAL_WITNESS_VALIDATOR)
            self.assertEqual(result["evidence_text_sha256"]["evidence-1"], hashlib.sha256(state["evidence"][0]["text"].encode()).hexdigest())
            self.assertEqual(json.loads((path / "factual_review.json").read_text())["raw"], raw)


if __name__ == "__main__":
    unittest.main()
