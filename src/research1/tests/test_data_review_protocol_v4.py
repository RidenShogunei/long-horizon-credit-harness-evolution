"""Narrow metadata context and asynchronous polling observation support."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research1.data_pipeline.policy import (FACTUAL_REVIEW_POLICY, FACTUAL_WITNESS_VALIDATOR,
    _review_witnesses, review_plan)
from research1.tests import test_data_review_protocol_v3 as fixtures
from research1.tests.test_data_sft import record


class ReviewProtocolV4Tests(unittest.TestCase):
    def state(self, observation, command="", duration=3.0):
        state = fixtures.ReviewProtocolV3Tests().state([command], observation)
        envelope = json.loads(state["evidence"][0]["text"])
        envelope["omitted_step_ids"] = [12, 13]
        envelope["steps"][0]["tool_calls"][0]["arguments"]["duration"] = duration
        state["evidence"][0]["text"] = json.dumps(envelope, ensure_ascii=False, separators=(",", ":"))
        return state

    def witness(self, text, kind="observed_tool_output"):
        return {"evidence_ref": "evidence-1", "text": text, "kind": kind}

    def test_serialized_metadata_context_retains_its_own_offsets_and_hash(self):
        state = self.state("No output yet")
        original = state["evidence"][0]["text"]
        quote = '"omitted_step_ids":[12,13]'
        result = _review_witnesses([self.witness(quote, "context")], state, "edit")[0]
        location = result["validation"]["locations"][0]
        self.assertEqual(location["field"], "serialized_context")
        self.assertEqual(location["source_representation"], "serialized_json")
        self.assertEqual(location["source_start"], original.index(quote))
        self.assertEqual(location["source_end"], original.index(quote) + len(quote))
        self.assertEqual(original[location["source_start"]:location["source_end"]], quote)
        self.assertFalse(location["execution_support"])
        self.assertNotIn("execution_origin", location)
        self.assertEqual(result["validation"]["serialized_evidence_sha256"], hashlib.sha256(original.encode()).hexdigest())
        self.assertEqual(state["evidence"][0]["text"], original)
        with self.assertRaisesRegex(ValueError, "substantive observed execution"):
            _review_witnesses([self.witness(quote, "context")], state, "keep")

    def test_serialized_context_never_becomes_execution_evidence(self):
        state = self.state("No output yet")
        quote = '"omitted_step_ids":[12,13]'
        for kind in ("observed_tool_output", "execution_status", "check_status", "positive_execution"):
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "exact supplied"):
                _review_witnesses([self.witness(quote, kind)], state, "keep")
        # Nested step metadata is visible context, but never execution proof.
        nested = '"step_id":4,"source_index":3'
        context = _review_witnesses([self.witness(nested, "context")], state, "edit")[0]
        self.assertEqual(context["validation"]["locations"][0]["field"], "serialized_context")
        self.assertFalse(context["validation"]["substantive_observed_fact"])
        with self.assertRaisesRegex(ValueError, "exact supplied"):
            _review_witnesses([self.witness(nested)], state, "keep")

    def test_polling_binds_capture_without_claiming_a_producer_or_freshness(self):
        quote = "Successfully installed cffi-2.1.1 pycparser-3.12"
        observation = "New Terminal Output:\n\n" + quote + "\nroot@container:/app#\n"
        state = self.state(observation)
        result = _review_witnesses([self.witness(quote)], state, "keep")[0]
        validation = result["validation"]
        self.assertFalse(validation["success_established"])
        location = validation["locations"][0]
        self.assertEqual(location["source_start"], observation.index(quote))
        self.assertEqual(location["step_id"], 4)
        self.assertEqual(location["source_granularity"], "step_terminal_observation")
        self.assertEqual(location["recorded_step_tool_call_ids"], ["call_4_1"])
        self.assertIsNone(location["producer_tool_call_id"])
        self.assertEqual(location["producer_relation"], "not_provided_by_selector")
        self.assertFalse(location["freshness_established"])
        self.assertFalse(location["success_established"])
        self.assertEqual(location["observation_segment_sha256"], hashlib.sha256(observation.encode()).hexdigest())
        self.assertEqual(validation["serialized_evidence_sha256"], hashlib.sha256(state["evidence"][0]["text"].encode()).hexdigest())
        self.assertNotIn("command_sha256", location)

    def test_polling_status_or_check_text_remains_a_report_without_success_proof(self):
        for quote, kind in (("HTTP/1.1 200 OK; payload observed", "execution_status"),
                            ("All tests passed in 0.12 seconds", "check_status"),
                            ("All tests passed in 0.12 seconds", "positive_execution")):
            with self.subTest(kind=kind):
                result = _review_witnesses([self.witness(quote, kind)], self.state(quote), "keep")[0]
                self.assertTrue(result["validation"]["reported_only"])
                self.assertFalse(result["validation"]["success_established"])
                location = result["validation"]["locations"][0]
                self.assertIsNone(location["producer_tool_call_id"])
                self.assertFalse(location["freshness_established"])
        with self.assertRaisesRegex(ValueError, "unproven"):
            _review_witnesses([self.witness('{"status":"success"}', "execution_status")], self.state('{"status":"success"}'), "keep")

    def test_observation_needs_typed_step_identity_not_an_invented_duration_guarantee(self):
        quote = "Successfully installed cffi-2.1.1 pycparser-3.12"
        cases = [self.state(quote, duration=0), self.state(quote, duration=-1),
                 self.state(quote, duration=True), self.state(quote, command="C-c", duration=3)]
        for index, state in enumerate(cases):
            with self.subTest(case=index):
                result = _review_witnesses([self.witness(quote)], state, "keep")[0]
                location = result["validation"]["locations"][0]
                self.assertIsNone(location["producer_tool_call_id"])
                self.assertFalse(location["freshness_established"])
                self.assertFalse(result["validation"]["success_established"])
        missing = self.state(quote)
        envelope = json.loads(missing["evidence"][0]["text"])
        del envelope["steps"][0]["tool_calls"][0]["tool_call_id"]
        missing["evidence"][0]["text"] = json.dumps(envelope)
        with self.assertRaisesRegex(ValueError, "unproven"):
            _review_witnesses([self.witness(quote)], missing, "keep")

    def test_actual_gcc_command_echo_remains_context_not_output(self):
        command = "gcc -shared -fPIC -O2 -o /app/libmatrix.so /app/matrix.c\n"
        quote = "root@container:/app# " + command.rstrip()
        state = self.state(quote + "\nroot@container:/app#\n", command=command)
        with self.assertRaisesRegex(ValueError, "unproven"):
            _review_witnesses([self.witness(quote)], state, "keep")
        result = _review_witnesses([self.witness(quote, "context")], state, "edit")[0]
        self.assertNotIn("execution_origin", result["validation"]["locations"][0])
        with self.assertRaisesRegex(ValueError, "unproven"):
            _review_witnesses([self.witness(quote)], self.state(quote), "keep")

    def test_future_prompt_explains_echo_metadata_and_polling_limits(self):
        quote = "Successfully installed cffi-2.1.1 pycparser-3.12"
        state = self.state(quote)
        raw = json.dumps({"status": "pending_review", "reason": "Still uncertain.",
            "evidence_refs": ["evidence-1"], "witnesses": [self.witness(quote)]})
        metadata = {"raw_field": "content", "finish_reason": "stop", "temperature": 0, "model": "review"}
        with tempfile.TemporaryDirectory() as td, patch("research1.distill._tongji_chat", return_value=(raw, metadata)):
            result = review_plan(state, record(action="keep")["plan"], Path(td))
            saved = json.loads((Path(td) / "factual_review.json").read_text())
            self.assertEqual(result["status"], "pending_review")
            self.assertEqual(result["review_policy"], FACTUAL_REVIEW_POLICY)
            self.assertEqual(result["witness_validator"], FACTUAL_WITNESS_VALIDATOR)
            self.assertEqual(saved["raw"], raw)
            self.assertIn("captured terminal text at step granularity", saved["prompt"])
            self.assertIn("A command echo alone cannot justify keep", saved["prompt"])
            self.assertIn("serialized envelope metadata such as omitted_step_ids", saved["prompt"])
            self.assertIn("not which command produced it", saved["prompt"])


if __name__ == "__main__":
    unittest.main()
