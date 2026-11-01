"""Bounded factual review with execution provenance, not success phrase gating."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research1.data_pipeline.policy import (FACTUAL_REVIEW_POLICY, FACTUAL_WITNESS_VALIDATOR, POLICY_ID,
    LEGACY_PLAN_POLICY_ID, _review_witnesses, review_plan, render_messages, collect_candidates)
from research1.tests.test_data_sft import record


class ReviewProtocolV3Tests(unittest.TestCase):
    def state(self, commands, observation, *, source="agent"):
        state = record(action="keep")["state"]
        step = {"step_id": 4, "source_index": 3, "source": source,
                "visible_step_sha256": "a" * 64,
                "observation_excerpt": {"segments": [{"start": 0, "end": len(observation), "text": observation}]},
                "tool_calls": [{"tool_call_id": "call_4_" + str(index + 1), "position": index + 1,
                    "function_name": "bash_command", "arguments": {"keystrokes": command,
                    "keystrokes_excerpt": {"truncated": False, "segments": [{"start": 0,
                        "end": len(command), "text": command}]}}} for index, command in enumerate(commands)]}
        state["evidence"][0]["text"] = json.dumps({"steps": [step]}, ensure_ascii=False)
        return state

    def witness(self, text, kind="observed_tool_output"):
        return {"evidence_ref": "evidence-1", "text": text, "kind": kind}

    def test_real_curl_json_is_an_observed_body_not_proven_success(self):
        command = "curl -s -X POST http://localhost:8000/api/post-to-pre -d '{\"post_order\":[2,1,3]}'\n"
        body = '{"pre_order": [3, 1, 2]}'
        observation = "New Terminal Output:\n\nroot@x:/app# " + command + body + "root@x:/app#\n"
        state = self.state([command], observation)
        result = _review_witnesses([self.witness(body)], state, "keep")[0]
        validation = result["validation"]
        self.assertEqual(validation["source_granularity"], "step_observation")
        self.assertEqual(result["kind"], "observed_tool_output")
        self.assertFalse(validation["success_established"])
        location = validation["locations"][0]
        self.assertEqual(location["step_id"], 4)
        self.assertEqual(location["visible_step_sha256"], "a" * 64)
        self.assertEqual(location["source_start"], observation.index(body))
        self.assertEqual(location["recorded_step_tool_call_ids"], ["call_4_1"])
        self.assertIsNone(location["producer_tool_call_id"])
        self.assertEqual(location["producer_relation"], "not_provided_by_selector")
        self.assertNotIn("command_sha256", location)
        self.assertEqual(location["observation_segment_sha256"], hashlib.sha256(observation.encode()).hexdigest())
        self.assertEqual(validation["serialized_evidence_sha256"], hashlib.sha256(state["evidence"][0]["text"].encode()).hexdigest())
        for kind in ("execution_status", "check_status", "positive_execution"):
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "unproven"):
                _review_witnesses([self.witness(body, kind)], state, "keep")

    def test_error_body_can_only_support_its_observed_fact(self):
        body = '{"error": "invalid request"}'
        state = self.state(["curl -s http://localhost/api\n"], body)
        self.assertFalse(_review_witnesses([self.witness(body)], state, "keep")[0]["validation"]["success_established"])
        with self.assertRaisesRegex(ValueError, "unproven"):
            _review_witnesses([self.witness(body, "execution_status")], state, "keep")

    def test_execution_and_check_reports_are_distinct(self):
        status, check = "HTTP/1.1 503 Service Unavailable", "9 passed in 0.12 seconds"
        for text, kind in ((status, "execution_status"), (check, "check_status")):
            state = self.state(["python -m pytest\n"], text)
            result = _review_witnesses([self.witness(text, kind)], state, "keep")[0]
            self.assertFalse(result["validation"]["success_established"])
        for text, kind in ((status, "check_status"), (check, "execution_status")):
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "unproven"):
                _review_witnesses([self.witness(text, kind)], self.state(["python -m pytest\n"], text), "keep")

    def test_multi_command_output_keeps_step_context_without_inventing_origin(self):
        first, second = "curl -s http://localhost/first\n", "curl -s http://localhost/second\n"
        body = '{"observed_value": 31}'
        observation = "root@x:/app# " + first + '{"other_value": 20}\nroot@x:/app# ' + second + body + "root@x:/app#\n"
        result = _review_witnesses([self.witness(body)], self.state([first, second], observation), "keep")[0]
        location = result["validation"]["locations"][0]
        self.assertEqual(location["recorded_step_tool_call_ids"], ["call_4_1", "call_4_2"])
        self.assertIsNone(location["producer_tool_call_id"])
        self.assertEqual(location["source_granularity"], "step_terminal_observation")
        self.assertFalse(location["freshness_established"])
        replay = self.state([], observation)
        with self.assertRaisesRegex(ValueError, "unproven"):
            _review_witnesses([self.witness(body)], replay, "keep")

    def test_wrapped_terminal_request_retains_step_source_not_guessed_producer(self):
        command = "curl -s http://localhost/api -d '{\"input\": [1, 2, 3]}'\n"
        body = '{"output": [3, 2, 1]}'
        wrapped = "root@x:/app# curl -s http://localhost/api -d '{\"input\": [1, 2,\n 3]}'\n" + body
        result = _review_witnesses([self.witness(body)], self.state([command], wrapped), "keep")
        location = result[0]["validation"]["locations"][0]
        self.assertEqual(location["recorded_step_tool_call_ids"], ["call_4_1"])
        self.assertIsNone(location["producer_tool_call_id"])
        self.assertEqual(location["observation_segment_sha256"], hashlib.sha256(wrapped.encode()).hexdigest())

    def test_executed_echo_printf_and_obvious_print_fakes_do_not_qualify(self):
        text = "All tests passed with no failures"
        commands = ["echo '" + text + "'\n", "printf '%s\\n' '" + text + "'\n",
                    "env echo '" + text + "'\n", "bash -c \"echo '" + text + "'\"\n",
                    "python -c 'print(\"" + text + "\")'\n"]
        for command in commands:
            observation = "root@x:/app# " + command + text + "\nroot@x:/app#\n"
            for kind in ("observed_tool_output", "check_status", "positive_execution"):
                with self.subTest(command=command, kind=kind), self.assertRaisesRegex(ValueError, "unproven"):
                    _review_witnesses([self.witness(text, kind)], self.state([command], observation), "keep")

    def test_command_echo_alone_cannot_count_as_output(self):
        body = '{"echoed_input": [1, 2, 3]}'
        command = "curl -s http://localhost/api -d '" + body + "'\n"
        state = self.state([command], "root@x:/app# " + command)
        with self.assertRaisesRegex(ValueError, "unproven"):
            _review_witnesses([self.witness(body)], state, "keep")

    def test_commands_messages_and_missing_provenance_cannot_replace_observations(self):
        body = '{"actual_response": [1, 2, 3]}'
        state = self.state(["curl -s http://localhost/api\n"], "No output yet")
        value = json.loads(state["evidence"][0]["text"])
        value["steps"][0]["message_excerpt"] = {"segments": [{"text": body}]}
        state["evidence"][0]["text"] = json.dumps(value)
        with self.assertRaisesRegex(ValueError, "unproven"):
            _review_witnesses([self.witness(body)], state, "keep")
        for field in ("tool_call_id", "visible_step_sha256"):
            state = self.state(["curl -s http://localhost/api\n"], body)
            changed = json.loads(state["evidence"][0]["text"])
            target = changed["steps"][0]["tool_calls"][0] if field == "tool_call_id" else changed["steps"][0]
            del target[field]
            state["evidence"][0]["text"] = json.dumps(changed)
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "unproven"):
                _review_witnesses([self.witness(body)], state, "keep")

    def test_unknown_refs_arbitrary_json_and_absence_never_qualify_by_themselves(self):
        sample = record(action="keep")
        for text, kind in (('{"status":"all tests passed"}', "positive_execution"),
                           ('{"result":"arbitrary JSON body"}', "observed_tool_output"),
                           ("No errors were visible anywhere", "observed_tool_output")):
            sample["state"]["evidence"][0]["text"] = text
            with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "unproven"):
                _review_witnesses([self.witness(text, kind)], sample["state"], "keep")
        with self.assertRaisesRegex(ValueError, "exact supplied"):
            _review_witnesses([dict(self.witness("All checks passed"), evidence_ref="unknown")], sample["state"], "keep")

    def test_review_versions_and_bounds_prompt_without_promoting_pending(self):
        command = "curl -s http://localhost/api\n"
        body = '{"actual_response": [1, 2, 3]}'
        state = self.state([command], body)
        plan = record(action="keep")["plan"]
        plan["problem_hypothesis"] = "No defect is evidenced in this bounded view; the executed request returned the quoted array."
        value = {"status": "pending_review", "reason": "The independent reviewer is uncertain.",
                 "evidence_refs": ["evidence-1"], "witnesses": [self.witness(body)]}
        raw = json.dumps(value)
        metadata = {"raw_field": "content", "finish_reason": "stop", "temperature": 0, "model": "review"}
        with tempfile.TemporaryDirectory() as td, patch("research1.distill._tongji_chat", return_value=(raw, metadata)) as chat:
            result = review_plan(state, plan, Path(td))
            self.assertEqual(result["status"], "pending_review")
            self.assertEqual(result["review_policy"], FACTUAL_REVIEW_POLICY)
            self.assertEqual(result["witness_validator"], FACTUAL_WITNESS_VALIDATOR)
            saved = json.loads((Path(td) / "factual_review.json").read_text())
            self.assertEqual(saved["raw"], raw)
            self.assertIn("Distinguish bounded abstention from global absence", saved["prompt"])
            self.assertIn("classification, not a proven cause", saved["prompt"])
            self.assertIn("does not establish HTTP success", saved["prompt"])
            self.assertEqual(chat.call_count, 1)

    def test_bounded_keep_needs_independent_pass_and_raw_is_not_rewritten(self):
        command, body = "curl -s http://localhost/api\n", '{"actual_response": [1, 2, 3]}'
        state = self.state([command], body)
        plan = record(action="keep")["plan"]
        plan["problem_hypothesis"] = "No defect is evidenced in this supplied bounded view: the observed response contains the quoted array."
        value = {"status": "PASS", "reason": "The bounded abstention is supported by the concrete observed output.",
                 "evidence_refs": ["evidence-1"], "witnesses": [self.witness(body)]}
        raw = json.dumps(value, indent=2)
        meta = {"raw_field": "content", "finish_reason": "stop", "temperature": 0, "model": "review"}
        with tempfile.TemporaryDirectory() as td, patch("research1.distill._tongji_chat", return_value=(raw, meta)):
            result = review_plan(state, plan, Path(td))
            self.assertEqual(result["status"], "PASS")
            self.assertFalse(result["witnesses"][0]["validation"]["success_established"])
            self.assertEqual(json.loads((Path(td) / "factual_review.json").read_text())["raw"], raw)

    def test_future_keep_requires_factual_checks_without_rewriting_old_instruction(self):
        state = record(action="keep")["state"]
        legacy = render_messages(state)[0]["content"]
        state["plan_policy_version"] = LEGACY_PLAN_POLICY_ID
        self.assertEqual(render_messages(state)[0]["content"], legacy)
        state["plan_policy_version"] = POLICY_ID
        future = render_messages(state)[0]["content"]
        self.assertIn("name concrete factual checks and remaining evidence limits", future)
        self.assertIn("an empty verification catalog alone is not a sufficient keep justification", future)
        self.assertNotEqual(future, legacy)
        state["plan_policy_version"] = "unknown"
        with self.assertRaisesRegex(ValueError, "instruction version"):
            render_messages(state)

    def test_teacher_saves_the_actual_frozen_instruction_version(self):
        sample = record(action="keep")
        raw = json.dumps({"candidates": [sample["plan"]]})
        meta = {"raw_field": "content", "finish_reason": "stop", "temperature": 0, "model": "teacher"}
        with tempfile.TemporaryDirectory() as td, patch("research1.distill._tongji_chat", return_value=(raw, meta)):
            directory = Path(td)
            collect_candidates(sample["state"], directory / "legacy")
            sample["state"]["plan_policy_version"] = POLICY_ID
            collect_candidates(sample["state"], directory / "future")
            self.assertEqual(json.loads((directory / "legacy/teacher.json").read_text())["policy_id"], LEGACY_PLAN_POLICY_ID)
            self.assertEqual(json.loads((directory / "future/teacher.json").read_text())["policy_id"], POLICY_ID)


if __name__ == "__main__":
    unittest.main()
