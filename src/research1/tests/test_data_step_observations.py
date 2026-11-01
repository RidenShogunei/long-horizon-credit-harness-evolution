"""Observed real-v5 failure classes, without replaying tasks or model calls."""
import hashlib
import json
import unittest
from research1.data_pipeline.observation_validator import audit_witnesses, validate_witnesses


class StepObservationTests(unittest.TestCase):
    def view(self, observation, command="python3 - <<'PY'\nprint('example')\nPY\n", commands=None):
        step = {"step_id": 5, "source_index": 4, "source": "agent", "visible_step_sha256": "a"*64,
            "selected_for": ["signal:runtime_warning", "additional_context"],
            "observation_excerpt": {"segments": [{"start": 100, "end": 100+len(observation), "text": observation}]},
            "tool_calls": [{"function_name": "bash_command", "tool_call_id": f"call_3_{index+1}",
                "arguments": {"keystrokes": text}} for index, text in enumerate(commands or [command])]}
        serialized = json.dumps({"trajectory_sha256": "b"*64, "steps": [step]}, separators=(",", ":"))
        return {"evidence": [{"id": "evidence-1", "text": serialized}]}

    def witness(self, text, kind="observed_tool_output", ref="evidence-1"):
        return {"evidence_ref": ref, "text": text, "kind": kind}

    def test_exact_view_step_observation_hashes_spans_and_unknown_producer(self):
        body = '{"actual_response": [3, 1, 2]}'
        observation = "New Terminal Output:\n" + body + "\n"
        view = self.view(observation, "curl -s http://localhost/api\n")
        result = validate_witnesses([self.witness(body)], view, "keep")[0]
        validation = result["validation"]; location = validation["locations"][0]
        self.assertEqual(validation["serialized_evidence_sha256"], hashlib.sha256(view["evidence"][0]["text"].encode()).hexdigest())
        self.assertEqual(location["visible_step_sha256"], "a"*64)
        self.assertEqual(location["observation_segment_sha256"], hashlib.sha256(observation.encode()).hexdigest())
        self.assertEqual(location["source_start"], 100+observation.index(body))
        self.assertEqual(location["source_end"], 100+observation.index(body)+len(body))
        self.assertEqual(location["recorded_step_tool_call_ids"], ["call_3_1"])
        self.assertIsNone(location["producer_tool_call_id"])
        self.assertFalse(location["freshness_established"])
        self.assertFalse(location["success_established"])
        self.assertNotIn("command_sha256", location)

    def test_actual_mixed_server_command_and_job_start_is_observed_fact(self):
        quote = "root@5625b2984687:/app# python3 server.py > server.log 2>&1 &\n[1] 515"
        command = "cat > server.py <<'PY'\nprint(\"it's valid Python\")\nPY\npython3 server.py > server.log 2>&1 &\n"
        observation = "New Terminal Output:\n>\n> PY\n" + quote + "\nroot@5625b2984687:/app#\n"
        result = audit_witnesses([self.witness(quote)], self.view(observation, command), "keep")
        self.assertTrue(result["eligible_witness_set"])
        self.assertTrue(result["substantive_execution_fact"])
        self.assertFalse(result["witnesses"][0]["validation"]["success_established"])

    def test_actual_payroll_responses_do_not_require_rebuilding_a_heredoc(self):
        command = "python3 - <<'PY'\nfrom app import app\n\nc=app.test_client()\nprint(c.get('/accounts'))\nPY\n"
        bodies = ["/accounts/batch/mgr001 200 {'accounts': [{'accountId': 'emp001'}]}",
            "/accounts/batch/mgr004 200 {'accounts': [], 'pagination': {'totalPages': 0}}",
            "/accounts/batch/mgr999 403 {'error': 'Access denied'}",
            "/accounts/batch/mgr001?page=0 400 {'error': 'Invalid pagination parameters'}"]
        observation = "root@x:/app# python3 - <<'PY'\n> from app import app\n>\n> PY\n" + "\n".join(bodies)
        report = audit_witnesses([self.witness(body) for body in bodies], self.view(observation, command), "keep")
        self.assertTrue(report["eligible_witness_set"])
        self.assertEqual(len(report["witnesses"]), 4)

    def test_actual_bloom_python_body_is_not_parsed_as_shell_and_stdout_is_found(self):
        command = "cat > bloom.py <<'PY'\nprint(\"it's valid Python\")\nPY\npython3 - <<'PY'\nassert True\nprint('all tests passed')\nPY\n"
        observation = "root@x:/app# cat > bloom.py <<'PY'\n> print(\"it's valid Python\")\n>\n> PY\n> assert True\n> print('all tests passed')\n> PY\nall tests passed\nroot@x:/app#\n"
        report = audit_witnesses([self.witness("all tests passed")], self.view(observation, command), "keep")
        self.assertTrue(report["eligible_witness_set"])
        locations = report["witnesses"][0]["validation"]["locations"]
        self.assertEqual([item["substantive_observed_fact"] for item in locations], [False, True])

    def test_pure_commands_are_weak_but_do_not_invalidate_other_real_outputs(self):
        command = "gcc -shared -o lib.so source.c\n"
        displayed = "root@x:/app# " + command.rstrip()
        body = '{"library_loaded": true}'
        view = self.view(displayed + "\n" + body, command)
        only = audit_witnesses([self.witness(displayed)], view, "keep")
        self.assertTrue(only["valid_references"])
        self.assertFalse(only["eligible_witness_set"])
        together = audit_witnesses([self.witness(displayed), self.witness(body)], view, "keep")
        self.assertTrue(together["eligible_witness_set"])
        self.assertFalse(together["witnesses"][0]["validation"]["substantive_observed_fact"])

    def test_obvious_echo_printf_only_cannot_supply_needed_fact(self):
        quote = "All tests passed with no failures"
        for command in ("echo '"+quote+"'\n", "printf '%s\\n' '"+quote+"'\n", "env echo '"+quote+"'\n",
                        "python -c 'print(\""+quote+"\")'\n"):
            with self.subTest(command=command):
                report = audit_witnesses([self.witness(quote)], self.view("root@x:/app# "+command+quote, command), "keep")
                self.assertTrue(report["valid_references"])
                self.assertFalse(report["eligible_witness_set"])
        # Unrelated genuine commands do not make an echo-only citation enough.
        commands = ["echo '"+quote+"'\n", "curl -s http://localhost/api\n"]
        report = audit_witnesses([self.witness(quote)], self.view(quote, commands=commands), "keep")
        self.assertFalse(report["eligible_witness_set"])

    def test_nested_metadata_is_serialized_context_only(self):
        body = '{"actual_response": [1, 2, 3]}'
        view = self.view(body)
        quote = '"selected_for":["signal:runtime_warning","additional_context"]'
        report = audit_witnesses([self.witness(quote, "context"), self.witness(body)], view, "keep")
        self.assertTrue(report["eligible_witness_set"])
        context = report["witnesses"][0]["validation"]
        self.assertFalse(context["substantive_observed_fact"])
        self.assertEqual(context["locations"][0]["field"], "serialized_context")
        self.assertFalse(audit_witnesses([self.witness(quote, "context")], view, "keep")["eligible_witness_set"])
        self.assertFalse(audit_witnesses([self.witness(quote)], view, "keep")["eligible_witness_set"])

    def test_external_prior_outcome_is_not_silently_aliased_to_evidence(self):
        body = '{"actual_response": [1, 2, 3]}'
        view = self.view(body)
        view["prior_proposal_outcomes"] = [{"failure_layer": "model", "finished_before_decision": True}]
        witnesses = [self.witness(body), self.witness('"failure_layer":"model"', "context"),
            self.witness('"finished_before_decision":true', "context")]
        report = audit_witnesses(witnesses, view, "keep")
        self.assertFalse(report["valid_references"])
        self.assertEqual([item["index"] for item in report["issues"]], [1, 2])

    def test_status_and_check_text_are_reported_only_not_success_proof(self):
        for text, kind, supported in (('{"status": "success"}', "execution_status", False),
                           ("HTTP/1.1 200 OK", "execution_status", True),
                           ("all tests passed", "check_status", True),
                           ("9 passed in 0.12 seconds", "check_status", True)):
            with self.subTest(kind=kind):
                report = audit_witnesses([self.witness(text, kind)], self.view(text), "keep")
                self.assertEqual(report["eligible_witness_set"], supported)
                self.assertFalse(report["witnesses"][0]["validation"]["success_established"])
                self.assertTrue(report["witnesses"][0]["validation"]["reported_only"])

    def test_plain_legacy_adapter_is_compatible_and_untyped_json_is_not_proof(self):
        text = "Ran actual proposal commands. FINAL CHECK PASSED: GET 200, POST 201."
        view = {"evidence": [{"id": "evidence-1", "text": text}]}
        result = validate_witnesses([self.witness("FINAL CHECK PASSED: GET 200, POST 201.", "positive_execution")], view, "keep")
        self.assertEqual(result[0]["validation"]["locations"][0]["field"], "plain_text")
        self.assertFalse(result[0]["validation"]["success_established"])
        view["evidence"][0]["text"] = '{"message":"All tests passed successfully"}'
        report = audit_witnesses([self.witness("All tests passed successfully", "positive_execution")], view, "keep")
        self.assertFalse(report["eligible_witness_set"])

    def test_missing_typed_hash_or_incorrect_span_blocks_execution_support(self):
        body = '{"actual_response": [1, 2, 3]}'
        for field in ("visible_step_sha256", "end"):
            view = self.view(body); envelope = json.loads(view["evidence"][0]["text"])
            if field == "visible_step_sha256":
                del envelope["steps"][0][field]
            else:
                envelope["steps"][0]["observation_excerpt"]["segments"][0][field] += 1
            view["evidence"][0]["text"] = json.dumps(envelope)
            with self.subTest(field=field):
                self.assertFalse(audit_witnesses([self.witness(body)], view, "keep")["eligible_witness_set"])

    def test_message_or_command_source_alone_is_not_an_observation(self):
        body = '{"actual_response": [1, 2, 3]}'
        view = self.view("Waiting for execution")
        envelope = json.loads(view["evidence"][0]["text"])
        envelope["steps"][0]["message_excerpt"] = {"segments": [{"text": body}]}
        view["evidence"][0]["text"] = json.dumps(envelope)
        self.assertFalse(audit_witnesses([self.witness(body)], view, "keep")["eligible_witness_set"])
        self.assertTrue(audit_witnesses([self.witness(body, "context")], view, "edit")["valid_references"])


if __name__ == "__main__":
    unittest.main()
