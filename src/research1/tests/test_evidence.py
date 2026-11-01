"""Evidence selector tests, no models or external data."""
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from research1.evidence import select_trajectory_evidence


class EvidenceTests(unittest.TestCase):
    def write(self, data):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        path = Path(tmp.name) / "trajectory.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return path

    def data(self):
        steps = [{"step_id": i + 1, "source": "agent", "message": "visible plan",
                  "tool_calls": [{"tool_call_id": f"call_{i}", "function_name": "bash_command",
                                  "arguments": {"keystrokes": "", "duration": 1}}],
                  "observation": {"results": [{"content": f"ordinary output {i}"}]},
                  "reasoning_content": "PRIVATE_REASONING", "metrics": {"hidden": "METRIC_SECRET"}}
                 for i in range(25)]
        steps[8]["observation"]["results"][0]["content"] = "Previous response had warnings:\nWARNINGS: - Command 2 should end with newline"
        steps[8]["tool_calls"].append({"tool_call_id": "call_8_wait", "function_name": "bash_command",
                                       "arguments": {"keystrokes": "", "duration": 55.0}})
        return {"steps": steps, "final_metrics": {"secret": "SCORER_SECRET"},
                "reference_solution": "REFERENCE_SECRET", "agent": {"extra": {"hidden_tests": "TEST_SECRET"}}}

    def test_middle_parser_warning_and_neighbor_actions_survive(self):
        path = self.write(self.data())
        evidence = select_trajectory_evidence(path)
        view = json.loads(evidence[0]["text"])
        self.assertTrue({1, 8, 9, 10, 25}.issubset(set(view["selected_step_ids"])))
        self.assertIn("Command 2 should end with newline", evidence[0]["text"])
        self.assertEqual(evidence, select_trajectory_evidence(path))
        self.assertEqual(evidence[0]["source_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())

    def test_hidden_fields_never_appear(self):
        evidence = select_trajectory_evidence(self.write(self.data()))[0]["text"]
        for sentinel in ("PRIVATE_REASONING", "METRIC_SECRET", "SCORER_SECRET", "REFERENCE_SECRET", "TEST_SECRET"):
            self.assertNotIn(sentinel, evidence)

    def test_earlier_decision_never_reads_later_messages_into_view(self):
        data = self.data()
        data["steps"][15]["message"] = "FUTURE_BRANCH_SECRET"
        view = json.loads(select_trajectory_evidence(self.write(data), decision_step_id=10)[0]["text"])
        self.assertNotIn("FUTURE_BRANCH_SECRET", json.dumps(view))
        self.assertEqual(view["visible_steps_at_cutoff"], 10)
        self.assertTrue(max(view["selected_step_ids"]) <= 10)

    def test_long_step_retains_signal_with_explicit_offsets_and_limits(self):
        data = self.data()
        content = "x" * 20000 + "\nERROR: actual parser failure\n" + "y" * 20000
        data["steps"][8]["observation"]["results"][0]["content"] = content
        evidence = select_trajectory_evidence(self.write(data), max_total_chars=14000, max_step_chars=1800, max_steps=8)
        text = evidence[0]["text"]
        self.assertLessEqual(len(text), 14000)
        self.assertIn("actual parser failure", text)
        view = json.loads(text)
        for step in view["steps"]:
            self.assertLessEqual(len(json.dumps(step, ensure_ascii=False, sort_keys=True, separators=(",", ":"))), 1800)
        step = next(s for s in view["steps"] if s["step_id"] == 9)
        field = step["observation_excerpt"]
        self.assertTrue(field["truncated"])
        for segment in field["segments"]:
            self.assertEqual(segment["text"], content[segment["start"]:segment["end"]])
        self.assertIn("does not establish absence", view["diagnosis_scope"])

    def test_signals_beyond_budget_are_reported_as_omitted(self):
        data = self.data()
        for step in data["steps"]:
            step["observation"]["results"][0]["content"] = "ERROR: timeout after 1s"
        view = json.loads(select_trajectory_evidence(self.write(data), max_steps=5)[0]["text"])
        self.assertTrue(view["omitted_signal_step_ids"])
        self.assertIn(1, view["selected_step_ids"])
        self.assertIn(25, view["selected_step_ids"])

    def test_missing_cutoff_rejected(self):
        with self.assertRaises(ValueError):
            select_trajectory_evidence(self.write(self.data()), decision_step_id=999)

    def test_middle_empty_wait_survives_large_first_and_last_calls(self):
        data = self.data()
        data["steps"][8]["tool_calls"] = [
            {"tool_call_id": "call_7_1", "function_name": "bash_command", "arguments": {"keystrokes": "large script\n" * 2000, "duration": 1.0}},
            {"tool_call_id": "call_7_2", "function_name": "bash_command", "arguments": {"keystrokes": "", "duration": 55.0}},
            {"tool_call_id": "call_7_3", "function_name": "bash_command", "arguments": {"keystrokes": "echo final\n", "duration": 0.5}},
        ]
        view = json.loads(select_trajectory_evidence(self.write(data))[0]["text"])
        step = next(s for s in view["steps"] if s["step_id"] == 9)
        self.assertEqual(step["referenced_command_positions"], [2])
        calls = step["tool_calls"]
        self.assertEqual([c["position"] for c in calls], [1, 2, 3])
        self.assertEqual([c["tool_call_id"] for c in calls], ["call_7_1", "call_7_2", "call_7_3"])
        self.assertEqual(calls[1]["arguments"]["keystrokes"], "")
        self.assertEqual(calls[1]["arguments"]["duration"], 55.0)
        self.assertTrue(calls[1]["arguments"]["keystrokes_metadata"]["is_empty"])
        self.assertFalse(calls[1]["arguments"]["keystrokes_metadata"]["ends_with_newline"])
        self.assertTrue(calls[0]["arguments"]["keystrokes_metadata"]["ends_with_newline"])
        self.assertFalse(calls[0]["arguments"]["keystrokes_metadata"]["is_empty"])
        self.assertTrue(calls[0]["arguments"]["keystrokes_excerpt"]["truncated"])

    def test_unavailable_referenced_call_rejected(self):
        data = self.data()
        data["steps"][8]["tool_calls"] = []
        with self.assertRaisesRegex(ValueError, "unavailable Command"):
            select_trajectory_evidence(self.write(data))

    def test_tiny_budget_cannot_silently_drop_actions(self):
        data = self.data()
        data["steps"][8]["tool_calls"] = [
            {"tool_call_id": f"call_{i}", "function_name": "bash_command",
             "arguments": {"keystrokes": "echo x\n", "duration": 1}} for i in range(25)]
        with self.assertRaisesRegex(ValueError, "critical arguments"):
            select_trajectory_evidence(self.write(data), max_total_chars=12000, max_step_chars=1000, max_steps=5)


if __name__ == "__main__":
    unittest.main()
