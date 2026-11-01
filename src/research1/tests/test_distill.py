"""Schema and parse tests for plan-I/O distillation. No network."""

from __future__ import annotations

import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from research1.distill import (
    SCHEMA_ID,
    VERIFIED_SCHEMA_ID,
    build_edit_prompt,
    export_observed_jsonl,
    filter_empty_keystroke_warnings,
    run_llm_review,
    _label_none,
    _message_text,
    build_plan_prompt,
    extract_file_body,
    parse_plan_output,
    relevant_component_excerpt,
    validate_plan_io_record,
    parse_observed_output,
    parse_review_verdict,
    phenomenon_grounded,
    score_false_concat_probe,
    score_observed,
    verify_false_concat_edit,
)

_SOURCE = '''\
class BaselineTools:
    def parse_llm_response(self, response: str):
        return response

    def execute(self, call, ctx):
        return call

def register(library):
    library.register(type_="tools")
'''


def _ok_record() -> dict:
    plan_input = {
        "error_trajectory_excerpt": "WARNINGS: Command 2 should end with newline",
        "finding_kind": "false_concat_warning_on_empty_keystrokes",
        "finding_detail": "empty keystrokes",
        "source_trajectory_path": "/tmp/trajectory.json",
        "module_snapshot_path": "/tmp/tools/baseline.py",
        "module_snapshot_sha256": "a" * 64,
        "module_relpath": "tools/baseline.py",
        "module_excerpt": "def parse_llm_response(self, response: str):\n    return response\n",
    }
    prompt = build_plan_prompt(plan_input)
    return {
        "schema": SCHEMA_ID,
        "record_id": "plan_io_test",
        "status": "ok",
        "input": plan_input,
        "planner_prompt": prompt,
        "planner_raw": (
            "TARGET: tools/baseline.py\n"
            "INTENT:\n"
            "Suppress the false concatenation warning.\n"
            "TRIGGER: BASELINE_TOOLS_FALSE_CONCAT_SUPPRESSED\n"
        ),
        "parsed_plan": parse_plan_output(
            "TARGET: tools/baseline.py\n"
            "INTENT:\n"
            "Suppress the false concatenation warning.\n"
            "TRIGGER: BASELINE_TOOLS_FALSE_CONCAT_SUPPRESSED\n"
        ),
        "metadata": {
            "provider": "tongji_agi",
            "model": "DeepSeek-V4.1-Flash",
            "temperature": 0.0,
            "task": "cron-env-debug",
            "source_trajectory_path": "/tmp/trajectory.json",
            "timestamp": "2026-10-30T00:00:00+00:00",
        },
    }


class DistillPlanSchemaTests(unittest.TestCase):
    def test_mixed_warnings_preserve_real_and_unrelated_diagnostics(self) -> None:
        def warning(index: int) -> str:
            return (
                f"- Command {index} should end with newline when followed by another command. "
                "Otherwise the two commands will be concatenated together on the same line."
            )

        commands = [SimpleNamespace(keystrokes=""), SimpleNamespace(keystrokes="echo hi")]
        empty, real = warning(1), warning(2)
        unrelated = "- Command 1 contains an unsupported key sequence."
        for items in ([empty, real, unrelated], [real, unrelated, empty]):
            before = "\n".join(items)
            result = filter_empty_keystroke_warnings(before, commands)
            self.assertNotIn(empty, result)
            self.assertIn(real, result)
            self.assertIn(unrelated, result)
        self.assertEqual([command.keystrokes for command in commands], ["", "echo hi"])

    def test_warning_filter_keeps_container_type_and_unknown_messages(self) -> None:
        false = "- Command 1 should end with newline when followed by another command."
        other = "Unrelated warning"
        self.assertEqual(filter_empty_keystroke_warnings([false, other], [{"keystrokes": ""}]), [other])
        self.assertEqual(filter_empty_keystroke_warnings((false, other), [("", 5.0)]), (other,))
        self.assertIsNone(filter_empty_keystroke_warnings(false, [{"keystrokes": ""}]))
        self.assertEqual(filter_empty_keystroke_warnings(false, [{}]), false)
        combined = false + " Another meaningful diagnostic on the same line."
        self.assertEqual(filter_empty_keystroke_warnings(combined, [{"keystrokes": ""}]), combined)

    def test_editor_prompt_preserves_non_target_parser_outputs(self) -> None:
        prompt = build_edit_prompt(plan={"target": "tools/baseline.py", "intent": "fix a diagnosed issue"}, source=_SOURCE)
        self.assertIn("command keystrokes, durations, ordering, parser errors", prompt)
        self.assertIn("never drop a whole multi-warning string", prompt)
        self.assertNotIn("must stop the false warning", prompt)

    def test_warning_only_export_is_auxiliary_and_cannot_target_sft(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            records = root / "run" / "records"
            records.mkdir(parents=True)
            (records / "one.json").write_text(json.dumps({
                "record_id": "one", "verified": {"label": "positive"},
                "metadata": {"task": "cron-env-debug"}, "parsed_plan": {"target": "tools/baseline.py"},
                "planner_prompt": "input", "planner_raw": "plan",
            }))
            dest = root / "auxiliary_warning_cleanup" / "candidates.jsonl"
            self.assertEqual(export_observed_jsonl(root / "run", dest), 1)
            row = json.loads(dest.read_text())
            self.assertFalse(row["training_eligible"])
            self.assertEqual(row["use"], "auxiliary_warning_cleanup")
            self.assertIsNone(row["long_horizon_label"])
            self.assertEqual(export_observed_jsonl(root / "run", dest), 0)
            for target in (root / "plan_sft" / "positives.jsonl", root / "research1" / "sft" / "verified.jsonl"):
                with self.assertRaises(ValueError):
                    export_observed_jsonl(root / "run", target)
                self.assertFalse(target.exists())

    def test_llm_review_does_not_mutate_source_or_export_training_targets(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "reports").mkdir()
            record_path = root / "runs" / "distill_observed_old" / "records" / "one.json"
            record_path.parent.mkdir(parents=True)
            record_path.write_text(json.dumps(_ok_record()))
            before = record_path.read_bytes()
            with patch("research1.distill.REPO_ROOT", root), patch("research1.distill.load_pilot", return_value={}), patch(
                "research1.distill._tongji_chat", return_value=("VERDICT: pass\nREASON: plausible\n", {})
            ):
                _, summary = run_llm_review()
            self.assertEqual(record_path.read_bytes(), before)
            self.assertFalse((root / "datasets" / "plan_sft").exists())
            row = json.loads(Path(summary["jsonl"]).read_text())
            self.assertEqual(row["use"], "auxiliary_review")
            self.assertFalse(row["training_eligible"])
            self.assertEqual(summary["exported_positives"], 0)

    def test_parse_plan_fields(self) -> None:
        parsed = parse_plan_output(
            "TARGET: tools/baseline.py\n"
            "INTENT:\n"
            "Drop the false newline warning on empty keystrokes.\n"
            "TRIGGER: NONE\n"
        )
        self.assertEqual(parsed["target"], "tools/baseline.py")
        self.assertIn("empty keystrokes", parsed["intent"])
        self.assertIsNone(parsed["trigger"])

    def test_parse_rejects_missing_target(self) -> None:
        with self.assertRaises(ValueError):
            parse_plan_output("INTENT:\nfix it\nTRIGGER: NONE\n")

    def test_prompt_embeds_excerpt_and_hash(self) -> None:
        record = _ok_record()
        errors = validate_plan_io_record(record)
        self.assertEqual(errors, [])
        self.assertIn(record["input"]["error_trajectory_excerpt"], record["planner_prompt"])
        self.assertIn(record["input"]["module_snapshot_sha256"], record["planner_prompt"])

    def test_ok_record_requires_parsed_intent(self) -> None:
        record = _ok_record()
        record["parsed_plan"] = {"target": "tools/baseline.py", "trigger": None}
        errors = validate_plan_io_record(record)
        self.assertTrue(any("intent" in err for err in errors))

    def test_applied_edit_requires_plan_path(self) -> None:
        record = _ok_record()
        record["edit"] = {"status": "applied", "diff_path": "/tmp/a.diff"}
        errors = validate_plan_io_record(record)
        self.assertTrue(any("plan_path" in err for err in errors))

    def test_message_text_falls_back_to_reasoning(self) -> None:
        text, field = _message_text({"content": "  ", "reasoning": "TARGET: none\n"})
        self.assertEqual(field, "reasoning")
        self.assertIn("TARGET", text)
        text, field = _message_text({"content": [{"type": "text", "text": "TARGET: tools/baseline.py"}]})
        self.assertEqual(field, "content")
        self.assertIn("tools/baseline.py", text)

    def test_none_on_unfixed_snapshot_stays_unlabeled(self) -> None:
        record = _ok_record()
        record["parsed_plan"] = {
            "target": "none",
            "intent": "The warning is not this component.",
            "trigger": None,
        }
        label = _label_none(record)
        self.assertEqual(label["schema"], VERIFIED_SCHEMA_ID)
        self.assertEqual(label["label"], "unlabeled")

    def test_score_passes_when_false_warning_is_gone(self) -> None:
        scored = score_false_concat_probe(
            "",
            "Command 1 should end with newline when followed by another command",
        )
        self.assertTrue(scored["passed"])
        self.assertTrue(scored["trigger_repeats_warning"])

    def test_score_rejects_a_missing_new_trigger(self) -> None:
        scored = score_false_concat_probe("", "concat_warning_suppressed_empty_keystrokes")
        self.assertFalse(scored["passed"])
        self.assertIn("new trigger", scored["reason"])

    def test_observed_score_requires_parent_to_emit_phenomenon(self) -> None:
        scored = score_observed("", "", "should end with newline", None)
        self.assertEqual(scored["label"], "unlabeled")

    def test_observed_score_passes_when_phenomenon_disappears(self) -> None:
        scored = score_observed(
            "Command 1 should end with newline",
            "",
            "should end with newline",
            "should end with newline",
        )
        self.assertEqual(scored["label"], "positive")
        scored_new = score_observed(
            "Command 1 should end with newline",
            "",
            "should end with newline",
            "suppressed empty wait",
        )
        self.assertEqual(scored_new["label"], "positive")

    def test_review_verdict_parses_pass_and_fail(self) -> None:
        passed = parse_review_verdict("VERDICT: pass\nREASON: The plan drops the false warning.\n")
        self.assertEqual(passed["verdict"], "pass")
        failed = parse_review_verdict("VERDICT: fail\nREASON: The excerpt does not implicate tools.\n")
        self.assertEqual(failed["verdict"], "fail")
        with self.assertRaises(ValueError):
            parse_review_verdict("VERDICT: maybe\nREASON: unsure\n")

    def test_phenomenon_must_be_quoted_from_excerpt(self) -> None:
        parsed = parse_observed_output(
            "TARGET: tools/baseline.py\n"
            "INTENT:\n"
            "Drop the false warning.\n"
            "PHENOMENON: should end with newline\n"
            "TRIGGER: NONE\n"
        )
        self.assertEqual(parsed["phenomenon"], "should end with newline")
        self.assertTrue(phenomenon_grounded(parsed["phenomenon"], "Command 1 should end with newline"))
        self.assertFalse(phenomenon_grounded("paraphrased warning", "Command 1 should end with newline"))

    def test_score_rejects_a_remaining_false_warning(self) -> None:
        scored = score_false_concat_probe(
            "Otherwise the two commands will be concatenated together on the same line.",
            "concat_warning_suppressed_empty_keystrokes",
        )
        self.assertFalse(scored["passed"])
        self.assertFalse(scored["false_concat_gone"])
        self.assertTrue(callable(verify_false_concat_edit))

    def test_extract_fenced_module_without_sentinels(self) -> None:
        src = (
            "class BaselineTools:\n"
            "    def parse_llm_response(self, response):\n"
            "        return response\n"
            "\n"
            "def register(library):\n"
            "    pass\n"
        )
        body = extract_file_body("note\n```python\n" + src + "```\n")
        self.assertIn("class BaselineTools", body)
        self.assertTrue(body.endswith("\n"))

    def test_excerpt_keeps_parse_and_execute(self) -> None:
        excerpt = relevant_component_excerpt(_SOURCE)
        self.assertIn("def parse_llm_response", excerpt)
        self.assertIn("def execute", excerpt)
        self.assertNotIn("def register", excerpt)


if __name__ == "__main__":
    unittest.main()
