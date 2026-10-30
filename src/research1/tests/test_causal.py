"""Unit tests for the fixed editor and task-selection rules. No Harbor."""

from __future__ import annotations

import unittest

import json
import tempfile
from pathlib import Path

from research1.causal import (
    BRANCH_STRATEGIES,
    continuation_finding,
    editor_from_replacement,
    false_concat_findings,
    functional_ast_changed,
    is_heavy_source_build,
    is_private_base_image,
    outcome_for,
    parse_planner_output,
    score_direction,
    select_expansion_tasks,
    sufficient_findings,
    FILE_BEGIN,
    FILE_END,
)

_BEFORE = '''\
class BaselineTools:
    def parse_llm_response(self, response: str):
        return response

    def execute(self, call, ctx):
        return call

def register(library):
    library.register(type_="tools")
'''


class CausalEditorTests(unittest.TestCase):
    def test_comment_only_is_rejected(self) -> None:
        after = "# research1-p2b-marker: A1\n" + _BEFORE
        changed, reason = functional_ast_changed(_BEFORE, after)
        self.assertFalse(changed)
        self.assertEqual(reason, "comment-or-docstring-only")
        diff, err = editor_from_replacement(
            before=_BEFORE,
            replacement=after,
            trigger="duration was capped at 60s",
        )
        self.assertIsNone(diff)
        self.assertTrue(err)

    def test_functional_string_is_accepted(self) -> None:
        after = _BEFORE.replace(
            "return response",
            'return "tools: duration was capped at 60s"',
        )
        diff, err = editor_from_replacement(
            before=_BEFORE,
            replacement=after,
            trigger="tools: duration was capped at 60s",
        )
        self.assertEqual(err, "")
        self.assertIsNotNone(diff)
        assert diff is not None
        self.assertIn("tools/baseline.py", diff)
        self.assertIn("duration was capped at 60s", diff)

    def test_planner_markers_round_trip(self) -> None:
        raw = (
            "PLAN:\nfix the cap feedback\n"
            "TRIGGER: tools: duration was capped at 60s\n"
            f"{FILE_BEGIN}\n"
            "print(1)\n"
            f"{FILE_END}\n"
        )
        parsed = parse_planner_output(raw)
        self.assertEqual(parsed["trigger"], "tools: duration was capped at 60s")
        self.assertIn("print(1)", parsed["file_text"])
        self.assertIn("fix the cap", parsed["plan_text"])

    def test_score_direction_equal_is_not_no_behavior(self) -> None:
        self.assertEqual(score_direction(0.0), "no_score_change_measured")
        self.assertEqual(score_direction(0.25), "improved")
        self.assertEqual(score_direction(-0.25), "regressed")
        self.assertEqual(
            outcome_for(applied=True, triggered=True, direction="no_score_change_measured"),
            "success",
        )
        self.assertEqual(
            outcome_for(applied=True, triggered=False, direction="no_score_change_measured"),
            "no-effect",
        )
        self.assertEqual(
            outcome_for(applied=False, triggered=False, direction="improved"),
            "apply-failure",
        )

    def test_selection_skips_heavy_covered_private(self) -> None:
        train = ["cron", "ffmpeg", "complex", "draft", "browser", "wasm", "magick", "crypto", "next"]
        sizes = {
            "cron": 10,
            "ffmpeg": 20,
            "complex": 30,
            "draft": 40,
            "browser": 50,
            "wasm": 60,
            "magick": 70,
            "crypto": 80,
            "next": 5,
        }
        chosen = select_expansion_tasks(
            train,
            sizes=sizes,
            heavy={"ffmpeg", "magick"},
            covered={"cron", "browser"},
            private={"draft"},
            window_n=8,
            limit=3,
        )
        self.assertEqual(chosen, ["complex", "wasm", "crypto"])

    def test_false_concat_requires_empty_keystrokes(self) -> None:
        payload = {
            "steps": [
                {
                    "source": "agent",
                    "step_id": 3,
                    "tool_calls": [
                        {"arguments": {"keystrokes": "echo ok\n", "duration": 0.1}},
                        {"arguments": {"keystrokes": "", "duration": 30}},
                    ],
                    "observation": {
                        "results": [
                            {
                                "content": (
                                    "Previous response had warnings:\n"
                                    "WARNINGS: - Command 2 should end with newline when followed "
                                    "by another command.\n\nNew Terminal Output:\nroot@host:/app# "
                                )
                            }
                        ]
                    },
                }
            ]
        }
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "trajectory.json"
            path.write_text(json.dumps(payload))
            found = false_concat_findings(path)
        self.assertEqual(len(found), 1)
        self.assertTrue(found[0]["sufficient"])
        self.assertEqual(found[0]["kind"], "false_concat_warning_on_empty_keystrokes")

    def test_image_rules(self) -> None:
        self.assertTrue(is_heavy_source_build("RUN apt-get install -y build-essential\n"))
        self.assertFalse(is_heavy_source_build("FROM python:3.13-slim\n"))
        self.assertTrue(is_private_base_image("skylensage-registry.cn-hangzhou.cr.aliyuncs.com/instances/1124:t"))
        self.assertFalse(is_private_base_image("python:3.13-slim-bookworm"))

    def test_branch_strategies_are_distinct(self) -> None:
        self.assertIn("FALSE_CONCAT_SUPPRESSED", BRANCH_STRATEGIES["A"])
        self.assertIn("EMPTY_WAIT_NORMALIZED", BRANCH_STRATEGIES["B"])
        self.assertNotEqual(BRANCH_STRATEGIES["A"], BRANCH_STRATEGIES["B"])

    def test_continuation_skips_handled_false_concat_when_absent(self) -> None:
        findings = [
            {
                "kind": "newline_warning_on_wait",
                "sufficient": False,
                "detail": "already classified as non-defect when not empty",
            }
        ]
        self.assertEqual(sufficient_findings(findings), [])
        self.assertIsNone(
            continuation_finding(
                findings,
                already_handled_kinds={"false_concat_warning_on_empty_keystrokes"},
            )
        )

    def test_continuation_keeps_residual_false_concat(self) -> None:
        findings = [
            {
                "kind": "false_concat_warning_on_empty_keystrokes",
                "sufficient": True,
                "detail": "still present after incomplete fix",
            }
        ]
        got = continuation_finding(
            findings,
            already_handled_kinds={"false_concat_warning_on_empty_keystrokes"},
        )
        self.assertIsNotNone(got)
        assert got is not None
        self.assertEqual(got["kind"], "false_concat_warning_on_empty_keystrokes")


if __name__ == "__main__":
    unittest.main()
