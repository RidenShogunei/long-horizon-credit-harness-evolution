"""Unit tests for the fixed editor and task-selection rules. No Harbor."""

from __future__ import annotations

import unittest

import json
import tempfile
from pathlib import Path
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

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
    probe_parse_warning,
    run_dual_branch,
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
    def test_dual_branch_records_each_branch_rewards_without_leakage(self) -> None:
        """Exercise the branch loop with all model/Harbor calls replaced."""
        with tempfile.TemporaryDirectory() as td, ExitStack() as mocks:
            root = Path(td)
            parent = root / "parent"
            (parent / "tools").mkdir(parents=True)
            (parent / "tools" / "baseline.py").write_text(_BEFORE)
            for label in ("A", "B"):
                module_dir = root / f"H0_{label}1"
                (module_dir / "tools").mkdir(parents=True)
                (module_dir / "tools" / "baseline.py").write_text(_BEFORE.replace("return response", f"return '{label}'"))

            def plan(**kwargs):
                label = kwargs["label"]
                (kwargs["run_dir"] / "plans").mkdir(parents=True, exist_ok=True)
                return {"ok": True, "replacement": _BEFORE, "plan_text": label,
                        "trigger": f"TRIGGER_{label}", "diff_text": label}

            def edit(**kwargs):
                label = kwargs["plan"].candidate_label
                return SimpleNamespace(applied=True, child_version_id=kwargs["child_version_id"],
                                       attempt_id=f"attempt_{label}", reject_reason="")

            replacements = {
                "load_pilot": lambda _: {},
                "run_preflight": lambda _: SimpleNamespace(status="PASS"),
                "resolve_path": lambda cfg, name: root / name,
                "_best_false_concat": lambda _: {"kind": "false_concat", "detail": "evidence"},
                "with_llm_provider_env": lambda _: SimpleNamespace(),
                "apply_provider_to_environ": lambda _: None,
                "ensure_h0_version": lambda *args: SimpleNamespace(modules_root=str(parent)),
                "_excerpt_for_planner": lambda *args: "bounded evidence",
                "_plan_one_edit": plan,
                "make_plan": lambda **kwargs: SimpleNamespace(plan_id=f"plan_{kwargs['candidate_label']}", **kwargs),
                "apply_plan_patch": edit,
                "load_state": lambda versions, version: SimpleNamespace(modules_root=str(root / version)),
                "_eval_version_task": lambda **kwargs: [SimpleNamespace(reward=0.5)],
                "_eval_one": lambda **kwargs: SimpleNamespace(reward=0.8 if kwargs["version_id"] == "H0_A1" else 0.2),
                "_measure_trigger": lambda **kwargs: {"triggered": True, "trigger_hits": [], "loaded_modules_roots": [], "loaded_modules_root_ok": True},
                "aggregate_task_means": lambda trials: {"cron-env-debug": sum(trial.reward for trial in trials) / len(trials)},
                "write_transition": lambda *args, **kwargs: None,
                "mark_step": lambda *args: None,
                "_collect_findings": lambda _: [],
                "_write_dual_report": lambda *args: None,
            }
            for name, replacement in replacements.items():
                mocks.enter_context(patch(f"research1.causal.{name}", side_effect=replacement))
            mocks.enter_context(patch.dict("os.environ", {"NO_PROXY": "agi.tongji.edu.cn"}))
            _, result = run_dual_branch(prior_h0=parent, reps=1, max_trigger_reps=1)
            branches = {branch["label"]: branch for branch in result["branches"]}
            self.assertEqual(branches["A"]["raw_rewards"], [0.8])
            self.assertEqual(branches["B"]["raw_rewards"], [0.2])
            self.assertAlmostEqual(branches["A"]["observed_delta"], 0.3)
            self.assertAlmostEqual(branches["B"]["observed_delta"], -0.3)

    def test_parse_probe_keeps_error_commands_and_completion_fields(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "tools").mkdir()
            (root / "tools" / "baseline.py").write_text(
                "from types import SimpleNamespace\n"
                "class BaselineTools:\n"
                "    def __init__(self, parser_name): pass\n"
                "    def parse_llm_response(self, response):\n"
                "        return SimpleNamespace(warning='keep warning', error='keep error', "
                "is_task_complete=True, commands=[SimpleNamespace(keystrokes='echo hi\\n', duration_sec=2)])\n"
            )
            result = probe_parse_warning(root, {})
        self.assertEqual(result["warning"], "keep warning")
        self.assertEqual(result["error"], "keep error")
        self.assertTrue(result["is_task_complete"])
        self.assertEqual(result["commands"], [["echo hi\n", 2.0]])

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
