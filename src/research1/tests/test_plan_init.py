"""Admission rules for one plan-init sample. No network."""

from __future__ import annotations

import unittest

from research1.plan_init import (
    classify_coverage,
    coverage_skip_reason,
    frozen_false_concat_checks,
    judge_behavior,
    observed_delta,
    sft_record_allowed,
)


def _decision(*, behavior: dict, status: str = "PASS", reward: float | None = 0.0, action: str = "edit") -> dict:
    return {
        "action": action,
        "behavior": behavior,
        "child_task_run": {"status": status, "reward": reward},
    }


def _probes(warnings: dict) -> dict:
    return {check["id"]: {"warning": warnings.get(check["id"],
            "Command 1 should end with newline" if check["id"] in {"nonempty_space", "nonempty_tab"} else ""),
            "commands": [[c["keystrokes"], min(c["duration"], 60)] for c in check["payload"]["commands"]],
            "error": "", "is_task_complete": False}
            for check in frozen_false_concat_checks()}


class PlanInitGateTests(unittest.TestCase):
    def test_space_and_tab_are_nonempty_inputs_with_real_warnings(self) -> None:
        checks = frozen_false_concat_checks()
        before = _probes({"empty_wait_mid_batch": "should end with newline"})
        after = _probes({"nonempty_missing_newline": "should end with newline",
                         "mixed_empty_and_real": "Command 2 should end with newline",
                         "nonempty_space": "", "nonempty_tab": ""})
        result = judge_behavior(parent_warnings=before, child_warnings=after,
                                checks=checks, edit_applied=True)
        self.assertFalse(result["sft_eligible"])
        self.assertTrue(result["harmful_behavior"])
        self.assertEqual({c["id"] for c in result["checks"] if not c["ok"]},
                         {"nonempty_space", "nonempty_tab"})

    def test_fix_and_preservation_admit_without_score_gain(self) -> None:
        checks = frozen_false_concat_checks()
        parent = {
            "empty_wait_mid_batch": "Command 2 should end with newline",
            "nonempty_missing_newline": "Command 1 should end with newline",
            "well_formed_commands": "",
            "mixed_empty_and_real": "Command 1 should end with newline\nCommand 2 should end with newline",
        }
        child = {
            "empty_wait_mid_batch": "",
            "nonempty_missing_newline": "Command 1 should end with newline",
            "well_formed_commands": "",
            "mixed_empty_and_real": "Command 2 should end with newline",
        }
        behavior = judge_behavior(
            parent_warnings=_probes(parent),
            child_warnings=_probes(child),
            checks=checks,
            edit_applied=True,
        )
        self.assertTrue(behavior["sft_eligible"])
        self.assertFalse(behavior["harmful_behavior"])
        decision = _decision(behavior=behavior, reward=0.0)
        self.assertTrue(sft_record_allowed(decision))
        self.assertEqual(observed_delta(0.0, 0.0), 0.0)

    def test_wiping_real_warnings_is_harmful_and_not_sft(self) -> None:
        checks = frozen_false_concat_checks()
        parent = {
            "empty_wait_mid_batch": "should end with newline",
            "nonempty_missing_newline": "should end with newline",
            "well_formed_commands": "",
            "mixed_empty_and_real": "Command 1 should end with newline\nCommand 2 should end with newline",
        }
        child = {key: "" for key in parent}
        behavior = judge_behavior(
            parent_warnings=_probes(parent),
            child_warnings=_probes(child),
            checks=checks,
            edit_applied=True,
        )
        self.assertTrue(behavior["harmful_behavior"])
        self.assertFalse(behavior["sft_eligible"])
        self.assertFalse(sft_record_allowed(_decision(behavior=behavior, reward=1.0)))

    def test_incomplete_task_run_does_not_promote(self) -> None:
        behavior = {"sft_eligible": True, "harmful_behavior": False}
        decision = _decision(behavior=behavior, status="INCOMPLETE", reward=None)
        self.assertFalse(sft_record_allowed(decision))
        self.assertIsNone(observed_delta(0.0, None))

    def test_private_base_without_a_local_image_is_skipped(self) -> None:
        text = "FROM skylensage-registry.cn-hangzhou.cr.aliyuncs.com/instances/1124:t-bench-python-3-13\n"
        self.assertIn("private", coverage_skip_reason(text, set()))
        self.assertEqual(coverage_skip_reason(text, {text.split()[1]}), "")
        self.assertEqual(coverage_skip_reason("FROM python:3.13-slim-bookworm\n", set()), "")

    def test_missing_trajectory_with_a_score_is_keep(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as td:
            got = classify_coverage(Path(td), 0.0)
        self.assertEqual(got["action"], "keep")
        self.assertEqual(got["mechanisms"], [])

    def test_keep_is_not_an_sft_edit(self) -> None:
        behavior = {"sft_eligible": False, "harmful_behavior": False}
        decision = _decision(behavior=behavior, action="keep")
        self.assertFalse(sft_record_allowed(decision))

    def test_mixed_real_warning_must_survive(self) -> None:
        checks = frozen_false_concat_checks()
        before = _probes({"empty_wait_mid_batch": "should end with newline",
                          "nonempty_missing_newline": "should end with newline",
                          "mixed_empty_and_real": "Command 1 should end with newline\nCommand 2 should end with newline"})
        after = _probes({"nonempty_missing_newline": "should end with newline"})
        result = judge_behavior(parent_warnings=before, child_warnings=after, checks=checks, edit_applied=True)
        self.assertFalse(result["sft_eligible"])
        self.assertTrue(result["harmful_behavior"])

    def test_changed_commands_cannot_pass_warning_only_checks(self) -> None:
        before = _probes({"empty_wait_mid_batch": "should end with newline"})
        after = _probes({"nonempty_missing_newline": "should end with newline",
                         "mixed_empty_and_real": "Command 2 should end with newline"})
        after["well_formed_commands"]["commands"] = []
        result = judge_behavior(parent_warnings=before, child_warnings=after,
                                checks=frozen_false_concat_checks(), edit_applied=True)
        self.assertFalse(result["sft_eligible"])

    def test_legacy_warning_strings_are_insufficient_for_readmission(self) -> None:
        result = judge_behavior(parent_warnings={"empty_wait_mid_batch": "should end with newline"},
                                child_warnings={"empty_wait_mid_batch": ""},
                                checks=frozen_false_concat_checks(), edit_applied=True)
        self.assertFalse(result["sft_eligible"])
        self.assertFalse(result["complete_contract"])


if __name__ == "__main__":
    unittest.main()
