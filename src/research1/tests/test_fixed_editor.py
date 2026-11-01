"""Regression checks for bounded implementation transport and policy locking."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research1.fixed_editor import apply_replacements, compact_editor


class FixedEditorTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        modules = self.root / "parent"
        (modules / "tools").mkdir(parents=True)
        (modules / "tools/baseline.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
        self.directory = self.root / "batch" / "attempt"
        self.state = {"modules_root": str(modules), "view": {"config": {"editor_model": "fixed"}},
                      "checks": {"allowed_targets": ["tools/baseline.py"],
                                 "allowed_changed_globals": ["x"]}}
        self.plan = {"action": "edit", "target_module": "tools", "locations": ["tools/baseline.py"],
                     "intended_change": "Use the supplied configured constant", "preserve_behaviors": ["Keep y unchanged"]}
        self.metadata = {"provider": "test", "model": "fixed", "temperature": 0.0,
                         "raw_field": "content", "finish_reason": "stop"}
        self.raw = json.dumps({"edits": [{"old": "x = 1", "new": "x = 3"}]})

    def test_multiple_replacements_use_original_offsets_and_reject_invalid_edits(self):
        source = "x = 1\ny = 2\n"
        edited = apply_replacements(source, [{"old": "1", "new": "1000"}, {"old": "2", "new": "1"}])
        self.assertEqual(edited, "x = 1000\ny = 1\n")
        with self.assertRaisesRegex(ValueError, "exactly once"):
            apply_replacements("x = 1\ny = 1\n", [{"old": "1", "new": "2"}])
        with self.assertRaisesRegex(ValueError, "Overlapping"):
            apply_replacements("x = 123\n", [{"old": "123", "new": "4"}, {"old": "23", "new": "5"}])
        with self.assertRaises(SyntaxError):
            apply_replacements(source, [{"old": "x = 1", "new": "def ("}])

    def test_supplied_plan_and_scope_drive_json_editor_without_extra_repair_recipe(self):
        with patch("research1.distill._tongji_chat", return_value=(self.raw, self.metadata)) as chat, \
             patch("research1.distill._apply_edit", return_value={"status": "applied", "child_version_id": "child"}) as apply:
            result = compact_editor(self.plan, self.state, self.directory)
        chat.assert_called_once()
        arguments = chat.call_args.kwargs
        self.assertEqual(arguments["max_tokens"], 6000)
        self.assertTrue(arguments["json_output"])
        self.assertEqual(arguments["reasoning_effort"], "none")
        self.assertEqual(apply.call_args.kwargs["replacement"], "x = 3\ny = 2\n")
        saved = json.loads((self.directory / "editor_input.json").read_text())
        self.assertEqual(saved["plan"], self.plan)
        self.assertIn("Keep y unchanged", saved["prompt"])
        self.assertNotIn("empty", saved["prompt"])
        self.assertNotIn("newline", saved["prompt"])
        self.assertEqual(Path(result["modules_root"]), self.directory / "versions/child/modules")
        self.assertEqual((Path(self.state["modules_root"]) / "tools/baseline.py").read_text(), "x = 1\ny = 2\n")

    def test_reasoning_or_truncated_outputs_are_saved_and_never_applied(self):
        for changed in ({"raw_field": "reasoning"}, {"finish_reason": "length"}):
            with self.subTest(changed=changed), \
                 patch("research1.distill._tongji_chat", return_value=(self.raw, dict(self.metadata, **changed))), \
                 patch("research1.distill._apply_edit") as apply:
                with self.assertRaisesRegex(ValueError, "complete final content"):
                    compact_editor(self.plan, self.state, self.directory)
                apply.assert_not_called()
                self.assertEqual(json.loads((self.directory / "editor_output.json").read_text())["raw"], self.raw)

    def test_provider_model_or_temperature_drift_blocks_apply_and_preserves_lock(self):
        self.directory.parent.mkdir(parents=True)
        lock = self.directory.parent / "fixed_editor_policy.json"
        signature = {key: self.metadata[key] for key in ("provider", "model", "temperature")}
        lock.write_text(json.dumps(signature))
        for changed in ({"provider": "other"}, {"model": "other"}, {"temperature": 0.7}):
            with self.subTest(changed=changed), \
                 patch("research1.distill._tongji_chat", return_value=(self.raw, dict(self.metadata, **changed))), \
                 patch("research1.distill._apply_edit") as apply:
                result = compact_editor(self.plan, self.state, self.directory)
                self.assertEqual(result["status"], "pending")
                apply.assert_not_called()
                self.assertEqual(json.loads(lock.read_text()), signature)

    def test_first_attempt_still_requires_frozen_model_and_zero_temperature(self):
        for changed in ({"model": "other"}, {"temperature": 0.7}):
            with self.subTest(changed=changed), \
                 patch("research1.distill._tongji_chat", return_value=(self.raw, dict(self.metadata, **changed))), \
                 patch("research1.distill._apply_edit") as apply:
                result = compact_editor(self.plan, self.state, self.directory)
                self.assertEqual(result["status"], "pending")
                apply.assert_not_called()
                self.assertFalse((self.directory.parent / "fixed_editor_policy.json").exists())
