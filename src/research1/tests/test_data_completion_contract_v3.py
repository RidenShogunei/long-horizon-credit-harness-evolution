"""Optional v3 completion-contract probes.

Synthetic edits in this file are temporary test fixtures only. They are not
teacher examples, repair artifacts, or H0 data, and they must not be applied
outside this test.
"""
import hashlib
import inspect
from pathlib import Path
import tempfile
import unittest

from research1.data_pipeline import modules as legacy
from research1.data_pipeline.modules_v3 import frozen_registry, probe_module
from research1.tests.test_data_completion_recheck import ORIGINAL_NEXT, PARENT, RECHECK_NEXT


LEGACY_MODULES_SHA256 = "4a04740d5ae00ad847bcf3616afe00915ed81d2a4f4266b13a2e68f7d95d9b27"


def _confirmation_prefix() -> str:
    """Prefix actually returned by the miniature parent's first confirmation."""
    namespace = {}
    exec(PARENT, namespace)
    sample = namespace["BaselineAgentLoop"]()._completion_confirmation("MARKER")
    if not sample.endswith("MARKER"):
        raise AssertionError("parent confirmation shape changed")
    return sample[: -len("MARKER")]


_PREFIX = _confirmation_prefix()

# Inject only when the observation is the real first confirmation. The second
# build returns ordinary terminal text, which does not carry this prefix.
GOOD_NEXT = f'''        if not parse.is_task_complete:
            return tools.format_continuation_prompt(observation_text, ctx)
        if not str(observation_text).startswith({_PREFIX!r}):
            return observation_text
        instruction = getattr(getattr(ctx, "state", None), "instruction", "") or ""
        request = "Compare behavioral checks and untested cases with the original requirements before confirming."
        return observation_text + "\\n" + request + "\\nOriginal task:\\n" + instruction[:4096]'''

COUNTER_NEXT = '''        if not parse.is_task_complete:
            return tools.format_continuation_prompt(observation_text, ctx)
        if ctx.state.consecutive_complete_signals != 1:
            return observation_text
        instruction = getattr(getattr(ctx, "state", None), "instruction", "") or ""
        request = "Compare behavioral checks and untested cases with the original requirements before confirming."
        return observation_text + "\\n" + request + "\\nOriginal task:\\n" + instruction[:4096]'''

MUTATING_NEXT = GOOD_NEXT.replace(
    '        instruction = getattr(getattr(ctx, "state", None), "instruction", "") or ""',
    '        ctx.state.instruction = "overwritten"\n        instruction = getattr(getattr(ctx, "state", None), "instruction", "") or ""',
    1,
)


class CompletionContractV3Tests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "agent_loop").mkdir()
        self.path = self.root / "agent_loop/baseline.py"
        self.path.write_text(PARENT, encoding="utf-8")

    def probe(self, source=PARENT):
        self.path.write_text(source, encoding="utf-8")
        return probe_module(self.root, "completion_requirement_recheck")

    def test_registry_v3_and_legacy_modules_unchanged(self):
        raw = Path(legacy.__file__).read_bytes()
        self.assertEqual(hashlib.sha256(raw).hexdigest(), LEGACY_MODULES_SHA256)
        self.assertNotIn(b"modules_v3", raw)
        self.assertNotIn(b"second_completion_prompt_exact", raw)
        self.assertEqual(legacy.REGISTRY_VERSION, "research1.module-contracts.v2")
        old = legacy.frozen_registry()
        new = frozen_registry()
        self.assertEqual(old["version"], "research1.module-contracts.v2")
        self.assertEqual(new["version"], "research1.module-contracts.v3")
        self.assertEqual(set(new["packs"]), set(old["packs"]))
        for name, pack in old["packs"].items():
            if name == "completion_requirement_recheck":
                continue
            self.assertEqual(new["packs"][name], pack)
            self.assertEqual(pack["id"], name + ".v1")
        old_pack = old["packs"]["completion_requirement_recheck"]
        new_pack = new["packs"]["completion_requirement_recheck"]
        self.assertEqual(old_pack["id"], "completion_requirement_recheck.v1")
        self.assertEqual(new_pack["id"], "completion_requirement_recheck.v2")
        self.assertNotIn("second_completion_prompt_exact", old_pack["preserve_checks"])
        self.assertEqual(new_pack["goal_checks"], old_pack["goal_checks"])
        self.assertEqual(
            new_pack["preserve_checks"],
            [
                "noncomplete_prompt_exact",
                "completion_observation_preserved",
                "second_completion_prompt_exact",
                "completion_prompt_bounded",
                "instructionless_completion_safe",
                "instruction_source_unchanged",
                "two_signal_confirmation",
                "false_resets_confirmation",
                "tool_results_ordered",
                "all_tool_feedback_visible",
                "real_warning_visible",
                "parse_feedback",
                "session_dead_stops",
                "session_error_continues",
            ],
        )
        for key in (
            "module",
            "mechanism_id",
            "functions",
            "globals",
            "signal",
            "diagnostic_review_required",
            "allowed_paths",
            "allowed_imports",
            "max_files",
            "max_changed_lines",
        ):
            self.assertEqual(new_pack[key], old_pack[key])
        self.assertIn("First-completion-only requirement review", new_pack["coverage"])
        self.assertIn("8192", new_pack["coverage"])
        self.assertIn("including separators", new_pack["coverage"])
        self.assertNotIn("bloom", str(new_pack).lower())
        self.assertNotIn("payroll", str(new_pack).lower())
        self.assertIn(
            "from research1.data_pipeline.modules_v3 import _probe_in_process",
            inspect.getsource(probe_module),
        )
        self.assertIn(
            "from research1.data_pipeline.modules import _probe_in_process",
            inspect.getsource(legacy.probe_module),
        )
        self.assertNotIn("modules_v3", inspect.getsource(legacy.probe_module))

    def test_parent_fails_goal_and_passes_all_preservation(self):
        result = self.probe()
        pack = frozen_registry()["packs"]["completion_requirement_recheck"]
        self.assertFalse(result["checks"]["completion_requirements_rechecked"])
        self.assertTrue(result["checks"]["second_completion_prompt_exact"] is True)
        guards = pack["preserve_checks"]
        self.assertTrue(all(result["checks"][name] is True for name in guards), result)
        self.assertEqual(set(result["checks"]), set(guards + ["completion_requirements_rechecked"]))
        self.assertIn("not solver compliance", result["scope"])
        self.assertIn("first-completion-only", result["scope"].lower())

    def test_first_confirmation_prefix_and_frozen_instruction_passes(self):
        result = self.probe(PARENT.replace(ORIGINAL_NEXT, GOOD_NEXT))
        self.assertTrue(all(value is True for value in result["checks"].values()), result)

    def test_every_completion_injection_fails_second_prompt_exact(self):
        result = self.probe(PARENT.replace(ORIGINAL_NEXT, RECHECK_NEXT))
        self.assertTrue(result["checks"]["completion_requirements_rechecked"] is True)
        self.assertFalse(result["checks"]["second_completion_prompt_exact"])
        others = [name for name in result["checks"] if name != "second_completion_prompt_exact"]
        self.assertTrue(all(result["checks"][name] is True for name in others), result)

    def test_task_state_completion_counter_cannot_pass(self):
        result = self.probe(PARENT.replace(ORIGINAL_NEXT, COUNTER_NEXT))
        self.assertFalse(result["checks"]["completion_requirements_rechecked"])
        self.assertFalse(all(value is True for value in result["checks"].values()))

    def test_frozen_task_state_mutation_fails(self):
        result = self.probe(PARENT.replace(ORIGINAL_NEXT, MUTATING_NEXT))
        self.assertFalse(result["checks"]["instruction_source_unchanged"])
        self.assertFalse(all(value is True for value in result["checks"].values()))

    def test_oversized_injection_fails_bound(self):
        source = PARENT.replace(ORIGINAL_NEXT, GOOD_NEXT.replace("instruction[:4096]", "instruction"))
        result = self.probe(source)
        self.assertTrue(result["checks"]["completion_requirements_rechecked"] is True)
        self.assertTrue(result["checks"]["second_completion_prompt_exact"] is True)
        self.assertFalse(result["checks"]["completion_prompt_bounded"])

    def test_instruction_copy_alone_is_not_requirement_review(self):
        source = PARENT.replace(
            ORIGINAL_NEXT,
            f'''        if not parse.is_task_complete:
            return tools.format_continuation_prompt(observation_text, ctx)
        if not str(observation_text).startswith({_PREFIX!r}):
            return observation_text
        return observation_text + str(getattr(ctx.state, "instruction", "") or "")[:4096]''',
        )
        result = self.probe(source)
        self.assertFalse(result["checks"]["completion_requirements_rechecked"])
        self.assertTrue(result["checks"]["second_completion_prompt_exact"] is True)

    def test_continuation_corruption_fails_preservation(self):
        changed = GOOD_NEXT.replace(
            "return tools.format_continuation_prompt(observation_text, ctx)",
            'return "discarded terminal"',
        )
        result = self.probe(PARENT.replace(ORIGINAL_NEXT, changed))
        self.assertTrue(result["checks"]["completion_requirements_rechecked"] is True)
        self.assertFalse(result["checks"]["noncomplete_prompt_exact"])

    def test_unknown_mechanism_remains_unsupported(self):
        with self.assertRaises(RuntimeError):
            probe_module(self.root, "not_a_mechanism")


if __name__ == "__main__":
    unittest.main()
