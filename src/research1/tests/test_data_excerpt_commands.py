"""Only provable retained command lines can add a context downgrade."""
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("excerpt_commands_under_test", Path(__file__).resolve().parents[1] / "data_pipeline/excerpt_commands.py")
guard = importlib.util.module_from_spec(spec); spec.loader.exec_module(guard)


def excerpt(text, *, start=0, source_chars=None):
    value = {"segments": [{"start": start, "end": start + len(text), "text": text}]}
    if source_chars is not None: value["source_chars"] = source_chars
    return value


def step(command, observation, *, command_start=0, command_chars=None, observation_start=0, observation_chars=None):
    return {"observation_excerpt": excerpt(observation, start=observation_start, source_chars=observation_chars),
            "tool_calls": [{"arguments": {"keystrokes_excerpt": excerpt(command, start=command_start, source_chars=command_chars)}}]}


class ExcerptCommandTests(unittest.TestCase):
    def test_excerpt_only_original_counterexample_downgrades_exact_complete_line(self):
        value = step("pytest -q\n", "pytest -q\n")
        self.assertEqual(guard.known_complete_keystroke_lines(value), {"pytest -q"})
        self.assertTrue(guard.observation_excerpt_command_only(value, 0))

    def test_arbitrary_command_looking_observation_is_not_inferred_to_be_command(self):
        self.assertFalse(guard.observation_excerpt_command_only(step("ls\n", "pytest -q\n"), 0))
        self.assertFalse(guard.observation_excerpt_command_only(step("", "pytest -q\n"), 0))

    def test_partial_prefix_or_suffix_line_does_not_supply_known_input(self):
        prefix = step("pytest -q", "pytest -q\n", command_chars=99)
        suffix = step("pytest -q\n", "pytest -q\n", command_start=21, command_chars=99)
        self.assertFalse(guard.observation_excerpt_command_only(prefix, 0))
        self.assertFalse(guard.observation_excerpt_command_only(suffix, 0))

    def test_newline_inside_fragment_proves_later_complete_lines(self):
        value = step("unknown prefix\npytest -q\nunknown suffix", "pytest -q\n", command_start=21, command_chars=999)
        self.assertEqual(guard.known_complete_keystroke_lines(value), {"pytest -q"})
        self.assertTrue(guard.observation_excerpt_command_only(value, 0))

    def test_explicit_source_end_allows_complete_final_line_without_newline(self):
        value = step("pytest -q", "pytest -q", command_chars=9, observation_chars=9)
        self.assertTrue(guard.observation_excerpt_command_only(value, 0))
        value["tool_calls"][0]["arguments"]["keystrokes_excerpt"].pop("source_chars")
        self.assertFalse(guard.observation_excerpt_command_only(value, 0))

    def test_fragments_are_never_joined_even_when_adjacent(self):
        value = step("", "pytest -q\n")
        value["tool_calls"][0]["arguments"]["keystrokes_excerpt"] = {"source_chars": 10, "segments": [
            {"start": 0, "end": 7, "text": "pytest "}, {"start": 7, "end": 10, "text": "-q\n"}]}
        self.assertEqual(guard.known_complete_keystroke_lines(value), set())
        self.assertFalse(guard.observation_excerpt_command_only(value, 0))

    def test_mixed_real_output_is_never_downgraded_as_command_only(self):
        value = step("pytest -q\n", "New Terminal Output:\nroot@host:/app# pytest -q\npytest -q\n4 passed in 0.3s\n")
        self.assertFalse(guard.observation_excerpt_command_only(value, 0))
        value["observation_excerpt"] = excerpt("pytest -q\nreturned payload: 7\n")
        self.assertFalse(guard.observation_excerpt_command_only(value, 0))

    def test_existing_decorations_and_exact_known_line_only_are_context(self):
        value = step("pytest -q\n", "New Terminal Output:\nroot@host:/app#\n>\npytest -q\n")
        self.assertTrue(guard.observation_excerpt_command_only(value, 0))

    def test_partial_observation_line_is_not_newly_downgraded(self):
        value = step("pytest -q\n", "pytest -q\n", observation_start=22, observation_chars=99)
        self.assertFalse(guard.observation_excerpt_command_only(value, 0))
        value = step("pytest -q\n", "pytest -q", observation_chars=999)
        self.assertFalse(guard.observation_excerpt_command_only(value, 0))

    def test_invalid_offsets_do_not_create_known_lines(self):
        value = step("pytest -q\n", "pytest -q\n")
        value["tool_calls"][0]["arguments"]["keystrokes_excerpt"]["segments"][0]["end"] = 2
        self.assertFalse(guard.observation_excerpt_command_only(value, 0))


if __name__ == "__main__": unittest.main()
