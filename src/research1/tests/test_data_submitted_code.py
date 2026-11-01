"""Submitted-source navigation must stay exact, generic, bounded and nonexecuting."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest

try:
    from research1.data_pipeline import submitted_code as submitted
except ImportError:
    for parent in Path(__file__).resolve().parents:
        if (parent / "server_fix" / "src").is_dir():
            sys.path.insert(0, str(parent / "server_fix" / "src"))
            break
    helper = Path(__file__).resolve().parents[1] / "data_pipeline" / "submitted_code.py"
    spec = importlib.util.spec_from_file_location("offline_submitted_code", helper)
    submitted = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(submitted)

from research1.data_pipeline.fact_cards import validate_fact_card


def make_state(command, observation="", *, segment_start=0):
    def excerpt(text, start=0):
        return {"source_chars": start + len(text), "truncated": start != 0,
                "segments": [{"start": start, "end": start + len(text), "text": text}]}
    step = {"source": "agent", "step_id": 7, "source_index": 6, "visible_step_sha256": "a" * 64,
        "observation_excerpt": excerpt(observation) if observation else {"source_chars": 0, "truncated": False, "segments": []},
        "tool_calls": [{"function_name": "bash_command", "arguments": {"keystrokes_excerpt": excerpt(command, segment_start)}}]}
    return {"plan_policy_version": "proposal_single_plan_v6", "evidence": [{"id": "original:trajectory", "partition": "proposal",
        "text": json.dumps({"steps": [step]}, ensure_ascii=False)}], "module_code": {}}


class SubmittedCodeTests(unittest.TestCase):
    def quotes(self, card):
        return [fact["quote"] for fact in card["facts"]] if card else []

    def test_all_supported_statement_kinds_are_exact_original_input_quotes(self):
        command = "cat > /app/program.py <<'PY'\nx = 1\ny: int = 2\ndef f():\n    return x + y\nassert f() == 3\nPY\n"
        state = make_state(command)
        card, notes = submitted.build_submitted_code_index(state)
        self.assertEqual(self.quotes(card), ["x = 1", "y: int = 2", "return x + y", "assert f() == 3"])
        self.assertEqual(notes["candidate_total"], 4)
        self.assertFalse(notes["truncated"])
        self.assertEqual(validate_fact_card(state, card), card)
        self.assertTrue(all(fact["source_role"] == "recorded_tool_input" for fact in card["facts"]))
        for fact in card["facts"]:
            self.assertEqual(command[fact["start"]:fact["end"]], fact["quote"])

    def test_python_stdin_and_double_quoted_or_unquoted_delimiters(self):
        command = 'python3 - <<"FIRST"\na = 1\nFIRST\npython - <<SECOND\nb = 2\nSECOND\n'
        card, notes = submitted.build_submitted_code_index(make_state(command))
        self.assertEqual(self.quotes(card), ["a = 1", "b = 2"])
        self.assertEqual(notes["recognized_blocks"], 2)
        self.assertEqual([block["delimiter_quoted"] for block in notes["blocks"]], [True, False])

    def test_utf8_ast_columns_convert_to_exact_character_offsets(self):
        command = "cat > unicode.py <<'PY'\n名称 = '雪'; result = 名称\nassert 名称 == '雪'\nPY\n"
        state = make_state(command, segment_start=100)
        card, _ = submitted.build_submitted_code_index(state)
        self.assertEqual(self.quotes(card), ["名称 = '雪'", "result = 名称", "assert 名称 == '雪'"])
        self.assertEqual(validate_fact_card(state, card), card)
        self.assertTrue(all(fact["locator"]["segment_start"] == 100 for fact in card["facts"]))
        for fact in card["facts"]:
            self.assertEqual(command[fact["start"]:fact["end"]], fact["quote"])

    def test_malformed_or_indented_closing_delimiter_is_not_repaired(self):
        for closing in (" PY", "PY ", "'PY'"):
            with self.subTest(closing=closing):
                card, notes = submitted.build_submitted_code_index(make_state("cat > x.py <<'PY'\nx = 1\n" + closing + "\n"))
                self.assertIsNone(card)
                self.assertEqual(notes["unclosed_heredocs"], 1)

    def test_invalid_python_block_is_not_partially_indexed(self):
        command = "cat > broken.py <<'PY'\nx = 1\nif:\n    y = 2\nPY\n"
        card, notes = submitted.build_submitted_code_index(make_state(command))
        self.assertIsNone(card)
        self.assertEqual(notes["syntax_invalid_blocks"], 1)
        self.assertEqual(notes["candidate_total"], 0)

    def test_terminal_echoes_and_prompt_prefixes_are_not_stripped_or_parsed(self):
        observation = "cat > observed.py <<'PY'\nx = 1\nPY\n"
        state = make_state("echo harmless\n", observation)
        self.assertIsNone(submitted.build_submitted_code_card(state))
        prefixed = "> cat > input.py <<'PY'\n> x = 1\n> PY\n"
        self.assertIsNone(submitted.build_submitted_code_card(make_state(prefixed)))

    def test_unknown_outer_heredoc_body_cannot_become_a_command(self):
        command = "cat > notes.txt <<'OUTER'\npython - <<'INNER'\nx = 1\nINNER\nOUTER\n"
        card, notes = submitted.build_submitted_code_index(make_state(command))
        self.assertIsNone(card)
        self.assertEqual(notes["unsupported_heredoc_headers"], 1)
        self.assertEqual(notes["recognized_blocks"], 0)

    def test_round_robin_cap_covers_late_blocks_without_keyword_selection(self):
        first = "".join("item_" + str(i) + " = " + str(i) + "\n" for i in range(40))
        command = "cat > many.py <<'A'\n" + first + "A\npython - <<'B'\nassert 7 == 7\nB\n"
        state = make_state(command)
        card, notes = submitted.build_submitted_code_index(state, max_facts=24)
        self.assertEqual(notes["candidate_total"], 41)
        self.assertEqual(notes["selected_count"], 24)
        self.assertTrue(notes["truncated"])
        self.assertEqual([block["selected_count"] for block in notes["blocks"]], [23, 1])
        self.assertIn("assert 7 == 7", self.quotes(card))
        self.assertEqual((card, notes), submitted.build_submitted_code_index(state, max_facts=24))
        self.assertEqual([fact["start"] for fact in card["facts"]], sorted(fact["start"] for fact in card["facts"]))

    def test_long_statements_are_counted_not_shortened(self):
        command = "cat > long.py <<'PY'\nx = '" + "z" * 601 + "'\ny = 2\nPY\n"
        card, notes = submitted.build_submitted_code_index(make_state(command))
        self.assertEqual(self.quotes(card), ["y = 2"])
        self.assertEqual(notes["statement_nodes_total"], 2)
        self.assertEqual(notes["overlength_statements"], 1)

    def test_only_ast_selected_statement_types_are_indexed_without_executing(self):
        command = "python3 - <<'PY'\nraise RuntimeError('must never run')\nprint('x = 999')\nx = 1\nPY\n"
        card, _ = submitted.build_submitted_code_index(make_state(command))
        self.assertEqual(self.quotes(card), ["x = 1"])

    def test_partial_segment_is_not_joined_or_filled_in(self):
        card, notes = submitted.build_submitted_code_index(make_state("x = 1\nPY\n", segment_start=80))
        self.assertIsNone(card)
        self.assertEqual(notes["recognized_blocks"], 0)
        card, notes = submitted.build_submitted_code_index(make_state("python - <<'PY'\nx = 1\n"))
        self.assertIsNone(card)
        self.assertEqual(notes["unclosed_heredocs"], 1)

    def test_fact_cap_obeys_existing_32_fact_card_schema(self):
        command = "python - <<PY\n" + "".join(f"x{i} = {i}\n" for i in range(40)) + "PY\n"
        card, notes = submitted.build_submitted_code_index(make_state(command), max_facts=32)
        self.assertEqual(len(card["facts"]), 32)
        self.assertTrue(notes["truncated"])
        for invalid in (0, 33, True, "24"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    submitted.build_submitted_code_card(make_state(command), max_facts=invalid)


if __name__ == "__main__":
    unittest.main()
