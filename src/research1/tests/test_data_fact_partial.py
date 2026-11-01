"""Regression for v8: eleven exact facts survive one rewritten terminal quote."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("fact_partial_under_test", ROOT / "data_pipeline/fact_cards.py")
cards = importlib.util.module_from_spec(spec); spec.loader.exec_module(cards)
fixture_spec = importlib.util.spec_from_file_location("frozen_v8_facts", Path(__file__).with_name("test_data_fact_partial_fixture.py"))
fixture = importlib.util.module_from_spec(fixture_spec); fixture_spec.loader.exec_module(fixture)


class PartialFactTests(unittest.TestCase):
    def setUp(self):
        self.state = copy.deepcopy(fixture.STATE)
        self.sources = cards.flat_proposal_sources(self.state)
        self.raw = fixture.OLD_OUTPUT["raw"]

    def test_real_row_11_failure_keeps_all_eleven_original_exact_facts(self):
        result = cards.parse_fact_reply(self.state, self.sources, self.raw)
        self.assertEqual(result["accepted_count"], 11)
        self.assertEqual(result["rejected_rows"], [{"index": 11, "reason": "nonliteral_quote"}])
        self.assertEqual(result["status"], "PARTIAL")
        raw_rows = json.loads(self.raw)["quotes"]
        self.assertEqual([fact["quote"] for fact in result["card"]["facts"]], [row["quote"] for row in raw_rows[:11]])
        self.assertTrue(any("normalized" in fact["quote"] for fact in result["card"]["facts"]))

    def test_ten_valid_rows_survive_each_independent_row_failure(self):
        original = json.loads(self.raw)["quotes"][:10]
        for bad, reason in (({"source_id": "unknown", "quote": "literal"}, "unknown_source"),
                            ({"source_id": original[0]["source_id"], "quote": ""}, "invalid_quote_type_or_length"),
                            ({"source_id": original[0]["source_id"], "quote": "x" * 601}, "invalid_quote_type_or_length"),
                            ({"source_id": original[0]["source_id"], "quote": "literal", "diagnosis": "new claim"}, "malformed_row_or_extra_claim")):
            result = cards.parse_fact_reply(self.state, self.sources, json.dumps({"quotes": [*original, bad]}))
            self.assertEqual(result["accepted_count"], 10)
            self.assertEqual(result["rejected_rows"], [{"index": 10, "reason": reason}])

    def test_top_level_schema_still_rejects_whole_reply(self):
        for raw in ("{", json.dumps({"quotes": [], "diagnosis": "new claim"}), json.dumps({"quotes": [None] * 13})):
            with self.assertRaises(ValueError): cards.parse_fact_reply(self.state, self.sources, raw)

    def old_directory(self, root):
        old = root / "old"; old.mkdir()
        (old / "fact_input.json").write_text(json.dumps(fixture.OLD_INPUT, indent=2) + "\n", encoding="utf-8")
        (old / "fact_output.json").write_text(json.dumps(fixture.OLD_OUTPUT, indent=2) + "\n", encoding="utf-8")
        path = root / "old_state.json"; path.write_text(json.dumps(self.state), encoding="utf-8")
        return old, path

    def test_replay_is_no_model_call_byte_exact_idempotent_and_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); old, old_state = self.old_directory(root); new = root / "new"
            old_input, old_output = (old / "fact_input.json").read_bytes(), (old / "fact_output.json").read_bytes()
            with patch("research1.distill._tongji_chat") as chat:
                one = cards.replay_fact_card(self.state, old, new, old_state_path=old_state)
                two = cards.replay_fact_card(self.state, old, new, old_state_path=old_state)
            chat.assert_not_called(); self.assertEqual(one, two)
            self.assertEqual((new / "fact_input.json").read_bytes(), old_input)
            self.assertEqual((new / "fact_output.json").read_bytes(), old_output)
            self.assertEqual((old / "fact_output.json").read_bytes(), old_output)
            report = json.loads((new / "fact_validation.json").read_text())
            self.assertEqual(report["status"], "PARTIAL"); self.assertEqual(report["accepted_count"], 11)
            self.assertEqual(report["provenance"]["new_model_calls"], 0)
            self.assertFalse(report["provenance"]["raw_reply_rewritten"])

    def test_replay_rejects_even_serialized_envelope_whitespace_mutation(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); old, old_state = self.old_directory(root)
            changed = copy.deepcopy(self.state); changed["evidence"][0]["text"] += " "
            self.assertEqual(cards.flat_proposal_sources(changed), self.sources)
            with patch("research1.distill._tongji_chat") as chat:
                with self.assertRaisesRegex(ValueError, "serialized proposal-evidence bytes"):
                    cards.replay_fact_card(changed, old, root / "new", old_state_path=old_state)
            chat.assert_not_called()

    def test_replay_rejects_old_prompt_or_metadata_mutation(self):
        for change in ("prompt", "metadata"):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp); old, old_state = self.old_directory(root)
                path = old / ("fact_input.json" if change == "prompt" else "fact_output.json")
                saved = json.loads(path.read_text())
                if change == "prompt": saved["prompt"] += "changed"
                else: saved["metadata"]["model"] = "another"
                path.write_text(json.dumps(saved))
                with self.assertRaises(ValueError): cards.replay_fact_card(self.state, old, root / "new", old_state_path=old_state)

    def test_collect_cached_raw_preserves_valid_subset_without_repair(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); old, _ = self.old_directory(root)
            original = (old / "fact_output.json").read_bytes()
            with patch("research1.distill._tongji_chat") as chat:
                card = cards.collect_fact_card(self.state, old)
            chat.assert_not_called(); self.assertEqual(len(card["facts"]), 11)
            self.assertEqual((old / "fact_output.json").read_bytes(), original)
            self.assertEqual(json.loads((old / "fact_card.json").read_text())["validation_status"], "PARTIAL")

    def test_full_projection_is_unchanged_until_explicit_compact_opt_in(self):
        self.state["fact_index"] = [cards.build_interface_card(self.state)]
        old = cards.project_fact_index(self.state)
        self.assertEqual(old, {"schema": cards.SCHEMA, "cards": self.state["fact_index"], "limitations": list(cards.LIMITS)})
        self.state["fact_index_projection"] = "compact_v1"
        compact = cards.project_fact_index(self.state)
        self.assertEqual(compact["projection"], "compact_v1")
        self.assertLess(len(cards._canonical(compact)), 15_000)
        self.assertGreater(len(cards._canonical(old)), 40_000)
        for raw_fact, reduced in zip(old["cards"][0]["facts"], compact["cards"][0]["facts"]):
            self.assertEqual([row["signature"] for row in raw_fact["definitions"]], [row["text"] for row in reduced["signatures"]])
            self.assertEqual([row["symbol"] for row in raw_fact["definitions"]], [row["symbol"] for row in reduced["signatures"]])
            self.assertEqual(raw_fact["lexical_references_truncated"], reduced["lexical_references_truncated"])

    def test_compact_still_validates_full_card_and_preserves_original_quotations(self):
        result = cards.parse_fact_reply(self.state, self.sources, self.raw)
        self.state["fact_index"] = [result["card"]]
        self.state["fact_index_projection"] = "compact_v1"
        reduced = cards.project_fact_index(self.state)
        self.assertEqual([fact["quote"] for fact in reduced["cards"][0]["facts"]], [fact["quote"] for fact in result["card"]["facts"]])
        self.state["fact_index"][0]["facts"][0]["quote"] += "paraphrase"
        with self.assertRaises(ValueError): cards.project_fact_index(self.state)


if __name__ == "__main__": unittest.main()
