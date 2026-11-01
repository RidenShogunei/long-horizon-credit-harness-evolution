"""Admission boundaries: old collectors cannot silently mint training data."""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

from research1 import EXIT_FAIL, EXIT_INCOMPLETE
from research1.cli import build_parser
from research1.value_io import label_continuations


class CollectionSafetyTests(unittest.TestCase):
    def test_completed_mc_paths_write_one_idempotent_target(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / 'continuations.jsonl'
            target = Path(td) / 'targets.jsonl'
            rows = []
            for seed, score in ((1, 0.8), (2, 0.6)):
                leaf = f'A2_{seed}'
                versions = ['H0', 'A1', leaf]
                rows.append({'path_id': f'path_{seed}', 'candidate_id': 'A', 'policy_id': 'frozen',
                             'horizon': 1, 'first_implementation_hash': 'a' * 64, 'continuation_seed': seed,
                             'version_order': versions, 'score_tables': {'H0': {'t': .5}, 'A1': {'t': .7}, leaf: {'t': score}},
                             'trial_counts': {v: {'t': 1} for v in versions},
                             'expected_tasks': ['t'], 'required_repetitions': 1})
            source.write_text(''.join(json.dumps(row) + '\n' for row in rows), encoding='utf-8')
            for _ in range(2):
                code, _ = label_continuations(source, target)
                self.assertEqual(code, 0)
            labels = [json.loads(row) for row in target.read_text(encoding='utf-8').splitlines()]
            self.assertEqual(len(labels), 1)
            self.assertEqual(labels[0]['label']['n_paths'], 2)
            self.assertAlmostEqual(labels[0]['label']['mean_terminal_gain'], .2)

    def test_warning_only_entry_points_require_explicit_legacy_opt_in(self):
        for command in ('distill-verified', 'distill-observed', 'distill-observed-batches', 'distill-llm-review'):
            with self.subTest(command=command):
                args = build_parser().parse_args([command])
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    result = args.func(args)
                self.assertEqual(result, EXIT_FAIL)
                self.assertIn('legacy', json.loads(output.getvalue())['reason'])

    def test_absent_continuations_do_not_create_a_target_file(self):
        with tempfile.TemporaryDirectory() as td:
            target = Path(td) / 'targets.jsonl'
            code, _ = label_continuations(Path(td) / 'missing.jsonl', target)
            self.assertEqual(code, EXIT_INCOMPLETE)
            self.assertFalse(target.exists())

    def test_incomplete_continuation_input_leaves_previous_targets_unchanged(self):
        with tempfile.TemporaryDirectory() as td:
            source = Path(td) / 'continuations.jsonl'
            target = Path(td) / 'targets.jsonl'
            source.write_text(json.dumps({'candidate_id': 'A'}) + '\n', encoding='utf-8')
            prior = json.dumps({'label_id': 'prior', 'note': 'preserve previous results'}) + '\n'
            target.write_text(prior, encoding='utf-8')
            code, _ = label_continuations(source, target)
            self.assertEqual(code, EXIT_INCOMPLETE)
            self.assertEqual(target.read_text(encoding='utf-8'), prior)
