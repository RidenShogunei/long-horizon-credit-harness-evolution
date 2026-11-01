import json
from pathlib import Path
import tempfile
import unittest
from research1.audit import audit_run


class ReturnAuditTests(unittest.TestCase):
    def test_missing_task_is_incomplete_and_never_writes_a_return(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / 'score_tables.json').write_text(json.dumps({
                'scores': {'H0': {'a': 0, 'b': 1}, 'A1': {'b': 1}},
                'branches': [{'branch_id': 'A', 'version_order': ['H0', 'A1']}],
            }), encoding='utf-8')
            result = audit_run(root)
            self.assertEqual(result['status'], 'INCOMPLETE')
            self.assertFalse(result['returns_complete'])
            self.assertFalse((root / 'returns').exists())

    def test_old_marker_labels_do_not_become_formal_supervision(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / 'score_tables.json').write_text(json.dumps({
                'scores': {'H0': {'a': 0}, 'A1': {'a': 1}},
                'branches': [{'branch_id': 'A', 'version_order': ['H0', 'A1'],
                              'stored_return': {'terminal_gain': 1, 'cumulative_regression': 0, 'u_lambda': 1}}],
            }), encoding='utf-8')
            result = audit_run(root)
            self.assertEqual(result['status'], 'INCOMPLETE')
            self.assertFalse(result['returns_complete'])
