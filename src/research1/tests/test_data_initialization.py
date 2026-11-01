"""Independent SFT collection must not inherit the MC noise prerequisite."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from research1.data_pipeline.initialization import run_initialization
from research1.data_pipeline.execution import tree_hash
from research1.data_pipeline.ledger import Ledger
from research1.data_pipeline.util import digest


class InitializationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name); self.modules = self.root / 'modules'
        for name in ('agent_loop', 'observation', 'tools', 'context_mgmt', 'verification'):
            path = self.modules / name; path.mkdir(parents=True)
            (path / 'baseline.py').write_text('x = 1\n')
        self.campaign = self.root / 'campaign'; self.campaign.mkdir()
        self.protocol = {'protocol_id': 'test-init-v1', 'execution_mode': 'initialization',
            'modules_root': str(self.modules), 'modules_hash': tree_hash(self.modules),
            'policies': {}, 'budget': {'regular': 18, 'retries': 2, 'total': 20},
            'task_sets': {'proposal': ['first', 'second', 'heldout'],
                'proposal_train': ['first', 'second'], 'proposal_validation': ['heldout'],
                'reward': ['reward'], 'dev': ['dev'], 'debug': ['debug']},
            'initialization': {'max_parents': 3, 'task_order': ['first', 'second', 'heldout'], 'max_candidates': 2},
            'evidence_budget': {'max_total_chars': 96000, 'max_step_chars': 16000, 'max_steps': 12}}
        (self.campaign / 'protocol.json').write_text(json.dumps(self.protocol))
        self.calls, self.teacher_calls, self.edit_calls = [], [], []
        self.protocol_patch = patch('research1.data_pipeline.protocol.load_protocol', return_value=self.protocol)
        self.protocol_patch.start(); self.addCleanup(self.protocol_patch.stop)
        self.contract_patch = patch('research1.data_pipeline.modules.build_contract',
            return_value={'checks_id': 'fixture', 'selected': []})
        self.contract_patch.start(); self.addCleanup(self.contract_patch.stop)

    def evaluator(self, campaign, protocol, ledger, code, task, role, scope, repeat):
        self.assertEqual(role, 'proposal'); self.assertIn(task, self.protocol['task_sets']['proposal'])
        key = digest({'code': tree_hash(code), 'task': task, 'role': role, 'scope': scope, 'repeat': repeat})
        old = ledger.get_job(key)
        if old:
            return old['result']
        self.calls.append((task, role, scope)); ledger.reserve_job(key, {'role': role}); ledger.start_job(key, 'test')
        result = {'status': 'PASS', 'reward': 0.0, 'failure_layer': 'model', 'evaluation_id': key,
            'evaluation_spec': {'role': role, 'task': task, 'modules_root': str(code), 'code_hash': tree_hash(code)},
            'evidence': [{'id': 'evidence', 'text': 'Executed expected command and observed output'}]}
        ledger.complete_job(key, result)
        return result

    def plan(self, state, action='edit'):
        return {'action': action, 'target_module': 'tools', 'locations': ['tools/baseline.py'] if action == 'edit' else [],
            'evidence_refs': [state['evidence'][0]['id']], 'problem_hypothesis': 'Observed configured constant',
            'intended_change': 'Use configured constant' if action == 'edit' else 'Keep observed behavior',
            'preserve_behaviors': ['Preserve command transport'], 'verification_plan': ['Frozen fixture'],
            'verification_kind': 'configured_constant'}

    def proposer(self, state, directory):
        self.teacher_calls.append(copy.deepcopy(state))
        self.assertNotIn('reward', json.dumps(state['history']))
        self.assertEqual(state['proposal_task_ids'], [state['evidence'][0]['proposal_task_id']])
        return [self.plan(state), self.plan(state, 'keep')]

    def implementer(self, plan, state, directory):
        # Every teacher candidate must already be persisted before any child.
        pending = Ledger(self.campaign).records('samples')
        own = [r for r in pending if r['root_lineage_id'] == state['root_lineage_id']]
        self.assertEqual(len(own), 2)
        snapshot = json.loads((self.campaign / 'status.json').read_text())
        self.assertGreaterEqual(snapshot['counts']['raw_plans'], 2)
        self.edit_calls.append(str(directory))
        child = directory / 'child'; shutil.copytree(state['modules_root'], child)
        (child / 'tools/baseline.py').write_text('x = 2\n')
        patch_path = directory / 'edit.diff'; patch_path.write_text('fixture patch\n')
        return {'status': 'applied', 'modules_root': str(child), 'code_hash': tree_hash(child),
            'parent_code_hash': state['code_hash'], 'patch_path': str(patch_path),
            'patch_sha256': hashlib.sha256(patch_path.read_bytes()).hexdigest()}

    def verifier(self, *args):
        return {'status': 'PASS', 'covers_plan': True, 'executable': True,
            'checks_id': 'fixture', 'mechanism_id': 'configured_constant'}

    def reviewer(self, plan, state):
        self.assertNotIn('qualification', state)
        return {'status': 'PASS', 'reason': 'Independent fixture factual review', 'independent': True,
            'reviewer_id': 'fixture-review', 'evidence_refs': [state['evidence'][0]['id']]}

    def run_init(self, **extra):
        kwargs = {'evaluator': self.evaluator, 'proposer': self.proposer, 'implementer': self.implementer,
            'verifier': self.verifier, 'reviewer': self.reviewer, 'diagnostician': lambda state, directory: []}
        kwargs.update(extra)
        return run_initialization(self.campaign, **kwargs)

    def test_zero_score_plans_persist_without_any_noise_reward_or_dev_and_resume(self):
        result = self.run_init()
        self.assertEqual(result['status'], 'INIT_BATCH_COMPLETE', result)
        self.assertFalse(result['calibration_complete']); self.assertFalse(result['data_collection_complete'])
        self.assertEqual(result['counts']['raw_plans'], 6)
        self.assertEqual(result['counts']['qualified_edit'], 3)
        self.assertEqual(result['counts']['qualified_keep'], 3)
        self.assertEqual(len(self.calls), 6)
        self.assertTrue(all(role == 'proposal' for task, role, scope in self.calls))
        self.assertEqual({s['split'] for s in self.teacher_calls}, {'train', 'validation'})
        before = (len(self.calls), len(self.teacher_calls), len(self.edit_calls))
        again = self.run_init()
        self.assertEqual(again['status'], 'INIT_BATCH_COMPLETE')
        self.assertEqual(before, (len(self.calls), len(self.teacher_calls), len(self.edit_calls)))
        pointer = json.loads((self.campaign / 'exports/sft/CURRENT.json').read_text())
        manifest = json.loads(Path(pointer['manifest']).read_text())
        self.assertFalse(manifest['training_ready'])
        self.assertEqual(manifest['counts']['train']['edit'], 2)
        self.assertEqual(Ledger(self.campaign).records('values'), [])

    def test_operational_unscored_parent_does_not_block_next_frozen_source(self):
        failed_calls = []
        def evaluate(*args):
            if args[4] == 'first':
                failed_calls.append(args[6])
                return {'status': 'INCOMPLETE', 'reward': None, 'failure_layer': 'infrastructure',
                    'wall_limit_reached': True, 'retryable': False}
            return self.evaluator(*args)
        first = self.run_init(evaluator=evaluate)
        self.assertEqual(first['status'], 'INIT_BATCH_PARTIAL')
        self.assertEqual(first['counts']['failed_parents'], 1)
        self.assertEqual(first['counts']['raw_plans'], 4)
        self.assertEqual(len(failed_calls), 1)
        self.run_init(evaluator=evaluate)
        self.assertEqual(len(failed_calls), 1)
        self.assertNotIn('first', {s['proposal_task_ids'][0] for s in self.teacher_calls})

    def test_global_auth_failure_stops_queue_before_any_teacher(self):
        result = self.run_init(evaluator=lambda *args: {'status': 'INCOMPLETE', 'reward': None,
            'failure_layer': 'infrastructure', 'fatal': True, 'error': 'HTTP 403'})
        self.assertEqual(result['status'], 'BLOCKED_CONFIGURATION')
        self.assertEqual(self.teacher_calls, [])
        self.assertEqual(Ledger(self.campaign).records('samples'), [])

    def test_unknown_contract_pending_does_not_invent_behavior_or_child_score(self):
        result = self.run_init(implementer=lambda *args: {'status': 'pending', 'reason': 'Unknown mechanism'})
        self.assertEqual(result['status'], 'INIT_BATCH_COMPLETE')
        self.assertEqual(result['counts']['qualified_edit'], 0)
        self.assertEqual(result['counts']['qualified_keep'], 3)
        self.assertEqual(result['counts']['pending'], 3)
        self.assertEqual(len(self.calls), 3)
        for record in Ledger(self.campaign).records('samples'):
            if record['plan']['action'] == 'edit':
                self.assertIsNone(record['qualification']['implementation_task_result'])

    def test_child_execution_required_and_child_timeout_does_not_block_other_parents(self):
        def evaluate(*args):
            if args[4] == 'first' and 'child' in args[6]:
                return {'status': 'INCOMPLETE', 'reward': None, 'failure_layer': 'infrastructure',
                    'wall_limit_reached': True, 'retryable': False}
            return self.evaluator(*args)
        result = self.run_init(evaluator=evaluate)
        self.assertEqual(result['status'], 'INIT_BATCH_PARTIAL')
        self.assertEqual(result['counts']['qualified_edit'], 2)
        self.assertEqual(result['counts']['qualified_keep'], 3)
        record = next(r for r in Ledger(self.campaign).records('samples') if r['state']['proposal_task_ids'] == ['first'] and r['plan']['action'] == 'edit')
        self.assertEqual(record['qualification']['status'], 'pending')

    def test_resume_detects_child_tampering_and_revokes_qualified_records(self):
        self.run_init()
        ledger = Ledger(self.campaign)
        edit = next(r for r in ledger.records('samples') if r['plan']['action'] == 'edit')
        (Path(edit['implementation']['modules_root']) / 'tools/baseline.py').write_text('x = 999\n')
        before = len(self.calls)
        result = self.run_init()
        self.assertEqual(result['status'], 'BLOCKED_ARTIFACT_CHANGED')
        self.assertEqual(len(self.calls), before)
        self.assertTrue(all(r['revoked'] for r in ledger.records('samples')))

    def test_diagnostic_or_factual_uncertainty_does_not_erase_teacher_plans(self):
        def bad_diagnosis(*args): raise ValueError('Attribution output malformed')
        def bad_review(*args): raise ValueError('Review output malformed')
        result = self.run_init(diagnostician=bad_diagnosis, reviewer=bad_review)
        self.assertEqual(result['counts']['raw_plans'], 6)
        self.assertEqual(result['counts']['qualified_edit'], 0)
        self.assertEqual(result['counts']['qualified_keep'], 0)
        self.assertEqual(result['counts']['pending'], 6)

    def test_reward_task_cannot_be_scheduled_as_initialization_source(self):
        self.protocol['initialization']['task_order'] = ['reward']
        with self.assertRaisesRegex(ValueError, 'proposal partition'):
            self.run_init()
        self.assertEqual(self.calls, [])

    def test_crash_after_teacher_keeps_raw_plans_and_resume_does_not_regenerate(self):
        def interrupted_editor(*args): raise KeyboardInterrupt('controller interrupted after teacher save')
        with self.assertRaises(KeyboardInterrupt):
            self.run_init(implementer=interrupted_editor)
        snapshot = json.loads((self.campaign / 'status.json').read_text())
        self.assertEqual(snapshot['counts']['raw_plans'], 2)
        self.assertEqual(len(self.teacher_calls), 1)
        self.assertEqual(len(self.calls), 1)
        result = self.run_init()
        self.assertEqual(result['status'], 'INIT_BATCH_COMPLETE')
        self.assertEqual(len(self.teacher_calls), 3)
        self.assertEqual(len(self.calls), 6)
        self.assertEqual(len(Ledger(self.campaign).records('samples')), 6)

    def test_finished_parent_feedback_is_bound_binary_and_never_child_feedback(self):
        def evaluated(*args):
            result = copy.deepcopy(self.evaluator(*args))
            path = self.root / 'results' / (digest(args[6]) + '.json')
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps({'task': args[4], 'scope': args[6]}))
            result['trial_result_path'] = str(path)
            result['result_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            return result
        result = self.run_init(evaluator=evaluated)
        self.assertEqual(result['status'], 'INIT_BATCH_COMPLETE', result)
        self.assertEqual(len(self.teacher_calls), 3)
        for state in self.teacher_calls:
            rows = state['prior_proposal_outcomes']
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['outcome'], 'FAIL')
            self.assertEqual(rows[0]['parent_code_hash'], tree_hash(self.modules))
            self.assertEqual(rows[0]['proposal_task_id'], state['proposal_task_ids'][0])
            self.assertTrue(rows[0]['finished_before_decision'])
            self.assertEqual(rows[0]['source_partition'], 'proposal')
            self.assertNotIn('reward', rows[0])
