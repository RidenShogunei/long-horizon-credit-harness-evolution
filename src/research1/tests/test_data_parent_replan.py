"""No provider requests, Harbor launches, Docker calls, or production mutations."""
import copy
import hashlib
import io
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

from research1.data_pipeline.engine import ArtifactChangedError
from research1.data_pipeline.initialization import run_initialization
from research1.data_pipeline.ledger import Ledger
from research1.data_pipeline.protocol import prepare, load_protocol, partition_tasks
from research1.data_pipeline.source_parents import (EXECUTION_SOURCES, freeze_parent_sources,
    validate_parent_sources, write_parent_snapshot)
from research1.data_pipeline.util import atomic_json, digest, tree_hash


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ReplanTests(unittest.TestCase):
    def setUp(self):
        context = patch('research1.data_pipeline.framework_context.build_framework_context', return_value=None)
        context.start(); self.addCleanup(context.stop)
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.dataset = self.root / 'dataset'; self.dataset.mkdir()
        self.tasks = ['task' + str(i) for i in range(120)]
        atomic_json(self.dataset / 'manifest.json', {'tasks': [{'name': t, 'split': 'train'} for t in self.tasks]})
        for task in self.tasks:
            env = self.dataset / 'tasks' / task / 'environment'; env.mkdir(parents=True)
            (env / 'Dockerfile').write_text('FROM fixture:1\n')
        h0 = self.root / 'h0'
        for module in ('agent_loop', 'observation', 'tools', 'context_mgmt', 'verification'):
            path = h0 / module / 'baseline.py'; path.parent.mkdir(parents=True); path.write_text('value = 1\n')
        atomic_json(self.root / 'configs/h0.snapshot.v0.json', {'modules_root': str(h0),
            'file_sha256': {p.relative_to(h0).as_posix(): sha(p) for p in h0.rglob('*.py')}})
        self.config = self.root / 'configs/pilot.yaml'
        self.config.write_text('dataset_tb: ' + self.dataset.as_posix() + '\nmodularrsi_root: ' + self.root.as_posix() + '\n')
        runtime = self.root / 'src/harbor/runtime.py'; runtime.parent.mkdir(parents=True); runtime.write_text('version = 1\n')
        for rel in EXECUTION_SOURCES:
            path = self.root / rel; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('frozen = 1\n')
        sets = partition_tasks(self.tasks, set(), seed=7,
            ready={t: {'cached': True, 'heavy_build': False} for t in self.tasks})
        self.order = [sets['proposal_train'][0], sets['proposal_validation'][0],
            sets['proposal_train'][1], sets['proposal_validation'][1]]
        self.source = self.root / 'runs/v3'
        self.old = prepare(self.source, self.config, root=self.root, seed=7, cached_images={'fixture:1'},
            mode='initialization', max_parents=4, parent_tasks=self.order)
        self.source_ledger = Ledger(self.source)
        self.original_results = []
        for index, task in enumerate(self.order):
            self.save_source_parent(index, task)
        atomic_json(self.source / 'status.json', {'status': 'INIT_BATCH_COMPLETE'})
        # A newer batch is the budget predecessor; v3 is only the source of old executions.
        self.latest = self.root / 'runs/v4'
        self.latest_protocol = prepare(self.latest, self.config, root=self.root, seed=7,
            cached_images={'fixture:1'}, supersedes=self.source, mode='initialization', max_parents=2)
        latest_ledger = Ledger(self.latest)
        for index in range(2):
            key = 'v4_spent_' + str(index)
            latest_ledger.reserve_job(key, {'role': 'proposal'}); latest_ledger.start_job(key, 'fixture')
            latest_ledger.fail_job(key, {'reward': None})
        atomic_json(self.latest / 'status.json', {'status': 'INIT_BATCH_PARTIAL'})
        self.campaign = self.root / 'runs/v5'
        self.image_patch = patch('research1.data_pipeline.source_parents._image_identity', side_effect=self.image)
        self.image_patch.start(); self.addCleanup(self.image_patch.stop)
        self.model_patch = patch('research1.distill._tongji_chat', side_effect=AssertionError('unexpected provider call'))
        self.model_patch.start(); self.addCleanup(self.model_patch.stop)
        self.calls, self.teacher_calls, self.review_calls, self.diagnostic_calls = [], [], [], []

    def image(self, task_path):
        return {'dockerfile_sha256': sha(task_path / 'environment/Dockerfile'),
            'image_ids': {'hb__' + task_path.name: 'sha256:fixture-' + task_path.name, 'fixture:1': 'sha256:base'}}

    def save_source_parent(self, index, task):
        directory = self.source / 'executions' / task; directory.mkdir(parents=True)
        official = directory / 'result.json'; atomic_json(official, {'reward': index % 2})
        trajectory = directory / 'trajectory.json'; atomic_json(trajectory, {'steps': []})
        solver, runtime = self.old['policies']['solver'], self.old['runtime']
        task_path = self.dataset / 'tasks' / task
        scope = 'init:parent:' + str(index)
        spec = {'schema': 'research1.evaluation.v2', 'protocol_hash': self.old['protocol_hash'],
            'modules_root': self.old['modules_root'], 'code_hash': self.old['modules_hash'],
            'task': task, 'task_path': str(task_path), 'task_hash': tree_hash(task_path), 'image': self.image(task_path),
            'model': solver['model'], 'provider_id': solver['provider'], 'temperature': solver['temperature'],
            'reasoning_effort': solver['reasoning_effort'], 'max_turns': runtime['max_turns'],
            'wall_seconds': runtime['wall_seconds'], 'role': 'proposal', 'scope': scope, 'repeat': 1}
        logical = digest(spec); key = logical + ':0'
        result = {'status': 'PASS', 'reward': float(index % 2), 'failure_layer': 'model',
            'evaluation_id': logical, 'job_key': key, 'evaluation_spec': spec, 'role': 'proposal', 'scope': scope,
            'trial_result_path': str(official), 'result_sha256': sha(official),
            'trajectory': str(trajectory), 'trajectory_sha256': sha(trajectory),
            'bound_assets': {'task': task, 'task_hash': tree_hash(task_path), 'image': self.image(task_path)},
            'evidence': [{'id': 'observation', 'text': 'Executed command and saw a coverage normalization error'}]}
        self.source_ledger.reserve_job(key, spec); self.source_ledger.start_job(key, 'fixture')
        self.source_ledger.complete_job(key, result)
        atomic_json(directory / 'task_run.json', result)
        state_id = 'old_state_' + str(index); lineage = 'original_root_' + str(index)
        state = {'state_id': state_id, 'root_lineage_id': lineage,
            'split': 'train' if task in self.old['task_sets']['proposal_train'] else 'validation'}
        self.source_ledger.put_record('states', state_id, state)
        atomic_json(self.source / 'states' / state_id / 'state.json', state)
        checkpoint = {'scope': scope, 'task': task, 'status': 'DONE', 'parent_result': result,
            'root_lineage_id': lineage, 'state_id': state_id, 'raw_plans': 1}
        self.source_ledger.put_record('init_parents', scope, checkpoint)
        self.source_ledger.put_record('samples', 'old_keep_' + str(index), {'sample_id': 'old_keep_' + str(index),
            'root_lineage_id': lineage, 'source_refs': [logical], 'qualification': {'status': 'pending'}, 'revoked': False})
        atomic_json(self.source / 'decisions' / str(index) / 'teacher.json', {'raw': 'old unchanged teacher ' + str(index)})
        atomic_json(self.source / 'reviews' / str(index) / 'review.json', {'status': 'pending_review'})
        self.original_results.append(copy.deepcopy(result))

    def prepare_replan(self):
        return prepare(self.campaign, self.config, root=self.root, seed=7, cached_images={'fixture:1'},
            mode='initialization', supersedes=self.latest, replan_from=self.source)

    def plan(self, state, action='keep'):
        return {'action': action, 'target_module': 'tools', 'locations': ['tools/baseline.py'] if action == 'edit' else [],
            'evidence_refs': [state['evidence'][0]['id']], 'problem_hypothesis': 'Observed execution output warrants a scoped hypothesis',
            'intended_change': 'Use scoped constant' if action == 'edit' else 'Keep within observed evidence limits',
            'preserve_behaviors': ['Keep tool transport'], 'verification_plan': ['Frozen fixture'],
            'verification_kind': 'fixture_mechanism'}

    def proposer(self, state, directory):
        self.teacher_calls.append(copy.deepcopy(state)); atomic_json(directory / 'teacher.json', {'raw': 'new teacher'})
        return [self.plan(state)]

    def reviewer(self, plan, state):
        self.review_calls.append(state['state_id'])
        return {'status': 'PASS', 'reason': 'Independent factual fixture', 'independent': True,
            'reviewer_id': 'new-policy-review', 'evidence_refs': [state['evidence'][0]['id']]}

    def diagnostician(self, state, directory):
        self.diagnostic_calls.append(state['state_id']); return []

    def evaluator(self, campaign, protocol, ledger, modules, task, role, scope, repeat):
        self.assertTrue(scope.startswith('init:child:'), 'A reused parent must never execute again')
        self.assertEqual(role, 'proposal'); self.calls.append((task, scope))
        spec = {'protocol_hash': protocol['protocol_hash'], 'role': role, 'task': task,
            'modules_root': str(modules), 'code_hash': tree_hash(modules), 'scope': scope}
        key = digest(spec); ledger.reserve_job(key, spec); ledger.start_job(key, 'fixture')
        result = {'status': 'PASS', 'reward': 0.0, 'failure_layer': 'model',
            'evaluation_id': key, 'evaluation_spec': spec}
        ledger.complete_job(key, result); return result

    def implementer(self, plan, state, directory):
        child = directory / 'child'; shutil.copytree(state['modules_root'], child)
        (child / 'tools/baseline.py').write_text('value = 2\n')
        patch_path = directory / 'saved.diff'; patch_path.write_text('fixture saved change\n')
        return {'status': 'applied', 'modules_root': str(child), 'code_hash': tree_hash(child),
            'parent_code_hash': state['code_hash'], 'patch_path': str(patch_path), 'patch_sha256': sha(patch_path)}

    def run_replan(self, **extra):
        kwargs = {'evaluator': self.evaluator, 'proposer': self.proposer, 'reviewer': self.reviewer,
            'diagnostician': self.diagnostician, 'implementer': self.implementer,
            'verifier': lambda *a: {'status': 'PASS', 'covers_plan': True, 'executable': True,
                'checks_id': 'fixture', 'mechanism_id': 'fixture_mechanism'}}
        kwargs.update(extra)
        with patch('research1.data_pipeline.modules.build_contract', return_value={'checks_id': 'fixture', 'selected': []}):
            return run_initialization(self.campaign, **kwargs)

    def test_complete_source_set_is_new_decisions_not_new_independent_executions(self):
        old_bytes = {str(p): p.read_bytes() for area in ('decisions', 'reviews') for p in (self.source / area).rglob('*.json')}
        protocol = self.prepare_replan()
        self.assertEqual(protocol['initialization']['task_order'], self.order)
        self.assertEqual(protocol['initialization']['max_parents'], 4)
        self.assertEqual(protocol['budget'], {'regular': 90, 'retries': 32, 'total': 122})
        self.assertEqual(protocol['stage_budget']['proposal'], 8)
        self.assertNotEqual(protocol['modules_root'], self.old['modules_root'])
        self.assertEqual(protocol['modules_hash'], self.old['modules_hash'])
        first = self.run_replan()
        self.assertEqual(first['status'], 'INIT_BATCH_COMPLETE', first)
        self.assertEqual(first['counts']['raw_plans'], 4)
        self.assertEqual(first['counts']['new_independent_parents'], 0)
        self.assertEqual(first['counts']['unique_parent_roots'], 4)
        self.assertEqual(first['counts']['reused_parent_executions'], 4)
        self.assertEqual(self.calls, [])
        self.assertEqual(len(self.diagnostic_calls), 4); self.assertEqual(len(self.review_calls), 4)
        ledger = Ledger(self.campaign)
        self.assertEqual(ledger.status()['budget']['used'], {})
        self.assertEqual(ledger.records('evaluation_specs'), [])
        for sample in ledger.records('samples'):
            parent = sample['qualification']['proposal_task_result']
            self.assertEqual(parent['evaluation_spec']['protocol_hash'], self.old['protocol_hash'])
            self.assertIn(parent, self.original_results)
            self.assertEqual(sample['source_kind'], 'fresh_teacher_plan_from_reused_parent')
            self.assertEqual(sample['parent_execution_provenance']['decision_protocol_hash'], protocol['protocol_hash'])
            self.assertTrue(sample['root_lineage_id'].startswith('original_root_'))
            self.assertEqual(sample['state']['split'], 'train' if parent['evaluation_spec']['task'] in self.old['task_sets']['proposal_train'] else 'validation')
        self.assertEqual(old_bytes, {path: Path(path).read_bytes() for path in old_bytes})
        self.assertTrue(all(s['qualification']['status'] == 'pending' for s in self.source_ledger.records('samples')))
        before = (len(self.teacher_calls), len(self.review_calls), len(self.diagnostic_calls))
        self.run_replan(); self.assertEqual(before, (len(self.teacher_calls), len(self.review_calls), len(self.diagnostic_calls)))
        self.assertEqual(prepare(self.campaign, self.config, root=self.root, seed=7, cached_images=set(),
            mode='initialization', supersedes=self.latest, replan_from=self.source), protocol)

    def test_edit_requires_new_child_job_under_new_protocol_and_same_task(self):
        protocol = self.prepare_replan()
        def propose(state, directory):
            self.teacher_calls.append(copy.deepcopy(state)); return [self.plan(state, 'edit')]
        status = self.run_replan(proposer=propose)
        self.assertEqual(status['counts']['qualified_edit'], 4, status)
        self.assertEqual(len(self.calls), 4)
        self.assertEqual(Ledger(self.campaign).status()['budget']['used']['regular'], 4)
        for sample in Ledger(self.campaign).records('samples'):
            parent = sample['qualification']['proposal_task_result']; child = sample['qualification']['implementation_task_result']
            self.assertEqual(parent['evaluation_spec']['protocol_hash'], self.old['protocol_hash'])
            self.assertEqual(child['evaluation_spec']['protocol_hash'], protocol['protocol_hash'])
            self.assertEqual(parent['evaluation_spec']['task'], child['evaluation_spec']['task'])
            self.assertNotEqual(parent['evaluation_spec']['code_hash'], child['evaluation_spec']['code_hash'])

    def test_interruption_after_plans_resume_without_parent_solver_or_teacher_replay(self):
        self.prepare_replan()
        with self.assertRaises(KeyboardInterrupt):
            self.run_replan(reviewer=lambda *a: (_ for _ in ()).throw(KeyboardInterrupt()))
        self.assertEqual(len(Ledger(self.campaign).records('samples')), 1)
        first_state = self.teacher_calls[0]['state_id']
        self.run_replan()
        self.assertEqual(len(self.teacher_calls), 4)
        self.assertEqual(self.teacher_calls[0]['state_id'], first_state)
        self.assertEqual(self.calls, [])
        self.assertEqual(len(Ledger(self.campaign).records('samples')), 4)

    def test_source_byte_change_revokes_new_samples_without_reexecuting(self):
        self.prepare_replan(); self.run_replan()
        Path(self.original_results[0]['trajectory']).write_text('{"changed":true}')
        status = self.run_replan()
        self.assertEqual(status['status'], 'BLOCKED_ARTIFACT_CHANGED', status)
        self.assertEqual(status['counts']['revoked'], 4)
        self.assertEqual(self.calls, []); self.assertEqual(len(self.teacher_calls), 4)

    def test_root_revocation_propagates_but_old_plan_revocation_does_not(self):
        self.prepare_replan(); self.run_replan()
        self.source_ledger.revoke('old_keep_0', 'Old demonstration withdrawn')
        self.assertEqual(self.run_replan()['status'], 'INIT_BATCH_COMPLETE')
        self.source_ledger.revoke('original_root_0', 'Parent root withdrawn')
        status = self.run_replan()
        self.assertEqual(status['status'], 'BLOCKED_ARTIFACT_CHANGED')
        self.assertEqual(status['counts']['revoked'], 4)
        self.assertEqual(len(self.teacher_calls), 4)

    def test_snapshot_cannot_be_subset_reordered_or_edited_on_resume(self):
        protocol = self.prepare_replan()
        snapshot = json.loads((self.campaign / 'source_parents.json').read_text())
        altered = copy.deepcopy(snapshot); altered['rows'] = altered['rows'][1:]
        with self.assertRaises(ArtifactChangedError): write_parent_snapshot(self.campaign, altered)
        for order in (self.order[::-1], self.order[:2]):
            changed = copy.deepcopy(protocol); changed['initialization']['task_order'] = order
            with self.assertRaises(ArtifactChangedError): validate_parent_sources(self.campaign, changed)
        atomic_json(self.campaign / 'source_parents.json', altered)
        self.assertEqual(self.run_replan()['status'], 'BLOCKED_ARTIFACT_CHANGED')

    def test_incompatible_execution_conditions_rejected_before_budget_handover(self):
        # Compatibility checks are strict for solver/config/runtime/assets,
        # while a new teacher/reviewer/contract protocol is permitted.
        target = copy.deepcopy(self.latest_protocol)
        target['policies']['teacher']['version'] = 'new-teacher'
        target['policies']['factual_reviewer']['version'] = 'new-review'
        self.assertEqual(len(freeze_parent_sources(self.source, target)['rows']), 4)
        for path in (('policies', 'solver', 'model'), ('policies', 'solver', 'temperature'),
                     ('policies', 'solver', 'reasoning_effort'), ('config_hash',), ('runtime_source_hash',),
                     ('runtime', 'max_turns'), ('modules_hash',)):
            changed = copy.deepcopy(target); ref = changed
            for key in path[:-1]: ref = ref[key]
            ref[path[-1]] = 'changed'
            with self.subTest(path=path), self.assertRaises(ValueError): freeze_parent_sources(self.source, changed)
        changed = copy.deepcopy(target)
        first = self.order[0]; changed['task_sets']['proposal_train'].remove(first); changed['task_sets']['proposal_validation'].append(first)
        with self.assertRaisesRegex(ValueError, 'root train/validation'): freeze_parent_sources(self.source, changed)
        (self.root / EXECUTION_SOURCES[0]).write_text('execution_runtime_changed = True\n')
        with self.assertRaisesRegex(ValueError, 'execution_sources'): self.prepare_replan()
        self.assertEqual(Ledger(self.latest).records('budget_handover'), [])

    def test_missing_or_revoked_source_preflight_leaves_budget_and_source_ledger_unchanged(self):
        trajectory = Path(self.original_results[0]['trajectory'])
        saved = trajectory.read_bytes(); trajectory.unlink()
        before = sha(self.source_ledger.path)
        with self.assertRaises(ArtifactChangedError): self.prepare_replan()
        self.assertEqual(sha(self.source_ledger.path), before)
        self.assertEqual(Ledger(self.latest).records('budget_handover'), [])
        trajectory.write_bytes(saved)
        self.source_ledger.revoke('original_root_0', 'Withdrawn before new preparation')
        before = sha(self.source_ledger.path)
        self.campaign = self.root / 'runs/v5_after_missing_preflight'
        with self.assertRaises(ArtifactChangedError): self.prepare_replan()
        self.assertEqual(sha(self.source_ledger.path), before)
        self.assertEqual(Ledger(self.latest).records('budget_handover'), [])

    def test_source_job_or_old_teacher_tampering_blocks_reuse(self):
        self.prepare_replan(); self.run_replan()
        old_teacher = self.source / 'decisions/0/teacher.json'; old_teacher.write_text('{"raw":"rewritten"}')
        self.assertEqual(self.run_replan()['status'], 'BLOCKED_ARTIFACT_CHANGED')

    def test_public_export_revalidates_source_revocation_and_publishes_exclusions(self):
        from research1.cli import build_parser
        from research1.data_pipeline.cli import command
        self.prepare_replan(); self.run_replan()
        pointer = self.campaign / 'exports/sft/CURRENT.json'
        original = json.loads(pointer.read_text())
        self.assertEqual(json.loads(Path(original['manifest']).read_text())['qualified_records'], 4)
        self.source_ledger.revoke('original_root_0', 'Source execution withdrawn after collection')
        args = build_parser().parse_args(['data', 'export', '--campaign', str(self.campaign)])
        output = io.StringIO()
        with redirect_stdout(output): self.assertEqual(command(args), 2)
        result = json.loads(output.getvalue())
        self.assertEqual(result['status'], 'BLOCKED_ARTIFACT_CHANGED')
        replacement = json.loads(pointer.read_text())
        self.assertNotEqual(original['dataset_revision'], replacement['dataset_revision'])
        manifest = json.loads(Path(replacement['manifest']).read_text())
        self.assertEqual(manifest['qualified_records'], 0)
        self.assertEqual(manifest['source_parent_validation']['status'], 'FAIL')
        self.assertEqual(len(self.teacher_calls), 4); self.assertEqual(self.calls, [])
        self.assertTrue(all(r['revoked'] for r in Ledger(self.campaign).records('samples')))

    def test_direct_sft_export_cannot_republish_stale_admissions_after_source_mutation(self):
        from research1.data_pipeline.sft import export_dataset
        self.prepare_replan(); self.run_replan()
        ledger = Ledger(self.campaign); stale = ledger.records('samples')
        Path(self.original_results[0]['trial_result_path']).write_text('{"reward":1}')
        with ledger.controller(): result = export_dataset(self.campaign, stale)
        self.assertEqual(result['qualified_records'], 0)
        self.assertEqual(result['source_parent_validation']['status'], 'FAIL')
        self.assertEqual(len(self.teacher_calls), 4)

    def test_original_completed_job_binding_and_target_cache_cannot_be_rewritten(self):
        self.prepare_replan(); self.run_replan()
        ledger = Ledger(self.campaign)
        cached = dict(ledger.record_items('init_parent_executions'))['init:parent:0']
        cached['value']['reward'] = 99
        ledger.put_record('init_parent_executions', 'init:parent:0', cached)
        self.assertEqual(self.run_replan()['status'], 'BLOCKED_ARTIFACT_CHANGED')
        # Altering the old immutable completed job also fails source validation,
        # even when its result file and source checkpoint remain unchanged.
        with self.source_ledger.connection() as conn:
            key = self.original_results[0]['job_key']
            result = copy.deepcopy(self.original_results[0]); result['reward'] = 99
            conn.execute('UPDATE jobs SET result=? WHERE key=?', (json.dumps(result), key))
        with self.assertRaises(ArtifactChangedError): validate_parent_sources(self.campaign, load_protocol(self.campaign))

    def test_provenance_is_in_export_sidecar_but_not_student_messages(self):
        from research1.data_pipeline.policy import render_messages
        protocol = self.prepare_replan(); self.run_replan()
        pointer = json.loads((self.campaign / 'exports/sft/CURRENT.json').read_text())
        manifest = json.loads(Path(pointer['manifest']).read_text())
        metadata = [json.loads(line) for line in Path(manifest['files']['metadata.jsonl']).read_text().splitlines()]
        self.assertEqual(len(metadata), 4)
        self.assertEqual(metadata[0]['parent_execution_provenance']['source_protocol_hash'], self.old['protocol_hash'])
        for sample in Ledger(self.campaign).records('samples'):
            state = sample['state']; messages = render_messages(state, sample['plan'])
            self.assertNotIn('parent_execution_provenance', json.dumps(messages))
            self.assertNotIn(protocol['protocol_hash'], json.dumps(messages))
            feedback = state['prior_proposal_outcomes'][0]
            parent = sample['qualification']['proposal_task_result']
            self.assertEqual(feedback['execution_id'], parent['evaluation_id'])
            self.assertEqual(feedback['outcome'], 'PASS' if parent['reward'] == 1 else 'FAIL')
            self.assertEqual(feedback['source_sha256'], parent['result_sha256'])

    def test_explicit_source_subset_and_missing_budget_predecessor_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'latest stopped budget predecessor'):
            prepare(self.campaign, self.config, root=self.root, mode='initialization', replan_from=self.source)
        with self.assertRaisesRegex(ValueError, 'subset'):
            prepare(self.campaign, self.config, root=self.root, mode='initialization', supersedes=self.latest,
                replan_from=self.source, parent_tasks=self.order[:1])

    def test_cli_exposes_source_and_budget_predecessor_separately(self):
        from research1.cli import build_parser
        from research1.data_pipeline.cli import command
        args = build_parser().parse_args(['data', 'prepare', '--campaign', str(self.campaign),
            '--mode', 'initialization', '--replan-from', str(self.source), '--supersedes', str(self.latest)])
        with patch('research1.data_pipeline.cli.prepare', return_value={'protocol_hash': 'new', 'task_sets': {}, 'budget': {}}) as prepare_call, redirect_stdout(io.StringIO()):
            self.assertEqual(command(args), 0)
        self.assertEqual(prepare_call.call_args.kwargs['replan_from'], self.source)
        self.assertEqual(prepare_call.call_args.kwargs['supersedes'], self.latest)


if __name__ == '__main__': unittest.main()
