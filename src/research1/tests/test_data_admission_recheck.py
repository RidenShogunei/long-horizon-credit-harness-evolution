import copy
import hashlib
import importlib.util
import io
import json
from contextlib import redirect_stdout
from pathlib import Path
import tempfile
import shutil
import unittest
from unittest.mock import patch

import research1.evidence as selector
from research1.data_pipeline import admission_recheck as recheck, policy
from research1.data_pipeline.engine import _public_state
from research1.data_pipeline.ledger import Ledger
from research1.data_pipeline.util import atomic_json, canonical, digest, tree_hash

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class RecheckTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory(); self.addCleanup(temp.cleanup)
        self.campaign = Path(temp.name) / 'campaign'; self.campaign.mkdir()
        modules = self.campaign / 'modules'
        for module in ('agent_loop', 'observation', 'tools', 'context_mgmt', 'verification'):
            path = modules / module / 'baseline.py'; path.parent.mkdir(parents=True); path.write_text('value = 1\n')
        self.tasks = ['first', 'second', 'third', 'heldout']
        for task in self.tasks:
            path = self.campaign / 'tasks' / task / 'README.md'; path.parent.mkdir(parents=True); path.write_text('fixture task\n')
        self.protocol = {'protocol_id': 'original-decision-v5', 'modules_root': str(modules), 'modules_hash': tree_hash(modules),
            'registry_sha256': digest({'packs': []}), 'budget': {'regular': 20, 'retries': 2, 'total': 22},
            'policies': {'teacher': {'model': 'fixture-model', 'provider': 'fixture-provider'}},
            'task_sets': {'proposal': self.tasks, 'proposal_train': self.tasks[:3], 'proposal_validation': self.tasks[3:], 'reward': [], 'dev': []},
            'task_asset_hashes': {task: tree_hash(self.campaign / 'tasks' / task) for task in self.tasks},
            'evidence_budget': {'max_total_chars': 96000, 'max_step_chars': 16000, 'max_steps': 12},
            'sft': {'plan_policy_version': 'proposal_single_plan_v3'},
            'source_sha256': {'src/research1/evidence.py': sha(selector.__file__)}}
        self.protocol['protocol_hash'] = digest(self.protocol)
        atomic_json(self.campaign / 'protocol.json', self.protocol)
        atomic_json(self.campaign / 'contracts.json', {'packs': []})
        atomic_json(self.campaign / 'status.json', {'status': 'INIT_BATCH_COMPLETE', 'counts': {'raw_plans': 7, 'unique_parent_roots': 4}})
        snapshot = self.campaign / 'source_snapshot/src/research1/evidence.py'; snapshot.parent.mkdir(parents=True)
        snapshot.write_bytes(Path(selector.__file__).read_bytes())
        self.ledger = Ledger(self.campaign)
        self.originals, self.raw_paths = [], []
        for index, task in enumerate(self.tasks):
            self.make_parent(index, task, 2 if index != 1 else 1)
        self.assertEqual(len(self.originals), 7)
        self.model = patch('research1.distill._tongji_chat', side_effect=AssertionError('Recheck called a model'))
        self.model.start(); self.addCleanup(self.model.stop)
        self.eval = patch('research1.data_pipeline.execution.evaluate', side_effect=AssertionError('Recheck evaluated a task'))
        self.eval.start(); self.addCleanup(self.eval.stop)

    def make_parent(self, index, task, candidates):
        directory = self.campaign / 'executions' / task; directory.mkdir(parents=True)
        observation = "root@x:/app# cat > bloom.py <<'PY'\n> print(\"it's valid Python\")\n> PY\nall tests passed\nroot@x:/app#\n"
        command = "cat > bloom.py <<'PY'\nprint(\"it's valid Python\")\nPY\npython3 - <<'PY'\nassert True\nprint('all tests passed')\nPY\n"
        trajectory = directory / 'trajectory.json'
        atomic_json(trajectory, {'steps': [{'step_id': 5, 'source': 'agent', 'message': 'Observe the result',
            'tool_calls': [{'function_name': 'bash_command', 'tool_call_id': 'call_3_1', 'arguments': {'keystrokes': command, 'duration': 1}}],
            'observation': {'results': [{'content': observation}]}}]})
        evidence = selector.select_trajectory_evidence(trajectory, **self.protocol['evidence_budget'])
        for row in evidence: row.update(id=task + ':' + row['id'], proposal_task_id=task)
        state = _public_state(Path(self.protocol['modules_root']), evidence, [], 'root_' + str(index),
            'validation' if task == 'heldout' else 'train', [task], self.protocol)
        state.update(review_model='fixture-model', review_provider='fixture-provider', max_candidates=2)
        state['contract'] = {'checks_id': 'original-keep'}
        spec = {'protocol_hash': self.protocol['protocol_hash'], 'role': 'proposal', 'task': task,
            'modules_root': state['modules_root'], 'code_hash': state['code_hash'],
            'task_path': str(self.campaign / 'tasks' / task), 'task_hash': self.protocol['task_asset_hashes'][task]}
        key = digest(spec); raw_result = directory / 'result.json'; atomic_json(raw_result, {'reward': 0})
        parent = {'status': 'PASS', 'reward': 0., 'failure_layer': 'model', 'evaluation_id': key,
            'job_key': key, 'evaluation_spec': spec, 'trial_result_path': str(raw_result), 'result_sha256': sha(raw_result),
            'trajectory': str(trajectory), 'trajectory_sha256': sha(trajectory)}
        self.ledger.reserve_job(key, spec); self.ledger.start_job(key, 'fixture'); self.ledger.complete_job(key, parent)
        plans = []
        for candidate in range(candidates):
            plans.append({'action': 'keep', 'target_module': 'agent_loop', 'locations': [],
                'evidence_refs': [evidence[0]['id']], 'problem_hypothesis': 'Observed terminal text in bounded view',
                'intended_change': 'Keep bounded behavior pending further evidence variant ' + str(candidate),
                'preserve_behaviors': ['Retain tool transport'], 'verification_plan': ['Preserve existing execution behavior'],
                'verification_kind': 'explicit_keep'})
        teacher_path = self.campaign / 'decisions' / task / 'teacher.json'
        prompt = 'Original teacher wording.\nFrozen proposal-only view:\n' + canonical(policy.public_view(state))
        metadata = {'model': 'fixture-model', 'provider': 'fixture-provider', 'raw_field': 'content', 'finish_reason': 'stop', 'temperature': 0}
        atomic_json(teacher_path, {'policy_id': 'proposal_single_plan_v3', 'prompt': prompt,
            'prompt_sha256': hashlib.sha256(prompt.encode()).hexdigest(), 'raw': canonical({'candidates': plans}),
            'metadata': metadata, 'visible_view_sha256': digest(policy.public_view(state))})
        for candidate, plan in enumerate(plans):
            # 4 raw PASS: two true observation sets, one genuinely wrong ref,
            # one metadata-only keep. The other three critics stay pending.
            raw_status = 'PASS' if index in {0, 1} or (index == 2 and candidate == 0) else 'pending_review'
            witness = {'evidence_ref': evidence[0]['id'], 'text': 'all tests passed', 'kind': 'observed_tool_output'}
            if index == 1: witness['text'] = 'not contained in this reference'
            if index == 2 and candidate == 0:
                witness = {'evidence_ref': evidence[0]['id'], 'text': '"selected_for":["boundary_context"]', 'kind': 'context'}
            value = {'status': raw_status, 'reason': 'Bounded terminal observation without a correctness or causality claim',
                'evidence_refs': [evidence[0]['id']], 'witnesses': [witness]}
            prompt = 'Factual review policy: original-review.v4. Original wording retained.\nFrozen visible view:\n' + canonical(policy.public_view(state)) + '\nCandidate:\n' + canonical(plan)
            prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
            reviewer_id = 'factual_review_' + digest({'prompt': prompt_sha, 'model': metadata['model'], 'provider': metadata['provider']})[:24]
            review_path = self.campaign / 'reviews' / task / str(candidate) / 'factual_review.json'
            atomic_json(review_path, {'prompt': prompt, 'prompt_sha256': prompt_sha, 'raw': canonical(value),
                'metadata': metadata, 'review_policy': 'original-review.v4', 'review_type': 'model_fact_review',
                'visible_view_sha256': digest(policy.public_view(state)), 'plan_sha256': digest(plan)})
            old_review = {'status': 'pending_review', 'reason': 'Original validator rejected program-level reconstruction',
                'independent': True, 'review_type': 'model_fact_review', 'review_policy': 'original-review.v4',
                'witness_validator': 'original-witness.v4', 'reviewer_id': reviewer_id, 'review_path': str(review_path),
                'plan_sha256': digest(plan), 'visible_view_sha256': digest(policy.public_view(state)),
                'evidence_text_sha256': {row['id']: hashlib.sha256(row['text'].encode()).hexdigest() for row in evidence}, 'evidence_refs': []}
            atomic_json(review_path.with_name('factual_review_result.json'), old_review)
            sample = {'sample_id': task + '_' + str(candidate), 'candidate_id': task + ':' + str(candidate),
                'state': state, 'plan': plan, 'root_lineage_id': state['root_lineage_id'], 'protocol_id': self.protocol['protocol_id'],
                'source_kind': 'fresh_teacher_plan', 'source_refs': [state['state_id'], parent['evaluation_id']],
                'teacher_record_path': str(teacher_path), 'teacher_path': str(teacher_path),
                'teacher_prompt_sha256': json.loads(teacher_path.read_text())['prompt_sha256'],
                'qualification': {'status': 'pending', 'factual_review': old_review, 'proposal_task_result': parent,
                    'task_result': parent, 'implementation_task_result': None,
                    'behavior': {'status': 'PASS', 'covers_plan': True, 'executable': True, 'mechanism_id': 'explicit_keep'}},
                'implementation': {'status': 'kept', 'modules_root': state['modules_root'], 'code_hash': state['code_hash']},
                'mechanism_id': 'explicit_keep', 'revoked': False}
            self.ledger.put_record('samples', sample['sample_id'], sample)
            self.originals.append(copy.deepcopy(sample)); self.raw_paths += [review_path, review_path.with_name('factual_review_result.json'), teacher_path]

    def run_recheck(self):
        with self.ledger.controller(): return recheck.recheck_factual(self.campaign, ledger=self.ledger)

    def current(self): return {r['sample_id']: r for r in self.ledger.records('samples')}

    def test_fixed_all_seven_recheck_fixes_parser_without_promoting_bad_or_pending_critics(self):
        before = {str(path): path.read_bytes() for path in self.raw_paths}
        result = self.run_recheck()
        self.assertEqual(result['samples_rechecked'], 7, result)
        self.assertEqual(result['critic_pass_revalidated'], 2)
        self.assertEqual(result['eligible'], 2)
        self.assertEqual(result['new_independent_reviews'], 0)
        samples = self.current()
        self.assertEqual({key for key, row in samples.items() if row['qualification']['status'] == 'verified'}, {'first_0', 'first_1'})
        for old in self.originals:
            now = samples[old['sample_id']]
            for key in ('state', 'plan', 'protocol_id', 'root_lineage_id'): self.assertEqual(old[key], now[key])
            self.assertFalse(now['qualification']['factual_review']['new_independent_review'])
            with self.ledger.connection() as conn:
                history = json.loads(conn.execute("SELECT payload FROM record_history WHERE kind='samples' AND key=? AND revision=1", (old['sample_id'],)).fetchone()[0])
            self.assertEqual(history, old)
        self.assertEqual(before, {path: Path(path).read_bytes() for path in before})
        self.assertEqual(self.ledger.status()['budget']['used']['regular'], 4)

    def test_identical_corpus_validator_is_cached_and_revocation_never_restores_admission(self):
        first = self.run_recheck(); rows = recheck._rows(self.ledger)
        with patch.object(policy, '_review_witnesses', side_effect=AssertionError('Repeated deterministic attempt')):
            second = self.run_recheck()
        self.assertTrue(second['cached']); self.assertEqual(second['run_id'], first['run_id'])
        self.assertEqual(recheck._rows(self.ledger), rows)
        self.ledger.revoke(first['run_id'], 'Withdraw this admission revision')
        again = self.run_recheck(); self.assertTrue(again['cached']); self.assertEqual(again['active_qualified'], 0)
        self.assertTrue(all(row['revoked'] for row in self.current().values()))

    def test_original_raw_fail_is_not_promoted(self):
        path = self.campaign / 'reviews/first/0/factual_review.json'
        saved = json.loads(path.read_text()); value = json.loads(saved['raw']); value['status'] = 'FAIL'
        saved['raw'] = canonical(value); atomic_json(path, saved)
        self.run_recheck()
        self.assertEqual(self.current()['first_0']['qualification']['status'], 'pending')

    def test_parent_bad_binding_and_metadata_alone_cannot_qualify_keep(self):
        record = self.current()['first_0']; record['qualification']['proposal_task_result']['evaluation_spec']['code_hash'] = 'b'*64
        self.ledger.put_record('samples', 'first_0', record)
        result = self.run_recheck(); self.assertEqual(result['eligible'], 1)
        self.assertEqual(self.current()['first_0']['qualification']['status'], 'pending')
        self.assertEqual(self.current()['third_0']['qualification']['status'], 'pending')

    def test_old_prompt_and_review_identity_are_required_without_rebuilding_new_wording(self):
        record = self.current()['first_0']; record['qualification']['factual_review']['reviewer_id'] = 'wrong-original-id'
        self.ledger.put_record('samples', 'first_0', record)
        self.run_recheck()
        self.assertEqual(self.current()['first_0']['qualification']['status'], 'pending')
        self.assertEqual(self.current()['first_1']['qualification']['status'], 'verified')

    def test_changed_raw_after_frozen_recheck_revokes_all_derived_admissions(self):
        self.run_recheck()
        path = self.campaign / 'reviews/first/0/factual_review.json'; path.write_text('{}')
        result = self.run_recheck(); self.assertEqual(result['status'], 'BLOCKED_ARTIFACT_CHANGED')
        self.assertTrue(all(row['revoked'] for row in self.current().values()))

    def test_complete_scope_cannot_be_extended_or_favorable_subset_selected(self):
        self.run_recheck(); record = copy.deepcopy(self.originals[0]); record['sample_id'] = 'new_late_plan'
        self.ledger.put_record('samples', record['sample_id'], record)
        self.assertEqual(self.run_recheck()['status'], 'BLOCKED_ARTIFACT_CHANGED')

    def test_interruption_publishes_no_partial_admissions_and_resumes_cached_outcomes(self):
        original = recheck._derive; calls = []
        def interrupt(item, *args):
            calls.append(item['sample_id'])
            if len(calls) == 3: raise KeyboardInterrupt()
            return original(item, *args)
        with patch.object(recheck, '_derive', side_effect=interrupt), self.assertRaises(KeyboardInterrupt): self.run_recheck()
        self.assertTrue(all(row['qualification']['status'] == 'pending' for row in self.current().values()))
        with patch.object(recheck, '_derive', wraps=original) as derive:
            self.assertEqual(self.run_recheck()['eligible'], 2)
        self.assertEqual(derive.call_count, 5)

    def test_cli_flag_is_explicit_generic_keep_only_mixed_stays_empty(self):
        from research1.cli import build_parser
        from research1.data_pipeline.cli import command
        args = build_parser().parse_args(['data', 'export', '--campaign', str(self.campaign), '--recheck-factual'])
        output = io.StringIO()
        with redirect_stdout(output): self.assertEqual(command(args), 0)
        result = json.loads(output.getvalue())
        self.assertEqual(result['factual_recheck']['samples_rechecked'], 7)
        self.assertFalse(result['training_ready']); self.assertEqual(result['counts']['train']['mixed'], 0)
        self.assertEqual(result['counts']['train']['keep'], 2)
        self.assertEqual(result['independent_lineages'], 1)
        self.assertEqual(json.loads((self.campaign / 'status.json').read_text())['counts']['unique_parent_roots'], 4)

    def test_source_guard_blocks_before_admission_and_export_cannot_revive(self):
        self.run_recheck()
        def block(campaign, protocol, ledger):
            for row in ledger.records('samples'): ledger.revoke(row['root_lineage_id'], 'Source withdrawn')
            return {'status': 'FAIL', 'reason': 'Source withdrawn'}
        with patch('research1.data_pipeline.source_parents.audit_sources_for_export', side_effect=block):
            self.assertEqual(self.run_recheck()['status'], 'BLOCKED_ARTIFACT_CHANGED')
        self.assertTrue(all(row['revoked'] for row in self.current().values()))

    def test_same_python_observation_was_false_rejected_by_old_program_reconstruction(self):
        old_path = Path(__file__).parent / 'test_data_legacy_witness_fixture.py'
        spec = importlib.util.spec_from_file_location('old_policy_recheck_test', old_path)
        old_policy = importlib.util.module_from_spec(spec); spec.loader.exec_module(old_policy)
        sample = self.originals[0]
        saved = json.loads(Path(sample['qualification']['factual_review']['review_path']).read_text())
        witnesses = json.loads(saved['raw'])['witnesses']
        with self.assertRaises(ValueError): old_policy._review_witnesses(witnesses, sample['state'], 'keep')
        checked = policy._review_witnesses(witnesses, sample['state'], 'keep')
        self.assertEqual(checked[0]['text'], witnesses[0]['text'])
        self.assertFalse(checked[0]['validation']['success_established'])
        self.assertTrue(any(name.endswith('observation_validator') for name in recheck.validator_fingerprint()['source_files']))

    def test_plain_export_checks_existing_derived_raw_hashes_without_new_revalidation(self):
        from research1.cli import build_parser
        from research1.data_pipeline.cli import command
        self.run_recheck()
        path = self.campaign / 'reviews/first/0/factual_review.json'; path.write_text('{}')
        args = build_parser().parse_args(['data', 'export', '--campaign', str(self.campaign)])
        output = io.StringIO()
        with patch.object(policy, '_review_witnesses', side_effect=AssertionError('Implicit review recheck')), redirect_stdout(output):
            self.assertEqual(command(args), 2)
        result = json.loads(output.getvalue())
        self.assertEqual(result['qualified_records'], 0)
        self.assertEqual(result['factual_recheck_validation']['status'], 'FAIL')

    def test_uncommitted_historical_raw_hashes_are_not_invented(self):
        self.assertNotIn('raw_review_file_sha256', self.originals[0]['qualification']['factual_review'])
        self.run_recheck()
        review = self.current()['first_0']['qualification']['factual_review']
        self.assertIn('raw_review_file_sha256_observed_at_recheck', review)
        self.assertNotIn('raw_review_file_sha256', review)
        corpus_pointer = self.ledger.records('factual_recheck_corpora')[0]
        corpus = json.loads(Path(corpus_pointer['path']).read_text())
        self.assertEqual(corpus['hash_observation_time'], 'first_explicit_recheck_not_original_model_call')

    def test_existing_edit_gate_requires_exact_saved_child_task_code_and_protocol(self):
        record = copy.deepcopy(self.originals[0]); state = record['state']; plan = record['plan']
        plan.update(action='edit', locations=['agent_loop/baseline.py'], verification_kind='fixture_pack')
        teacher = Path(record['teacher_record_path']); saved = json.loads(teacher.read_text())
        saved['raw'] = canonical({'candidates': [plan]}); atomic_json(teacher, saved)
        child = self.campaign / 'child'; shutil.copytree(state['modules_root'], child)
        (child / 'agent_loop/baseline.py').write_text('value = 2\n')
        patch_path = self.campaign / 'saved.diff'; patch_path.write_text('fixture stored patch\n')
        record['implementation'] = {'status': 'applied', 'parent_code_hash': state['code_hash'],
            'modules_root': str(child), 'code_hash': tree_hash(child), 'patch_path': str(patch_path), 'patch_sha256': sha(patch_path)}
        record['mechanism_id'] = 'fixture_pack'
        q = record['qualification']; q['status'] = 'verified'
        q['factual_review'].update(status='PASS', evidence_refs=plan['evidence_refs'])
        q['behavior'].update(checks_id=state['contract']['checks_id'], mechanism_id='fixture_pack')
        parent = q['proposal_task_result']; spec = copy.deepcopy(parent['evaluation_spec'])
        spec.update(modules_root=str(child), code_hash=tree_hash(child))
        official = self.campaign / 'child_result.json'; atomic_json(official, {'reward': 0})
        result = copy.deepcopy(parent); result.update(evaluation_spec=spec, evaluation_id=digest(spec), job_key=digest(spec),
            trial_result_path=str(official), result_sha256=sha(official))
        self.ledger.reserve_job(result['job_key'], spec); self.ledger.start_job(result['job_key'], 'fixture'); self.ledger.complete_job(result['job_key'], result)
        q['implementation_task_result'] = result; q['task_result'] = result
        recheck._existing_admission_checks(record, self.protocol, self.ledger, [])
        for field, bad in (('task', 'second'), ('code_hash', 'c'*64), ('protocol_hash', 'other-protocol')):
            changed = copy.deepcopy(record); changed['qualification']['implementation_task_result']['evaluation_spec'][field] = bad
            with self.subTest(field=field), self.assertRaises(ValueError):
                recheck._existing_admission_checks(changed, self.protocol, self.ledger, [])

    def test_active_batch_cannot_freeze_a_favorable_partial_corpus(self):
        atomic_json(self.campaign / 'status.json', {'status': 'RUNNING'})
        with self.assertRaisesRegex(ValueError, 'finished/stopped'): self.run_recheck()
        self.assertEqual(self.ledger.records('factual_recheck_corpora'), [])


if __name__ == '__main__': unittest.main()
