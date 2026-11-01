"""Fact-ids applicability binds via saved factual witness IDs, not re-quoted trajectory text."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from research1.data_pipeline import applicability as legacy_app
from research1.data_pipeline import applicability_v3 as app_v3
from research1.data_pipeline import modules_v3 as adapter
from research1.data_pipeline import selector_review as selector
from research1.data_pipeline.policy import canonical, digest, public_view, review_plan
from research1.tests import test_data_probe_evidence_admission as fixtures


class ApplicabilityFactIdsV3Tests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ProbeEvidenceAdmissionTests('test_verified_new_source_note_is_explicit_without_fake_agent_or_success')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.state['evidence'].append({'id': 'plain-context', 'text': 'Original bounded requirement context.', 'partition': 'proposal'})
        with patch.object(adapter, 'probe_module', side_effect=fixtures.probe):
            self.fixture.contract = adapter.build_contract(self.fixture.state)
        self.fixture.state['contract'] = self.fixture.contract
        self.state = self.fixture.publish()
        self.state['factual_review_policy'] = selector.PROBE_POLICY_ID
        self.state['applicability_policy'] = app_v3.POLICY_ID
        self.root = self.fixture.root
        self.meta = self.fixture.meta
        self.plan = self.fixture.plan
        self.contract = self.state['contract']

    def catalog(self, state=None):
        state = state or self.state
        return selector.build_witness_catalog(public_view(state), review_policy=selector.PROBE_POLICY_ID, state=state)

    def factual_reply(self, state=None, ids=None):
        state = state or self.state
        catalog = self.catalog(state)
        ids = ids or [next(row['id'] for row in catalog['items'] if row['kind'] == 'local_api_observation')]
        lookup = {row['id']: row for row in catalog['items']}
        return {'status': 'PASS', 'reason': 'The verified local API return supports this bounded factual review; future effects remain unproven.',
            'evidence_refs': list(dict.fromkeys(lookup[identifier]['evidence_ref'] for identifier in ids)),
            'witness_ids': ids, 'code_witnesses': [{'path': 'verification/baseline.py', 'text': 'async def should_terminate('}]}

    def factual_pass(self, state=None, reply=None, directory='factual'):
        state = state or self.state
        reply = reply or self.factual_reply(state)
        with patch('research1.distill._tongji_chat', return_value=(json.dumps(reply), self.meta)):
            return review_plan(state, self.plan, self.root / directory)

    def applicability_reply(self, witness_ids=None, **updates):
        pack = self.contract['registry']['packs'][self.plan['verification_kind']]
        catalog = self.catalog()
        if witness_ids is None:
            witness_ids = [next(row['id'] for row in catalog['items'] if row['kind'] == 'local_api_observation')]
        value = {'status': 'PASS',
            'reason': 'Visible source supports the data path and unchanged frozen checks cover the bounded edit; utility remains unproven.',
            'mechanism_id': self.plan['verification_kind'], 'witness_ids': witness_ids,
            'code_refs': [{'path': 'verification/baseline.py', 'text': 'async def should_terminate('}],
            'covered_checks': pack['goal_checks'] + pack['preserve_checks']}
        value.update(updates)
        return value

    def bind(self, review, reply=None, directory='binding'):
        reply = reply or self.applicability_reply()
        with patch('research1.distill._tongji_chat', return_value=(json.dumps(reply), self.meta)) as chat:
            binding = app_v3.review_applicability(self.state, self.plan, review, self.root / directory)
        return binding, chat

    def test_api_only_witness_ids_bind_without_trajectory_requote(self):
        review = self.factual_pass()
        self.assertEqual(review['status'], 'PASS', review)
        catalog = self.catalog()
        api_id = next(row['id'] for row in catalog['items'] if row['kind'] == 'local_api_observation')
        trajectory = next(row['id'] for row in catalog['items'] if row['kind'] != 'local_api_observation')
        binding, chat = self.bind(review, self.applicability_reply([api_id]))
        self.assertEqual(binding['status'], 'PASS', binding)
        self.assertEqual(binding['policy'], app_v3.POLICY_ID)
        self.assertEqual(binding['factual_witness_ids'], [api_id])
        self.assertEqual(binding['factual_witness_catalog_sha256'], digest(app_v3._factual_witness_catalog(review)))
        self.assertNotIn(trajectory, binding['factual_witness_ids'])
        raw = json.loads(Path(binding['applicability_output_path']).read_text(encoding='utf-8'))['raw']
        parsed_raw = json.loads(raw)
        self.assertNotIn('witnesses', parsed_raw)
        self.assertNotIn('evidence_refs', parsed_raw)
        with patch('research1.distill._tongji_chat') as repeated:
            self.assertTrue(app_v3.validate_binding(binding, self.state, self.plan, self.contract))
        repeated.assert_not_called()
        self.assertEqual(chat.call_args.kwargs['role'], 'data_plan_applicability_fact_ids')
        self.assertFalse(chat.call_args.kwargs['allow_http_fallback'])

    def test_instructions_state_bounded_improvement_without_future_score_proof(self):
        review = self.factual_pass()
        input_record, _, _, _ = app_v3._request(self.state, self.plan, review, self.contract)
        prompt = input_record['prompt']
        self.assertIn('future task-score improvement are not required before implementation', prompt)
        self.assertIn('baseline behavior works as designed', prompt)
        self.assertIn('proven causal responsibility for the original failed task', prompt)

    def test_legacy_v1_instructions_and_parser_bytes_unchanged(self):
        review = self.factual_pass()
        state = copy.deepcopy(self.state)
        state.pop('applicability_policy', None)
        request, _ = legacy_app._request(state, self.plan, review, self.contract)
        self.assertEqual(request['policy'], legacy_app.APPLICABILITY_POLICY)
        self.assertEqual(hashlib.sha256(legacy_app._INSTRUCTIONS.encode()).hexdigest(),
            'f3a162d540fc6fce99961ba3b6cf7219c53bff9a74e4dc1228fa41105e843958')

    def test_old_witness_quotation_shape_stays_pending(self):
        review = self.factual_pass()
        pack = self.contract['registry']['packs'][self.plan['verification_kind']]
        legacy_shape = {'status': 'PASS', 'reason': 'Looks plausible.', 'mechanism_id': self.plan['verification_kind'],
            'evidence_refs': ['proposal1'], 'witnesses': [{'evidence_ref': 'proposal1', 'text': self.fixture.output.strip()}],
            'code_refs': [{'path': 'verification/baseline.py', 'text': 'async def should_terminate('}],
            'covered_checks': pack['goal_checks'] + pack['preserve_checks']}
        binding, _ = self.bind(review, legacy_shape, 'legacy-shape')
        self.assertEqual(binding['status'], 'pending')
        self.assertIn('fact-ids review shape', binding['reason'])

    def test_unknown_duplicate_or_missing_ids_and_bad_code_refs_fail(self):
        review = self.factual_pass()
        catalog = self.catalog()
        api_id = next(row['id'] for row in catalog['items'] if row['kind'] == 'local_api_observation')
        for reply, label in [
            (self.applicability_reply(['unknown-id']), 'unknown'),
            (self.applicability_reply([api_id, api_id]), 'duplicate'),
            (self.applicability_reply([]), 'empty'),
            (self.applicability_reply([api_id], code_refs=[{'path': 'verification/baseline.py', 'text': 'missing symbol'}]), 'code'),
            (self.applicability_reply([api_id], covered_checks=[]), 'coverage'),
        ]:
            binding, _ = self.bind(review, reply, label)
            self.assertEqual(binding['status'], 'pending', label)

    def test_changed_factual_receipt_or_applicability_raw_invalidates_binding(self):
        review = self.factual_pass()
        binding, _ = self.bind(review)
        self.assertTrue(app_v3.validate_binding(binding, self.state, self.plan, self.contract))
        path = Path(self.state['predecision_probe_artifacts']['receipt']['path'])
        before = path.read_bytes()
        path.write_bytes(before + b' ')
        self.assertFalse(app_v3.validate_binding(binding, self.state, self.plan, self.contract))
        path.write_bytes(before)
        raw_path = Path(binding['applicability_output_path'])
        saved = json.loads(raw_path.read_text(encoding='utf-8'))
        reply = json.loads(saved['raw'])
        reply['witness_ids'] = [self.catalog()['items'][0]['id']]
        saved['raw'] = json.dumps(reply)
        raw_path.write_text(json.dumps(saved), encoding='utf-8')
        self.assertFalse(app_v3.validate_binding(binding, self.state, self.plan, self.contract))

    def test_immutable_input_and_missing_policy_reject_without_resend(self):
        review = self.factual_pass()
        binding, _ = self.bind(review, directory='stable')
        with patch('research1.distill._tongji_chat') as chat:
            repeated = app_v3.review_applicability(self.state, self.plan, review, self.root / 'stable')
        chat.assert_not_called()
        self.assertEqual(repeated, binding)
        changed = copy.deepcopy(self.plan)
        changed['intended_change'] = 'Different behavior'
        with self.assertRaisesRegex(ValueError, 'preconditions changed'):
            app_v3.review_applicability(self.state, changed, review, self.root / 'stable')
        state = copy.deepcopy(self.state)
        state.pop('applicability_policy')
        with patch('research1.distill._tongji_chat') as chat:
            value = app_v3.review_applicability(state, self.plan, review, self.root / 'no-policy')
        chat.assert_not_called()
        self.assertEqual(value['status'], 'pending')

    def test_started_without_raw_is_unknown_and_not_resent(self):
        review = self.factual_pass()
        directory = self.root / 'started-only'
        with patch('research1.distill._tongji_chat', side_effect=RuntimeError('offline')):
            first = app_v3.review_applicability(self.state, self.plan, review, directory)
        self.assertEqual(first['status'], 'pending')
        self.assertTrue((directory / 'applicability_started.json').is_file())
        self.assertFalse((directory / 'applicability_output.json').exists())
        with patch('research1.distill._tongji_chat') as chat:
            second = app_v3.review_applicability(self.state, self.plan, review, directory)
        chat.assert_not_called()
        self.assertIn('CALL_OUTCOME_UNKNOWN', second['reason'])

    def test_v3_request_includes_factual_catalog_and_publisher_hashes(self):
        review = self.factual_pass()
        request, proof, catalog, catalog_hash = app_v3._request(self.state, self.plan, review, self.contract)
        self.assertEqual(request['policy'], app_v3.POLICY_ID)
        self.assertEqual(request['factual_raw_record_sha256'], proof['factual_raw_record_sha256'])
        self.assertEqual(request['factual_result_record_sha256'], proof['factual_result_record_sha256'])
        self.assertEqual(request['factual_witness_catalog_sha256'], catalog_hash)
        self.assertEqual(len(catalog['items']), len(review['witness_ids']))
        self.assertEqual([row['id'] for row in catalog['items']], review['witness_ids'])


if __name__ == '__main__':
    unittest.main()
