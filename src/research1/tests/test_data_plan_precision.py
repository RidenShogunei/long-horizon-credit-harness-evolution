"""Opt-in plan field catalog and claim_audit bind only with explicit v8 precision profile."""
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from research1.data_pipeline import applicability_v3 as app_v3
from research1.data_pipeline import modules_v3 as adapter
from research1.data_pipeline import plan_precision
from research1.data_pipeline import selector_review as selector
from research1.data_pipeline.policy import canonical, digest, public_view, probe_source_strength, render_messages, review_plan
from research1.tests import test_data_probe_evidence_admission as fixtures


class PlanPrecisionFieldAuditTests(unittest.TestCase):
    def setUp(self):
        self.fixture = fixtures.ProbeEvidenceAdmissionTests(
            'test_verified_new_source_note_is_explicit_without_fake_agent_or_success')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.fixture.state['evidence'].append(
            {'id': 'plain-context', 'text': 'Original bounded requirement context.', 'partition': 'proposal'})
        with patch.object(adapter, 'probe_module', side_effect=fixtures.probe):
            self.fixture.contract = adapter.build_contract(self.fixture.state)
        self.fixture.state['contract'] = self.fixture.contract
        self.state = self.fixture.publish()
        self.state['factual_review_policy'] = selector.PROBE_POLICY_ID
        self.root = self.fixture.root
        self.meta = self.fixture.meta
        self.plan = self.fixture.plan
        self.contract = self.state['contract']

    def enable_precision(self, state=None):
        target = state or self.state
        target['plan_precision_policy'] = plan_precision.POLICY_ID
        return target

    def catalog(self, state=None):
        state = state or self.state
        return selector.build_witness_catalog(
            public_view(state), review_policy=selector.PROBE_POLICY_ID, state=state)

    def field_catalog(self, plan=None):
        return plan_precision.build_plan_field_catalog(plan or self.plan)

    def claim_rows(self, catalog=None, **overrides):
        catalog = catalog or self.field_catalog()
        rows = []
        for item in catalog['items']:
            verdict = overrides.get(item['id'], overrides.get(item['field_path'], 'supported'))
            rows.append({'field_id': item['id'], 'verdict': verdict,
                'reason': 'Checked concrete assertions against the same supplied sources.'})
        return rows

    def factual_reply(self, state=None, ids=None, claim_audit=None, **extra):
        state = state or self.state
        witness_catalog = self.catalog(state)
        ids = ids or [next(row['id'] for row in witness_catalog['items'] if row['kind'] == 'local_api_observation')]
        lookup = {row['id']: row for row in witness_catalog['items']}
        value = {'status': 'PASS',
            'reason': 'The verified local API return supports this bounded factual review; future effects remain unproven.',
            'evidence_refs': list(dict.fromkeys(lookup[identifier]['evidence_ref'] for identifier in ids)),
            'witness_ids': ids,
            'code_witnesses': [{'path': 'verification/baseline.py', 'text': 'async def should_terminate('}]}
        if state.get('plan_precision_policy') == plan_precision.POLICY_ID:
            value['claim_audit'] = claim_audit if claim_audit is not None else self.claim_rows()
        value.update(extra)
        return value

    def review(self, state=None, reply=None, directory='review'):
        state = state or self.state
        reply = reply or self.factual_reply(state)
        with patch('research1.distill._tongji_chat', return_value=(json.dumps(reply), self.meta)) as chat:
            result = review_plan(state, self.plan, self.root / directory)
        return result, chat

    def test_absent_profile_keeps_original_v8_request_prompt_and_raw_shape(self):
        view = public_view(self.state)
        request, rebuilt = selector._request(self.state, self.plan)
        payload = {'policy': selector.PROBE_POLICY_ID, 'original_view': view,
            'witness_catalog': rebuilt, 'plan': self.plan}
        expected = selector._PROBE_INSTRUCTIONS + probe_source_strength(view) + canonical(payload)
        self.assertEqual(request['prompt'], expected)
        self.assertNotIn('precision_policy', request)
        self.assertNotIn('plan_field_catalog_sha256', request)
        self.assertNotIn('plan_precision', view)
        bad = self.factual_reply()
        bad['claim_audit'] = self.claim_rows()
        result, _ = self.review(reply=bad, directory='extra-key')
        self.assertEqual(result['status'], 'pending_review')
        self.assertIn('frozen response shape', result['reason'])

    def test_public_view_and_teacher_share_plan_precision_without_future_fields(self):
        baseline = render_messages(self.state)[0]['content']
        self.enable_precision()
        view = public_view(self.state)
        self.assertEqual(view['plan_precision']['policy'], plan_precision.POLICY_ID)
        self.assertTrue(view['plan_precision']['rules'])
        system = render_messages(self.state)[0]['content']
        self.assertTrue(system.startswith(baseline))
        self.assertNotIn('future_trajectory', system)
        forbidden = {'acceptance', 'future_results', 'reward', 'patch_text'}
        for token in forbidden:
            self.assertNotIn(token, json.dumps(view['plan_precision']))
        self.assertNotIn('plan_field_catalog', view)
        user_payload = json.loads(render_messages(self.state)[1]['content'])
        self.assertEqual(user_payload.get('plan_precision'), view['plan_precision'])

    def test_public_rules_are_student_facing_without_audit_implementation_language(self):
        rules = plan_precision.public_rules()
        self.assertEqual(len(rules), 5)
        combined = '\n'.join(rules).lower()
        for phrase in ('indexed', 'catalog', 'claim_audit', 'field audit', 'factual review', 'automatic',
                       'atomic', 'enumerated', 'coverage indexes'):
            self.assertNotIn(phrase, combined, msg=phrase)
        self.assertIn('your plan', combined)
        self.assertIn('runtime dataflow', combined)
        self.assertIn('defined or captured', combined)
        self.assertIn('remain unproven', combined)
        self.assertIn('decision-time sources', combined)
        self.enable_precision()
        view_rules = public_view(self.state)['plan_precision']['rules']
        self.assertEqual(view_rules, rules)

    def test_unknown_precision_policy_rejected(self):
        state = copy.deepcopy(self.state)
        state['plan_precision_policy'] = 'unknown-precision'
        with self.assertRaises(ValueError):
            public_view(state)

    def test_precision_requires_v8_and_adds_catalog_to_request(self):
        self.enable_precision()
        request, _ = selector._request(self.state, self.plan)
        self.assertEqual(request['precision_policy'], plan_precision.POLICY_ID)
        self.assertEqual(request['plan_field_catalog_sha256'], plan_precision.catalog_sha256(self.field_catalog()))
        self.assertIn('claim_audit', request['prompt'])
        self.assertIn('every catalog field', request['prompt'].lower())
        state = copy.deepcopy(self.state)
        self.enable_precision(state)
        state['factual_review_policy'] = selector.POLICY_ID
        with self.assertRaises(ValueError):
            selector._request(state, self.plan)

    def test_field_catalog_is_stable_complete_and_preserves_exact_text(self):
        catalog = self.field_catalog()
        again = self.field_catalog()
        self.assertEqual(catalog, again)
        paths = [row['field_path'] for row in catalog['items']]
        self.assertIn('problem_hypothesis', paths)
        self.assertIn('intended_change', paths)
        self.assertTrue(any(path.startswith('preserve_behaviors[') for path in paths))
        self.assertTrue(any(path.startswith('verification_plan[') for path in paths))
        for row in catalog['items']:
            if row['field_path'] == 'problem_hypothesis':
                self.assertEqual(row['text'], self.plan['problem_hypothesis'])
            self.assertEqual(row['text_sha256'], hashlib.sha256(row['text'].encode()).hexdigest())

    def test_pass_requires_every_catalog_row_and_rejects_bogus_ids(self):
        self.enable_precision()
        catalog = self.field_catalog()
        reply = self.factual_reply(claim_audit=self.claim_rows(catalog)[:-1])
        result, _ = self.review(reply=reply, directory='missing-row')
        self.assertEqual(result['status'], 'pending_review')
        rows = self.claim_rows(catalog)
        rows.append(dict(rows[-1]))
        result, _ = self.review(reply=self.factual_reply(claim_audit=rows), directory='duplicate-row')
        self.assertEqual(result['status'], 'pending_review')
        bogus = self.claim_rows(catalog)
        bogus[0] = dict(bogus[0], field_id='plan_field_deadbeef')
        result, _ = self.review(reply=self.factual_reply(claim_audit=bogus), directory='bogus-id')
        self.assertEqual(result['status'], 'pending_review')

    def test_incorrect_or_uncertain_field_blocks_pass_despite_true_api_witness(self):
        self.enable_precision()
        catalog = self.field_catalog()
        target = catalog['items'][0]['id']
        for verdict, label in (('incorrect', 'incorrect-field'), ('uncertain', 'uncertain-field')):
            reply = self.factual_reply(claim_audit=self.claim_rows(catalog, **{target: verdict}))
            result, _ = self.review(reply=reply, directory=label)
            self.assertEqual(result['status'], 'pending_review', result)
            self.assertIn('incorrect or uncertain', result['reason'])

    def test_unproven_expected_effect_and_planned_action_allow_pass(self):
        self.enable_precision()
        catalog = self.field_catalog()
        hypothesis = next(row['id'] for row in catalog['items'] if row['field_path'] == 'problem_hypothesis')
        verification = next(row['id'] for row in catalog['items'] if row['field_path'].startswith('verification_plan'))
        rows = self.claim_rows(catalog, **{hypothesis: 'unproven_expected_effect', verification: 'planned_action'})
        result, _ = self.review(reply=self.factual_reply(claim_audit=rows), directory='hypothesis-planned')
        self.assertEqual(result['status'], 'PASS', result)
        self.assertEqual(result['precision_policy'], plan_precision.POLICY_ID)
        self.assertEqual(result['plan_field_catalog_sha256'], plan_precision.catalog_sha256(catalog))
        self.assertEqual(len(result['claim_audit']), len(catalog['items']))

    def test_saved_pass_reconstructs_precision_fields_and_tampering_invalidates(self):
        self.enable_precision()
        result, _ = self.review(directory='saved-pass')
        self.assertEqual(result['status'], 'PASS', result)
        with patch('research1.distill._tongji_chat') as chat:
            proof = selector.verify_saved_selected_review(result, self.state, self.plan)
        chat.assert_not_called()
        self.assertTrue(proof['factual_raw_record_sha256'])
        tampered = copy.deepcopy(result)
        tampered['claim_audit'][0]['verdict'] = 'incorrect'
        with self.assertRaises(ValueError):
            selector.verify_saved_selected_review(tampered, self.state, self.plan)
        tampered = copy.deepcopy(result)
        tampered['plan_field_catalog_sha256'] = '0' * 64
        with self.assertRaises(ValueError):
            selector.verify_saved_selected_review(tampered, self.state, self.plan)

    def test_app3_binding_still_works_with_precision_profile(self):
        self.enable_precision()
        self.state['applicability_policy'] = app_v3.POLICY_ID
        review, _ = self.review(directory='app3-factual')
        self.assertEqual(review['status'], 'PASS', review)
        pack = self.contract['registry']['packs'][self.plan['verification_kind']]
        catalog = self.catalog()
        api_id = next(row['id'] for row in catalog['items'] if row['kind'] == 'local_api_observation')
        reply = {'status': 'PASS',
            'reason': 'Visible source supports the data path; utility remains unproven.',
            'mechanism_id': self.plan['verification_kind'], 'witness_ids': [api_id],
            'code_refs': [{'path': 'verification/baseline.py', 'text': 'async def should_terminate('}],
            'covered_checks': pack['goal_checks'] + pack['preserve_checks']}
        with patch('research1.distill._tongji_chat', return_value=(json.dumps(reply), self.meta)) as chat:
            binding = app_v3.review_applicability(self.state, self.plan, review, self.root / 'app3-binding')
        self.assertEqual(binding['status'], 'PASS', binding)
        with patch('research1.distill._tongji_chat') as repeated:
            self.assertTrue(app_v3.validate_binding(binding, self.state, self.plan, self.contract))
        repeated.assert_not_called()
        chat.assert_called_once()

    def test_v7_no_profile_prompt_bytes_unchanged(self):
        from research1.tests import test_data_sft as sft
        sample = sft.record(action='edit')
        sample['state'].update(factual_review_policy=selector.POLICY_ID, review_model='fixed', review_provider='fixed')
        view = public_view(sample['state'])
        catalog = selector.build_witness_catalog(view)
        payload = {'policy': selector.POLICY_ID, 'original_view': view, 'witness_catalog': catalog, 'plan': sample['plan']}
        expected = selector._INSTRUCTIONS + probe_source_strength(view) + canonical(payload)
        request, _ = selector._request(sample['state'], sample['plan'])
        self.assertEqual(request['prompt'], expected)


if __name__ == '__main__':
    unittest.main()
