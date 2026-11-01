"""Generic process-improvement objective, immutable legacy instructions."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research1.data_pipeline.diagnostics import DIAGNOSTIC_POLICY, review_diagnostics
from research1.data_pipeline.policy import (
    LEGACY_PLAN_POLICY_ID, PREVIOUS_PLAN_POLICY_ID, PREVIOUS_PROCESS_POLICY_ID, POLICY_ID, FACTUAL_REVIEW_POLICY,
    collect_candidates, render_messages, review_plan,
)
from research1.tests.test_data_sft import record


class ProcessObjectiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.sample = record(action='keep')
        self.state, self.plan = self.sample['state'], self.sample['plan']
        self.metadata = {'raw_field': 'content', 'finish_reason': 'stop', 'temperature': 0,
                         'provider': 'fixed', 'model': 'fixed'}

    def test_legacy_v2_v3_and_v4_system_bytes_are_frozen(self):
        hashes = {
            LEGACY_PLAN_POLICY_ID: '2560fe7119b14a27b3aeeb78930184bb3e5cf143f938af6a32ccf0afacf37626',
            PREVIOUS_PLAN_POLICY_ID: 'f73f30e2488f45ac30872bb10515892778b0d89124d66278d1b5071d188eec3b',
            PREVIOUS_PROCESS_POLICY_ID: '9fd23239104b467ef2507c3f37e60792093a0f98db968ad210631736f77de237',
        }
        original = copy.deepcopy(self.state)
        pairs = []
        for version, expected in hashes.items():
            state = dict(self.state, plan_policy_version=version)
            messages = render_messages(state, self.plan)
            self.assertEqual(hashlib.sha256(messages[0]['content'].encode()).hexdigest(), expected)
            pairs.append(messages[1:])
        self.assertEqual(pairs[0], pairs[1])
        self.assertEqual(render_messages(self.state, self.plan), render_messages(dict(self.state, plan_policy_version=LEGACY_PLAN_POLICY_ID), self.plan))
        self.assertEqual(self.state, original)

    def test_previous_process_objective_is_still_renderable_without_changes(self):
        self.assertEqual(PREVIOUS_PROCESS_POLICY_ID, 'proposal_single_plan_v4')
        messages = render_messages(dict(self.state, plan_policy_version=PREVIOUS_PROCESS_POLICY_ID))
        system = messages[0]['content']
        self.assertIn('even when the baseline operates as designed', system)
        self.assertIn('a proven implementation defect or demonstrated benefit is not required in advance', system)
        self.assertIn('observed facts and current-code facts', system)
        self.assertIn('testable mechanism hypothesis', system)
        self.assertIn('applicable behavior and preservation checks', system)
        self.assertIn('Treat future utility and causal benefit as unproven', system)
        self.assertIn('Choose keep when no sufficiently grounded change is available', system)
        self.assertIn('Neither action is preferred', system)
        clarification = system.split('Objective clarification:', 1)[1].lower()
        for forbidden in ('bloom', 'payroll', 'gmail', 'normalization', 'agent_loop', 'verification module', 'quota'):
            self.assertNotIn(forbidden, clarification)

    def test_each_teacher_record_keeps_actual_frozen_instruction_identity(self):
        raw = json.dumps({'candidates': [self.plan]})
        for version in (LEGACY_PLAN_POLICY_ID, PREVIOUS_PLAN_POLICY_ID, PREVIOUS_PROCESS_POLICY_ID, POLICY_ID):
            state = dict(self.state, plan_policy_version=version)
            target = self.directory / version
            with patch('research1.distill._tongji_chat', return_value=(raw, self.metadata)) as call:
                self.assertEqual(collect_candidates(state, target), [self.plan])
            saved = json.loads((target/'teacher.json').read_text())
            self.assertEqual(saved['policy_id'], version)
            self.assertTrue(call.call_args.args[0].startswith(render_messages(state)[0]['content']))
            self.assertEqual(saved['raw'], raw)

    def test_new_objective_cannot_resample_an_existing_immutable_teacher_request(self):
        raw = json.dumps({'candidates': [self.plan]})
        target = self.directory / 'immutable'
        state = dict(self.state, plan_policy_version=PREVIOUS_PLAN_POLICY_ID)
        with patch('research1.distill._tongji_chat', return_value=(raw, self.metadata)):
            collect_candidates(state, target)
        before = (target/'teacher.json').read_bytes()
        state['plan_policy_version'] = POLICY_ID
        with patch('research1.distill._tongji_chat') as call:
            with self.assertRaisesRegex(ValueError, 'Teacher input changed'):
                collect_candidates(state, target)
        call.assert_not_called()
        self.assertEqual((target/'teacher.json').read_bytes(), before)

    def test_diagnostic_objective_is_generic_and_keeps_empty_attribution_valid(self):
        with patch('research1.distill._tongji_chat', return_value=('{"diagnostics":[]}', self.metadata)) as call:
            self.assertEqual(review_diagnostics(self.state, self.directory/'diagnostic'), [])
        prompt = call.call_args.args[0]
        self.assertEqual(DIAGNOSTIC_POLICY, 'proposal-parent-attribution.v4')
        saved = json.loads((self.directory/'diagnostic/diagnostic_input.json').read_text())
        self.assertEqual(saved['policy'], DIAGNOSTIC_POLICY)
        self.assertIn('improvement hypothesis', prompt)
        self.assertIn('baseline operates as designed', prompt)
        self.assertIn('Separate observed facts, current-code facts and unproven expected effects', prompt)
        self.assertIn('alone is also insufficient', prompt)
        self.assertIn('return an empty list', prompt)
        self.assertNotIn('defect occurred', prompt)
        self.assertNotIn('merely application code echoed', prompt)
        preamble = prompt.split('\n', 1)[0].lower()
        for forbidden in ('bloom', 'payroll', 'gmail', 'normalization', 'quota'):
            self.assertNotIn(forbidden, preamble)

    def test_diagnostic_guard_still_rejects_outcome_as_mechanism_evidence(self):
        self.state['permitted_modules'] = ['verification']
        self.state['module_code'] = {'verification/baseline.py': 'def gate():\n    return True\n'}
        witness = self.state['evidence'][0]['text']
        item = {'module': 'verification', 'mechanism_id': 'verification_error_gate',
            'signal': 'completion_after_tool_error', 'evidence_refs': ['evidence-1'], 'witness': witness,
            'reason': 'The task reward is zero so this registered mechanism must be defective.'}
        with patch('research1.distill._tongji_chat', return_value=(json.dumps({'diagnostics': [item]}), self.metadata)):
            self.assertEqual(review_diagnostics(self.state, self.directory/'outcome'), [])
        result = json.loads((self.directory/'outcome/diagnostic_review.json').read_text())
        self.assertEqual(len(result['rejected']), 1)

    def test_matching_critic_reviews_facts_without_requiring_proven_benefit(self):
        example = record(action='edit')
        example['state']['plan_policy_version'] = POLICY_ID
        value = {'status': 'pending_review', 'reason': 'Concrete mechanism linkage remains uncertain.',
                 'evidence_refs': ['evidence-1']}
        with patch('research1.distill._tongji_chat', return_value=(json.dumps(value), self.metadata)) as call:
            result = review_plan(example['state'], example['plan'], self.directory/'critic')
        self.assertEqual(FACTUAL_REVIEW_POLICY, 'proposal-factual-review.v5')
        self.assertEqual(result['review_policy'], FACTUAL_REVIEW_POLICY)
        self.assertEqual(result['status'], 'pending_review')
        prompt = call.call_args.args[0]
        self.assertIn('review concrete observations and current-code claims separately from its expected benefit', prompt)
        self.assertIn('do not require a proven implementation defect or proof of future performance benefit', prompt)
        self.assertIn('Explicitly qualified expected effects are hypotheses, not observed outcomes', prompt)
        self.assertIn('Unsupported causal guarantees or invented facts must remain pending_review', prompt)

    def test_new_critic_prompt_does_not_rewrite_a_cached_request(self):
        value = {'status': 'pending_review', 'reason': 'Review remains uncertain.', 'evidence_refs': ['evidence-1']}
        target = self.directory/'critic-cache'
        with patch('research1.distill._tongji_chat', return_value=(json.dumps(value), self.metadata)):
            review_plan(self.state, self.plan, target)
        saved_path = target/'factual_review.json'
        saved = json.loads(saved_path.read_text())
        saved['prompt'] = saved['prompt'].replace(FACTUAL_REVIEW_POLICY, 'proposal-factual-review.v4', 1)
        saved['prompt_sha256'] = hashlib.sha256(saved['prompt'].encode()).hexdigest()
        saved_path.write_text(json.dumps(saved), encoding='utf-8')
        before = saved_path.read_bytes()
        with patch('research1.distill._tongji_chat') as call:
            with self.assertRaisesRegex(ValueError, 'input changed'):
                review_plan(self.state, self.plan, target)
        call.assert_not_called()
        self.assertEqual(saved_path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
