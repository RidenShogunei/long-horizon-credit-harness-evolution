"""A controlled instruction revision preserves old inputs and permits abstention.

These checks guard the collection/deployment boundary and immutable artifacts;
they do not establish whether a teacher can actually propose a useful edit.
"""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research1.data_pipeline.policy import POLICY_ID, collect_candidates, render_messages, validate_plan
from research1.data_pipeline.diagnostics import review_diagnostics
from research1.tests.test_data_sft import record


class PlanningWorkflowV5Tests(unittest.TestCase):
    def setUp(self):
        self.state = record(action='keep')['state']
        self.state['plan_policy_version'] = POLICY_ID
        self.metadata = {'raw_field':'content','finish_reason':'stop','temperature':0,
                         'provider':'fixed','model':'fixed'}

    def test_current_objective_begins_with_workflow_and_separates_registry_from_ideas(self):
        self.assertEqual(POLICY_ID, 'proposal_single_plan_v6')
        system = render_messages(self.state)[0]['content']
        positions = [system.index(part) for part in ('First compare', 'Then connect', 'Form a mechanism', 'Finally choose')]
        self.assertEqual(positions, sorted(positions))
        self.assertLess(positions[0], system.index('Do not infer'))
        self.assertIn('current-code facts and unproven expected effects clearly distinguished', system)
        self.assertIn('a proven implementation defect, demonstrated benefit or proven cause', system)
        self.assertIn('A listed interface does not prove applicability', system)
        self.assertIn('verification_kind=null when no listed interface covers it', system)
        self.assertIn('Such a plan remains pending executable verification', system)
        self.assertIn('Neither edit nor keep is preferred', system)
        self.assertIn('what observation would undermine', system)
        for task_hint in ('bloom','payroll','gmail','normalization','completion_requirement_recheck'):
            self.assertNotIn(task_hint, system.lower())

    def test_empty_catalog_neither_blocks_a_raw_edit_nor_manufactures_a_mechanism(self):
        self.state['verification_catalog'] = []
        plan = record(action='edit')['plan']
        plan['verification_kind'] = None
        before = copy.deepcopy(self.state)
        self.assertEqual(validate_plan(plan, self.state), plan)
        self.assertEqual(self.state, before)
        with tempfile.TemporaryDirectory() as directory:
            raw = json.dumps({'candidates':[plan]})
            with patch('research1.distill._tongji_chat',return_value=(raw,self.metadata)):
                self.assertEqual(collect_candidates(self.state,Path(directory)),[plan])
            saved = json.loads((Path(directory)/'teacher.json').read_text())
            self.assertEqual(saved['raw'],raw)
            self.assertIsNone(saved.get('qualification'))

    def test_keep_remains_legal_and_collection_does_not_fill_candidate_quota(self):
        plan = record(action='keep')['plan']
        self.state['max_candidates'] = 2
        with tempfile.TemporaryDirectory() as directory:
            with patch('research1.distill._tongji_chat',return_value=(json.dumps({'candidates':[plan]}),self.metadata)) as chat:
                self.assertEqual(collect_candidates(self.state,Path(directory)),[plan])
            self.assertEqual(chat.call_count,1)

    def test_attribution_is_explicitly_not_a_generation_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch('research1.distill._tongji_chat',return_value=('{"diagnostics":[]}',self.metadata)) as chat:
                self.assertEqual(review_diagnostics(self.state,Path(directory)),[])
            prompt = chat.call_args.args[0]
            self.assertIn('not to decide whether the planner may propose any improvement',prompt)
            self.assertIn('it is not evidence for keep',prompt)
            self.assertIn('do not invent a registered match',prompt)
            self.assertFalse(json.loads((Path(directory)/'diagnostic_review.json').read_text())['causal_claim'])


if __name__ == '__main__':
    unittest.main()
