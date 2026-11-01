"""Expanded proposal visibility must not silently expand edit admission."""
import copy
import unittest

from research1.data_pipeline.modules import planning_interfaces, proposal_catalog, _pack
from research1.data_pipeline.policy import public_view
from research1.tests.test_data_sft import record


class GenerationCatalogTests(unittest.TestCase):
    def test_new_planner_sees_all_interfaces_when_attribution_is_empty(self):
        state = record()['state']; state['plan_policy_version'] = 'proposal_single_plan_v5'
        state['permitted_modules'] = ['agent_loop', 'observation', 'tools', 'context_mgmt', 'verification']
        contract = {'selected': []}
        catalog = proposal_catalog(state, contract)
        self.assertEqual(len(catalog), 6)
        self.assertEqual(contract, {'selected': []})
        state['verification_catalog'] = catalog
        state['verification_interfaces'] = planning_interfaces(state['permitted_modules'])
        self.assertEqual(public_view(state)['verification_interfaces'], state['verification_interfaces'])

    def test_catalog_visibility_does_not_supply_an_admission_contract(self):
        sample = record(action='edit'); state = sample['state']
        state['plan_policy_version'] = 'proposal_single_plan_v5'
        state['verification_catalog'] = proposal_catalog(state, {'selected': []})
        state['verification_interfaces'] = planning_interfaces(state['permitted_modules'])
        self.assertIsNone(_pack(sample['plan'], state, {'selected': []}))

    def test_old_input_keeps_original_empty_catalog_and_no_added_fields(self):
        for version in (None, 'proposal_single_plan_v2', 'proposal_single_plan_v3', 'proposal_single_plan_v4'):
            state = record()['state']; state['plan_policy_version'] = version
            state['verification_catalog'] = proposal_catalog(state, {'selected': []})
            before = copy.deepcopy(public_view(state))
            self.assertEqual(before['verification_catalog'], [])
            self.assertNotIn('framework_context', before)
            self.assertNotIn('verification_interfaces', before)

    def test_fabricated_capability_is_rejected(self):
        state = record()['state']
        state['verification_interfaces'] = planning_interfaces(state['permitted_modules'])
        state['verification_interfaces'][0]['coverage'] = 'All improvements are guaranteed'
        with self.assertRaises(ValueError):
            public_view(state)
