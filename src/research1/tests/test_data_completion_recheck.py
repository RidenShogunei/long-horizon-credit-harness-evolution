"""Real subprocess probes for a bounded requirement-review prompt interface.

The miniature runtime keeps the actual parent function contracts/control flow.
No scripted model answer is treated as evidence of improved solver correctness.
"""
import copy
from pathlib import Path
import tempfile
import unittest

from research1.data_pipeline.modules import build_contract, frozen_registry, probe_module


PARENT = '''from types import SimpleNamespace

class BaselineAgentLoop:
    def _build_observation_text(self, *, parse, obs_result, tool_results, state, pending_completion):
        parts = [tr.output if tr.output else "[tool error] " + tr.error for tr in tool_results if tr.output or tr.error]
        if obs_result.text:
            parts.append(obs_result.text)
        combined = "\\n\\n".join(parts)
        if parse.is_task_complete:
            state.consecutive_complete_signals += 1
            if pending_completion:
                return combined, pending_completion
            return self._completion_confirmation(combined), True
        state.consecutive_complete_signals = 0
        if parse.warning:
            combined = "Previous response had warnings:\\nWARNINGS: " + parse.warning + "\\n\\n" + combined
        return combined, False

    def _completion_confirmation(self, terminal_output):
        return "You declared the task complete. Please verify with a final check.\\n" + terminal_output

    def _next_prompt(self, observation_text, parse, tools, ctx):
        return observation_text if parse.is_task_complete else tools.format_continuation_prompt(observation_text, ctx)

    async def _execute_commands(self, parse, tools, ctx, *, iteration):
        results = []
        for call in parse.commands:
            results.append(await tools.execute(call, ctx))
        return results, results[-1] if results else None

    def _build_parse_error_prompt(self, parse, tools):
        return tools._parser_name + ": " + parse.error + " " + parse.warning

    async def _check_session_alive(self, ctx):
        try:
            if not await ctx.shared.tmux_session.is_session_alive():
                return SimpleNamespace(failure_tag="session_dead")
        except Exception:
            return None
        return None
'''

ORIGINAL_NEXT = '''        return observation_text if parse.is_task_complete else tools.format_continuation_prompt(observation_text, ctx)'''

RECHECK_NEXT = '''        if not parse.is_task_complete:
            return tools.format_continuation_prompt(observation_text, ctx)
        instruction = getattr(getattr(ctx, "state", None), "instruction", "") or ""
        request = "Compare behavioral checks and untested cases with the original requirements before confirming."
        return observation_text + "\\n" + request + "\\nOriginal task:\\n" + instruction[:4096]'''


class CompletionRecheckTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / 'agent_loop').mkdir()
        self.path = self.root / 'agent_loop/baseline.py'
        self.path.write_text(PARENT, encoding='utf-8')

    def probe(self, source=PARENT):
        self.path.write_text(source, encoding='utf-8')
        return probe_module(self.root, 'completion_requirement_recheck')

    def test_registered_scope_is_runtime_prompt_plumbing_not_score_causation(self):
        registry = frozen_registry()
        self.assertEqual(registry['version'], 'research1.module-contracts.v2')
        pack = registry['packs']['completion_requirement_recheck']
        self.assertEqual(pack['module'], 'agent_loop')
        self.assertEqual(pack['signal'], 'explicit_requirement_coverage_gap')
        self.assertTrue(pack['diagnostic_review_required'])
        self.assertEqual(pack['allowed_paths'], ['agent_loop/baseline.py'])
        self.assertIn('not solver compliance', pack['coverage'])
        self.assertNotIn('bloom', str(pack).lower())
        self.assertNotIn('payroll', str(pack).lower())

    def test_parent_reproduces_missing_request_with_all_preservation_guards(self):
        result = self.probe()
        self.assertFalse(result['checks']['completion_requirements_rechecked'])
        guards = frozen_registry()['packs']['completion_requirement_recheck']['preserve_checks']
        self.assertTrue(all(result['checks'][name] is True for name in guards), result)
        self.assertEqual(set(result['checks']), set(guards + ['completion_requirements_rechecked']))
        self.assertIn('not solver compliance', result['scope'])

    def test_dynamic_bounded_request_satisfies_runtime_interface_and_preservation(self):
        result = self.probe(PARENT.replace(ORIGINAL_NEXT, RECHECK_NEXT))
        self.assertTrue(all(result['checks'].values()), result)

    def test_copying_instruction_alone_is_not_a_behavioral_coverage_request(self):
        source = PARENT.replace(ORIGINAL_NEXT, '''        if not parse.is_task_complete:
            return tools.format_continuation_prompt(observation_text, ctx)
        return observation_text + str(getattr(ctx.state, "instruction", "") or "")[:4096]''')
        self.assertFalse(self.probe(source)['checks']['completion_requirements_rechecked'])

    def test_fixed_task_instruction_cannot_satisfy_dynamic_context_goal(self):
        fixed = RECHECK_NEXT.replace('instruction[:4096]', '"REQ-C17: render rows in ascending order; retain Unicode identifiers and handle an empty collection."')
        self.assertFalse(self.probe(PARENT.replace(ORIGINAL_NEXT, fixed))['checks']['completion_requirements_rechecked'])

    def test_unbounded_instruction_copy_fails_resource_guard(self):
        result = self.probe(PARENT.replace(ORIGINAL_NEXT, RECHECK_NEXT.replace('instruction[:4096]', 'instruction')))
        self.assertTrue(result['checks']['completion_requirements_rechecked'])
        self.assertFalse(result['checks']['completion_prompt_bounded'])

    def test_continuation_corruption_fails_preservation(self):
        changed = RECHECK_NEXT.replace('return tools.format_continuation_prompt(observation_text, ctx)', 'return "discarded terminal"')
        result = self.probe(PARENT.replace(ORIGINAL_NEXT, changed))
        self.assertTrue(result['checks']['completion_requirements_rechecked'])
        self.assertFalse(result['checks']['noncomplete_prompt_exact'])

    def test_instruction_mutation_does_not_satisfy_preservation(self):
        changed = RECHECK_NEXT.replace('        request = ', '        ctx.state.instruction = "overwritten"\n        request = ')
        result = self.probe(PARENT.replace(ORIGINAL_NEXT, changed))
        self.assertTrue(result['checks']['completion_requirements_rechecked'])
        self.assertFalse(result['checks']['instruction_source_unchanged'])

    def state(self):
        witness = 'The completion follow-up only compiled/imported the module although a public requirement lacked a matching behavioral check.'
        return {'state_id': 'source-parent', 'root_lineage_id': 'source-lineage', 'modules_root': str(self.root),
            'permitted_modules': ['agent_loop'], 'module_code': {'agent_loop/baseline.py': PARENT},
            'history': [], 'evidence': [{'id': 'visible-parent', 'text': witness}],
            'diagnostics': [{'module': 'agent_loop', 'mechanism_id': 'completion_requirement_recheck',
                'signal': 'explicit_requirement_coverage_gap', 'evidence_refs': ['visible-parent'],
                'witness': witness, 'review': {'status': 'PASS', 'reviewer_id': 'independent-fixture-review'}}]}

    def test_applicable_contract_still_requires_independent_exact_visible_attribution(self):
        state = self.state()
        self.assertEqual(build_contract(state)['selected'][0]['mechanism_id'], 'completion_requirement_recheck')
        for change in ({'review': {}}, {'witness': 'invented text'}, {'evidence_refs': ['reward-task']},
                       {'signal': 'task_scored_zero'}):
            bad = copy.deepcopy(state)
            bad['diagnostics'][0].update(change)
            self.assertEqual(build_contract(bad)['selected'], [], change)
        state['diagnostics'] = []
        self.assertEqual(build_contract(state)['selected'], [])

    def test_already_rechecking_parent_cannot_claim_missing_request_reproduction(self):
        self.path.write_text(PARENT.replace(ORIGINAL_NEXT, RECHECK_NEXT), encoding='utf-8')
        contract = build_contract(self.state())
        self.assertEqual(contract['selected'], [])
        self.assertIn('parent target not reproduced', contract['pending_reasons'][0])


if __name__ == '__main__':
    unittest.main()
