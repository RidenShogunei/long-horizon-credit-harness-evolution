"""Mechanism matching, immutable implementations and real subprocess fixtures."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research1.data_pipeline.modules import (
    MODULES, build_contract, code_hash, frozen_registry, implement, probe_module,
    tree_hashes, verify,
)


VERIFIER = '''class BaselineVerification:
    def __init__(self, required_consecutive=2):
        self.required = max(1, required_consecutive)
    async def should_terminate(self, state, ctx):
        if state.consecutive_complete_signals >= self.required:
            return True, "confirmed"
        return False, ""
'''


class DataModuleTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.modules = self.root / 'parent'
        (self.modules / 'verification').mkdir(parents=True)
        (self.modules / 'verification/baseline.py').write_text(VERIFIER)
        self.witness = 'completion confirmed with [tool error] execution failed'
        self.state = {'state_id':'parent1','root_lineage_id':'lineage',
            'modules_root':str(self.modules), 'permitted_modules':list(MODULES),
            'module_code':{'verification/baseline.py':VERIFIER},
            'evidence':[{'id':'proposal1','text':self.witness}], 'history':[],
            'diagnostics':[{'module':'verification','mechanism_id':'verification_error_gate',
                'signal':'completion_after_tool_error','evidence_refs':['proposal1'],
                'witness':self.witness,'review':{'status':'PASS','reviewer_id':'independent-test'}}]}
        self.plan = {'action':'edit','target_module':'verification',
            'locations':['verification/baseline.py'],'evidence_refs':['proposal1'],
            'problem_hypothesis':'completion confirmed after a failed tool',
            'intended_change':'Retract confirmation on an explicitly failed last tool result',
            'preserve_behaviors':['normal two-consecutive completion'],
            'verification_plan':['frozen goal and preservation checks'],
            'verification_kind':'verification_error_gate'}
        self.fixed = VERIFIER.replace('        if state.consecutive_complete_signals',
            '        if getattr(getattr(state, "last_tool_result", None), "success", True) is False:\n            return False, "tool failed"\n        if state.consecutive_complete_signals')
        self.raw = json.dumps({'files':[{'path':'verification/baseline.py',
            'edits':[{'old':VERIFIER,'new':self.fixed}]}]})
        self.meta = {'provider':'fixed','model':'fixed','temperature':0.0,'raw_field':'content','finish_reason':'stop'}

    def prepared(self):
        self.state['contract'] = build_contract(self.state)
        self.assertEqual(len(self.state['contract']['selected']),1)
        return self.state['contract']

    def applied(self):
        self.prepared()
        with patch('research1.distill._tongji_chat',return_value=(self.raw,self.meta)):
            return implement(self.plan,self.state,self.root/'attempt')

    def test_registry_covers_five_modules_with_executable_goals_and_preservation(self):
        registry = frozen_registry()
        self.assertEqual({p['module'] for p in registry['packs'].values()},set(MODULES))
        for pack in registry['packs'].values():
            self.assertTrue(pack['goal_checks'])
            self.assertGreaterEqual(len(pack['preserve_checks']),4)
            self.assertTrue(set(pack['goal_checks']).isdisjoint(pack['preserve_checks']))
        registry['packs'].clear()
        self.assertEqual(len(frozen_registry()['packs']),6)

    def test_real_subprocess_reproduces_parent_and_fixed_verification_gate(self):
        before = probe_module(self.modules,'verification_error_gate')
        self.assertFalse(before['checks']['failed_tool_retracts_completion'])
        self.assertTrue(all(v for k,v in before['checks'].items() if k!='failed_tool_retracts_completion'))
        (self.modules/'verification/baseline.py').write_text(self.fixed)
        after = probe_module(self.modules,'verification_error_gate')
        self.assertTrue(all(after['checks'].values()))

    def test_claimed_diagnostic_requires_independent_visible_witness_and_reproduction(self):
        for change in ({'review':{}},{'witness':'not in proposal'},{'evidence_refs':['reward1']},{'signal':'generic error'}):
            with self.subTest(change=change):
                state = copy.deepcopy(self.state)
                state['diagnostics'][0].update(change)
                self.assertEqual(build_contract(state)['selected'],[])
        (self.modules/'verification/baseline.py').write_text(self.fixed)
        contract = build_contract(self.state)
        self.assertEqual(contract['selected'],[])
        self.assertIn('not reproduced',contract['pending_reasons'][0])

    def test_unknown_mechanism_is_pending_before_any_model_call(self):
        self.state['contract']=build_contract(self.state)
        plan=dict(self.plan,verification_kind='invented-after-child')
        with patch('research1.distill._tongji_chat') as chat:
            attempt=implement(plan,self.state,self.root/'unknown')
            chat.assert_not_called()
        self.assertEqual(attempt['status'],'pending')
        result=verify(plan,self.state,attempt,self.state['contract'])
        self.assertFalse(result['covers_plan'])
        self.assertEqual(result['status'],'pending')

    def test_contract_rechecks_decoded_witness_before_real_parent_probe(self):
        witness=self.witness+' for "quoted command"\nlast observation'
        self.state['diagnostics'][0]['witness']=witness
        original=json.dumps({'steps':[{'step_id':7,'observation_excerpt':{'segments':[{'text':witness}]}}]})
        self.state['evidence'][0]['text']=original
        self.assertNotIn(witness,original)
        self.assertEqual(len(build_contract(self.state)['selected']),1)
        self.assertEqual(self.state['evidence'][0]['text'],original)
        self.state['diagnostics'][0]['witness']=witness.replace('\n','\\n')
        self.assertEqual(build_contract(self.state)['selected'],[])

    def test_implementation_freezes_patch_and_never_regenerates_on_resume(self):
        contract=self.prepared()
        with patch('research1.distill._tongji_chat',return_value=(self.raw,self.meta)) as chat:
            result=implement(self.plan,self.state,self.root/'attempt')
            repeated=implement(self.plan,self.state,self.root/'attempt')
        self.assertEqual(chat.call_count,1)
        self.assertEqual(result,repeated)
        self.assertEqual(result['status'],'applied')
        self.assertEqual(tree_hashes(self.modules),contract['parent_hashes'])
        self.assertEqual(result['code_hash'],code_hash(Path(result['modules_root'])))
        checked=verify(self.plan,self.state,result,contract)
        self.assertEqual(checked['status'],'PASS')
        self.assertTrue(checked['covers_plan'])
        self.assertTrue(checked['executable'])

    def test_tampered_patch_or_child_is_rejected_not_replayed(self):
        result=self.applied()
        Path(result['patch_path']).write_text('different patch')
        with self.assertRaisesRegex(ValueError,'patch has changed'):
            implement(self.plan,self.state,self.root/'attempt')
        checked=verify(self.plan,self.state,result,self.state['contract'])
        self.assertEqual(checked['status'],'FAIL')

    def test_interrupted_tree_recovery_uses_saved_editor_output_without_model(self):
        result=self.applied()
        (self.root/'attempt/implementation.json').unlink()
        (Path(result['modules_root'])/'verification/baseline.py').write_text('partial bytes')
        with patch('research1.distill._tongji_chat') as chat:
            recovered=implement(self.plan,self.state,self.root/'attempt')
        chat.assert_not_called()
        self.assertEqual(recovered['status'],'applied')
        self.assertEqual(recovered['code_hash'],result['code_hash'])
        self.assertTrue(recovered['preserved_partial_trees'])
        self.assertEqual((Path(result['modules_root'])/'verification/baseline.py').read_text(),'partial bytes')

    def test_changed_contract_or_evidence_cannot_borrow_old_qualification(self):
        self.prepared()
        self.state['contract']['selected'][0]['evidence_refs']=['invented']
        with patch('research1.distill._tongji_chat') as chat:
            self.assertEqual(implement(self.plan,self.state,self.root/'tampered')['status'],'pending')
        chat.assert_not_called()

    def test_scope_rejects_unrelated_module_and_constant_only_changes(self):
        self.prepared()
        malformed={'files':[{'path':'../outside.py','edits':[{'old':'x','new':'y'}]}]}
        with patch('research1.distill._tongji_chat',return_value=(json.dumps(malformed),self.meta)):
            with self.assertRaisesRegex(ValueError,'outside frozen'):
                implement(self.plan,self.state,self.root/'outside')
        new=VERIFIER+'\n# misleading marker\n'
        raw=json.dumps({'files':[{'path':'verification/baseline.py','edits':[{'old':VERIFIER,'new':new}]}]})
        with patch('research1.distill._tongji_chat',return_value=(raw,self.meta)):
            result=implement(self.plan,self.state,self.root/'marker')
        self.assertEqual(result['status'],'rejected')

    def test_reasoning_or_truncated_output_cannot_be_implementation(self):
        self.prepared()
        for metadata in (dict(self.meta,raw_field='reasoning'),dict(self.meta,finish_reason='length'),dict(self.meta,temperature=.7)):
            directory=self.root/('bad-'+str(metadata['raw_field'])+str(metadata['finish_reason'])+str(metadata['temperature']))
            with patch('research1.distill._tongji_chat',return_value=(self.raw,metadata)):
                with self.assertRaisesRegex(ValueError,'complete final JSON'):
                    implement(self.plan,self.state,directory)
            self.assertTrue((directory/'editor_output.json').is_file())

    def test_first_implementation_call_requires_frozen_provider(self):
        self.prepared()
        self.state['editor_provider']='frozen-provider'
        directory=self.root/'wrong-provider'
        with patch('research1.distill._tongji_chat',return_value=(self.raw,self.meta)):
            with self.assertRaisesRegex(ValueError,'provider differs'):
                implement(self.plan,self.state,directory)
        self.assertTrue((directory/'editor_output.json').is_file())
        self.assertFalse((directory/'implementation.json').exists())
        self.assertFalse((directory.parent/'fixed_implementer_policy.json').exists())

    def test_cached_implementation_still_requires_current_frozen_provider(self):
        result=self.applied()
        self.state['editor_provider']='different-frozen-provider'
        with patch('research1.distill._tongji_chat') as chat:
            with self.assertRaisesRegex(ValueError,'provider differs'):
                implement(self.plan,self.state,self.root/'attempt')
        chat.assert_not_called()
        checked=verify(self.plan,self.state,result,self.state['contract'])
        self.assertEqual(checked['status'],'pending')
        self.assertFalse(checked['covers_plan'])

    def test_tools_signal_requires_warning_referenced_to_same_empty_command(self):
        text={'steps':[{'observation_excerpt':{'segments':[{'text':'Command 2 should end with newline when followed by another command'}]},
            'tool_calls':[{'arguments':{'keystrokes':'echo hi','duration':.5}},{'arguments':{'keystrokes':'','duration':5}}]}]}
        state=dict(self.state,diagnostics=[],evidence=[{'id':'empty','text':json.dumps(text)}])
        pack=frozen_registry()['packs']['false_concat_preservation']
        checks={n:False for n in pack['goal_checks']}
        checks.update({n:True for n in pack['preserve_checks']})
        with patch('research1.data_pipeline.modules.probe_module',return_value={'checks':checks}):
            self.assertEqual(build_contract(state)['selected'][0]['mechanism_id'],'false_concat_preservation')
            text['steps'][0]['tool_calls'][1]['arguments']['keystrokes']=' '
            state['evidence'][0]['text']=json.dumps(text)
            self.assertEqual(build_contract(state)['selected'],[])


if __name__=='__main__':
    unittest.main()
