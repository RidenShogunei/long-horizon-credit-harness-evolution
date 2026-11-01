"""New protocol bindings require real frozen probes and immutable review bytes."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research1.data_pipeline import applicability as app
from research1.data_pipeline import modules as mod
from research1.data_pipeline.policy import public_view, review_plan
from research1.tests.test_data_modules import VERIFIER


class PlanContractBindingTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory=Path(self.temporary.name)
        self.parent=self.directory/'parent'
        (self.parent/'verification').mkdir(parents=True)
        (self.parent/'verification/baseline.py').write_text(VERIFIER,encoding='utf-8')
        self.witness='completion confirmed with [tool error] execution failed'
        self.state={'state_id':'parent1','root_lineage_id':'r','split':'train',
            'modules_root':str(self.parent),'module_code':{'verification/baseline.py':VERIFIER},
            'permitted_modules':list(mod.MODULES),'evidence':[{'id':'proposal1','text':self.witness,'partition':'proposal'}],
            'history':[],'diagnostics':[],'plan_policy_version':app.PLAN_POLICY,
            'teacher_model':'fixed','teacher_provider':'fixed','review_model':'fixed','review_provider':'fixed'}
        self.plan={'action':'edit','target_module':'verification','locations':['verification/baseline.py'],
            'evidence_refs':['proposal1'],'problem_hypothesis':'Completion follows the explicitly failed tool.',
            'intended_change':'Retract completion when the last tool result explicitly failed.',
            'preserve_behaviors':['normal consecutive completion'],
            'verification_plan':['Use the frozen target and preservation cases.'],
            'verification_kind':'verification_error_gate'}
        self.meta={'provider':'fixed','model':'fixed','temperature':0,'raw_field':'content','finish_reason':'stop'}
        self.fact_reply={'status':'PASS','reason':'Original visible facts support this scoped mechanism hypothesis.',
            'evidence_refs':['proposal1'],'witnesses':[{'evidence_ref':'proposal1','text':self.witness,'kind':'context'}]}

    def probe(self,root,mechanism,**kwargs):
        pack=mod.frozen_registry()['packs'][mechanism]
        return {'mechanism_id':mechanism,'checks':{**{name:False for name in pack['goal_checks']},
            **{name:True for name in pack['preserve_checks']}}}

    def freeze(self):
        with patch.object(mod,'probe_module',side_effect=self.probe) as probe:
            contract=mod.build_contract(self.state)
        self.state['contract']=contract
        return contract,probe

    def facts(self):
        with patch('research1.distill._tongji_chat',return_value=(json.dumps(self.fact_reply),self.meta)):
            review=review_plan(self.state,self.plan,self.directory/'factual')
        self.assertEqual(review['status'],'PASS')
        self.state['plan_factual_review']=review
        return review

    def reply(self,**updates):
        pack=mod.frozen_registry()['packs'][self.plan['verification_kind']]
        value={'status':'PASS','reason':'The proposed explicit-failure condition is reachable and all listed frozen cases cover its scope; future benefit is unproven.',
            'mechanism_id':self.plan['verification_kind'],'evidence_refs':['proposal1'],
            'witnesses':[{'evidence_ref':'proposal1','text':self.witness}],
            'code_refs':[{'path':'verification/baseline.py','text':'async def should_terminate('}],
            'covered_checks':pack['goal_checks']+pack['preserve_checks']}
        value.update(updates)
        return value

    def binding(self,reply=None,directory=None):
        with patch('research1.distill._tongji_chat',return_value=(json.dumps(reply or self.reply()),self.meta)) as chat:
            value=app.review_applicability(self.state,self.plan,self.state['plan_factual_review'],directory or self.directory/'binding')
        return value,chat

    def test_legacy_build_contract_keys_and_selected_gates_do_not_change(self):
        for version in [None,'proposal_single_plan_v2','proposal_single_plan_v3','proposal_single_plan_v4','proposal_single_plan_v5']:
            state=dict(self.state,plan_policy_version=version)
            with patch.object(mod,'probe_module') as probe:
                value=mod.build_contract(state)
            probe.assert_not_called()
            self.assertEqual(set(value),{'registry','selected','pending_reasons','parent_hashes','evidence_hashes','checks_id'})
            self.assertIsNone(mod._pack(self.plan,state,value))

    def test_v6_freezes_all_probes_without_requiring_parent_diagnostic(self):
        contract,probe=self.freeze()
        self.assertEqual(probe.call_count,len(mod.frozen_registry()['packs']))
        self.assertEqual(contract['selected'],[])
        self.assertEqual(contract['binding_mode'],app.BINDING_MODE)
        self.assertEqual(len(contract['parent_probes']),6)
        self.assertEqual(len(contract['preprobed']),6)
        self.assertIsNone(mod._pack(self.plan,self.state,contract))

    def test_missing_target_or_failed_preservation_not_preprobed(self):
        def probe(root,mechanism,**kwargs):
            value=self.probe(root,mechanism)
            pack=mod.frozen_registry()['packs'][mechanism]
            value['checks'][pack['preserve_checks'][0]]=False
            return value
        with patch.object(mod,'probe_module',side_effect=probe):
            contract=mod.build_contract(self.state)
        self.assertEqual(contract['preprobed'],[])
        self.assertEqual(len(contract['parent_probes']),6)

    def test_asserted_factual_pass_is_not_enough_and_makes_no_call(self):
        self.freeze()
        fake={'status':'PASS','independent':True,'reason':'trust me'}
        with patch('research1.distill._tongji_chat') as chat:
            value=app.review_applicability(self.state,self.plan,fake,self.directory/'fake')
        chat.assert_not_called()
        self.assertEqual(value['status'],'pending')
        self.assertFalse(app.validate_factual_review(fake,self.state,self.plan))

    def test_saved_independent_pass_binds_only_existing_tests_and_scope(self):
        contract,_=self.freeze();self.facts()
        binding,chat=self.binding()
        self.assertEqual(binding['status'],'PASS',binding)
        self.assertFalse(chat.call_args.kwargs['allow_http_fallback'])
        self.assertEqual(chat.call_args.kwargs['reasoning_effort'],'none')
        self.assertTrue(app.validate_binding(binding,self.state,self.plan,contract))
        self.state['plan_contract_binding']=binding
        self.assertEqual(mod._pack(self.plan,self.state,contract)['mechanism_id'],self.plan['verification_kind'])
        visible=json.dumps(public_view(self.state))
        self.assertNotIn('plan_contract_binding',visible)
        self.assertNotIn('factual_review.json',visible)
        self.assertEqual(contract['selected'],[])

    def test_cached_result_does_not_recall_model_and_input_mutation_is_rejected(self):
        self.freeze();self.facts();binding,_=self.binding()
        with patch('research1.distill._tongji_chat') as chat:
            repeated=app.review_applicability(self.state,self.plan,self.state['plan_factual_review'],self.directory/'binding')
        chat.assert_not_called();self.assertEqual(binding,repeated)
        changed=dict(self.plan,intended_change='Another behavior')
        before=(self.directory/'binding/applicability_result.json').read_bytes()
        with self.assertRaisesRegex(ValueError,'preconditions changed'):
            app.review_applicability(self.state,changed,self.state['plan_factual_review'],self.directory/'binding')
        self.assertEqual(before,(self.directory/'binding/applicability_result.json').read_bytes())

    def test_invalid_unknown_or_uncovered_raw_replies_stay_pending_without_repair(self):
        self.freeze();self.facts()
        variants=[self.reply(witnesses=[{'evidence_ref':'proposal1','text':'invented'}]),
            self.reply(code_refs=[{'path':'verification/baseline.py','text':'ctx.nonexistent'}]),
            self.reply(covered_checks=[]),self.reply(status='FAIL'),
            dict(self.reply(),invented_new_tests=['newtest'])]
        for index,reply in enumerate(variants):
            path=self.directory/f'invalid{index}'
            binding,chat=self.binding(reply,path)
            self.assertEqual(binding['status'],'pending')
            self.assertEqual(chat.call_count,1)
            self.assertEqual(json.loads((path/'applicability_output.json').read_text())['raw'],json.dumps(reply))

    def test_unknown_mechanism_or_old_protocol_never_calls_applicability(self):
        self.freeze();self.facts()
        for plan,state in [(dict(self.plan,verification_kind='unknown'),self.state),
                           (self.plan,dict(self.state,plan_policy_version='proposal_single_plan_v5'))]:
            with patch('research1.distill._tongji_chat') as chat:
                value=app.review_applicability(state,plan,self.state['plan_factual_review'],self.directory/str(len(str(plan))))
            chat.assert_not_called();self.assertEqual(value['status'],'pending')

    def test_stale_plan_view_source_contract_or_hand_filled_binding_cannot_apply(self):
        contract,_=self.freeze();self.facts();binding,_=self.binding()
        self.assertFalse(app.validate_binding(dict(binding,status='PASS',reason='hand filled'),self.state,self.plan,contract))
        self.assertFalse(app.validate_binding(binding,self.state,dict(self.plan,problem_hypothesis='changed'),contract))
        changed=copy.deepcopy(self.state);changed['evidence'][0]['text']+=' other observation'
        self.assertFalse(app.validate_binding(binding,changed,self.plan,contract))
        changed=copy.deepcopy(contract);changed['registry']['packs'][self.plan['verification_kind']]['max_changed_lines']+=1
        changed['checks_id']='checks_'+mod._digest({key:value for key,value in changed.items() if key!='checks_id'})[:24]
        self.assertFalse(app.validate_binding(binding,self.state,self.plan,changed))
        (self.parent/'verification/baseline.py').write_text(VERIFIER+'\n# changed\n',encoding='utf-8')
        self.assertFalse(app.validate_binding(binding,self.state,self.plan,contract))

    def test_changed_original_raw_status_invalidates_even_if_result_still_says_pass(self):
        contract,_=self.freeze();self.facts();binding,_=self.binding()
        raw_path=Path(binding['applicability_output_path'])
        value=json.loads(raw_path.read_text());reply=json.loads(value['raw']);reply['status']='FAIL'
        value['raw']=json.dumps(reply);raw_path.write_text(json.dumps(value),encoding='utf-8')
        self.assertFalse(app.validate_binding(binding,self.state,self.plan,contract))
        factual_path=Path(self.state['plan_factual_review']['review_path'])
        value=json.loads(factual_path.read_text());reply=json.loads(value['raw']);reply['status']='pending_review'
        value['raw']=json.dumps(reply);factual_path.write_text(json.dumps(value),encoding='utf-8')
        self.assertFalse(app.validate_factual_review(self.state['plan_factual_review'],self.state,self.plan))

    def test_failed_call_and_started_without_reply_are_not_resent(self):
        self.freeze();self.facts()
        path=self.directory/'failed'
        with patch('research1.distill._tongji_chat',side_effect=RuntimeError('connection failed')) as chat:
            value=app.review_applicability(self.state,self.plan,self.state['plan_factual_review'],path)
        self.assertEqual(chat.call_count,1);self.assertEqual(value['status'],'pending')
        self.assertTrue((path/'applicability_started.json').is_file())
        self.assertTrue((path/'applicability_error.json').is_file())
        with patch('research1.distill._tongji_chat') as chat:
            repeated=app.review_applicability(self.state,self.plan,self.state['plan_factual_review'],path)
        chat.assert_not_called();self.assertIn('CALL_OUTCOME_UNKNOWN',repeated['reason'])

    def test_preselected_v6_still_requires_separately_saved_plan_factual_pass(self):
        self.state['diagnostics']=[{'module':'verification','mechanism_id':'verification_error_gate',
            'signal':'completion_after_tool_error','evidence_refs':['proposal1'],'witness':self.witness,
            'review':{'status':'PASS','reviewer_id':'parent-review'}}]
        contract,_=self.freeze()
        self.assertEqual(len(contract['selected']),1)
        self.assertIsNone(mod._pack(self.plan,self.state,contract))
        self.facts()
        self.assertIsNotNone(mod._pack(self.plan,self.state,contract))

    def test_binding_path_cannot_expand_actual_implementation_scope(self):
        contract,_=self.freeze();self.facts();binding,_=self.binding()
        self.state['plan_contract_binding']=binding
        fixed=VERIFIER.replace('        if state.consecutive_complete_signals',
            '        if getattr(getattr(state, "last_tool_result", None), "success", True) is False:\n            return False, "tool failed"\n        if state.consecutive_complete_signals')
        raw=json.dumps({'files':[{'path':'verification/baseline.py','edits':[{'old':VERIFIER,'new':fixed}]}]})
        with patch('research1.distill._tongji_chat',return_value=(raw,self.meta)):
            attempt=mod.implement(self.plan,self.state,self.directory/'implementation')
        self.assertEqual(attempt['status'],'applied')
        with patch('research1.distill._tongji_chat') as chat:
            repeated=mod.implement(self.plan,self.state,self.directory/'implementation')
        chat.assert_not_called();self.assertEqual(attempt,repeated)
        result=mod.verify(self.plan,self.state,attempt,contract)
        self.assertEqual(result['status'],'PASS')
        self.assertTrue(result['covers_plan'])


if __name__=='__main__':unittest.main()
