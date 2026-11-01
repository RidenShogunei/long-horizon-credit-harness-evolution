"""Published local API records keep their source strength and physical binding."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research1.data_pipeline import applicability as app,modules_v3 as mod,selector_review as selector
from research1.data_pipeline import predecision_probe_io as publication
from research1.data_pipeline.policy import public_view,render_messages,review_plan,canonical
from research1.plan_init import frozen_false_concat_checks
from research1.tests.test_data_modules import VERIFIER
from research1.tests import test_data_admission_artifacts as positive_fixture
from research1.tests import test_data_plan_contract_binding as binding_fixture


def probe(root,mechanism,**kwargs):
    pack=mod.frozen_registry()['packs'][mechanism]
    result={'mechanism_id':mechanism,'checks':{**{name:False for name in pack['goal_checks']},
        **{name:True for name in pack['preserve_checks']}}}
    if mechanism=='false_concat_preservation':
        result['results']={row['id']:{'commands':[['fixture-command',0.1]],'warning':'fixture warning',
            'error':None,'is_task_complete':False} for row in frozen_false_concat_checks()}
    return result


class ProbeEvidenceAdmissionTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name);parent=self.root/'parent';(parent/'verification').mkdir(parents=True)
        (parent/'verification/baseline.py').write_text(VERIFIER,encoding='utf-8')
        self.output='Original task terminal payload after explicit tool failure\n'
        trace={'steps':[{'step_id':1,'source':'agent','source_index':0,'visible_step_sha256':'a'*64,
            'tool_calls':[{'tool_call_id':'call1','arguments':{'keystrokes':'python3 check.py\n'}}],
            'observation_excerpt':{'segments':[{'start':0,'end':len(self.output),'text':self.output}]}}]}
        self.state={'state_id':'s','root_lineage_id':'r','split':'train','modules_root':str(parent),
            'module_code':{'verification/baseline.py':VERIFIER},'permitted_modules':list(mod.MODULES),
            'evidence':[{'id':'proposal1','text':json.dumps(trace),'partition':'proposal'}],
            'history':[],'diagnostics':[],'plan_policy_version':app.PLAN_POLICY,
            'verification_registry_version':mod.REGISTRY_VERSION,'factual_review_policy':selector.POLICY_ID,
            'review_model':'fixed','review_provider':'fixed','teacher_model':'fixed','teacher_provider':'fixed'}
        self.protocol_hash='f'*64
        self.state['decision_protocol_hash']=self.protocol_hash
        self.plan={'action':'edit','target_module':'verification','locations':['verification/baseline.py'],
            'evidence_refs':['proposal1'],'problem_hypothesis':'Original observed failure and API facts support a bounded hypothesis.',
            'intended_change':'Retract completion on explicitly failed last tool result.',
            'preserve_behaviors':['normal completion'],'verification_plan':['Unchanged checks.'],
            'verification_kind':'verification_error_gate'}
        self.meta={'provider':'fixed','model':'fixed','temperature':0,'raw_field':'content','finish_reason':'stop'}
        with patch.object(mod,'probe_module',side_effect=probe):self.contract=mod.build_contract(self.state)
        self.state['contract']=self.contract

    def publish(self):
        self.published,self.final=publication.publish_parent_probe_io(self.state,self.contract,self.root/'publication',protocol_hash=self.protocol_hash)
        return self.published

    def factual(self,state,ids=None,directory='fact'):
        catalog=selector.build_witness_catalog(public_view(state))
        ids=ids or [next(row['id'] for row in catalog['items'] if row['field']=='observation_excerpt')]
        refs=list(dict.fromkeys(row['evidence_ref'] for row in catalog['items'] if row['id'] in ids))
        value={'status':'PASS','reason':'Original evidence supports the bounded plan; local API evidence does not prove task causation.',
            'evidence_refs':refs,'witness_ids':ids,
            'code_witnesses':[{'path':'verification/baseline.py','text':'async def should_terminate('}]}
        with patch('research1.distill._tongji_chat',return_value=(json.dumps(value),self.meta)):
            return review_plan(state,self.plan,self.root/directory)

    def test_verified_new_source_note_is_explicit_without_fake_agent_or_success(self):
        old_system=render_messages(self.state)[0]['content'];state=self.publish()
        view=public_view(state);self.assertEqual(view['probe_evidence_policy'],publication.SCHEMA)
        system=render_messages(state)[0]['content'];self.assertTrue(system.startswith(old_system))
        self.assertIn('original task actually triggered',system)
        self.assertIn('not agent trajectory steps',system)
        catalog=selector.build_witness_catalog(view)
        added={row['id'] for row in state['evidence'][len(self.state['evidence']):]}
        local=[row for row in catalog['items'] if row['evidence_ref'] in added]
        self.assertTrue(local)
        self.assertTrue(all(row['kind']=='context' and row['field']=='plain_text' for row in local))
        self.assertTrue(all(row['success_established'] is False and row['producer_established'] is False for row in local))
        self.assertNotIn('predecision_probe_artifacts',canonical(view))
        self.assertNotIn(str(self.root/'publication'),canonical(view))

    def test_selector_and_applicability_append_note_only_for_new_source_flag(self):
        original_request,_=selector._request(self.state,self.plan)
        self.assertNotIn('Additional source-strength rule:',original_request['prompt'])
        state=self.publish();request,catalog=selector._request(state,self.plan)
        self.assertIn('Additional source-strength rule:',request['prompt'])
        local_id=next(row['id'] for row in catalog['items'] if row['field']=='plain_text')
        rejected=self.factual(state,[local_id],'local-only')
        self.assertEqual(rejected['status'],'pending_review')
        review=self.factual(state,directory='real-observation')
        self.assertEqual(review['status'],'PASS',review)
        request,_=app._request(state,self.plan,review,state['contract'])
        self.assertIn('Additional source-strength rule:',request['prompt'])
        self.assertIn('do not establish that the original task',request['prompt'])

    def test_public_input_rejects_raw_receipt_projection_or_final_contract_mutation(self):
        state=self.publish()
        for key in ('raw_contract','receipt'):
            path=Path(state['predecision_probe_artifacts'][key]['path']);before=path.read_bytes()
            path.write_bytes(before+b' ')
            with self.assertRaises(ValueError):public_view(state)
            path.write_bytes(before)
        changed=copy.deepcopy(state);changed['evidence'][-1]['text']+=' altered'
        with self.assertRaises(ValueError):public_view(changed)
        changed=copy.deepcopy(state);changed['contract']['checks_id']='invented'
        with self.assertRaises(ValueError):public_view(changed)

    def test_factual_binding_cannot_survive_changed_physical_publication(self):
        state=self.publish();review=self.factual(state)
        value={'status':'PASS','reason':'The original module data path and unchanged checks cover this plan.',
            'mechanism_id':self.plan['verification_kind'],'evidence_refs':['proposal1'],
            'witnesses':[{'evidence_ref':'proposal1','text':self.output.strip()}],
            'code_refs':[{'path':'verification/baseline.py','text':'async def should_terminate('}],
            'covered_checks':self.final['registry']['packs'][self.plan['verification_kind']]['goal_checks']+
                self.final['registry']['packs'][self.plan['verification_kind']]['preserve_checks']}
        with patch('research1.distill._tongji_chat',return_value=(json.dumps(value),self.meta)):
            binding=app.review_applicability(state,self.plan,review,self.root/'binding')
        self.assertEqual(binding['status'],'PASS',binding)
        self.assertTrue(app.validate_binding(binding,state,self.plan,self.final))
        path=Path(state['predecision_probe_artifacts']['receipt']['path']);path.write_bytes(path.read_bytes()+b' ')
        self.assertFalse(app.validate_binding(binding,state,self.plan,self.final))

    def test_private_future_and_review_metadata_do_not_enter_probe_view(self):
        state=self.publish();view=public_view(state)
        state['future_results']={'value':'DO_NOT_EXPOSE'};state['plan_contract_binding']={'review':'DO_NOT_EXPOSE'}
        self.assertEqual(view,public_view(state));self.assertNotIn('DO_NOT_EXPOSE',canonical(public_view(state)))

    def test_positive_seal_includes_both_physical_publication_artifacts(self):
        original_class=binding_fixture.PlanContractBindingTests
        def new_fixture(name):
            case=original_class(name);old_setup=case.setUp;old_freeze=case.freeze
            def setup():
                old_setup();case.state['verification_registry_version']=mod.REGISTRY_VERSION
                case.state['decision_protocol_hash']='e'*64
                case.probe=probe
            def freeze():
                contract,spy=old_freeze()
                published,final=publication.publish_parent_probe_io(case.state,contract,case.directory/'probe-io',protocol_hash='e'*64)
                case.state.clear();case.state.update(published)
                return final,spy
            case.setUp=setup;case.freeze=freeze;return case
        case=positive_fixture.AdmissionArtifactTests('test_real_saved_fixture_passes_without_model_probe_or_task_execution')
        with patch.object(binding_fixture,'PlanContractBindingTests',side_effect=new_fixture), \
                patch.object(binding_fixture,'mod',mod),patch.object(positive_fixture,'modules',mod):
            case.setUp()
        self.addCleanup(case.doCleanups)
        files=case.record['admission_artifacts']['files']
        for key in ('raw_contract','receipt'):
            self.assertEqual(files['predecision_probe_'+key],case.record['state']['predecision_probe_artifacts'][key])
            path=Path(files['predecision_probe_'+key]['path']);before=path.read_bytes()
            self.assertEqual(hashlib.sha256(before).hexdigest(),files['predecision_probe_'+key]['sha256'])
            path.write_bytes(before+b' ')
            self.assertFalse(positive_fixture.artifacts.validate_positive_artifacts(case.record)[0])
            path.write_bytes(before)
        self.assertTrue(positive_fixture.artifacts.validate_positive_artifacts(case.record)[0])


if __name__=='__main__':unittest.main()
