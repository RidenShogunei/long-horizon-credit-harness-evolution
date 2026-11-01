"""Admission-only selectors/fixture visibility preserve original planning bytes."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research1.data_pipeline import applicability as app,fixture_context as fixtures,modules
from research1.data_pipeline import protocol,selector_review,teacher_replay
from research1.data_pipeline.policy import canonical,digest,public_view,review_plan
from research1.tests.test_data_modules import VERIFIER
from research1.tests import test_data_teacher_replay as replay_fixtures


class AdmissionSelectorIntegrationTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name);self.parent=self.root/'parent'
        (self.parent/'verification').mkdir(parents=True)
        (self.parent/'verification/baseline.py').write_text(VERIFIER,encoding='utf-8')
        self.payload='Observed completion followed explicit tool failure\n'
        self.observation={'steps':[{'step_id':1,'source':'agent','source_index':0,'visible_step_sha256':'a'*64,
            'tool_calls':[{'tool_call_id':'call1','arguments':{'keystrokes':'python3 reproduce.py\n'}}],
            'observation_excerpt':{'segments':[{'start':0,'end':len(self.payload),'text':self.payload}]}}]}
        self.state={'state_id':'s','root_lineage_id':'r','split':'train','modules_root':str(self.parent),
            'module_code':{'verification/baseline.py':VERIFIER},'permitted_modules':list(modules.MODULES),
            'evidence':[{'id':'proposal1','text':json.dumps(self.observation),'partition':'proposal'}],
            'history':[],'diagnostics':[],'plan_policy_version':app.PLAN_POLICY,
            'review_model':'fixed','review_provider':'fixed','teacher_model':'fixed','teacher_provider':'fixed',
            'factual_review_policy':selector_review.POLICY_ID}
        self.plan={'action':'edit','target_module':'verification','locations':['verification/baseline.py'],
            'evidence_refs':['proposal1'],'problem_hypothesis':'Observed completion after explicit failure.',
            'intended_change':'Retract completion when the last tool result explicitly failed.',
            'preserve_behaviors':['normal two-signal completion'],'verification_plan':['Unchanged frozen checks.'],
            'verification_kind':'verification_error_gate'}
        self.meta={'provider':'fixed','model':'fixed','temperature':0,'raw_field':'content','finish_reason':'stop'}
        with patch.object(modules,'probe_module',side_effect=self.parent_probe):
            self.state['contract']=modules.build_contract(self.state)
        # Freeze outside the parent-probe mock: actual loaded fixture identity
        # must match the original snapshot, not a monkeypatched test double.
        self.fixture=self.fixture_snapshot(self.root/'original_campaign')
        self.state.update(applicability_policy=app.FIXTURE_POLICY,applicability_fixture_source=self.fixture)

    def parent_probe(self,root,mechanism,**kwargs):
        pack=modules.frozen_registry()['packs'][mechanism]
        return {'mechanism_id':mechanism,'checks':{**{name:False for name in pack['goal_checks']},
            **{name:True for name in pack['preserve_checks']}}}

    def fixture_snapshot(self,campaign):
        path=campaign/'source_snapshot'/fixtures.RELATIVE_SOURCE;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(Path(modules.__file__).read_bytes())
        manifest={'source_sha256':{fixtures.RELATIVE_SOURCE:hashlib.sha256(path.read_bytes()).hexdigest()},
                  'registry_sha256':digest(modules.frozen_registry())}
        manifest['protocol_hash']=digest(manifest)
        (campaign/'protocol.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
        return fixtures.freeze_probe_contract(path)

    def factual_pass(self):
        catalog=selector_review.build_witness_catalog(public_view(self.state))
        identifier=next(row['id'] for row in catalog['items'] if row['kind']=='observed_tool_output')
        value={'status':'PASS','reason':'Original observation and source support the bounded hypothesis.',
            'evidence_refs':['proposal1'],'witness_ids':[identifier],
            'code_witnesses':[{'path':'verification/baseline.py','text':'async def should_terminate('}]}
        with patch('research1.distill._tongji_chat',return_value=(json.dumps(value),self.meta)):
            review=review_plan(self.state,self.plan,self.root/'factual')
        self.assertEqual(review['status'],'PASS',review)
        self.state['plan_factual_review']=review
        return review

    def applicability_reply(self):
        pack=self.state['contract']['registry']['packs'][self.plan['verification_kind']]
        return {'status':'PASS','reason':'Original visible source is feasible and unchanged frozen checks cover the plan.',
            'mechanism_id':self.plan['verification_kind'],'evidence_refs':['proposal1'],
            'witnesses':[{'evidence_ref':'proposal1','text':self.payload.strip()}],
            'code_refs':[{'path':'verification/baseline.py','text':'async def should_terminate('}],
            'covered_checks':pack['goal_checks']+pack['preserve_checks']}

    def test_selector_factual_and_fixture_inputs_do_not_enter_original_public_view(self):
        original=copy.deepcopy(self.state)
        original.pop('factual_review_policy');original.pop('applicability_policy');original.pop('applicability_fixture_source')
        before=public_view(original);self.assertEqual(public_view(self.state),before)
        review=self.factual_pass()
        saved=json.loads(Path(review['review_path']).read_text(encoding='utf-8'))
        self.assertNotIn(self.fixture['context_id'],saved['prompt'])
        self.assertNotIn('original_fixture_source',saved['prompt'])
        request,_=app._request(self.state,self.plan,review,self.state['contract'])
        payload=json.loads(request['prompt'].rsplit('\n',1)[-1])
        self.assertEqual(payload['policy'],app.FIXTURE_POLICY)
        self.assertEqual(payload['original_fixture_source'],self.fixture)
        self.assertEqual(payload['parent_view'],before)
        self.assertEqual(public_view(self.state),before)

    def test_v1_request_bytes_preserve_original_prefix_and_payload_without_fixtures(self):
        review=self.factual_pass();state=copy.deepcopy(self.state);state.pop('applicability_policy')
        contract=state['contract'];pack=contract['registry']['packs'][self.plan['verification_kind']]
        parent=next(row for row in contract['preprobed'] if row['mechanism_id']==self.plan['verification_kind'])
        expected={'policy':'proposal-plan-applicability.v1','parent_view':public_view(state),'plan':self.plan,
            'factual_review_sha256':digest(review),
            'frozen_checks':{'checks_id':contract['checks_id'],'registry_version':contract['registry']['version'],
                'mechanism_id':pack['mechanism_id'],'module':pack['module'],
                'scope':{key:pack[key] for key in ('allowed_paths','functions','globals','allowed_imports','max_files','max_changed_lines')},
                'goal_checks':pack['goal_checks'],'preserve_checks':pack['preserve_checks'],
                'coverage':pack['coverage'],'parent_probe':parent['parent_probe']}}
        request,_=app._request(state,self.plan,review,contract)
        self.assertEqual(hashlib.sha256(app._INSTRUCTIONS.encode()).hexdigest(),
            'f3a162d540fc6fce99961ba3b6cf7219c53bff9a74e4dc1228fa41105e843958')
        self.assertEqual(request['prompt'],app._INSTRUCTIONS+canonical(expected))
        self.assertEqual(request['request_id'],digest(expected))
        self.assertNotIn(self.fixture['context_id'],request['prompt'])

    def test_new_v2_binding_reconstructs_selector_and_exact_fixture_records(self):
        review=self.factual_pass()
        with patch('research1.distill._tongji_chat',return_value=(json.dumps(self.applicability_reply()),self.meta)):
            binding=app.review_applicability(self.state,self.plan,review,self.root/'applicability')
        self.assertEqual(binding['status'],'PASS',binding);self.assertEqual(binding['policy'],app.FIXTURE_POLICY)
        self.state['plan_contract_binding']=binding
        with patch('research1.distill._tongji_chat') as chat:
            self.assertTrue(app.validate_binding(binding,self.state,self.plan,self.state['contract']))
        chat.assert_not_called()
        modified=copy.deepcopy(self.state);modified['applicability_fixture_source']['text']+='\n# changed'
        self.assertEqual(public_view(modified),public_view(self.state))
        self.assertFalse(app.validate_binding(binding,modified,self.plan,self.state['contract']))
        modified=copy.deepcopy(self.state);modified.pop('applicability_policy')
        self.assertFalse(app.validate_binding(binding,modified,self.plan,self.state['contract']))

    def test_original_fixture_file_change_invalidates_the_new_binding(self):
        review=self.factual_pass()
        with patch('research1.distill._tongji_chat',return_value=(json.dumps(self.applicability_reply()),self.meta)):
            binding=app.review_applicability(self.state,self.plan,review,self.root/'applicability')
        self.assertEqual(binding['status'],'PASS',binding)
        path=Path(self.fixture['path']);path.write_bytes(path.read_bytes()+b'\n# altered fixture\n')
        with patch('research1.distill._tongji_chat') as chat:
            self.assertFalse(app.validate_binding(binding,self.state,self.plan,self.state['contract']))
        chat.assert_not_called()

    def test_teacher_replay_adds_only_private_admission_context_without_generation(self):
        case=replay_fixtures.TeacherReplayTests('test_replay_preserves_exact_view_prompt_plan_contract_and_original_bytes')
        case.setUp();self.addCleanup(case.doCleanups)
        fixture=self.fixture_snapshot(case.source)
        target=copy.deepcopy(case.protocol)
        target['policies']['factual_reviewer']['version']=selector_review.POLICY_ID
        target['policies']['applicability_reviewer']={'version':app.FIXTURE_POLICY,'fixture_source':fixture}
        before={path:path.read_bytes() for path in case.source.glob('*.json')}
        with patch('research1.distill._tongji_chat',side_effect=AssertionError('No calls permitted')):
            loaded=teacher_replay.load_teacher_replay(target,case.row)
        for arm in teacher_replay.ARMS:
            state=loaded['arms'][arm]['state']
            self.assertEqual(public_view(state),public_view(case.states[arm]))
            self.assertEqual(state['factual_review_policy'],selector_review.POLICY_ID)
            self.assertEqual(state['applicability_policy'],app.FIXTURE_POLICY)
            self.assertEqual(state['applicability_fixture_source'],fixture)
            self.assertEqual(teacher_replay._prompt(state),teacher_replay._prompt(case.states[arm]))
        self.assertEqual(before,{path:path.read_bytes() for path in case.source.glob('*.json')})

    def test_prepare_admission_options_require_explicit_original_teacher_replay(self):
        for options in [{'admission_review_policy':selector_review.POLICY_ID},{'applicability_fixtures':True}]:
            with self.assertRaisesRegex(ValueError,'require saved original teacher plans'):
                protocol.prepare(self.root/'refused',self.root/'absent-config.json',root=self.root,**options)
        self.assertFalse((self.root/'refused').exists())


if __name__=='__main__':unittest.main()
