"""Offline saved-v3 admission: real physical publisher/factual proofs, no API."""
import copy
import hashlib
import json
from pathlib import Path
import shutil
import unittest
from unittest.mock import patch

from research1.data_pipeline import teacher_replay as replay, modules_v3, edit_trial, selector_review
from research1.data_pipeline.engine import ArtifactChangedError
from research1.data_pipeline.execution import tree_hash
from research1.data_pipeline.fixture_context import freeze_probe_contract
from research1.data_pipeline.ledger import Ledger
from research1.data_pipeline.policy import public_view, review_plan
from research1.data_pipeline.protocol import _validate_registry_options, V3_REGISTRY
from research1.data_pipeline.util import digest
from research1.tests import test_data_probe_evidence_admission as probe_fixtures
from research1.tests import test_data_edit_trial as trial_fixtures


class SameV3ReplayTests(unittest.TestCase):
    def setUp(self):
        case=probe_fixtures.ProbeEvidenceAdmissionTests();case.setUp();self.addCleanup(case.doCleanups)
        trial=trial_fixtures.EditTrialTests();trial.setUp();self.addCleanup(trial.doCleanups)
        self.trial=trial;self.case=case;self.source=case.root;self.target=case.root/'target'
        self.target.mkdir()
        relative='src/research1/data_pipeline/modules_v3.py'
        fixture_path=self.source/'source_snapshot'/relative;fixture_path.parent.mkdir(parents=True)
        shutil.copyfile(modules_v3.__file__,fixture_path)
        original_protocol={'verification_registry_version':V3_REGISTRY,
            'source_sha256':{relative:hashlib.sha256(fixture_path.read_bytes()).hexdigest()},
            'registry_sha256':digest(modules_v3.frozen_registry()),'budget':{'regular':88,'retries':32,'total':120},
            'grounded_edit_trial':{'teacher_max_tokens':16000,'publish_probe_io':True},
            'policies':{'teacher':{'model':'fixed','provider':'fixed'},'factual_reviewer':{'version':replay.FACTUAL_POLICY}}}
        original_protocol['protocol_hash']=digest(original_protocol)
        (self.source/'protocol.json').write_text(json.dumps(original_protocol))
        case.protocol_hash=original_protocol['protocol_hash'];case.state['decision_protocol_hash']=case.protocol_hash
        fixture=freeze_probe_contract(fixture_path)
        state=case.publish();state.update(factual_review_policy=replay.FACTUAL_POLICY,
            applicability_fixture_source=fixture,applicability_policy='proposal-plan-applicability.v2.fixtures',
            code_hash=tree_hash(Path(state['modules_root'])),proposal_task_ids=['bloom'])
        self.row=copy.deepcopy(trial.source);self.row.update(root_lineage_id=state['root_lineage_id'],split=state['split'])
        self.row['parent_result']['evaluation_spec'].update(modules_root=state['modules_root'],code_hash=state['code_hash'])
        state['parent_execution_provenance']={'original_evaluation_id':self.row['parent_result']['evaluation_id'],
            'original_result_sha256':self.row['parent_result']['result_sha256'],
            'original_trajectory_sha256':self.row['parent_result']['trajectory_sha256'],
            'source_snapshot_sha256':'a'*64}
        self.plan=case.plan;self.original_states={};self.reviews={};arms={}
        ledger=Ledger(self.source)
        for arm,effort in replay.ARMS.items():
            current=copy.deepcopy(state);current.update(state_id='source-'+arm,teacher_arm_id=arm,
                teacher_reasoning_effort=effort,max_candidates=1,teacher_max_tokens=16000,allow_teacher_http_fallback=False,
                teacher_record_path=str(self.source/'decisions'/'edittrial'/arm/'teacher.json'))
            teacher_path=Path(current['teacher_record_path']);teacher_path.parent.mkdir(parents=True)
            prompt=replay._prompt(current)
            record={'prompt':prompt,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),
                'visible_view_sha256':digest(public_view(current)),'policy_id':current['plan_policy_version'],
                'request_settings':{'reasoning_effort':effort,'max_tokens':16000,'temperature':0,'allow_http_fallback':False},
                'metadata':{**case.meta,'reasoning_effort':effort},'raw':json.dumps({'candidates':[self.plan]})}
            teacher_path.write_text(json.dumps(record))
            state_path=self.source/'states'/current['state_id']/'state.json';state_path.parent.mkdir(parents=True)
            state_path.write_text(json.dumps(current));self.original_states[arm]=current
            catalog=selector_review.build_witness_catalog(public_view(current),review_policy=replay.FACTUAL_POLICY,state=current)
            item=next(row for row in catalog['items'] if row['kind']=='local_api_observation')
            reply={'status':'PASS','reason':'Bounded API evidence only, no task-causation claim.',
                'evidence_refs':[item['evidence_ref']],'witness_ids':[item['id']],
                'code_witnesses':[{'path':'verification/baseline.py','text':'async def should_terminate('}]}
            with patch('research1.distill._tongji_chat',return_value=(json.dumps(reply),case.meta)):
                review=review_plan(current,self.plan,self.source/'reviews'/'edittrial'/arm)
            self.assertEqual(review['status'],'PASS',review);self.reviews[arm]=review
            binding={'status':'pending','reason':'Original reviewer parsing limitation'}
            sample_state={**current,'plan_factual_review':review,'plan_contract_binding':binding}
            sample={'sample_id':'sample-original-'+arm,'state_id':current['state_id'],'state':sample_state,
                'teacher_record_path':str(teacher_path),'source_kind':'fresh_teacher_plan_from_reused_parent','plan':self.plan,
                'qualification':{'status':'pending','factual_review':review,'applicability':binding,
                    'implementation_task_result':None},'implementation':{},'revoked':False}
            ledger.put_record('samples',sample['sample_id'],sample)
            arms[arm]={'state_path':str(state_path),'teacher_path':str(teacher_path)}
        contract_path=self.source/'edittrial_preteacher_contract.json';contract_path.write_text(json.dumps(state['contract']))
        artifacts={}
        for label,path in [('contract',contract_path)]+[(arm+'_'+key,Path(value)) for arm,values in arms.items() for key,value in values.items()]:
            artifacts[label]={'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        source=replay.freeze_factual_replay_source({'campaign':str(self.source),'arms':arms,'contract_path':str(contract_path),'artifacts':artifacts})
        self.protocol=copy.deepcopy(trial.protocol);self.protocol.update(verification_registry_version=V3_REGISTRY,
            modules_root=state['modules_root'],modules_hash=state['code_hash'],protocol_hash='target-protocol',
            sft={'plan_policy_version':current['plan_policy_version']},policies={
                'teacher':{'model':'fixed','provider':'fixed'},'factual_reviewer':{'version':replay.FACTUAL_POLICY},
                'implementer':{'version':'five-module-exact-replacements.v3'},
                'applicability_reviewer':{'version':replay.APPLICABILITY_POLICY,'fixture_source':fixture}})
        self.protocol['grounded_edit_trial'].update(teacher_source=source,teacher_max_tokens=16000,replay_factual_from_source=True)
        (self.target/'protocol.json').write_text(json.dumps(self.protocol))
        self.no_api=patch('research1.distill._tongji_chat',side_effect=AssertionError('No API allowed after fixture setup'))
        self.no_api.start();self.addCleanup(self.no_api.stop)

    def load(self):return replay.load_teacher_replay(self.protocol,self.row)

    def test_exact_public_view_raw_plan_final_contract_and_factual_results_survive(self):
        before={v['path']:Path(v['path']).read_bytes() for v in self.protocol['grounded_edit_trial']['teacher_source']['artifacts'].values()}
        loaded=self.load()
        for arm in replay.ARMS:
            state=loaded['arms'][arm]['state']
            self.assertEqual(public_view(state),public_view(self.original_states[arm]))
            self.assertEqual(state['decision_protocol_hash'],self.original_states[arm]['decision_protocol_hash'])
            self.assertEqual(state['original_factual_review'],self.reviews[arm])
            self.assertEqual(loaded['arms'][arm]['plans'],[self.plan])
            self.assertEqual(set(replay.validate_replayed_factual_state(state,self.plan)),{
                'factual_review_sha256','factual_raw_path','factual_raw_record_sha256',
                'factual_result_path','factual_result_record_sha256'})
        self.assertEqual(before,{path:Path(path).read_bytes() for path in before})

    def test_revocation_is_live_even_after_successful_source_load(self):
        state=self.load()['arms']['standard']['state']
        Ledger(self.source).revoke('sample-original-standard','Independent contradiction')
        with self.assertRaisesRegex(ArtifactChangedError,'revoked'):
            replay.validate_replayed_factual_state(state,self.plan)

    def test_started_editor_without_raw_blocks(self):
        ledger=Ledger(self.source)
        ledger.put_record('edittrial_call_starts','edittrial_implementations:reasoned',{'raw_path':'missing'})
        with self.assertRaisesRegex(ArtifactChangedError,'already started'):
            self.load()

    def test_reserved_child_before_any_result_blocks(self):
        self.load()
        Ledger(self.source).reserve_job('reserved-child',{'scope':'edittrial:child:standard','role':'proposal'})
        with self.assertRaisesRegex(ArtifactChangedError,'already reserved'):
            self.load()

    def test_source_sample_only_allows_exact_postreview_metadata(self):
        self.load() # Actual driver adds precisely these two fields after state.json freeze.
        ledger=Ledger(self.source);row=ledger.records('samples')[0]
        row['state']['plan_contract_binding']={'status':'PASS'}
        ledger.put_record('samples',row['sample_id'],row)
        with self.assertRaisesRegex(ArtifactChangedError,'identity/plan changed'):self.load()

    def test_manifest_tamper_fixture_or_factual_bytes_blocks_without_generation(self):
        source=self.protocol['grounded_edit_trial']['teacher_source'];path=Path(source['arms']['reasoned']['factual_review_path'])
        path.write_bytes(path.read_bytes()+b' ')
        with self.assertRaisesRegex(ArtifactChangedError,'bytes changed'):self.load()

    def test_new_trial_attempt_never_invalidates_original_source_trial(self):
        loaded=self.load();Ledger(self.target).put_record('edittrial_call_starts','edittrial_implementations:standard',{})
        replay.validate_replayed_factual_state(loaded['arms']['standard']['state'],self.plan)

    def test_full_driver_reuses_fact_reviews_and_only_calls_new_applicability(self):
        calls=[]
        def forbidden(*args,**kwargs):raise AssertionError('No generation, factual review or probes')
        def app(state,plan,review,directory):
            calls.append(state['teacher_arm_id']);self.assertEqual(review,self.reviews[state['teacher_arm_id']])
            return {'status':'pending','reason':'Independent offline app3 remains undecided'}
        kwargs=dict(evaluator=forbidden,proposer=forbidden,fact_collector=forbidden,interface_builder=forbidden,
            contract_builder=forbidden,evidence_selector=forbidden,reviewer=forbidden,implementer=forbidden,
            verifier=forbidden,applicability_reviewer=app,source_validator=lambda *args:[self.row])
        with patch('research1.data_pipeline.protocol.load_protocol',return_value=self.protocol):
            result=edit_trial.run_grounded_edit_trial(self.target,**kwargs)
            self.assertEqual(result['status'],'GROUNDED_EDIT_TRIAL_NO_QUALIFIED_EDIT',result)
            second=edit_trial.run_grounded_edit_trial(self.target,**kwargs)
        self.assertEqual(calls,['standard','reasoned']);self.assertEqual(result['counts'],second['counts'])
        ledger=Ledger(self.target)
        self.assertEqual(len(ledger.records('edittrial_replayed_factual_reviews')),2)
        self.assertFalse(ledger.records('edittrial_factual_reviews'));self.assertFalse(ledger.records('edittrial_call_starts'))
        self.assertEqual(ledger.status()['budget']['used'],{})
        self.assertTrue(all(row['original_sample_id'].startswith('sample-original-') for row in ledger.records('samples')))

    def test_registry_option_requires_explicit_same_v3_opt_in(self):
        options=dict(teacher_replay_from=self.source,teacher_max_tokens=16000,replay_factual_from_source=True,
            admission_review_policy=replay.FACTUAL_POLICY,applicability_review_policy=replay.APPLICABILITY_POLICY,
            applicability_fixtures=True)
        self.assertEqual(_validate_registry_options(V3_REGISTRY,{'tasks':['bloom']},**options),V3_REGISTRY)
        with self.assertRaises(ValueError):_validate_registry_options(None,{'tasks':['bloom']},**options)
        with self.assertRaises(ValueError):_validate_registry_options(V3_REGISTRY,{'tasks':['bloom']},**dict(options,publish_probe_io=True))


if __name__=='__main__':unittest.main()
