"""New receipt-backed API facts are independent of agent-trajectory success.

All captures/model replies here are explicit offline fixtures, not collection
examples. V7 remains unchanged and never gains API eligibility retrospectively.
"""
import copy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from research1.data_pipeline import applicability,modules_v3 as adapter,selector_review as selector
from research1.data_pipeline import predecision_probe_io as publisher
from research1.data_pipeline.policy import canonical,public_view,probe_source_strength,review_plan
from research1.data_pipeline.engine import _public_state
from research1.tests import test_data_probe_evidence_admission as fixtures


class ProbeIOFactualReviewV8Tests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixtures.ProbeEvidenceAdmissionTests('test_verified_new_source_note_is_explicit_without_fake_agent_or_success')
        self.fixture.setUp();self.addCleanup(self.fixture.doCleanups)
        self.fixture.state['evidence'].append({'id':'plain-context','text':'Original bounded requirement context.','partition':'proposal'})
        with patch.object(adapter,'probe_module',side_effect=fixtures.probe):
            self.fixture.contract=adapter.build_contract(self.fixture.state)
        self.fixture.state['contract']=self.fixture.contract
        self.state=self.fixture.publish();self.state['factual_review_policy']=selector.PROBE_POLICY_ID
        self.root=self.fixture.root;self.meta=self.fixture.meta;self.plan=self.fixture.plan

    def catalog(self,state=None,policy=None):
        state=state or self.state;policy=policy or state['factual_review_policy']
        return selector.build_witness_catalog(public_view(state),review_policy=policy,state=state)

    def reply(self,state=None,ids=None,code=None):
        state=state or self.state;catalog=self.catalog(state)
        ids=ids or [next(row['id'] for row in catalog['items'] if row['kind']=='local_api_observation')]
        lookup={row['id']:row for row in catalog['items']}
        return {'status':'PASS','reason':'The verified local API return supports this bounded factual review; future effects remain unproven.',
            'evidence_refs':list(dict.fromkeys(lookup[identifier]['evidence_ref'] for identifier in ids)),
            'witness_ids':ids,'code_witnesses':code if code is not None else
                [{'path':'verification/baseline.py','text':'async def should_terminate('}]}

    def review(self,state=None,reply=None,directory='review'):
        state=state or self.state;reply=reply or self.reply(state)
        with patch('research1.distill._tongji_chat',return_value=(json.dumps(reply),self.meta)) as chat:
            result=review_plan(state,self.plan,self.root/directory)
        return result,chat

    def test_api_only_selection_can_pass_without_an_unrelated_agent_quote(self):
        result,chat=self.review()
        self.assertEqual(result['status'],'PASS',result)
        self.assertEqual(result['review_policy'],selector.PROBE_POLICY_ID)
        self.assertEqual(len(result['witnesses']),1)
        witness=result['witnesses'][0]
        self.assertEqual(witness['kind'],'local_api_observation')
        validation=witness['validation']
        self.assertEqual(validation['validator'],selector.LOCAL_API_VALIDATOR)
        self.assertFalse(validation['original_agent_step'])
        self.assertFalse(validation['task_success_established'])
        self.assertFalse(validation['shell_success_established'])
        self.assertFalse(validation['captured_input_established'])
        self.assertEqual(set(validation['artifact_proofs']),{'raw_contract','receipt'})
        self.assertIn('definition-derived',validation['input_scope'])
        self.assertIn('NEVER an agent trajectory step',chat.call_args.args[0])
        with patch('research1.distill._tongji_chat') as repeated:
            proof=selector.verify_saved_selected_review(result,self.state,self.plan)
            self.assertTrue(applicability.validate_factual_review(result,self.state,self.plan))
        repeated.assert_not_called();self.assertTrue(proof['factual_raw_record_sha256'])

    def test_public_refs_are_derived_from_physical_proof_not_asserted_state(self):
        proof=publisher.validate_probe_io_state(self.state);view=public_view(self.state)
        self.assertEqual(view['predecision_probe_evidence_refs'],proof['evidence_ids_added'])
        altered=copy.deepcopy(self.state);altered['predecision_probe_evidence_refs']=['invented']
        self.assertEqual(public_view(altered),view)
        forged=copy.deepcopy(view);forged['predecision_probe_evidence_refs']=['invented']
        with self.assertRaises(ValueError):
            selector.build_witness_catalog(forged,review_policy=selector.PROBE_POLICY_ID,state=self.state)
        with self.assertRaises(ValueError):
            selector.build_witness_catalog(view,review_policy=selector.PROBE_POLICY_ID)

    def test_missing_receipt_marker_or_tampered_projection_never_become_api_facts(self):
        for field in ('predecision_probe_io_policy','predecision_probe_artifacts'):
            state=copy.deepcopy(self.state);state.pop(field)
            with patch('research1.distill._tongji_chat') as chat:
                with self.assertRaises((ValueError,KeyError)):
                    review_plan(state,self.plan,self.root/field)
            chat.assert_not_called()
        state=copy.deepcopy(self.state);state['evidence'][-1]['text']+=' changed'
        with self.assertRaises(ValueError):self.catalog(state)

    def test_plain_context_or_source_alone_is_not_an_observed_fact(self):
        identifier=next(row['id'] for row in self.catalog()['items'] if row['evidence_ref']=='plain-context')
        result,_=self.review(reply=self.reply(ids=[identifier]))
        self.assertEqual(result['status'],'pending_review')
        self.assertIn('physically verified known-API capture',result['reason'])
        bad=self.reply();bad['code_witnesses']=[{'path':'verification/baseline.py','text':'fabricated code'}]
        result,_=self.review(reply=bad,directory='bad-source')
        self.assertEqual(result['status'],'pending_review')

    def test_unknown_producer_even_with_matching_parser_case_keys_stays_context(self):
        state=copy.deepcopy(self.fixture.state);contract=copy.deepcopy(self.fixture.contract)
        known=contract['parent_probes']['false_concat_preservation']['parent_probe']['results']
        unknown='verification_error_gate'
        contract['parent_probes'][unknown]['parent_probe']['results']=copy.deepcopy(known)
        for row in contract['preprobed']:
            if row['mechanism_id']==unknown:row['parent_probe']=copy.deepcopy(contract['parent_probes'][unknown]['parent_probe'])
        contract['checks_id']='checks_'+adapter._digest({key:value for key,value in contract.items() if key!='checks_id'})[:24]
        state['contract']=contract
        state,_=publisher.publish_parent_probe_io(state,contract,self.root/'unknown-producer',protocol_hash=self.fixture.protocol_hash)
        state['factual_review_policy']=selector.PROBE_POLICY_ID
        catalog=self.catalog(state)
        unknown_rows=[row for row in catalog['items'] if row['evidence_ref'] in public_view(state)['predecision_probe_evidence_refs']
            and json.loads(row['text']).get('invoked_api') is None]
        self.assertTrue(unknown_rows);self.assertTrue(all(row['kind']=='context' for row in unknown_rows))
        result,_=self.review(state,self.reply(state,[unknown_rows[0]['id']]),'unknown-review')
        self.assertEqual(result['status'],'pending_review')

    def test_same_view_v7_keeps_original_catalog_prompt_and_trajectory_gate(self):
        state=copy.deepcopy(self.state);state['factual_review_policy']=selector.POLICY_ID
        view=public_view(state);catalog=selector.build_witness_catalog(view)
        refs=set(view['predecision_probe_evidence_refs'])
        local=[row for row in catalog['items'] if row['evidence_ref'] in refs]
        self.assertTrue(local);self.assertTrue(all(row['kind']=='context' for row in local))
        request,rebuilt=selector._request(state,self.plan)
        payload={'policy':selector.POLICY_ID,'original_view':view,'witness_catalog':catalog,'plan':self.plan}
        self.assertEqual(request['prompt'],selector._INSTRUCTIONS+probe_source_strength(view)+canonical(payload))
        self.assertEqual(rebuilt,catalog)
        result,_=self.review(state,self.reply(state,[local[0]['id']]),'old-v7')
        self.assertEqual(result['status'],'pending_review')

    def test_saved_api_pass_reconstructs_raw_ids_and_both_actual_artifact_hashes(self):
        result,_=self.review()
        with patch('research1.distill._tongji_chat') as chat:
            selector.verify_saved_selected_review(result,self.state,self.plan)
        chat.assert_not_called()
        path=Path(self.state['predecision_probe_artifacts']['raw_contract']['path']);before=path.read_bytes()
        path.write_bytes(before+b' ')
        with self.assertRaises(ValueError):selector.verify_saved_selected_review(result,self.state,self.plan)
        path.write_bytes(before)
        raw_path=Path(result['review_path']);before_raw=raw_path.read_bytes();saved=json.loads(before_raw)
        reply=json.loads(saved['raw'])
        reply['witness_ids']=[next(row['id'] for row in self.catalog()['items'] if row['evidence_ref']=='plain-context')]
        reply['evidence_refs']=['plain-context'];saved['raw']=json.dumps(reply)
        saved['raw_sha256']=hashlib.sha256(saved['raw'].encode()).hexdigest()
        raw_path.write_text(json.dumps(saved),encoding='utf-8')
        with self.assertRaises(ValueError):selector.verify_saved_selected_review(result,self.state,self.plan)
        raw_path.write_bytes(before_raw)
        wrong=copy.deepcopy(result);wrong['witnesses'][0]['validation']['original_agent_step']=True
        with self.assertRaises(ValueError):selector.verify_saved_selected_review(wrong,self.state,self.plan)
        other=copy.deepcopy(self.state);other['factual_review_policy']=selector.POLICY_ID
        with self.assertRaises(ValueError):selector.verify_saved_selected_review(result,other,self.plan)

    def test_engine_passes_only_the_explicit_new_review_policy_metadata(self):
        protocol={'protocol_id':'p','policies':{'factual_reviewer':{'version':selector.PROBE_POLICY_ID}},
            'sft':{'plan_policy_version':applicability.PLAN_POLICY},'verification_registry_version':adapter.REGISTRY_VERSION}
        state=_public_state(Path(self.fixture.state['modules_root']),self.fixture.state['evidence'],[],
            'r','train',['task'],protocol)
        self.assertEqual(state['factual_review_policy'],selector.PROBE_POLICY_ID)
        self.assertNotIn('factual_review_policy',public_view(state))


if __name__=='__main__':unittest.main()
