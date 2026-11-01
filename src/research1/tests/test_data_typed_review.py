"""Typed source/observation boundaries and immutable saved factual proof."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research1.data_pipeline import typed_review as typed
from research1.data_pipeline.policy import digest,review_plan
from research1.tests.test_data_sft import record
from research1.tests.test_data_framework_context import _make_runtime,_protocol,fc


class TypedReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        sample=record(action='edit');self.state=sample['state'];self.plan=sample['plan']
        self.state.update(factual_review_policy=typed.POLICY_ID,review_model='fixed',review_provider='fixed')
        self.output='Returned payload: {"value": 7}\n'
        trace={'steps':[{'step_id':1,'source':'agent','source_index':0,'visible_step_sha256':'a'*64,
            'tool_calls':[{'tool_call_id':'call1','arguments':{'keystrokes':'python3 check.py\n'}}],
            'observation_excerpt':{'segments':[{'start':0,'end':len(self.output),'text':self.output}]}}]}
        self.state['evidence']=[{'id':'evidence-1','text':json.dumps(trace),'partition':'proposal'}]
        self.meta={'provider':'fixed','model':'fixed','temperature':0,'raw_field':'content','finish_reason':'stop'}

    def reply(self,**updates):
        value={'status':'PASS','reason':'The quoted proposal fact and current source support the scoped hypothesis; expected benefit is unproven.',
            'evidence_refs':['evidence-1'],'witnesses':[{'evidence_ref':'evidence-1',
                'text':self.output.strip(),'kind':'observed_tool_output'}],
            'code_witnesses':[{'path':'tools/baseline.py','text':'def parse():\n    return 1'}]}
        value.update(updates);return value

    def run_review(self,reply=None,directory=None,metadata=None):
        raw=json.dumps(reply or self.reply())
        with patch('research1.distill._tongji_chat',return_value=(raw,metadata or self.meta)) as chat:
            result=typed.review_plan_typed(self.state,self.plan,directory or self.root/'typed')
        return result,chat,raw

    def test_separate_observation_and_source_pass_are_bound_to_original_records(self):
        result,chat,raw=self.run_review()
        self.assertEqual(result['status'],'PASS',result)
        self.assertEqual(result['review_policy'],typed.POLICY_ID)
        self.assertEqual(result['witnesses'][0]['kind'],'observed_tool_output')
        self.assertEqual(result['code_witnesses'][0]['validation']['evidence_type'],'source_text_only')
        self.assertEqual(chat.call_args.kwargs['reasoning_effort'],'none')
        self.assertFalse(chat.call_args.kwargs['allow_http_fallback'])
        saved=json.loads(Path(result['review_path']).read_text(encoding='utf-8'))
        self.assertEqual(saved['raw'],raw)
        self.assertEqual(saved['raw_sha256'],hashlib.sha256(raw.encode()).hexdigest())
        self.assertEqual(saved['metadata_sha256'],digest(self.meta))
        proof=typed.verify_saved_typed_review(result,self.state,self.plan)
        self.assertEqual(set(proof),{'factual_review_sha256','factual_raw_path','factual_raw_record_sha256',
            'factual_result_path','factual_result_record_sha256'})
        self.assertEqual(proof['factual_review_sha256'],digest(result))

    def test_readonly_framework_source_has_its_real_path_and_content_hash(self):
        runtime=_make_runtime(self.root/'runtime')
        self.state['framework_context']=fc.build_framework_context(runtime,_protocol())
        quote={'path':fc._PROTOCOLS,'text':'class RuntimeState:'}
        reply=self.reply(code_witnesses=self.reply()['code_witnesses']+[quote])
        result,_,_=self.run_review(reply)
        self.assertEqual(result['status'],'PASS',result)
        refs=result['code_witnesses'][1]['validation']['matches']
        self.assertTrue(any(row['source_kind']=='framework_fragment' and row['path']==fc._PROTOCOLS for row in refs))
        typed.verify_saved_typed_review(result,self.state,self.plan)

    def test_source_quote_in_trajectory_slot_is_rejected_without_relocation(self):
        reply=self.reply(witnesses=[{'evidence_ref':'evidence-1','text':'def parse():\n    return 1','kind':'context'}])
        result,_,raw=self.run_review(reply)
        self.assertEqual(result['status'],'pending_review')
        self.assertNotIn('witnesses',result)
        self.assertEqual(result['code_witnesses'],[])
        saved=json.loads(Path(result['review_path']).read_text(encoding='utf-8'))
        self.assertEqual(saved['raw'],raw)
        with self.assertRaises(ValueError):typed.verify_saved_typed_review(result,self.state,self.plan)

    def test_code_quote_not_present_or_outside_visible_source_stays_pending(self):
        variants=[self.reply(code_witnesses=[{'path':'tools/baseline.py','text':'ctx.no_such_field'}]),
            self.reply(code_witnesses=[{'path':'hidden/tests.py','text':'def parse():'}]),
            self.reply(code_witnesses=[]),dict(self.reply(),relocated_witnesses=['unsupported'])]
        for index,value in enumerate(variants):
            result,chat,_=self.run_review(value,self.root/f'invalid{index}')
            self.assertEqual(result['status'],'pending_review');self.assertEqual(chat.call_count,1)

    def test_source_witness_cannot_supply_keep_execution_fact(self):
        self.plan=record(action='keep')['plan']
        result,_,_=self.run_review(self.reply(witnesses=[{'evidence_ref':'evidence-1',
            'text':self.output.strip(),'kind':'context'}]))
        self.assertEqual(result['status'],'pending_review')
        self.assertIn('KEEP needs',result['reason'])

    def test_keep_actual_observation_pass_does_not_require_a_code_change_or_optimality(self):
        self.plan=record(action='keep')['plan']
        result,chat,_=self.run_review(self.reply(code_witnesses=[]))
        self.assertEqual(result['status'],'PASS',result)
        self.assertIn('does not prove no improvement exists or that keep is best',chat.call_args.args[0])
        typed.verify_saved_typed_review(result,self.state,self.plan)

    def test_cached_review_and_proof_never_call_model_or_rewrite_original_bytes(self):
        result,_,_=self.run_review();path=Path(result['review_path']);before=path.read_bytes()
        with patch('research1.distill._tongji_chat') as chat:
            repeated=typed.review_plan_typed(self.state,self.plan,path.parent)
            typed.verify_saved_typed_review(repeated,self.state,self.plan)
        chat.assert_not_called();self.assertEqual(result,repeated);self.assertEqual(before,path.read_bytes())
        changed=dict(self.plan,intended_change='A different change')
        with self.assertRaisesRegex(ValueError,'request changed'):
            typed.review_plan_typed(self.state,changed,path.parent)
        self.assertEqual(before,path.read_bytes())

    def test_old_untyped_directory_is_never_rewritten_or_auto_upgraded(self):
        old=self.root/'legacy'
        reply=self.reply();reply.pop('code_witnesses')
        legacy_state=dict(self.state);legacy_state.pop('factual_review_policy',None)
        with patch('research1.distill._tongji_chat',return_value=(json.dumps(reply),self.meta)):
            legacy=review_plan(legacy_state,self.plan,old)
        self.assertEqual(legacy['status'],'PASS')
        before={path.name:path.read_bytes() for path in old.iterdir()}
        with patch('research1.distill._tongji_chat') as chat:
            with self.assertRaisesRegex(ValueError,'historical/untyped'):
                typed.review_plan_typed(self.state,self.plan,old)
        chat.assert_not_called();self.assertEqual(before,{path.name:path.read_bytes() for path in old.iterdir()})

    def test_request_failure_has_started_marker_and_is_never_resent(self):
        directory=self.root/'failed'
        with patch('research1.distill._tongji_chat',side_effect=RuntimeError('connection failed')) as chat:
            result=typed.review_plan_typed(self.state,self.plan,directory)
        self.assertEqual(chat.call_count,1);self.assertEqual(result['status'],'pending_review')
        self.assertTrue((directory/'factual_review_started.json').is_file())
        self.assertFalse((directory/'factual_review.json').exists())
        with patch('research1.distill._tongji_chat') as chat:
            repeated=typed.review_plan_typed(self.state,self.plan,directory)
        chat.assert_not_called();self.assertEqual(result,repeated)

    def test_started_request_without_any_reply_is_unknown_and_not_resent(self):
        self.run_review();directory=self.root/'typed'
        (directory/'factual_review.json').unlink();(directory/'factual_review_result.json').unlink()
        with patch('research1.distill._tongji_chat') as chat:
            result=typed.review_plan_typed(self.state,self.plan,directory)
        chat.assert_not_called();self.assertEqual(result['status'],'pending_review')
        self.assertIn('CALL_OUTCOME_UNKNOWN',result['reason'])

    def test_saved_pass_cannot_be_hand_filled_or_borrowed_across_plans_views(self):
        result,_,_=self.run_review()
        with self.assertRaises(ValueError):typed.verify_saved_typed_review({'status':'PASS'},self.state,self.plan)
        changed=dict(self.plan,problem_hypothesis='Different original facts')
        with self.assertRaises(ValueError):typed.verify_saved_typed_review(result,self.state,changed)
        state=copy.deepcopy(self.state);state['module_code']['tools/baseline.py']+='\n# changed'
        with self.assertRaises(ValueError):typed.verify_saved_typed_review(result,state,self.plan)
        altered=copy.deepcopy(result);altered['code_witnesses'][0]['text']='invented'
        with self.assertRaises(ValueError):typed.verify_saved_typed_review(altered,self.state,self.plan)

    def test_raw_pass_and_wrong_slot_are_reparsed_even_with_recomputed_file_hashes(self):
        result,_,_=self.run_review();path=Path(result['review_path'])
        saved=json.loads(path.read_text(encoding='utf-8'));value=json.loads(saved['raw'])
        value['witnesses']=[{'evidence_ref':'evidence-1','text':'def parse():','kind':'context'}]
        saved['raw']=json.dumps(value);saved['raw_sha256']=hashlib.sha256(saved['raw'].encode()).hexdigest()
        path.write_text(json.dumps(saved),encoding='utf-8')
        with self.assertRaises(ValueError):typed.verify_saved_typed_review(result,self.state,self.plan)

    def test_metadata_raw_hash_start_provenance_and_result_bytes_are_verified(self):
        for field,newvalue in [('raw_sha256','0'*64),('metadata_sha256','0'*64),('started_record_sha256','0'*64)]:
            directory=self.root/field;result,_,_=self.run_review(directory=directory)
            path=Path(result['review_path']);saved=json.loads(path.read_text(encoding='utf-8'))
            saved[field]=newvalue;path.write_text(json.dumps(saved),encoding='utf-8')
            with self.assertRaises(ValueError):typed.verify_saved_typed_review(result,self.state,self.plan)

    def test_reasoning_truncation_or_provider_changes_preserve_raw_but_stay_pending(self):
        for index,metadata in enumerate([dict(self.meta,raw_field='reasoning'),dict(self.meta,finish_reason='length'),dict(self.meta,provider='other')]):
            result,_,raw=self.run_review(directory=self.root/f'meta{index}',metadata=metadata)
            self.assertEqual(result['status'],'pending_review')
            self.assertEqual(json.loads(Path(result['review_path']).read_text(encoding='utf-8'))['raw'],raw)

    def test_new_policy_is_explicit_and_future_private_state_is_not_rendered(self):
        state=dict(self.state);state.pop('factual_review_policy')
        with patch('research1.distill._tongji_chat') as chat:
            with self.assertRaisesRegex(ValueError,'explicitly frozen'):
                typed.review_plan_typed(state,self.plan,self.root/'wrong-policy')
        chat.assert_not_called()
        self.state['future_results']={'secret_future':'DO_NOT_EXPOSE'}
        self.state['plan_contract_binding']={'private_review':'DO_NOT_EXPOSE'}
        result,chat,_=self.run_review()
        self.assertEqual(result['status'],'PASS')
        self.assertNotIn('DO_NOT_EXPOSE',chat.call_args.args[0])


if __name__=='__main__':unittest.main()
