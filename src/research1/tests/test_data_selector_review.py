"""Exhaustive deterministic selectors never promote commands to success."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research1.data_pipeline import selector_review as selected
from research1.data_pipeline import typed_review
from research1.data_pipeline.policy import public_view,_witness_sources
from research1.tests.test_data_sft import record


class SelectorReviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary=tempfile.TemporaryDirectory();self.addCleanup(self.temporary.cleanup)
        self.root=Path(self.temporary.name)
        sample=record(action='edit');self.state=sample['state'];self.plan=sample['plan']
        self.state.update(factual_review_policy=selected.POLICY_ID,review_model='fixed',review_provider='fixed')
        self.meta={'provider':'fixed','model':'fixed','temperature':0,'raw_field':'content','finish_reason':'stop'}
        payload='Returned payload: {"value": 7}\n'
        def excerpt(text):return {'segments':[{'start':0,'end':len(text),'text':text}]}
        def step(number,text,command,message=None):
            value={'step_id':number,'source':'agent','source_index':number-1,'visible_step_sha256':str(number)*64,
                'observation_excerpt':excerpt(text),'tool_calls':[{'tool_call_id':f'call{number}',
                'arguments':{'keystrokes':command,'keystrokes_excerpt':excerpt(command)}}]}
            if message is not None:value['message_excerpt']=excerpt(message)
            return value
        steps=[{'step_id':1,'source':'user','source_index':0,'visible_step_sha256':'1'*64,
                    'message_excerpt':excerpt('Original task requirements.')},
            step(2,'root@host:/app# python3 check.py\n'+payload,'python3 check.py\n',payload),
            step(3,'root@host:/app# python3 -m py_compile module.py\nroot@host:/app#\n',
                 'python3 -m py_compile module.py\n'),
            step(4,'root@host:/app# echo all tests passed\nall tests passed\n','echo all tests passed\n'),
            step(5,'root@host:/app#\n','')]
        self.state['evidence']=[{'id':'evidence-1','text':json.dumps({'steps':steps}),'partition':'proposal'},
            {'id':'feedback','text':'Recorded proposal classification only.','partition':'proposal'}]

    def catalog(self):return selected.build_witness_catalog(public_view(self.state))

    def entry(self,field,step):
        return next(item for item in self.catalog()['items'] if item['field']==field and item['step_id']==step)

    def reply(self,**updates):
        value={'status':'PASS','reason':'The selected original observation and current code support the scoped hypothesis; future benefit is unproven.',
            'evidence_refs':['evidence-1'],'witness_ids':[self.entry('observation_excerpt',2)['id']],
            'code_witnesses':[{'path':'tools/baseline.py','text':'def parse():\n    return 1'}]}
        value.update(updates);return value

    def run_review(self,reply=None,directory=None,metadata=None):
        raw=json.dumps(reply or self.reply())
        with patch('research1.distill._tongji_chat',return_value=(raw,metadata or self.meta)) as chat:
            result=selected.review_plan_selected(self.state,self.plan,directory or self.root/'selected')
        return result,chat,raw

    def test_catalog_is_complete_exact_ordered_stable_and_not_success_filtered(self):
        view=public_view(self.state);catalog=self.catalog()
        sources=[(evidence,source) for evidence in view['evidence'] for source in _witness_sources(evidence['text'])]
        self.assertEqual(len(catalog['items']),len(sources));self.assertEqual(catalog,self.catalog())
        for item,(evidence,source) in zip(catalog['items'],sources):
            self.assertEqual(item['evidence_ref'],evidence['id']);self.assertEqual(item['text'],source['text'])
            self.assertEqual(item['offset_start'],0);self.assertEqual(item['offset_end'],len(source['text']))
            self.assertEqual(item['text_sha256'],hashlib.sha256(source['text'].encode()).hexdigest())
            self.assertFalse(item['success_established']);self.assertFalse(item['producer_established'])
            self.assertFalse(item['freshness_established'])
        self.assertEqual({item['evidence_ref'] for item in catalog['items']},{'evidence-1','feedback'})
        self.assertEqual(len({item['id'] for item in catalog['items']}),len(sources))

    def test_commands_messages_echo_claims_and_idle_prompts_never_get_observation_kind(self):
        self.assertEqual(self.entry('tool_calls/0/keystrokes_excerpt',2)['kind'],'context')
        # Identical text is present in a real observation at step2, but this
        # original message location must still remain context.
        self.assertEqual(self.entry('message_excerpt',2)['kind'],'context')
        self.assertEqual(self.entry('observation_excerpt',3)['kind'],'context')
        self.assertEqual(self.entry('observation_excerpt',4)['kind'],'context')
        self.assertEqual(self.entry('observation_excerpt',5)['kind'],'context')
        self.assertEqual(self.entry('observation_excerpt',2)['kind'],'observed_tool_output')

    def test_selected_real_observation_is_materialized_then_revalidated_by_v5(self):
        result,chat,raw=self.run_review()
        self.assertEqual(result['status'],'PASS',result)
        self.assertEqual(result['witness_ids'],self.reply()['witness_ids'])
        witness=result['witnesses'][0]
        self.assertEqual(witness['text'],self.entry('observation_excerpt',2)['text'])
        self.assertEqual(witness['kind'],'observed_tool_output')
        self.assertFalse(witness['validation']['success_established'])
        self.assertTrue(witness['validation']['substantive_observed_fact'])
        self.assertEqual(witness['validation']['validator'],'proposal-factual-witness.v5')
        self.assertEqual(chat.call_args.kwargs['reasoning_effort'],'none')
        self.assertFalse(chat.call_args.kwargs['allow_http_fallback'])
        self.assertEqual(json.loads(Path(result['review_path']).read_text())['raw'],raw)
        proof=selected.verify_saved_selected_review(result,self.state,self.plan)
        self.assertEqual(set(proof),{'factual_review_sha256','factual_raw_path','factual_raw_record_sha256',
            'factual_result_path','factual_result_record_sha256'})

    def test_command_only_or_echo_ids_cannot_establish_pass_even_with_correct_source(self):
        for index,identifier in enumerate([self.entry('tool_calls/0/keystrokes_excerpt',2)['id'],
                self.entry('observation_excerpt',3)['id'],self.entry('observation_excerpt',4)['id']]):
            result,_,_=self.run_review(self.reply(witness_ids=[identifier]),self.root/f'command{index}')
            self.assertEqual(result['status'],'pending_review');self.assertNotIn('witnesses',result)
            self.assertIn('actual substantive original observation',result['reason'])

    def test_unknown_duplicate_ids_wrong_refs_and_wrong_old_slots_are_not_relocated(self):
        old=self.reply();old['witnesses']=[{'evidence_ref':'evidence-1','kind':'observed_tool_output','text':'def parse():'}]
        old.pop('witness_ids')
        variants=[self.reply(witness_ids=['invented']),self.reply(witness_ids=self.reply()['witness_ids']*2),
            self.reply(evidence_refs=['feedback']),old,dict(self.reply(),kind='observed_tool_output')]
        for index,reply in enumerate(variants):
            result,chat,raw=self.run_review(reply,self.root/f'invalid{index}')
            self.assertEqual(result['status'],'pending_review');self.assertEqual(chat.call_count,1)
            self.assertEqual(json.loads(Path(result['review_path']).read_text())['raw'],raw)

    def test_source_still_uses_typed_exact_paths_and_does_not_substitute_for_observation(self):
        result,_,_=self.run_review(self.reply(code_witnesses=[{'path':'tools/baseline.py','text':'invented'}]))
        self.assertEqual(result['status'],'pending_review')
        result,_,_=self.run_review(self.reply(code_witnesses=[{'path':'hidden/test.py','text':'def parse():'}]),self.root/'outside')
        self.assertEqual(result['status'],'pending_review')

    def test_keep_with_real_selected_observation_pass_is_not_optimality_proof(self):
        self.plan=record(action='keep')['plan']
        result,chat,_=self.run_review(self.reply(code_witnesses=[]))
        self.assertEqual(result['status'],'PASS',result)
        self.assertIn('does not prove keep is best',chat.call_args.args[0])
        selected.verify_saved_selected_review(result,self.state,self.plan)

    def test_cached_results_and_saved_proof_do_zero_transport_and_preserve_bytes(self):
        result,_,_=self.run_review();directory=Path(result['review_path']).parent
        before={path.name:path.read_bytes() for path in directory.iterdir()}
        with patch('research1.distill._tongji_chat') as chat:
            repeated=selected.review_plan_selected(self.state,self.plan,directory)
            selected.verify_saved_selected_review(result,self.state,self.plan)
        chat.assert_not_called();self.assertEqual(result,repeated)
        self.assertEqual(before,{path.name:path.read_bytes() for path in directory.iterdir()})

    def test_catalog_tampering_and_context_reclassification_are_rejected(self):
        result,_,_=self.run_review();directory=Path(result['review_path']).parent
        catalog_path=directory/'factual_witness_catalog.json'
        catalog=json.loads(catalog_path.read_text());catalog['items'][0]['kind']='observed_tool_output'
        catalog_path.write_text(json.dumps(catalog),encoding='utf-8')
        with self.assertRaises(ValueError):selected.verify_saved_selected_review(result,self.state,self.plan)
        with patch('research1.distill._tongji_chat') as chat:
            with self.assertRaisesRegex(ValueError,'artifact changed'):
                selected.review_plan_selected(self.state,self.plan,directory)
        chat.assert_not_called()

    def test_raw_selected_ids_are_reparsed_even_if_the_raw_hash_is_recomputed(self):
        result,_,_=self.run_review();path=Path(result['review_path'])
        saved=json.loads(path.read_text());value=json.loads(saved['raw'])
        value['witness_ids']=[self.entry('tool_calls/0/keystrokes_excerpt',2)['id']]
        saved['raw']=json.dumps(value);saved['raw_sha256']=hashlib.sha256(saved['raw'].encode()).hexdigest()
        path.write_text(json.dumps(saved),encoding='utf-8')
        with self.assertRaises(ValueError):selected.verify_saved_selected_review(result,self.state,self.plan)

    def test_plan_view_catalog_or_forged_pass_cannot_borrow_saved_review(self):
        result,_,_=self.run_review()
        with self.assertRaises(ValueError):selected.verify_saved_selected_review({'status':'PASS'},self.state,self.plan)
        changed=dict(self.plan,intended_change='Different change')
        with self.assertRaises(ValueError):selected.verify_saved_selected_review(result,self.state,changed)
        state=copy.deepcopy(self.state);state['evidence'][0]['text']+=' '
        with self.assertRaises(ValueError):selected.verify_saved_selected_review(result,state,self.plan)
        state=copy.deepcopy(self.state);state['module_code']['tools/baseline.py']+='\n# changed'
        with self.assertRaises(ValueError):selected.verify_saved_selected_review(result,state,self.plan)

    def test_raw_input_start_metadata_hashes_and_result_are_verified(self):
        for name in ['input_record_sha256','catalog_record_sha256','started_record_sha256','metadata_sha256']:
            result,_,_=self.run_review(directory=self.root/name);path=Path(result['review_path'])
            saved=json.loads(path.read_text());saved[name]='0'*64;path.write_text(json.dumps(saved),encoding='utf-8')
            with self.assertRaises(ValueError):selected.verify_saved_selected_review(result,self.state,self.plan)

    def test_failed_or_started_without_reply_is_not_resent(self):
        directory=self.root/'failed'
        with patch('research1.distill._tongji_chat',side_effect=RuntimeError('connection failed')) as chat:
            result=selected.review_plan_selected(self.state,self.plan,directory)
        self.assertEqual(chat.call_count,1);self.assertEqual(result['status'],'pending_review')
        self.assertFalse((directory/'factual_review.json').exists())
        with patch('research1.distill._tongji_chat') as chat:
            repeated=selected.review_plan_selected(self.state,self.plan,directory)
        chat.assert_not_called();self.assertEqual(result,repeated)

    def test_old_typed_prompt_raw_and_results_are_never_rewritten(self):
        directory=self.root/'old';state=dict(self.state,factual_review_policy=typed_review.POLICY_ID)
        old={'status':'PASS','reason':'Original bounded fact and source','evidence_refs':['evidence-1'],
            'witnesses':[{'evidence_ref':'evidence-1','text':'Returned payload: {"value": 7}','kind':'observed_tool_output'}],
            'code_witnesses':self.reply()['code_witnesses']}
        with patch('research1.distill._tongji_chat',return_value=(json.dumps(old),self.meta)):
            typed_review.review_plan_typed(state,self.plan,directory)
        before={path.name:path.read_bytes() for path in directory.iterdir()}
        with patch('research1.distill._tongji_chat') as chat:
            with self.assertRaises(ValueError):selected.review_plan_selected(self.state,self.plan,directory)
        chat.assert_not_called();self.assertEqual(before,{path.name:path.read_bytes() for path in directory.iterdir()})

    def test_opt_in_policy_and_visible_projection_exclude_private_future_records(self):
        state=dict(self.state);state.pop('factual_review_policy')
        with patch('research1.distill._tongji_chat') as chat:
            with self.assertRaisesRegex(ValueError,'explicitly frozen'):
                selected.review_plan_selected(state,self.plan,self.root/'wrong-policy')
        chat.assert_not_called()
        self.state['future_results']={'secret':'DO_NOT_EXPOSE'}
        self.state['plan_contract_binding']={'private_review':'DO_NOT_EXPOSE'}
        result,chat,_=self.run_review()
        self.assertEqual(result['status'],'PASS')
        self.assertNotIn('DO_NOT_EXPOSE',chat.call_args.args[0])


if __name__=='__main__':unittest.main()
