import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research1.data_pipeline.diagnostics import review_diagnostics


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.witness='task_complete confirmed followed by [tool error] failed command'
        self.state={'state_id':'s','root_lineage_id':'r','split':'train',
            'module_code':{'verification/baseline.py':'def gate():\n    return True\n'},
            'permitted_modules':['verification'],
            'evidence':[{'id':'e1','text':self.witness,'partition':'proposal'}],
            'history':[],
            'candidate':{'intended_change':'must not leak candidate'},
            'reward_scores':{'hidden':1},'child_diff':'must not leak patch'}
        self.diagnostic={'module':'verification','mechanism_id':'verification_error_gate',
            'signal':'completion_after_tool_error','evidence_refs':['e1'],'witness':self.witness,
            'reason':'The referenced terminal interaction explicitly confirms completion after the tool execution failed; this is a bounded attribution hypothesis.'}
        self.metadata={'provider':'fixed','model':'fixed','temperature':0,'raw_field':'content','finish_reason':'stop'}

    def run_review(self,rows,directory=None,metadata=None):
        with patch('research1.distill._tongji_chat',return_value=(json.dumps({'diagnostics':rows}),metadata or self.metadata)) as chat:
            result=review_diagnostics(self.state,directory or self.root/'review')
        return result,chat

    def test_only_parent_visible_projection_sent_and_review_cache_never_recalls_model(self):
        result,chat=self.run_review([self.diagnostic])
        self.assertEqual(len(result),1)
        prompt=chat.call_args.args[0]
        self.assertNotIn('must not leak candidate',prompt)
        self.assertNotIn('must not leak patch',prompt)
        self.assertNotIn('reward_scores',prompt)
        self.assertIn(self.witness,prompt)
        self.assertTrue(result[0]['review']['independent'])
        self.assertEqual(result[0]['review']['review_type'],'model_parent_attribution')
        with patch('research1.distill._tongji_chat') as repeated:
            self.assertEqual(review_diagnostics(self.state,self.root/'review'),result)
        repeated.assert_not_called()

    def test_empty_uncertain_reply_is_valid_and_does_not_manufacture_attribution(self):
        result,_=self.run_review([])
        self.assertEqual(result,[])
        report=json.loads((self.root/'review/diagnostic_review.json').read_text())
        self.assertFalse(report['causal_claim'])

    def test_decoded_diagnostic_witness_preserves_exact_original_evidence(self):
        self.witness = 'completion confirmed with [tool error] execution failed: "quoted command"\nnext line'
        original = json.dumps({'steps':[{'step_id':7,'observation_excerpt':{'segments':[{'text':self.witness}]}}]})
        self.state['evidence'][0]['text']=original
        self.diagnostic['witness']=self.witness
        self.assertNotIn(self.witness,original)
        result,_=self.run_review([self.diagnostic])
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]['witness_validation']['matched_evidence_refs'],['e1'])
        self.assertEqual(self.state['evidence'][0]['text'],original)
        wrong=dict(self.diagnostic,witness=self.witness.replace('\n','\\n'))
        self.assertEqual(self.run_review([wrong],self.root/'escaped')[0],[])

    def test_unknown_mechanism_bad_witness_and_reward_reason_stay_rejected(self):
        variants=[dict(self.diagnostic,mechanism_id='unknown'),
            dict(self.diagnostic,witness='never appeared'),
            dict(self.diagnostic,evidence_refs=['dev']),
            dict(self.diagnostic,reason='The reward is zero so this module must be defective.'),
            dict(self.diagnostic,reason='There are no errors and it is globally defect-free in all other steps.')]
        for index,item in enumerate(variants):
            with self.subTest(index=index):
                directory=self.root/str(index)
                result,_=self.run_review([item],directory)
                self.assertEqual(result,[])
                self.assertEqual(len(json.loads((directory/'diagnostic_review.json').read_text())['rejected']),1)
                self.assertTrue((directory/'diagnostic_output.json').exists())

    def test_reasoning_and_truncated_replies_saved_not_trusted(self):
        for index,metadata in enumerate([dict(self.metadata,raw_field='reasoning'),dict(self.metadata,finish_reason='length')]):
            directory=self.root/str(index)
            with self.assertRaisesRegex(ValueError,'complete final JSON'):
                self.run_review([self.diagnostic],directory,metadata)
            self.assertTrue((directory/'diagnostic_output.json').exists())

    def test_changed_parent_or_provider_cannot_reuse_review(self):
        self.run_review([self.diagnostic])
        changed=copy.deepcopy(self.state);changed['evidence'][0]['text']+=' different parent observation'
        with self.assertRaisesRegex(ValueError,'input changed'):
            review_diagnostics(changed,self.root/'review')
        with patch('research1.distill._tongji_chat',return_value=(json.dumps({'diagnostics':[]}),dict(self.metadata,provider='other'))):
            with self.assertRaisesRegex(ValueError,'changed within branch'):
                review_diagnostics(self.state,self.root/'other')

    def test_reward_partition_and_global_history_fields_fail_before_model(self):
        for key,value in [('evidence',[{'id':'e1','text':'hidden','partition':'reward'}]),('history',[{'future_results':1}])]:
            state=dict(self.state,**{key:value})
            with patch('research1.distill._tongji_chat') as chat:
                with self.assertRaises(ValueError):
                    review_diagnostics(state,self.root/'blocked')
            chat.assert_not_called()

    def test_first_attribution_call_requires_frozen_provider(self):
        self.state['diagnostic_provider']='frozen-provider'
        directory=self.root/'wrong-provider'
        with self.assertRaisesRegex(ValueError,'provider differs'):
            self.run_review([self.diagnostic],directory)
        self.assertTrue((directory/'diagnostic_output.json').is_file())
        self.assertFalse((directory/'diagnostic_review.json').exists())
        self.assertFalse((directory.parent/'diagnostic_policy_signature.json').exists())


if __name__=='__main__':unittest.main()
