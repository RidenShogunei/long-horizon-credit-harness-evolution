from copy import deepcopy
import json,unittest
from research1.tests.test_data_sft import record
from research1.data_pipeline.policy import render_messages
from research1.data_pipeline.initialization import _add_addressable_feedback

class PriorFeedbackTests(unittest.TestCase):
    def test_new_interface_feedback_has_its_own_reference_without_rewriting_legacy_evidence(self):
        legacy=self.state();original=deepcopy(legacy)
        _add_addressable_feedback(legacy)
        self.assertEqual(legacy,original)
        state=self.state();state['plan_policy_version']='proposal_single_plan_v4'
        trace=deepcopy(state['evidence'])
        _add_addressable_feedback(state);_add_addressable_feedback(state)
        self.assertEqual(state['evidence'][:-1],trace)
        self.assertEqual(len(state['evidence']),len(trace)+1)
        feedback=state['evidence'][-1]
        self.assertIn('proposal_feedback_exact-execution',feedback['id'])
        decoded=json.loads(feedback['text'])
        self.assertEqual(decoded['failure_layer'],'model')
        self.assertEqual(decoded['outcome'],'FAIL')
        self.assertNotIn('reward',decoded)
        self.assertEqual(feedback['source_type'],'recorded_proposal_feedback')

    def state(self):
        state=record()['state'];state['code_hash']='a'*64
        state['prior_proposal_outcomes']=[{'proposal_task_id':state['proposal_task_ids'][0],
            'execution_id':'exact-execution','outcome':'FAIL','failure_layer':'model',
            'parent_code_hash':'a'*64,'source_sha256':'b'*64,'finished_before_decision':True,'source_partition':'proposal'}]
        return state

    def test_prior_binary_failure_is_identical_teacher_and_student_evidence(self):
        state=self.state();inference=render_messages(state);training=render_messages(state,record()['plan'])
        self.assertEqual(inference,training[:-1])
        visible=json.loads(inference[1]['content'])
        self.assertEqual(visible['prior_proposal_outcomes'][0]['outcome'],'FAIL')
        self.assertNotIn('reward',visible['prior_proposal_outcomes'][0])

    def test_future_other_partition_other_parent_or_untyped_score_is_rejected(self):
        for changes in ({'finished_before_decision':False},{'source_partition':'reward'},
                        {'proposal_task_id':'dev-task'},{'parent_code_hash':'c'*64},{'score':1}):
            with self.subTest(changes=changes):
                state=self.state();state['prior_proposal_outcomes'][0].update(changes)
                with self.assertRaises(ValueError):render_messages(state)
