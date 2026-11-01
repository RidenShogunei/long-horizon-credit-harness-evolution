import json
import unittest
from research1.tests import test_data_engine as fixtures
from research1.data_pipeline.execution import bind_trial,BindingError

class SolverPolicyTests(unittest.TestCase):
    def setUp(self):
        self.case=fixtures.TrialBindingTests('test_uses_exact_concrete_trial_valid_zero_not_job_aggregate')
        self.case.setUp();self.addCleanup(self.case.doCleanups)
        self.case.spec['reasoning_effort']='none'

    def result(self,reasoning):
        case=self.case;case.config['agent']['kwargs']['reasoning_effort']='none'
        (case.trial/'config.json').write_text(json.dumps(case.config))
        case.result['config']=case.config;(case.trial/'result.json').write_text(json.dumps(case.result))
        (case.trial/'agent').mkdir();(case.trial/'agent/trajectory.json').write_text(json.dumps({'steps':[{'source':'agent','reasoning_content':reasoning,'message':'{}'}]}))
        return bind_trial(case.output,case.spec)

    def test_explicit_nonthinking_binding_and_actual_zero_reasoning(self):
        result=self.result('')
        self.assertEqual(result['status'],'PASS');self.assertEqual(result['reward'],0)
        self.assertEqual(result['solver_reasoning_observation']['reasoning_characters'],0)

    def test_backend_ignoring_policy_is_fatal_without_fabricating_score(self):
        result=self.result('synthetic noncompliant reasoning')
        self.assertEqual(result['status'],'INCOMPLETE');self.assertEqual(result['reward'],0)
        self.assertTrue(result['fatal']);self.assertFalse(result['retryable'])
        self.assertFalse(result['policy_valid'])

    def test_missing_requested_flag_cannot_count_as_bound_nonthinking_run(self):
        with self.assertRaises(BindingError):bind_trial(self.case.output,self.case.spec)

