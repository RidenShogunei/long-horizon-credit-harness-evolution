import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from research1.data_pipeline.policy import collect_candidates, render_messages
from research1.tests.test_data_sft import record


class GroundedTeacherSettingsTests(unittest.TestCase):
    def test_v5_instruction_is_preserved_byte_for_byte(self):
        state=record()['state'];state['plan_policy_version']='proposal_single_plan_v5'
        self.assertEqual(hashlib.sha256(render_messages(state)[0]['content'].encode()).hexdigest(),
            'fc1889d81e2d78f8fff505313e24b829c3f2cbe68008483557bd2e8396af1ef6')
        self.assertNotIn('fact_index',json.loads(render_messages(state)[1]['content']))

    def test_same_prompt_cannot_reuse_a_different_reasoning_request(self):
        sample=record(action='keep');state=sample['state']
        state.update(plan_policy_version='proposal_single_plan_v6',teacher_reasoning_effort='none',
                     allow_teacher_http_fallback=False)
        metadata={'raw_field':'content','finish_reason':'stop','temperature':0,'reasoning_effort':'none'}
        with tempfile.TemporaryDirectory() as directory:
            with patch('research1.distill._tongji_chat',return_value=(json.dumps({'candidates':[sample['plan']]}),metadata)):
                collect_candidates(state,Path(directory))
            state['teacher_reasoning_effort']='low'
            with patch('research1.distill._tongji_chat') as call:
                with self.assertRaisesRegex(ValueError,'settings changed'):
                    collect_candidates(state,Path(directory))
                call.assert_not_called()

    def test_low_reasoning_is_explicit_and_does_not_change_the_visible_input(self):
        sample=record(action='keep');state=sample['state']
        state.update(plan_policy_version='proposal_single_plan_v6',teacher_reasoning_effort='low',
                     allow_teacher_http_fallback=False)
        before=render_messages(dict(state,teacher_reasoning_effort='none'))
        self.assertEqual(render_messages(state),before)
        metadata={'raw_field':'content','finish_reason':'stop','temperature':0,'reasoning_effort':'low'}
        with tempfile.TemporaryDirectory() as directory:
            with patch('research1.distill._tongji_chat',return_value=(json.dumps({'candidates':[sample['plan']]}),metadata)) as call:
                collect_candidates(state,Path(directory))
            self.assertEqual(call.call_args.kwargs['reasoning_effort'],'low')
            self.assertFalse(call.call_args.kwargs['allow_http_fallback'])
