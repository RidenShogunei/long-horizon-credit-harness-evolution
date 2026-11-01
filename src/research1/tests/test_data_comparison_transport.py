"""Explicit rejection must not invisibly duplicate a controlled model request."""
import io
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from research1.distill import _tongji_chat


class ComparisonTransportTests(unittest.TestCase):
    def provider(self):
        return SimpleNamespace(resolve_provider=lambda **kw: SimpleNamespace(
            api_base='https://agi.tongji.edu.cn/v1', api_model_id='fixed',
            provider_id='fixed', api_key='test-placeholder'), record_usage=lambda *a, **kw: None)

    def test_strict_controlled_call_does_not_fallback_on_http_400(self):
        error = HTTPError('https://agi.tongji.edu.cn/v1/chat/completions', 400, 'rejected', {}, io.BytesIO(b'unsupported'))
        with patch.dict('sys.modules', {'llm_provider_lib':self.provider()}), \
                patch('urllib.request.urlopen', side_effect=error) as request:
            with self.assertRaisesRegex(RuntimeError, 'HTTP 400'):
                _tongji_chat('fixed prompt', max_tokens=8, role='unit_test',
                             reasoning_effort='none', allow_http_fallback=False)
        self.assertEqual(request.call_count, 1)

    def test_success_records_actual_attempt_count_and_reasoning_setting(self):
        data = {'choices':[{'message':{'content':'{}'},'finish_reason':'stop'}], 'usage':{}}
        response = io.BytesIO(json.dumps(data).encode())
        with patch.dict('sys.modules', {'llm_provider_lib':self.provider()}), \
                patch('urllib.request.urlopen', return_value=response) as request:
            raw, metadata = _tongji_chat('fixed prompt', max_tokens=8, role='unit_test',
                json_output=True, reasoning_effort='none', allow_http_fallback=False)
        body = json.loads(request.call_args.args[0].data)
        self.assertEqual(raw, '{}')
        self.assertEqual(metadata['http_attempts'], 1)
        self.assertEqual(metadata['reasoning_effort'], 'none')
        self.assertEqual(body['reasoning_effort'], 'none')
        self.assertNotIn('thinking', body)
