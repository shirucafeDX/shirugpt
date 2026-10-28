"""実SDKのSSE受信、分割引数、途中切断、停止、計測と状態継続。"""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pandas as pd
from google import genai
from google.genai import types
from google.oauth2.credentials import Credentials
from google.genai._gaos.lib.compat_errors import APITimeoutError
from streamlit.testing.v1 import AppTest

from src import interactions, context_memory as memory, analysis_safety as safety
from src.agent import GeminiAgent, ToolContext
from src.bq_tools import QueryRun
from src.chat_history import pack, unpack
from src.execution import ExecutionControl, ExecutionStopped
from src.interaction_stream import StreamResponseError


def user(text):
    return types.Content(role='user', parts=[types.Part.from_text(text=text)])


def usage():
    return {'total_input_tokens': 50, 'total_output_tokens': 10, 'total_thought_tokens': 5,
            'total_cached_tokens': 4, 'total_tokens': 65}


def frame(event):
    return ('event: ' + event['event_type'] + '\ndata: ' + json.dumps(event, ensure_ascii=False) + '\n\n').encode()


def events(identifier, steps, *, complete=True, status=None):
    yield {'event_type': 'interaction.created', 'interaction': {'id': identifier, 'status': 'in_progress'}}
    for index, step in enumerate(steps):
        if step['type'] == 'function_call':
            yield {'event_type': 'step.start', 'index': index, 'step': {**step, 'arguments': {}}}
            args = json.dumps(step['arguments'], ensure_ascii=False)
            for part in (args[:3], args[3:11], args[11:]):
                yield {'event_type': 'step.delta', 'index': index, 'delta': {'type': 'arguments_delta', 'arguments': part}}
        else:
            yield {'event_type': 'step.start', 'index': index, 'step': {'type': 'model_output'}}
            for text in ('こんにちは', '。応答です。'):
                yield {'event_type': 'step.delta', 'index': index, 'delta': {'type': 'text', 'text': text}}
        yield {'event_type': 'step.stop', 'index': index}
    if complete:
        yield {'event_type': 'interaction.completed', 'interaction': {'id': identifier,
            'status': status or ('requires_action' if any(s['type'] == 'function_call' for s in steps) else 'completed'), 'usage': usage()}}


class Bytes(httpx.SyncByteStream):
    def __init__(self, chunks, *, failure=None, before_chunk=None):
        self.chunks, self.failure, self.before_chunk = chunks, failure, before_chunk
        self.closed = False

    def __iter__(self):
        for chunk in self.chunks:
            if self.before_chunk:
                self.before_chunk()
            yield chunk
        if self.failure:
            raise self.failure

    def close(self):
        self.closed = True


class StreamingTests(unittest.TestCase):
    def setUp(self):
        self.settings = SimpleNamespace(gemini_api='interactions', gemini_model='gemini-3.8-flash',
            gcp_project='test', allowed_datasets=('data',), vertex_location='global',
            max_tool_iterations=8, thinking_budget=None, interaction_max_input_tokens=32000, max_history_tokens=16000)
        self.ctx = ToolContext(settings=self.settings, tools=Mock(), usage_logger=Mock())
        self.ctx.start_turn()
        self.requests = []
        self.streams = []
        self.contents = [user('こんにちは')]

    def sdk(self, replies):
        queue = iter(replies)
        def handler(request):
            self.requests.append(json.loads(request.content))
            value = next(queue)
            if isinstance(value, httpx.Response):
                return value
            stream = value if isinstance(value, Bytes) else Bytes([frame(e) for e in value])
            self.streams.append(stream)
            return httpx.Response(200, headers={'content-type': 'text/event-stream', 'x-request-id': 'request-test'}, stream=stream)
        return genai.Client(enterprise=True, project='test', location='global', credentials=Credentials(token='test'),
            http_options=types.HttpOptions(api_version='v1beta1', retry_options=types.HttpRetryOptions(attempts=0),
                                          client_args={'transport': httpx.MockTransport(handler)}))

    def test_text_deltas_are_assembled_only_at_completion_and_metrics_persist(self):
        progress = []
        with self.sdk([list(events('s1', [{'type': 'model_output'}]))]) as client:
            answer = GeminiAgent(self.settings, client=client).run(self.contents, self.ctx, on_progress=progress.append)
        self.assertEqual(answer.text, 'こんにちは。応答です。')
        self.assertTrue(self.requests[0]['stream'])
        self.assertTrue(any('応答' in text for text in progress))
        self.assertFalse(any('こんにちは' in text for text in progress))
        self.assertTrue(self.streams[0].closed)
        metrics = next(e for e in self.ctx.turn_events if e['event'] == 'llm_call')
        self.assertTrue(metrics['stream_completed'])
        self.assertGreater(metrics['stream_events'], 2)
        self.assertLessEqual(metrics['first_event_seconds'], metrics['first_content_seconds'])
        self.assertEqual(metrics['x_request_id'], 'request-test')
        self.assertEqual(metrics['reported_usage'], usage())
        self.ctx.usage_logger.log_gemini_call.assert_called_once()
        self.assertEqual(self.ctx.turn_output_tokens, 15)
        saved = pack([{'role': 'assistant', 'text': answer.text, 'execution_events': list(self.ctx.turn_events)}], self.contents, self.ctx, None)
        restored = ToolContext(settings=self.settings, tools=Mock())
        messages, _, _ = unpack(saved, restored)
        self.assertEqual(messages[0]['execution_events'][0]['stream_events'], metrics['stream_events'])

    def test_completion_event_is_drained_before_next_request(self):
        class FinalizingBytes(Bytes):
            drained = False
            def __iter__(self):
                yield from super().__iter__()
                self.drained = True
        response = FinalizingBytes([frame(e) for e in events('drain', [{'type': 'model_output'}])])
        with self.sdk([response]) as client:
            GeminiAgent(self.settings, client=client).run(self.contents, self.ctx)
        self.assertTrue(response.drained)
        self.assertTrue(self.ctx.turn_events[-1]['stream_completed'])

    def test_fragmented_function_arguments_execute_once_and_keep_state_chain(self):
        sql = 'SELECT 1'
        result_id = memory.result_id_for('run_query', {'sql': sql})
        query = {'type': 'function_call', 'id': 'query', 'name': 'run_query', 'arguments': {'sql': sql}}
        final = {'type': 'function_call', 'id': 'final', 'name': 'finish_answer', 'arguments': {
            'answer': '確認した結果です', 'conversation_summary': '取得完了',
            'context_note': {'summary': '取得完了', 'references': [result_id]}}}
        self.ctx.tools.run_query.return_value = QueryRun(sql, sql, 1, 1, pd.DataFrame({'n': [1]}), {})
        partial_checks = []
        def before_chunk():
            partial_checks.append(self.ctx.tools.run_query.call_count)
        first = Bytes([frame(e) for e in events('s1', [query])], before_chunk=before_chunk)
        with self.sdk([first, list(events('s2', [final])), list(events('s3', [{'type': 'model_output'}]))]) as client:
            agent = GeminiAgent(self.settings, client=client)
            answer = agent.run(self.contents, self.ctx)
            self.assertEqual(answer.text, '確認した結果です')
            self.ctx.start_turn()
            self.contents.append(user('こんにちは'))
            agent.run(self.contents, self.ctx)
        self.assertTrue(all(count == 0 for count in partial_checks))
        self.ctx.tools.run_query.assert_called_once()
        self.assertEqual(self.requests[1]['previous_interaction_id'], 's1')
        self.assertEqual(self.requests[2]['previous_interaction_id'], 's2')
        self.assertEqual(self.requests[1]['input'][0]['call_id'], 'query')
        self.assertNotIn('csv', json.dumps(self.requests[2]['input']))
        self.assertTrue(all(s.closed for s in self.streams))

    def test_numeric_answer_and_followup_are_generated_by_model_with_state_continuity(self):
        self.contents = [user('取得した数値だけで答えてください。')]
        sql = 'SELECT 1 AS n'
        self.ctx.tools.run_query.return_value = QueryRun(sql, sql, 1, 1, pd.DataFrame({'n': [987654321]}), {})
        first = {'type': 'function_call', 'id': 'q', 'name': 'run_query', 'arguments': {'sql': sql}}
        def finish(identifier, answer):
            return {'type': 'function_call', 'id': identifier, 'name': 'finish_answer', 'arguments': {
                'answer': answer, 'conversation_summary': '保存値を説明',
                'context_note': {'summary': '保存値を説明', 'references': ['R1']}}}
        replies = [list(events('plain1', [first])),
                   list(events('plain2', [finish('f1', '987654321')])),
                   list(events('plain3', [finish('f2', '保存値は987654321です。')]))]
        with self.sdk(replies) as client:
            agent = GeminiAgent(self.settings, client=client)
            self.assertEqual(agent.run(self.contents, self.ctx).text, '987654321')
            self.ctx.start_turn()
            self.contents.append(user('その数値を説明してください'))
            self.assertEqual(agent.run(self.contents, self.ctx).text, '保存値は987654321です。')
        self.assertEqual(self.requests[1]['previous_interaction_id'], 'plain1')
        self.assertEqual(self.requests[2]['previous_interaction_id'], 'plain2')
        self.ctx.tools.run_query.assert_called_once()

    def test_missing_completion_does_not_execute_even_a_complete_function_call(self):
        call = {'type': 'function_call', 'id': 'q', 'name': 'run_query', 'arguments': {'sql': 'SELECT 1'}}
        with self.sdk([list(events('partial', [call], complete=False))]) as client:
            with self.assertRaises(StreamResponseError):
                GeminiAgent(self.settings, client=client).run(self.contents, self.ctx)
        self.ctx.tools.run_query.assert_not_called()
        self.ctx.usage_logger.log_gemini_call.assert_not_called()
        self.assertEqual(self.ctx.interaction_state, {})
        metric = self.ctx.turn_events[-1]
        self.assertTrue(metric['error'])
        self.assertEqual(metric['interaction_id'], 'partial')
        self.assertNotIn('input_tokens', metric)
        self.assertIn('失敗した処理: 1件', safety.snapshot(self.ctx)['execution_note'])
        self.assertTrue(self.streams[0].closed)

    def test_read_timeout_after_partial_response_is_not_retried_and_keeps_diagnostics(self):
        response = Bytes([frame({'event_type': 'interaction.created', 'interaction': {'id': 'waiting', 'status': 'in_progress'}})],
                         failure=httpx.ReadTimeout('no further events'))
        with self.sdk([response]) as client:
            with self.assertRaises(APITimeoutError):
                GeminiAgent(self.settings, client=client).run(self.contents, self.ctx)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.ctx.turn_events[-1]['last_event_type'], 'interaction.created')
        self.assertIn('first_event_seconds', self.ctx.turn_events[-1])
        self.assertNotIn('first_content_seconds', self.ctx.turn_events[-1])
        self.assertTrue(response.closed)

    def test_error_event_after_creation_is_not_automatically_retried(self):
        reply = [{'event_type': 'interaction.created', 'interaction': {'id': 'error', 'status': 'in_progress'}},
                 {'event_type': 'error', 'error': {'code': 'resource_exhausted', 'message': 'busy'}}]
        with self.sdk([reply]) as client:
            with self.assertRaises(StreamResponseError):
                GeminiAgent(self.settings, client=client).run(self.contents, self.ctx)
        self.assertEqual(len(self.requests), 1)
        self.assertEqual(self.ctx.turn_events[-1]['last_event_type'], 'error')

    def test_invalid_arguments_with_terminal_usage_are_billed_but_never_executed(self):
        reply = [{'event_type': 'interaction.created', 'interaction': {'id': 'bad', 'status': 'in_progress'}},
                 {'event_type': 'step.start', 'index': 0, 'step': {'type': 'function_call', 'id': 'q', 'name': 'run_query', 'arguments': {}}},
                 {'event_type': 'step.delta', 'index': 0, 'delta': {'type': 'arguments_delta', 'arguments': '{"sql":'}},
                 {'event_type': 'step.stop', 'index': 0},
                 {'event_type': 'interaction.completed', 'interaction': {'id': 'bad', 'status': 'requires_action', 'usage': usage()}}]
        with self.sdk([reply]) as client:
            with self.assertRaises(StreamResponseError):
                GeminiAgent(self.settings, client=client).run(self.contents, self.ctx)
        self.ctx.tools.run_query.assert_not_called()
        self.ctx.usage_logger.log_gemini_call.assert_called_once()
        self.assertEqual(self.ctx.turn_prompt_tokens, 50)
        self.assertEqual(self.ctx.turn_events[-1]['input_tokens'], 50)

    def test_completed_event_cannot_execute_a_step_without_stop(self):
        reply = [e for e in events('unclosed', [{'type': 'function_call', 'id': 'q', 'name': 'run_query', 'arguments': {'sql': 'SELECT 1'}}])
                 if e['event_type'] != 'step.stop']
        with self.sdk([reply]) as client:
            with self.assertRaises(StreamResponseError):
                GeminiAgent(self.settings, client=client).run(self.contents, self.ctx)
        self.ctx.tools.run_query.assert_not_called()

    def test_output_limit_does_not_execute_streamed_tool_and_is_counted_as_failure(self):
        call = {'type': 'function_call', 'id': 'q', 'name': 'run_query', 'arguments': {'sql': 'SELECT 1'}}
        with self.sdk([list(events('limited', [call], status='incomplete'))]) as client:
            result = GeminiAgent(self.settings, client=client).run(self.contents, self.ctx)
        self.assertIn('出力トークン上限', result.text)
        self.ctx.tools.run_query.assert_not_called()
        self.ctx.usage_logger.log_gemini_call.assert_called_once()
        self.assertIn('失敗した処理: 1件', safety.snapshot(self.ctx)['execution_note'])

    def test_stop_during_stream_closes_response_and_is_not_reported_as_ai_failure(self):
        control = ExecutionControl()
        self.ctx.execution_control = control
        def progress(message):
            if 'AIからの応答' in message:
                control.request_stop()
        with self.sdk([list(events('stopped', [{'type': 'model_output'}]))]) as client:
            with self.assertRaises(ExecutionStopped):
                GeminiAgent(self.settings, client=client).run(self.contents, self.ctx, on_progress=progress)
        self.assertTrue(self.streams[0].closed)
        self.assertFalse(self.ctx.turn_events[-1]['error'])
        self.assertEqual(self.ctx.turn_events[-1]['status'], 'stopped')
        self.ctx.usage_logger.log_gemini_call.assert_not_called()

    def test_timing_columns_are_visible_in_execution_detail(self):
        app = AppTest.from_string('''
from app import render_results
render_results({'turn_id': 'stream-turn', 'execution_events': [{'event': 'llm_call', 'seconds': 10,
 'first_event_seconds': .5, 'first_content_seconds': 2, 'stream_events': 12,
 'received_chars': 42, 'last_event_type': 'interaction.completed', 'error': False}]}, 0, set())
''').run(timeout=20)
        self.assertFalse(app.exception)
        row = app.dataframe[0].value.iloc[0]
        self.assertEqual(row['最初のイベントまで（秒）'], .5)
        self.assertEqual(row['本文・引数の受信開始まで（秒）'], 2)
        self.assertEqual(row['受信イベント数'], 12)
