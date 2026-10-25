"""Gemini APIへ送信するJSONの末尾と、不完全な履歴の回復を外部接続なしで検証。"""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
from google import genai
from google.oauth2.credentials import Credentials
from google.genai import types

from src.agent import GeminiAgent, ToolContext
from src.context_memory import build_request, ensure_user_ending, result_id_for


def content(role, text):
    return types.Content(role=role, parts=[types.Part.from_text(text=text)])


class RequestTailTests(unittest.TestCase):
    def test_model_ending_gets_user_continuation_without_mutating_original(self):
        original = [content('user', '店舗別の売上は？'), content('model', '途中までの回答')]
        original[-1].parts[0].thought_signature = b'keep-signature'
        snapshot = [c.model_dump() for c in original]
        request = ensure_user_ending(original)
        self.assertEqual(request[-1].role, 'user')
        self.assertIn('直前のユーザー', request[-1].parts[0].text)
        self.assertIs(request[-2], original[-1])
        self.assertEqual(request[-2].parts[0].thought_signature, b'keep-signature')
        self.assertEqual([c.model_dump() for c in original], snapshot)
        self.assertEqual(ensure_user_ending(request), request)

    def test_empty_user_suffix_does_not_mask_last_nonempty_model(self):
        original = [content('user', '質問'), content('model', '回答'),
                    types.Content(role='user', parts=[]), content('user', '   ')]
        request = ensure_user_ending(original)
        self.assertEqual(request[-1].role, 'user')
        self.assertTrue(request[-1].parts[0].text.strip())
        self.assertEqual(len(request), 3)
        self.assertEqual(len(original), 4)
        for empty in ([], [content('user', '   ')]):
            with self.subTest(empty=empty), self.assertRaises(ValueError):
                ensure_user_ending(empty)

    def test_mixed_tool_response_and_text_is_folded_into_json_without_changing_archive(self):
        tool = types.Part.from_function_response(name='list_datasets', response={'datasets': ['data']})
        tool.function_response.id = 'call-1'
        model = types.Content(role='model', parts=[types.Part(function_call=types.FunctionCall(
            name='list_datasets', id='call-1', args={}), thought_signature=b'keep-signature')])
        original = [content('user', '分析して'), model, types.Content(role='user', parts=[
            tool, types.Part.from_text(text='現在の未確認事項を確認する')])]
        before = [c.model_dump() for c in original]
        request = ensure_user_ending(original)
        self.assertEqual(request[-1].role, 'user')
        self.assertEqual(len(request[-1].parts), 1)
        response = request[-1].parts[0].function_response
        self.assertEqual(response.id, 'call-1')
        self.assertEqual(response.response['datasets'], ['data'])
        self.assertEqual(response.response['context_text'], ['現在の未確認事項を確認する'])
        self.assertEqual([c.model_dump() for c in original], before)
        self.assertIs(request[-2], model)
        self.assertEqual(ensure_user_ending(request), request)

    def test_pending_call_with_question_state_keeps_response_pair_and_no_text_part(self):
        model = types.Content(role='model', parts=[types.Part(function_call=types.FunctionCall(
            name='list_datasets', id='call-1', args={}), thought_signature=b'keep-signature')])
        ctx = ToolContext(tools=Mock(), settings=SimpleNamespace(), context_state={'questions': {
            'schema': {'status': 'open', 'question': '対象データはどこか？'}}})
        request = build_request([content('user', '分析して'), model], ctx)
        self.assertEqual(request[-1].role, 'user')
        self.assertEqual(len(request[-1].parts), 1)
        response = request[-1].parts[0].function_response
        self.assertEqual(response.id, 'call-1')
        self.assertEqual(response.response['status'], 'response_missing')
        self.assertEqual(response.response['current_question_state']['open_questions'][0]['id'], 'schema')
        self.assertEqual(request[-2].parts[0].thought_signature, b'keep-signature')

    def test_pending_calls_get_matching_missing_responses_without_execution(self):
        calls = [types.FunctionCall(name='run_query', id='q1', args={'sql': 'SELECT 1'}),
                 types.FunctionCall(name='list_tables', id='t1', args={'dataset': 'data'})]
        model = types.Content(role='model', parts=[
            types.Part(function_call=calls[0], thought_signature=b'keep'), types.Part(function_call=calls[1])])
        original = [content('user', '質問'), model]
        request = ensure_user_ending(original)
        self.assertIs(request[-2], model)
        self.assertEqual(request[-2].parts[0].thought_signature, b'keep')
        responses = [p.function_response for p in request[-1].parts]
        self.assertEqual([(r.name, r.id) for r in responses], [('run_query', 'q1'), ('list_tables', 't1')])
        self.assertTrue(all(r.response['status'] == 'response_missing' for r in responses))
        # 復旧を繰り返してもツール応答を重複させない。
        self.assertEqual(ensure_user_ending(request), request)
        self.assertEqual(len(original), 2)

    def test_valid_user_and_tool_responses_are_not_replaced(self):
        for last in [content('user', '続きの質問'), types.Content(role='user', parts=[
            types.Part.from_function_response(name='run_query', response={'result': {'row_count': 0}})])]:
            original = [content('model', '前の回答'), last]
            request = ensure_user_ending(original)
            self.assertEqual(request, original)
            self.assertIs(request[-1], last)

    def test_request_builder_recovers_model_ending_with_or_without_question_state(self):
        for questions in ({}, {'period': {'question': '期間は？', 'status': 'resolved'}}):
            ctx = ToolContext(tools=Mock(), settings=SimpleNamespace(), context_state={'questions': questions})
            original = [content('user', '質問'), content('model', '回答')]
            request = build_request(original, ctx)
            self.assertEqual(request[-1].role, 'user')
            self.assertEqual(original[-1].role, 'model')

    def test_first_question_tool_loop_serializes_user_endings_including_question_updates(self):
        captured = []
        dataset_id = result_id_for('list_datasets', {})
        table_id = result_id_for('list_tables', {'dataset': 'data'})
        note = lambda refs: {'summary': '- 利用できるデータを確認。', 'references': refs,
                             'open_questions': [], 'resolved_questions': []}
        opened = note([])
        opened['open_questions'] = [{'id': 'target_table', 'question': '対象テーブルはどれか？'}]
        closed = note([table_id])
        closed['resolved_questions'] = [{'id': 'target_table', 'conclusion': '来店テーブルを利用できる。'}]
        calls = [
            ('list_datasets', {'context_note': opened}),
            ('list_tables', {'dataset': 'data', 'context_note': note([dataset_id])}),
            ('finish_answer', {'answer': '来店データを分析できます。', 'context_note': closed,
                               'conversation_summary': '- 来店テーブルを利用可能。'}),
        ]

        def handler(request):
            payload = json.loads(request.content)
            self.assertEqual(payload['contents'][-1]['role'], 'user')
            self.assertTrue(payload['contents'][-1]['parts'])
            for turn in payload['contents']:
                if any('functionResponse' in part for part in turn['parts']):
                    self.assertTrue(all('functionResponse' in part for part in turn['parts']),
                                    'ツール応答とtextを同じturnに混在させない')
            name, args = calls[len(captured)]
            captured.append(payload)
            return httpx.Response(200, json={'candidates': [{'finishReason': 'STOP', 'content': {
                'role': 'model', 'parts': [{'functionCall': {'name': name, 'args': args, 'id': name},
                                          'thoughtSignature': 'c2lnbmF0dXJl'}]}}]})

        settings = SimpleNamespace(gemini_model='gemini-3.8-flash', thinking_budget=None, max_tool_iterations=5)
        tools = Mock()
        tools.list_datasets.return_value = {'datasets': ['data']}
        tools.list_tables.return_value = {'tables': ['来店']}
        ctx = ToolContext(tools=tools, settings=settings)
        original = [content('user', 'どのデータを分析できますか？')]
        with genai.Client(vertexai=True, project='test-project', location='global',
                          credentials=Credentials(token='local-test-token'), http_options=types.HttpOptions(
                              client_args={'transport': httpx.MockTransport(handler)})) as client:
            answer = GeminiAgent(settings, client=client).run(original, ctx)
        self.assertEqual(answer.text, '来店データを分析できます。')
        self.assertEqual(len(captured), 3)
        # 最初の結果を読み、次のツールを選ぶ呼び出しにもuser末尾と対応する応答を保つ。
        self.assertEqual(captured[1]['contents'][-1]['parts'][0]['functionResponse']['id'], 'list_datasets')
        self.assertEqual(len(captured[1]['contents'][-1]['parts']), 1)
        self.assertIn('open_questions', captured[1]['contents'][-1]['parts'][0]['functionResponse']['response']['current_question_state'])
        tools.list_datasets.assert_called_once()
        tools.list_tables.assert_called_once_with('data')
        self.assertEqual(ctx.context_state['questions']['target_table']['status'], 'resolved')

    def test_failed_api_call_logs_structure_without_question_sql_or_result_values(self):
        client = Mock()
        client.models.generate_content.side_effect = RuntimeError('API失敗')
        agent = GeminiAgent(SimpleNamespace(gemini_model='test', thinking_budget=None), client=client)
        original = [content('user', '質問の機密情報'), types.Content(role='model', parts=[
            types.Part(function_call=types.FunctionCall(name='run_query', args={'sql': 'SELECT secret'}))]),
            types.Content(role='user', parts=[types.Part.from_function_response(
                name='run_query', response={'result': '非公開の結果'})])]
        with self.assertLogs('src.agent', level='WARNING') as logs, self.assertRaises(RuntimeError):
            agent._generate(original)
        output = '\n'.join(logs.output)
        self.assertIn("['user', 'model', 'user']", output)
        self.assertIn('function_response', output)
        for sensitive in ('質問の機密情報', 'SELECT secret', '非公開の結果'):
            self.assertNotIn(sensitive, output)

    def test_sdk_serialized_payload_ends_in_user_for_real_and_recovered_history(self):
        captured = []

        def handler(request):
            payload = json.loads(request.content)
            captured.append(payload)
            # 実APIと同じ末尾条件でローカルのHTTP送信先を検証する。
            self.assertEqual(payload['contents'][-1]['role'], 'user')
            self.assertTrue(payload['contents'][-1]['parts'])
            for turn in payload['contents']:
                if any('functionResponse' in part for part in turn['parts']):
                    self.assertTrue(all('functionResponse' in part for part in turn['parts']),
                                    'ツール応答とtextを同じturnに混在させない')
            return httpx.Response(200, json={'candidates': [{'finishReason': 'STOP', 'content': {
                'role': 'model', 'parts': [{'text': '回答'}]}}]})

        settings = SimpleNamespace(gemini_model='gemini-3.8-flash', thinking_budget=None)
        with genai.Client(vertexai=True, project='test-project', location='global',
                          credentials=Credentials(token='local-test-token'), http_options=types.HttpOptions(
                              client_args={'transport': httpx.MockTransport(handler)})) as client:
            agent = GeminiAgent(settings, client=client)
            agent._generate([content('user', '通常の質問')])
            original = [content('user', '質問'), content('model', '前回の回答')]
            agent._generate(original)
            self.assertEqual(original[-1].role, 'model')
            tool = types.Content(role='model', parts=[types.Part(function_call=types.FunctionCall(
                name='run_query', id='query-1', args={'sql': 'SELECT 1'}), thought_signature=b'signature')])
            agent._generate([content('user', '質問'), tool])
        self.assertEqual(len(captured), 3)
        recovered = captured[-1]['contents']
        self.assertEqual(recovered[-2]['parts'][0]['thoughtSignature'], 'c2lnbmF0dXJl')
        self.assertEqual(recovered[-1]['parts'][0]['functionResponse']['id'], 'query-1')


if __name__ == '__main__':
    unittest.main()
