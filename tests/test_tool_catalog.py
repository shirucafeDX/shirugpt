"""ツール詳細の選択取得・切り替え・保存・API送信を外部接続なしで検証する。"""
import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from zipfile import ZipFile, ZIP_DEFLATED

import httpx
import pandas as pd
from google import genai
from google.genai import types
from google.oauth2.credentials import Credentials

from src.agent import GeminiAgent, ToolContext, ConfirmationPending, build_declarations
from src.bq_tools import BQToolError, ConfirmationRequired, QueryRun
from src.chat_history import pack, unpack
from src.context_memory import encoded, result_id_for
from src.tool_catalog import SELECT_TOOL, TOOL_SUMMARIES, active_declarations, validate_selection


def settings():
    return SimpleNamespace(gcp_project='test', allowed_datasets=('data',), gemini_model='test-model',
                           max_tool_iterations=8, thinking_budget=None, vertex_location='global')


def user(value='データを分析して'):
    return types.Content(role='user', parts=[types.Part.from_text(text=value)])


def note(refs=None):
    return {'summary': '- 利用できるデータを確認。', 'references': refs or [],
            'open_questions': [], 'resolved_questions': []}


def response(name=None, args=None, value='回答', call_id=None):
    parts = [types.Part(function_call=types.FunctionCall(name=name, args=args or {}, id=call_id or name),
                        thought_signature=b'preserve-signature')] if name else [types.Part.from_text(text=value)]
    return types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role='model', parts=parts), finish_reason='STOP')],
        usage_metadata=types.GenerateContentResponseUsageMetadata(prompt_token_count=10, candidates_token_count=2, total_token_count=12))


def names(config):
    return [d.name for tool in config.tools for d in tool.function_declarations]


class CatalogTests(unittest.TestCase):
    def setUp(self):
        self.settings = settings()
        self.client = Mock()
        self.tools = Mock()
        self.ctx = ToolContext(tools=self.tools, settings=self.settings)
        self.agent = GeminiAgent(self.settings, client=self.client)

    def test_initial_config_only_sends_short_catalog_and_selector(self):
        config = self.agent._config()
        self.assertEqual(names(config), [SELECT_TOOL])
        for name in TOOL_SUMMARIES:
            self.assertIn(name, config.system_instruction)
        self.assertEqual(set(TOOL_SUMMARIES), {d.name for d in build_declarations()})
        full_chars = len(encoded([d.model_dump(exclude_none=True) for d in build_declarations()]))
        initial_chars = len(encoded([d.model_dump(exclude_none=True) for d in config.tools[0].function_declarations]))
        self.assertLess(initial_chars, full_chars * .15)
        self.assertNotIn('context_note', config.tools[0].function_declarations[0].parameters.properties)

    def test_selected_definitions_are_swapped_and_reused_without_all_definitions(self):
        self.agent._call_tool(SELECT_TOOL, {'names': ['list_datasets', 'list_tables']}, self.ctx)
        self.assertEqual(names(self.agent._config(self.ctx.active_tool_names)), [SELECT_TOOL, 'list_datasets', 'list_tables'])
        payload = self.agent._call_tool(SELECT_TOOL, {'names': ['run_query', 'finish_answer']}, self.ctx)
        self.assertEqual(names(self.agent._config(self.ctx.active_tool_names)), [SELECT_TOOL, 'run_query', 'finish_answer'])
        self.assertEqual(payload['available_tools'], ['run_query', 'finish_answer'])
        self.assertNotIn('parameters', encoded(payload))
        self.tools.assert_not_called()
        # 選択に対して元の定義を書き換えず、再利用する。
        query = next(d for d in self.agent.declarations if d.name == 'run_query')
        self.assertIs(active_declarations(self.agent.declarations, ['run_query'])[1], query)

    def test_invalid_selection_never_changes_state_or_executes_tools(self):
        self.ctx.active_tool_names = ['run_query']
        for value in (None, [], 'run_query', [True], ['unknown'], [SELECT_TOOL],
                      ['run_query', 'run_query'], list(TOOL_SUMMARIES)[:4]):
            with self.subTest(value=value), self.assertRaises(BQToolError):
                self.agent._call_tool(SELECT_TOOL, {'names': value}, self.ctx)
            self.assertEqual(self.ctx.active_tool_names, ['run_query'])
        self.assertEqual(self.tools.mock_calls, [])

    def test_selection_is_a_local_tool_and_not_an_analysis_result(self):
        self.client.models.generate_content.side_effect = [
            response(SELECT_TOOL, {'names': ['list_datasets']}), response(value='回答')]
        contents = [user()]
        self.agent.run(contents, self.ctx)
        self.assertEqual(self.ctx.active_tool_names, ['list_datasets'])
        self.assertEqual(self.tools.mock_calls, [])
        self.assertEqual(self.ctx.context_state['results'], {})
        self.assertEqual(contents[2].parts[0].function_response.id, SELECT_TOOL)
        self.assertEqual(len(contents[2].parts), 1)
        self.assertEqual(contents[1].parts[0].thought_signature, b'preserve-signature')

    def test_full_loop_reuses_selected_tools_and_counts_selection_calls(self):
        self.ctx.usage_logger = Mock()
        self.tools.list_datasets.return_value = {'datasets': ['data']}
        self.tools.list_tables.return_value = {'tables': ['visits']}
        dataset_id = result_id_for('list_datasets', {})
        table_id = result_id_for('list_tables', {'dataset': 'data'})
        self.client.models.generate_content.side_effect = [
            response(SELECT_TOOL, {'names': ['list_datasets', 'list_tables']}),
            response('list_datasets', {'context_note': note()}),
            response('list_tables', {'dataset': 'data', 'context_note': note([dataset_id])}),
            response(SELECT_TOOL, {'names': ['finish_answer']}),
            response('finish_answer', {'answer': '来店テーブルを利用できます。',
                                      'context_note': note([table_id]), 'conversation_summary': '- 来店テーブルがある。'}),
        ]
        result = self.agent.run([user()], self.ctx)
        self.assertIn('来店テーブル', result.text)
        self.assertEqual([names(c.kwargs['config']) for c in self.client.models.generate_content.call_args_list], [
            [SELECT_TOOL], [SELECT_TOOL, 'list_datasets', 'list_tables'],
            [SELECT_TOOL, 'list_datasets', 'list_tables'], [SELECT_TOOL, 'list_datasets', 'list_tables'],
            [SELECT_TOOL, 'finish_answer']])
        self.tools.list_datasets.assert_called_once()
        self.tools.list_tables.assert_called_once_with('data')
        self.assertEqual(self.ctx.turn_prompt_tokens, 50)
        self.assertEqual(self.ctx.usage_logger.log_gemini_call.call_count, 5)

    def test_same_agent_keeps_each_conversations_selection_isolated(self):
        other = ToolContext(tools=Mock(), settings=self.settings)
        self.client.models.generate_content.side_effect = [
            response(SELECT_TOOL, {'names': ['run_query']}), response(value='回答A'),
            response(SELECT_TOOL, {'names': ['list_tables']}), response(value='回答B'), response(value='回答A2')]
        contents = [user()]
        self.agent.run(contents, self.ctx)
        self.agent.run([user()], other)
        contents.append(user('続きを教えて'))
        self.agent.run(contents, self.ctx)
        configs = [names(c.kwargs['config']) for c in self.client.models.generate_content.call_args_list]
        self.assertEqual(configs, [[SELECT_TOOL], [SELECT_TOOL, 'run_query'], [SELECT_TOOL],
                                   [SELECT_TOOL, 'list_tables'], [SELECT_TOOL, 'run_query']])
        instructions = [c.kwargs['config'].system_instruction for c in self.client.models.generate_content.call_args_list]
        self.assertNotIn('## SQL を書くときの規則', instructions[0])
        self.assertIn('## SQL を書くときの規則', instructions[1])
        self.assertNotIn('## SQL を書くときの規則', instructions[3])
        self.assertIn('## テーブル一覧の確認', instructions[3])
        self.assertIn('## SQL を書くときの規則', instructions[4])
        self.assertEqual(other.active_tool_names, ['list_tables'])
        self.assertEqual({d.name for d in self.agent.declarations}, set(TOOL_SUMMARIES))

    def test_selection_takes_precedence_over_parallel_execution_and_preserves_ids(self):
        first = response(SELECT_TOOL, {'names': ['list_datasets']})
        first.candidates[0].content.parts.append(types.Part(function_call=types.FunctionCall(name='list_datasets', id='data-call', args={})))
        self.client.models.generate_content.side_effect = [first, response(value='回答')]
        contents = [user()]
        self.agent.run(contents, self.ctx)
        self.assertEqual(self.tools.mock_calls, [])
        responses = [part.function_response for part in contents[2].parts]
        self.assertEqual([r.id for r in responses], [SELECT_TOOL, 'data-call'])
        self.assertIn('未実行', responses[1].response['error'])

    def test_selection_survives_save_and_approval_without_reselection(self):
        self.ctx.active_tool_names = ['run_query', 'finish_answer']
        sql = 'SELECT 1 AS n'
        run = QueryRun(sql, sql, 20, 20, pd.DataFrame({'n': [1]}), {'row_count': 1})
        self.tools.run_query.side_effect = [ConfirmationRequired(sql, 20, 10, []), run]
        self.client.models.generate_content.side_effect = [response('run_query', {'sql': sql}), response('finish_answer', {
            'answer': '1件です。', 'context_note': note([result_id_for('run_query', {'sql': sql})]),
            'conversation_summary': '- 1件と確認。'})]
        contents = [user()]
        pending = self.agent.run(contents, self.ctx)
        self.assertIsInstance(pending, ConfirmationPending)
        saved = pack([], contents, self.ctx, pending)
        restored = ToolContext(tools=self.tools, settings=self.settings)
        _, restored_contents, restored_pending = unpack(saved, restored)
        self.assertEqual(restored.active_tool_names, ['run_query', 'finish_answer'])
        restored.decisions[restored_pending.key] = 'approved'
        self.assertEqual(self.agent.run(restored_contents, restored).text, '1件です。')
        self.assertEqual(self.client.models.generate_content.call_count, 2)
        self.assertTrue(self.tools.run_query.call_args.kwargs['approved'])
        self.assertEqual(names(self.client.models.generate_content.call_args.kwargs['config']),
                         [SELECT_TOOL, 'run_query', 'finish_answer'])

    def test_old_corrupt_and_changed_model_snapshots_reset_selection(self):
        self.ctx.active_tool_names = ['run_query']
        raw = pack([], [user()], self.ctx, None)
        restored = ToolContext(tools=Mock(), settings=self.settings)
        unpack(raw, restored)
        self.assertEqual(restored.active_tool_names, ['run_query'])
        output = io.BytesIO()
        with ZipFile(io.BytesIO(raw)) as source, ZipFile(output, 'w', ZIP_DEFLATED) as target:
            for name in source.namelist():
                data = source.read(name)
                if name == 'conversation.json':
                    payload = json.loads(data)
                    del payload['context']['active_tool_names']
                    data = json.dumps(payload).encode()
                target.writestr(name, data)
        unpack(output.getvalue(), restored)
        self.assertEqual(restored.active_tool_names, [])
        changed = ToolContext(tools=Mock(), settings=SimpleNamespace(**{**vars(self.settings), 'gemini_model': 'another-model'}))
        unpack(raw, changed)
        self.assertEqual(changed.active_tool_names, [])
        self.ctx.active_tool_names = ['foreign']
        unpack(pack([], [user()], self.ctx, None), restored)
        self.assertEqual(restored.active_tool_names, [])

    def test_sdk_sends_details_only_after_selection_and_keeps_tool_response_parts(self):
        captured = []
        replies = [response(SELECT_TOOL, {'names': ['list_datasets', 'finish_answer']}),
                   response('list_datasets', {'context_note': note()}),
                   response('finish_answer', {'answer': '利用できます。', 'context_note': note([
                       result_id_for('list_datasets', {})]), 'conversation_summary': '- データを利用できる。'})]

        def handler(request):
            payload = json.loads(request.content)
            definitions = [d for t in payload['tools'] for d in t['functionDeclarations']]
            allowed = {d['name'] for d in definitions}
            reply = replies[len(captured)]
            called = reply.candidates[0].content.parts[0].function_call.name
            self.assertIn(called, allowed)
            self.assertEqual(payload['contents'][-1]['role'], 'user')
            for turn in payload['contents']:
                if any('functionResponse' in part for part in turn['parts']):
                    self.assertTrue(all('functionResponse' in part for part in turn['parts']))
            captured.append(payload)
            return httpx.Response(200, json=json.loads(reply.model_dump_json(by_alias=True, exclude_none=True)))

        self.tools.list_datasets.return_value = {'datasets': ['data']}
        with genai.Client(vertexai=True, project='test-project', location='global',
                          credentials=Credentials(token='local-test-token'), http_options=types.HttpOptions(
                              client_args={'transport': httpx.MockTransport(handler)})) as client:
            result = GeminiAgent(self.settings, client=client).run([user()], self.ctx)
        self.assertEqual(result.text, '利用できます。')
        defs = [[d['name'] for t in p['tools'] for d in t['functionDeclarations']] for p in captured]
        self.assertEqual(defs, [[SELECT_TOOL], [SELECT_TOOL, 'list_datasets', 'finish_answer'],
                               [SELECT_TOOL, 'list_datasets', 'finish_answer']])
        instructions = [''.join(part.get('text', '') for part in p['systemInstruction']['parts']) for p in captured]
        self.assertNotIn('## データセット一覧の確認', instructions[0])
        self.assertNotIn('conversation_summary', instructions[0])
        for instruction in instructions[1:]:
            self.assertIn('## データセット一覧の確認', instruction)
            self.assertIn('conversation_summary', instruction)
            self.assertNotIn('## SQL を書くときの規則', instruction)
        # 選択応答にパラメータ定義の重複を含めない。
        selected = captured[1]['contents'][-1]['parts'][0]['functionResponse']['response']
        self.assertNotIn('parameters', selected)


if __name__ == '__main__':
    unittest.main()
