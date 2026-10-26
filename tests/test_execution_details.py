"""失敗したターンの可視化と、確認済みテーブルの再利用。"""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from streamlit.testing.v1 import AppTest
from src.agent import GeminiAgent, ToolContext
from src.bq_tools import BQToolError
from src.chat_history import pack, unpack


class ExecutionDetailTests(unittest.TestCase):
    def context(self):
        settings = SimpleNamespace(gcp_project='test', default_dataset='202506', allowed_datasets=('202506',), gemini_model='test')
        ctx = ToolContext(settings=settings, tools=Mock())
        ctx.tools.list_tables.return_value = {'dataset': '202506', 'tables': [{'table': '来店'}]}
        ctx.tools.get_table_schema.return_value = {'columns': [{'name': 'store'}]}
        return GeminiAgent(settings, client=Mock()), ctx

    def test_failed_ai_request_is_timed_and_saved(self):
        from google.genai import types
        agent, ctx = self.context()
        agent.settings.max_tool_iterations = 1
        agent._generate = Mock(side_effect=TimeoutError('timed out'))
        ctx.start_turn()
        contents = [types.Content(role='user', parts=[types.Part.from_text(text='来店数')])]
        with self.assertRaises(TimeoutError):
            agent.run(contents, ctx)
        self.assertEqual(ctx.turn_events[-1]['event'], 'llm_call')
        self.assertTrue(ctx.turn_events[-1]['error'])
        self.assertGreaterEqual(ctx.turn_events[-1]['seconds'], 0)
        self.assertNotIn('input_tokens', ctx.turn_events[-1])

    def test_failed_turn_still_displays_id_and_timing_without_data(self):
        app = AppTest.from_string('''
from app import render_results
render_results({'turn_id': 'test-turn-123', 'elapsed_seconds': 2.5,
 'execution_events': [{'tool': 'run_query', 'seconds': .1, 'error': True}],
 'errors': ['SQLの実行に失敗しました']}, 0, set())
''').run(timeout=20)
        self.assertFalse(app.exception)
        self.assertEqual(app.expander[0].label, 'データ・実行詳細')
        self.assertIn('test-turn-123', '\n'.join(c.value for c in app.caption))
        self.assertTrue(app.dataframe[0].value['エラー'].iloc[0])
        self.assertIn('SQLの実行に失敗', app.text[0].value)

    def test_unknown_table_is_rejected_and_listing_is_reused(self):
        agent, ctx = self.context()
        for _ in range(2):
            with self.assertRaises(BQToolError):
                agent._call_tool('get_table_schema', {'table': 'visits'}, ctx)
        ctx.tools.get_table_schema.assert_not_called()
        ctx.tools.list_tables.assert_called_once_with('202506')
        agent._call_tool('get_table_schema', {'table': '来店'}, ctx)
        agent._call_tool('get_table_schema', {'table': '来店'}, ctx)
        ctx.tools.list_tables.assert_called_once()
        ctx.tools.get_table_schema.assert_called_once_with('202506', '来店')

    def test_restored_schema_skips_listing_and_unknown_sql_table_never_runs(self):
        agent, ctx = self.context()
        ctx.schema_cache['202506.来店'] = {'columns': []}
        ctx.start_turn()
        ctx.turn_events.append({'event': 'llm_call', 'seconds': 1.25, 'error': False})
        message = {'role': 'assistant', 'text': '結果', 'turn_id': ctx.turn_id,
                   'execution_events': list(ctx.turn_events)}
        restored = ToolContext(settings=ctx.settings, tools=ctx.tools)
        messages, _, _ = unpack(pack([message], [], ctx, None), restored)
        agent._call_tool('get_table_schema', {'table': '来店'}, restored)
        ctx.tools.list_tables.assert_not_called()
        self.assertEqual(messages[0]['execution_events'][0]['seconds'], 1.25)
        self.assertEqual(restored.turn_events, ctx.turn_events)
        with self.assertRaises(BQToolError):
            agent._run_query({'sql': 'SELECT * FROM `test.202506.visits`'}, restored)
        ctx.tools.run_query.assert_not_called()
