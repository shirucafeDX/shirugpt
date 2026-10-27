"""誤った列・失敗SQL・根拠なし回答を止め、引き継ぎ要約を保存する。"""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
from google.genai import types
from streamlit.testing.v1 import AppTest

from src import analysis_safety as safety, context_memory as memory
from src.agent import GeminiAgent, ToolContext, _sql_key
from src.bq_tools import QueryRun, BQToolError
from src.chat_history import pack, unpack
from src.sql_guard import SQLGuardError


def user(text):
    return types.Content(role='user', parts=[types.Part.from_text(text=text)])


def reply(name=None, args=None, text=None):
    return types.GenerateContentResponse(candidates=[types.Candidate(finish_reason='STOP',
        content=types.Content(role='model', parts=[types.Part(function_call=types.FunctionCall(name=name, args=args or {}))]
            if name else [types.Part.from_text(text=text)]))])


class AnalysisSafetyTests(unittest.TestCase):
    def setUp(self):
        self.settings = SimpleNamespace(gcp_project='test', default_dataset='202506',
            allowed_datasets=('202506',), gemini_model='test', thinking_budget=None,
            max_tool_iterations=8)
        self.ctx = ToolContext(settings=self.settings, tools=Mock())
        self.ctx.start_turn()
        self.agent = GeminiAgent(self.settings, client=Mock())
        self.listing = {'tables': [{'table': '来店'}, {'table': 'ユーザー'}, {'table': 'イベント'}]}
        self.schema = {'columns': [{'name': 'store', 'type': 'STRING'},
                                  {'name': 'ordered_at', 'type': 'DATETIME'},
                                  {'name': 'DU_id', 'type': 'INTEGER'}]}
        self.ctx.tools.list_tables.return_value = self.listing
        self.ctx.tools.get_table_schema.return_value = self.schema
        self.good_sql = 'SELECT store, COUNT(DISTINCT DU_id) AS n FROM `test.202506.来店` GROUP BY store'
        self.bad_sql = 'SELECT shop_name FROM `test.202506.来店`'
        self.ctx.tools.run_query.return_value = QueryRun(self.good_sql, self.good_sql, 10, 10,
            pd.DataFrame({'store': ['京大前店'], 'n': [12]}), {})

    def dispatch(self, name, args):
        return self.agent._dispatch(types.FunctionCall(name=name, args=args), self.ctx).function_response.response

    def test_relevant_schema_prefetched_once_and_attached_to_sql_request(self):
        contents = [user('京大店の来店数推移')]
        self.agent._prefetch_schemas(contents, self.ctx)
        self.ctx.tools.list_tables.assert_called_once_with('202506')
        self.ctx.tools.get_table_schema.assert_called_once_with('202506', '来店')
        self.ctx.start_turn()
        self.agent._prefetch_schemas(contents, self.ctx)
        self.ctx.tools.get_table_schema.assert_called_once()
        config = self.agent._config(['run_query'], self.ctx)
        self.assertIn('verified_schemas', config.system_instruction)
        self.assertIn('ordered_at', config.system_instruction)
        self.assertNotIn('ユーザー"', config.system_instruction)

    def test_restored_schema_needs_no_table_listing(self):
        self.ctx.schema_cache['202506.来店'] = self.schema
        self.agent._prefetch_schemas([user('来店数の推移')], self.ctx)
        self.ctx.tools.list_tables.assert_not_called()
        self.ctx.tools.get_table_schema.assert_not_called()
        self.assertIn('store', self.agent._config(['run_query'], self.ctx).system_instruction)

    def test_prefetch_fills_missing_related_schema_when_another_is_cached(self):
        self.ctx.schema_cache['202506.来店'] = self.schema
        self.ctx.listing_cache['tables:202506'] = self.listing
        self.agent._prefetch_schemas([user('来店とイベントを比較して')], self.ctx)
        self.ctx.tools.list_tables.assert_not_called()
        self.ctx.tools.get_table_schema.assert_called_once_with('202506', 'イベント')

    def test_prefetch_does_not_guess_nonexistent_table_or_run_for_advice(self):
        self.ctx.tools.list_tables.return_value = {'tables': [{'table': '日報'}]}
        self.agent._prefetch_schemas([user('来店数の推移')], self.ctx)
        self.ctx.tools.get_table_schema.assert_not_called()
        self.ctx.start_turn()
        self.ctx.listing_cache.clear()
        self.ctx.tools.list_tables.reset_mock()
        self.agent._prefetch_schemas([user('一般的な接客のアイデアを教えて')], self.ctx)
        self.ctx.tools.list_tables.assert_not_called()

    def test_wrong_column_is_rejected_before_query_and_repeated_failure_is_not_retried(self):
        result = self.dispatch('run_query', {'sql': self.bad_sql})
        self.assertIn('shop_name', result['error'])
        self.assertIn('store', result['error'])
        self.ctx.tools.run_query.assert_not_called()
        repeated = self.dispatch('run_query', {'sql': self.bad_sql + ' -- again'})
        self.assertIn('同じSQL', repeated['error'])
        self.ctx.tools.run_query.assert_not_called()
        corrected = self.dispatch('run_query', {'sql': self.good_sql})
        self.assertNotIn('error', corrected)
        self.ctx.tools.run_query.assert_called_once()
        self.assertTrue(safety.has_data_reference(self.ctx, [corrected['result_id']]))

    def test_cte_alias_join_and_case_insensitive_columns_validate(self):
        schemas = {'202506.来店': self.schema, '202506.ユーザー': {'columns': [
            {'name': 'store', 'type': 'STRING'}, {'name': 'name', 'type': 'STRING'}]}}
        for sql in [self.good_sql,
                    'WITH x AS (SELECT store AS s FROM `test.202506.来店`) SELECT s FROM x',
                    'SELECT a.store, b.name FROM `test.202506.来店` a JOIN `test.202506.ユーザー` b ON a.store=b.store']:
            refs = ['test.202506.来店'] + (['test.202506.ユーザー'] if 'JOIN' in sql else [])
            safety.verify_columns(sql, refs, schemas)
        with self.assertRaises(SQLGuardError):
            safety.verify_columns('SELECT x.missing FROM (SELECT store FROM `test.202506.来店`) x',
                                  ['test.202506.来店'], schemas)

    def test_metadata_and_failed_results_cannot_support_data_answer(self):
        safety.state(self.ctx)['data_requested'] = True
        meta = self.dispatch('list_tables', {'dataset': '202506'})['result_id']
        args = {'answer': '取得値を確認しました', 'conversation_summary': '248件だった',
                'context_note': {'summary': '248件だった', 'references': [meta]}}
        self.assertIn('result_id', self.dispatch('finish_answer', args)['error'])
        memory.state_for(self.ctx)['results']['failed'] = {'payload': {'error': 'SQL failed'}}
        args['context_note']['references'] = ['failed']
        self.assertIn('失敗', self.dispatch('finish_answer', args)['error'])
        self.assertIn('失敗', memory.validate_note(self.ctx, args['context_note']))

    def test_successful_query_and_valid_finish_answer_keep_summaries_and_references(self):
        rid = memory.result_id_for('run_query', {'sql': self.good_sql})
        self.agent._generate = Mock(side_effect=[reply('run_query', {'sql': self.good_sql}),
            reply('finish_answer', {'answer': '京大前店は{{n}}件です。', 'numeric_claims': [{'key': 'n', 'operation': 'cell', 'sources': [{'result_ref': rid, 'column': 'n', 'row_index': 0}]}], 'conversation_summary': '京大前店の来店数は12件。',
                'context_note': {'summary': '確認した来店数は12件。', 'references': [rid]}})])
        contents = [user('京大店の来店数推移')]
        result = self.agent.run(contents, self.ctx)
        self.assertEqual(result.text, '京大前店は12件です。')
        summary = safety.snapshot(self.ctx)
        self.assertEqual(summary['context_note']['references'], [rid])
        self.assertIn('12件', summary['summary'])
        self.assertIn('12件', summary['conversation_summary'])
        messages = [{'role': 'assistant', 'text': result.text, 'turn_id': self.ctx.turn_id, 'execution_summary': summary}]
        restored = ToolContext(settings=self.settings, tools=Mock())
        saved, _, _ = unpack(pack(messages, contents, self.ctx, None), restored)
        self.assertEqual(saved[0]['execution_summary'], summary)
        self.assertEqual(safety.snapshot(restored), summary)
        self.ctx.start_turn()
        self.assertIn('12件', saved[0]['execution_summary']['summary'])

    def test_plain_answer_cannot_bypass_rejected_finish_answer(self):
        self.agent._generate = Mock(side_effect=[reply('run_query', {'sql': self.bad_sql}),
            reply('finish_answer', {'answer': '248件です', 'conversation_summary': '248件',
                'context_note': {'summary': '248件', 'references': ['invented']}}), reply(text='248件です')])
        contents = [user('来店数の推移')]
        result = self.agent.run(contents, self.ctx)
        self.assertEqual(result.text, safety.BLOCKED_ANSWER)
        self.assertNotIn('248', result.text)
        self.ctx.tools.run_query.assert_not_called()
        summary = safety.snapshot(self.ctx)
        self.assertFalse(summary['summary'])
        self.assertTrue(any(step['status'] == 'failed' for step in summary['steps']))
        self.assertIn('根拠', summary['next_action'])

    def test_followup_cannot_use_unvalidated_prose_to_make_data_claims(self):
        self.dispatch('run_query', {'sql': self.good_sql})
        self.ctx.start_turn()
        self.agent._generate = Mock(return_value=reply(text='前年は999件でした'))
        result = self.agent.run([user('前年は？')], self.ctx)
        self.assertEqual(result.text, safety.BLOCKED_ANSWER)

    def test_general_advice_after_data_does_not_require_irrelevant_result_references(self):
        self.dispatch('run_query', {'sql': self.good_sql})
        self.ctx.start_turn()
        self.agent._prefetch_schemas([user('データを使わず、一般的な施策のアイデアを相談したい')], self.ctx)
        self.assertFalse(safety.needs_evidence(self.ctx))
        result = self.dispatch('finish_answer', {'answer': '仮説を立てて小さく試します。',
            'conversation_summary': '一般的な施策を相談した。',
            'context_note': {'summary': '一般論としての施策案。', 'references': []}})
        self.assertEqual(result['status'], 'final_answer')

    def test_saved_data_recall_can_be_used_as_evidence(self):
        source = self.dispatch('run_query', {'sql': self.good_sql})['result_id']
        self.ctx.start_turn()
        recalled = self.dispatch('recall_context', {'result_id': source})
        self.assertTrue(safety.has_data_reference(self.ctx, [recalled['result_id']]))
        self.assertIsNone(safety.validate_answer(self.ctx, {'references': [recalled['result_id']]}))
        self.ctx.tools.run_query.assert_called_once()

    def test_execution_detail_renders_ai_summary_failure_and_next_action(self):
        app = AppTest.from_string('''
from app import render_results
render_results({'turn_id': 'safe-turn', 'execution_summary': {
 'summary': '取得結果の要約', 'conversation_summary': '会話全体の要約',
 'context_note': {'summary': '作業メモ', 'references': ['result_real']},
 'steps': [{'tool': 'run_query', 'status': 'failed', 'error': 'shop_nameは存在しません'}],
 'next_action': 'storeを使って修正する'}}, 0, set())
''').run(timeout=20)
        self.assertFalse(app.exception)
        self.assertEqual([e.label for e in app.expander], ['データ・実行詳細', '引き継ぎ要約'])
        visible = '\n'.join(e.value for e in app.markdown) + '\n'.join(e.value for e in app.text)
        for text in ('取得結果の要約', '会話全体の要約', '作業メモ', 'shop_name', 'store'):
            self.assertIn(text, visible)
        self.assertIn('result_real', '\n'.join(e.value for e in app.caption))
