"""会話の継続・課金・終了診断・データセット境界の回帰テスト。GCP 接続不要。"""
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
from google.genai import types

from src.agent import GeminiAgent, ToolContext, ConfirmationPending
from src.bq_tools import QueryRun, ConfirmationRequired
from src.sql_guard import SQLGuardError, check_sql
from src.usage_log import UsageLogger


def response(text='回答', reason='STOP', call=None, content=True):
    parts = [types.Part.from_text(text=text)] if text else []
    if call:
        parts.append(types.Part(function_call=types.FunctionCall(name=call, args={'sql': 'SELECT 1'})))
    candidate = types.Candidate(
        content=types.Content(role='model', parts=parts) if content else None,
        finish_reason=reason,
    )
    return types.GenerateContentResponse(
        candidates=[candidate],
        usage_metadata=types.GenerateContentResponseUsageMetadata(
            prompt_token_count=10, candidates_token_count=3,
            thoughts_token_count=7, total_token_count=20,
        ),
    )


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.tools = Mock()
        self.logger = Mock()
        self.settings = SimpleNamespace(max_tool_iterations=3)
        self.ctx = ToolContext(tools=self.tools, settings=self.settings, usage_logger=self.logger)
        self.agent = GeminiAgent(self.settings, client=Mock())
        self.contents = [types.Content(role='user', parts=[types.Part.from_text(text='今月は？')])]

    def test_followup_contains_previous_final_answer_and_signature(self):
        original = response('前月比10%増です。')
        original.candidates[0].content.parts[0].thought_signature = b'signature'
        self.agent._generate = Mock(side_effect=[original, response('来店数の増加です。')])
        self.agent.run(self.contents, self.ctx)
        self.contents.append(types.Content(role='user', parts=[types.Part.from_text(text='その理由は？')]))
        snapshot = list(self.contents)
        self.agent.run(self.contents, self.ctx)
        self.assertEqual([c.role for c in snapshot], ['user', 'model', 'user'])
        self.assertEqual(snapshot[1].parts[0].text, '前月比10%増です。')
        self.assertEqual(snapshot[1].parts[0].thought_signature, b'signature')
        self.assertEqual(len(self.contents), 4)

    def test_tool_history_and_final_answer_remain_in_order(self):
        self.agent._generate = Mock(side_effect=[response('', call='list_datasets'), response('結果です。')])
        self.agent._dispatch = Mock(return_value=types.Part.from_function_response(name='list_datasets', response={'datasets': []}))
        self.agent.run(self.contents, self.ctx)
        self.assertEqual([c.role for c in self.contents], ['user', 'model', 'user', 'model'])
        self.assertIsNotNone(self.contents[2].parts[0].function_response)
        self.assertEqual(self.contents[-1].parts[0].text, '結果です。')

    def test_progress_describes_each_operation_before_it_executes(self):
        progress = []
        calls = [
            types.FunctionCall(name='list_datasets', args={}),
            types.FunctionCall(name='list_tables', args={'dataset': '店舗データ'}),
            types.FunctionCall(name='get_table_schema', args={'dataset': '店舗データ', 'table': '来店'}),
            types.FunctionCall(name='run_query', args={'sql': 'SELECT 1', 'purpose': '店舗別の来店数を集計'}),
        ]
        self.settings.max_tool_iterations = 5
        replies = []
        for call in calls:
            reply = response('')
            reply.candidates[0].content.parts = [types.Part(function_call=call)]
            replies.append(reply)
        self.agent._generate = Mock(side_effect=[*replies, response('完了')])
        observed = []

        def dispatch(call, ctx):
            observed.append(progress[-1])
            if call.name == 'run_query':
                ctx.turn_runs.append(QueryRun('SELECT 1', 'SELECT 1', 0, 0, pd.DataFrame({'n': [1]}), {}))
            return types.Part.from_function_response(name=call.name, response={})

        self.agent._dispatch = Mock(side_effect=dispatch)
        self.agent.run(self.contents, self.ctx, on_progress=progress.append)
        self.assertIn('調べ方', progress[0])
        self.assertIn('データセット', observed[0])
        self.assertIn('店舗データ', observed[1])
        self.assertIn('テーブル一覧', observed[1])
        self.assertIn('店舗データ.来店', observed[2])
        self.assertIn('列名・データ型', observed[2])
        self.assertIn('店舗別の来店数を集計', observed[3])
        self.assertTrue(progress[-1].startswith('データを取得・集計しています'))
        self.assertFalse(any('回答をまとめ' in text or '分析を進め' in text for text in progress))
        self.assertFalse(any('SELECT' in message for message in progress))

    def test_different_literal_spaces_execute_separately(self):
        def query(sql, **kwargs):
            return QueryRun(sql, sql, 0, 100, pd.DataFrame({'value': [sql]}), {})
        self.tools.run_query.side_effect = query
        first = self.agent._run_query({'sql': "SELECT 'A B' AS value"}, self.ctx)
        second = self.agent._run_query({'sql': "SELECT 'A  B' AS value"}, self.ctx)
        self.assertNotEqual(first['executed_sql'], second['executed_sql'])
        self.assertEqual(self.tools.run_query.call_count, 2)
        self.assertEqual(self.ctx.session_billed_bytes, 200)
        self.agent._run_query({'sql': "SELECT 'A  B' AS value"}, self.ctx)
        self.assertEqual(self.tools.run_query.call_count, 2)

    def diagnostic(self, reply):
        self.agent._generate = Mock(return_value=reply)
        answer = self.agent.run(self.contents, self.ctx)
        self.logger.log_error.assert_called_once()
        error = self.logger.log_error.call_args.kwargs
        return answer, error, json.loads(error['error_text'])

    def test_empty_stop_is_logged_and_fallback_is_kept_in_history(self):
        answer, error, detail = self.diagnostic(response(''))
        self.assertEqual(error['error_source'], 'empty_response')
        self.assertEqual(detail['finish_reason'], 'STOP')
        self.assertEqual(self.contents[-1].parts[0].text, answer.text)

    def test_missing_candidate_is_logged(self):
        _, error, _ = self.diagnostic(types.GenerateContentResponse(candidates=[]))
        self.assertEqual(error['error_source'], 'missing_candidate')

    def test_missing_content_at_output_limit_is_not_misclassified(self):
        _, error, detail = self.diagnostic(response('', reason='MAX_TOKENS', content=False))
        self.assertEqual(error['error_source'], 'output_limit')
        self.assertEqual(detail['finish_reason'], 'MAX_TOKENS')

    def test_partial_output_is_marked_and_logged_without_double_billing(self):
        answer, error, detail = self.diagnostic(response('途中まで', reason='MAX_TOKENS'))
        self.assertIn('途中で打ち切られ', answer.text)
        self.assertEqual(error['error_source'], 'output_limit')
        self.assertEqual(detail['thinking_tokens'], 7)
        self.assertEqual(detail['output_tokens'], 10)
        self.assertEqual(self.ctx.session_output_tokens, 10)
        self.logger.log_gemini_call.assert_called_once()
        self.assertNotIn('output_tokens', error)
        self.assertEqual(len(self.contents), 2)

    def test_truncated_call_is_not_left_unanswered_in_history(self):
        self.diagnostic(response('途中まで', reason='MAX_TOKENS', call='run_query'))
        self.assertTrue(all(not p.function_call for p in self.contents[-1].parts))
        self.tools.run_query.assert_not_called()

    def test_malformed_function_call_is_never_executed(self):
        _, error, _ = self.diagnostic(response('', reason='MALFORMED_FUNCTION_CALL', call='run_query'))
        self.assertEqual(error['error_source'], 'model_response_error')
        self.tools.run_query.assert_not_called()

    def test_iteration_limit_is_logged_and_conversation_can_continue(self):
        self.settings.max_tool_iterations = 1
        self.agent._generate = Mock(return_value=response('', call='list_datasets'))
        self.agent._dispatch = Mock(return_value=types.Part.from_function_response(name='list_datasets', response={}))
        answer = self.agent.run(self.contents, self.ctx)
        self.assertIn('途中で打ち切', answer.text)
        error = self.logger.log_error.call_args.kwargs
        self.assertEqual(error['error_source'], 'iteration_limit')
        detail = json.loads(error['error_text'])
        self.assertEqual(detail['finish_reason'], 'MAX_TOOL_ITERATIONS')
        self.assertEqual(detail['token_scope'], 'turn')
        self.assertEqual(detail['output_tokens'], 10)
        self.assertIsNotNone(self.contents[-2].parts[0].function_response)
        self.assertEqual(self.contents[-1].parts[0].text, answer.text)

    def test_confirmation_resume_does_not_duplicate_call_or_rebill(self):
        final = response('', call='finish_answer')
        from src import context_memory
        final.candidates[0].content.parts[0].function_call.args = {
            'answer': '完了', 'conversation_summary': '結果を確認した',
            'context_note': {'summary': '結果を確認した', 'references': [context_memory.result_id_for('run_query', {'sql': 'SELECT 1'})]}}
        self.agent._generate = Mock(side_effect=[response('', call='run_query'), final])
        self.tools.run_query.side_effect = [
            ConfirmationRequired('SELECT 1', 'SELECT 1', 100, 10),
            QueryRun('SELECT 1', 'SELECT 1', 100, 100, pd.DataFrame({'n': [1]}), {}),
        ]
        pending = self.agent.run(self.contents, self.ctx)
        self.assertIsInstance(pending, ConfirmationPending)
        self.assertEqual(len(self.contents), 2)
        self.ctx.decisions[pending.key] = 'approved'
        progress = []
        answer = self.agent.run(self.contents, self.ctx, on_progress=progress.append)
        self.assertEqual(progress[0], 'データを取得・集計しています…')
        self.assertEqual(progress[-1], '回答を作成しています…')
        self.assertFalse(any('回答をまとめ' in text or '分析を進め' in text for text in progress))
        self.assertEqual(answer.text, '完了')
        self.assertEqual([c.role for c in self.contents], ['user', 'model', 'user', 'model', 'user', 'model'])
        self.assertEqual(self.ctx.session_billed_bytes, 100)
        self.logger.log_bq_query.assert_called_once()


class SQLGuardTests(unittest.TestCase):
    def check(self, sql):
        return check_sql(sql, project='test', allowed_datasets=('allowed',), default_row_limit=1000)

    def test_aliases_comments_and_literals_are_not_tables(self):
        for sql in [
            'SELECT `l.name` FROM `test.allowed.t` AS l',
            "SELECT '`other.blocked.t`' AS label FROM `test.allowed.t`",
            'SELECT 1 /* FROM `other.blocked.t` */',
            'SELECT 1 -- JOIN `other.blocked.t`',
            'SELECT EXTRACT(YEAR FROM CURRENT_DATE())',
            'SELECT `first_visit_info.visit_date` FROM `test.allowed.t` first_visit_info',
        ]:
            with self.subTest(sql=sql): self.check(sql)

    def test_all_real_references_are_checked(self):
        for sql in [
            'SELECT * FROM `test.allowed.t` JOIN `test.blocked.t` USING(id)',
            'SELECT * FROM `test.allowed.t`, `test.blocked.t`',
            'SELECT * FROM test.allowed.t, other.blocked.t',
            'WITH x AS (SELECT * FROM `test.blocked.t`) SELECT * FROM x',
            'SELECT * FROM `test.allowed.t` WHERE EXISTS (SELECT 1 FROM `other.allowed.t`)',
            'SELECT * FROM `test.allowed.t` UNION ALL SELECT * FROM `test.blocked.t`',
            'SELECT `other.blocked.fn`(1)',
            'SELECT other.blocked.fn(1)',
            "SELECT * FROM EXTERNAL_QUERY('other.connection', 'select 1')",
        ]:
            with self.subTest(sql=sql), self.assertRaises(SQLGuardError): self.check(sql)

    def test_mutations_and_multiple_statements_remain_blocked(self):
        for sql in ['DELETE FROM `test.allowed.t` WHERE TRUE', 'SELECT 1; SELECT 2',
                    'CREATE TABLE `test.allowed.t` AS SELECT 1',
                    'WITH x AS (SELECT 1) DELETE FROM `test.allowed.t` WHERE TRUE']:
            with self.subTest(sql=sql), self.assertRaises(SQLGuardError): self.check(sql)

    def test_cte_array_and_japanese_columns_are_supported(self):
        sql = 'WITH x AS (SELECT `来店数`, arr FROM `test.allowed.来店`) SELECT `x.来店数` FROM x, UNNEST(x.arr) a'
        self.assertEqual(self.check(sql).referenced_tables, ('test.allowed.来店',))

    def test_inner_limit_does_not_disable_outer_limit(self):
        result = self.check('SELECT * FROM (SELECT * FROM `test.allowed.t` LIMIT 1)')
        self.assertTrue(result.limit_injected)
        self.assertTrue(result.sql.endswith('LIMIT 1000'))
        self.assertFalse(self.check('SELECT * FROM `test.allowed.t` LIMIT 20').limit_injected)

    def test_invalid_sql_fails_closed(self):
        with self.assertRaises(SQLGuardError): self.check('SELECT * FROM (')


class LoggingTests(unittest.TestCase):
    def test_error_context_does_not_populate_billing_token_columns(self):
        logger = UsageLogger(SimpleNamespace())
        logger._insert = Mock()
        logger.log_error(user_email='test', session_id='s', turn_id='t',
                         error_source='output_limit', error_text='{"thinking_tokens":7}')
        row = logger._insert.call_args.args[0]
        self.assertEqual(row['event_type'], 'error')
        self.assertNotIn('output_tokens', row)
        self.assertNotIn('thinking_tokens', row)


if __name__ == '__main__':
    unittest.main()
