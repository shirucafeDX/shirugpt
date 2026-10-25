"""分析の正確さ、送信予算、SQL境界、段階的な会話・保存を外部接続なしで検証。"""
import csv
from dataclasses import replace
import io
import json
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
from google.genai import types

from src import context_memory as memory
from src.agent import GeminiAgent, ToolContext, _sql_key
from src.analysis_layer import analyze
from src.bq_tools import BQToolError, QueryRun
from src.chat_history import pack, unpack
from src.config import ConfigError, load_settings
from src.execution import ExecutionControl, ExecutionStopped
from src.prompts import build_system_instruction
from src.token_budget import BudgetPolicy, TokenBudgetManager, bound_history, estimate_tokens


def rows(payload):
    return list(csv.DictReader(io.StringIO(payload['result']['csv'])))


def content(role, text):
    return types.Content(role=role, parts=[types.Part.from_text(text=text)])


def response(name=None, args=None):
    return types.GenerateContentResponse(candidates=[types.Candidate(finish_reason='STOP', content=types.Content(role='model', parts=[
        types.Part(function_call=types.FunctionCall(name=name, args=args or {}, id=name), thought_signature=b'original')
        ] if name else [types.Part.from_text(text='回答')]))])


class StatisticsTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({'group': ['a', 'a', 'b', 'b'], 'x': [1, 2, 4, 8],
                                   'y': [2, 4, 8, 16], 'month': ['2024-01', '2024-02'] * 2})

    def test_schema_and_description_do_not_return_original_rows(self):
        for operation in ('get_schema', 'get_row_count', 'describe_data'):
            result = analyze(self.frame, {'operation': operation})
            self.assertLessEqual(result['result']['sent_row_count'], 4)
            self.assertEqual(result['source_row_count'], 4)
        described = rows(analyze(self.frame, {'operation': 'describe_data', 'columns': ['x']}))[0]
        self.assertEqual(float(described['mean']), 3.75)
        self.assertEqual(float(described['median']), 3)
        self.assertEqual(float(described['q25']), 1.75)

    def test_aggregation_global_grouped_null_and_exact_large_integer_sum(self):
        spec = [{'column': 'x', 'function': 'sum', 'alias': 'total'}]
        result = analyze(self.frame, {'operation': 'aggregate', 'group_by': ['group'], 'aggregations': spec})
        self.assertEqual([int(r['total']) for r in rows(result)], [3, 12])
        self.assertEqual(int(rows(analyze(self.frame, {'operation': 'aggregate', 'aggregations': spec}))[0]['total']), 15)
        frame = pd.DataFrame({'g': [None, None], 'x': pd.Series([2**62, 2**62], dtype='Int64')})
        result = analyze(frame, {'operation': 'aggregate', 'group_by': ['g'], 'aggregations': spec})
        self.assertEqual(rows(result)[0]['total'], str(2**63))
        frame['x'] = pd.Series([pd.NA, pd.NA], dtype='Int64')
        result = analyze(frame, {'operation': 'aggregate', 'aggregations': spec})
        self.assertEqual(rows(result)[0]['total'], result['result']['null_marker'])

    def test_top_bottom_absolute_filter_distinct_and_sampling(self):
        for ascending, expected in ((False, '8'), (True, '1')):
            result = analyze(self.frame, {'operation': 'top_k', 'sort_by': ['x'], 'ascending': ascending, 'limit': 1})
            self.assertEqual(rows(result)[0]['x'], expected)
        frame = pd.DataFrame({'x': [-10, 5, 3]})
        self.assertEqual(rows(analyze(frame, {'operation': 'top_k', 'sort_by': ['x'], 'absolute': True}))[0]['x'], '-10')
        result = analyze(self.frame, {'operation': 'filter_data', 'filters': [{'column': 'group', 'op': 'eq', 'values': ['b']}], 'columns': ['x']})
        self.assertEqual([r['x'] for r in rows(result)], ['4', '8'])
        self.assertEqual(result['matched_row_count'], 2)
        distinct = rows(analyze(self.frame, {'operation': 'get_distinct_values', 'columns': ['group']}))
        self.assertEqual([int(r['result_rows']) for r in distinct], [2, 2])
        sampled = analyze(self.frame, {'operation': 'sample_data', 'columns': ['x'], 'limit': 2})
        self.assertEqual(len(rows(sampled)), 2)

    def test_group_comparison_and_correlations_handle_missing_and_constants(self):
        compared = rows(analyze(self.frame, {'operation': 'compare_groups', 'group_by': ['group'], 'metric': 'x', 'baseline': 'a'}))
        self.assertEqual(compared[0]['group'], 'b')
        self.assertEqual(float(compared[0]['delta']), 4.5)
        self.assertEqual(float(compared[0]['change_pct']), 300)
        for method in ('pearson', 'spearman'):
            pair = rows(analyze(self.frame, {'operation': 'correlation', 'columns': ['x', 'y'], 'method': method}))[0]
            self.assertAlmostEqual(float(pair['correlation']), 1)
            self.assertEqual(int(pair['paired_rows']), 4)
        frame = pd.DataFrame({'x': [1, 1, None, 1], 'y': [2, 3, 4, None]})
        result = analyze(frame, {'operation': 'correlation', 'columns': ['x', 'y']})
        self.assertEqual(rows(result)[0]['correlation'], result['result']['null_marker'])
        self.assertEqual(int(rows(result)[0]['paired_rows']), 2)

    def test_iqr_and_zscore_detect_outlier_without_mutating_data(self):
        frame = pd.DataFrame({'name': [str(i) for i in range(21)], 'x': [10] * 20 + [100]})
        before = frame.copy(deep=True)
        for method in ('iqr', 'z_score'):
            result = analyze(frame, {'operation': 'detect_outliers', 'metric': 'x', 'method': method})
            self.assertEqual(rows(result)[0]['name'], '20')
            self.assertEqual(len(rows(result)), 1)
        pd.testing.assert_frame_equal(frame, before)

    def test_timeseries_gaps_zero_baselines_and_duplicate_aggregation(self):
        frame = pd.DataFrame({'t': ['2024-01', '2024-02', '2024-04', '2024-05'], 'x': [0, 10, 20, 40]})
        result = analyze(frame, {'operation': 'analyze_timeseries', 'time_column': 't', 'metric': 'x', 'view': 'chronological', 'window': 2})
        data = rows(result)
        marker = result['result']['null_marker']
        self.assertEqual(data[1]['change_pct'], marker)  # 前期0
        self.assertEqual(data[2]['gap'], 'true')
        self.assertEqual(data[2]['delta'], marker)  # 3月の欠損を0補完しない
        self.assertEqual(data[2]['moving_average'], marker)
        self.assertEqual(float(data[3]['moving_average']), 30)
        self.assertEqual(float(data[3]['change_pct']), 100)
        duplicates = pd.concat([frame, frame])
        with self.assertRaises(ValueError):
            analyze(duplicates, {'operation': 'analyze_timeseries', 'time_column': 't', 'metric': 'x'})
        result = analyze(duplicates, {'operation': 'analyze_timeseries', 'time_column': 't', 'metric': 'x', 'aggregation': 'sum'})
        self.assertEqual(float(rows(result)[0]['value']), 80)

    def test_invalid_operations_columns_types_and_cancellation(self):
        for args in ({'operation': 'eval'}, {'operation': 'aggregate', 'aggregations': []},
                     {'operation': 'top_k', 'sort_by': ['missing']},
                     {'operation': 'correlation', 'columns': ['group', 'x']},
                     {'operation': 'filter_data', 'offset': -1}):
            with self.assertRaises(ValueError):
                analyze(self.frame, args)
        control = ExecutionControl()
        control.request_stop()
        with self.assertRaises(ExecutionStopped):
            analyze(self.frame, {'operation': 'describe_data'}, control=control)


class LocalSqlTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({'g': ['a', 'a', 'b'], 'x': [1, 3, 9]})

    def test_cte_aggregation_and_window_are_available(self):
        result = analyze(self.frame, {'operation': 'sql', 'sql': 'WITH grouped AS (SELECT g, SUM(x) AS n FROM data GROUP BY g) SELECT g,n,RANK() OVER (ORDER BY n DESC) AS rank FROM grouped ORDER BY rank'})
        self.assertEqual(rows(result), [{'g': 'b', 'n': '9', 'rank': '1'}, {'g': 'a', 'n': '4', 'rank': '2'}])

    def test_writes_external_files_functions_recursive_and_multiple_statements_rejected(self):
        for sql in ('DELETE FROM data', 'SELECT * FROM data; DROP TABLE data',
                    "ATTACH DATABASE '/tmp/escape' AS external", 'PRAGMA query_only=OFF',
                    'SELECT * FROM sqlite_master', 'SELECT * FROM other', 'SELECT * FROM main.data',
                    "SELECT load_extension('/tmp/anything')", 'SELECT randomblob(1000000000)',
                    'WITH RECURSIVE x(n) AS (SELECT 1 UNION ALL SELECT n+1 FROM x) SELECT * FROM x'):
            with self.subTest(sql=sql), self.assertRaises(ValueError):
                analyze(self.frame, {'operation': 'sql', 'sql': sql})
        self.assertEqual(self.frame['x'].tolist(), [1, 3, 9])

    def test_large_local_output_is_explicitly_limited(self):
        frame = pd.DataFrame({'x': range(5000)})
        result = analyze(frame, {'operation': 'sql', 'sql': 'SELECT * FROM data'})
        self.assertEqual(len(rows(result)), 50)
        self.assertTrue(result['result']['row_count_is_lower_bound'])
        self.assertFalse(result['result']['complete'])

    def test_sql_timeout_and_string_memory_limit(self):
        with patch('src.analysis_layer.time.monotonic', side_effect=[0, 0, 100, 100, 100, 100]):
            with self.assertRaises(ValueError):
                analyze(pd.DataFrame({'x': range(100)}), {'operation': 'sql', 'sql': 'SELECT SUM(a.x*b.x*c.x) FROM data a,data b,data c'})
        # 許可関数でも巨大な文字列を作れない。
        sql = "SELECT replace(replace(replace('aaaaaaaaaa','a','aaaaaaaaaa'),'a','aaaaaaaaaa'),'a','aaaaaaaaaa')"
        result = analyze(self.frame, {'operation': 'sql', 'sql': sql})
        self.assertLessEqual(estimate_tokens(result['result']), 8000)
        expression = "'a'"
        for _ in range(7):
            expression = f"replace({expression},'a','aaaaaaaaaa')"
        with self.assertRaises(ValueError):
            analyze(self.frame, {'operation': 'sql', 'sql': 'SELECT ' + expression})


class BudgetTests(unittest.TestCase):
    def test_a_small_b_medium_c_large_and_f_long_cells(self):
        manager = TokenBudgetManager()
        for n, delivery, size_class in ((20, 'csv', None), (500, 'schema', 'medium'), (5000, 'schema', 'large')):
            frame = pd.DataFrame({'g': [str(i) for i in range(n)], 'x': range(n)})
            result = manager.initial_result(frame)
            self.assertEqual(result['delivery'], delivery)
            if size_class:
                self.assertEqual(result['size_class'], size_class)
                self.assertNotIn('csv', result)
            self.assertLessEqual(estimate_tokens(result), 8000)
        result = manager.initial_result(pd.DataFrame({'huge': ['日本語' * 10000]}))
        self.assertEqual(result['delivery'], 'schema')
        self.assertEqual(result['reduction_reason'], 'token_budget')

    def test_analysis_preview_limits_tokens_and_keeps_original_frame(self):
        frame = pd.DataFrame({'id': range(500), 'long': ['文字' * 100] * 500})
        before = frame.copy(deep=True)
        result = analyze(frame, {'operation': 'top_k', 'sort_by': ['id']}, BudgetPolicy(max_data_tokens=2000, target_data_tokens=1000))
        self.assertLessEqual(estimate_tokens(result['result']), 2000)
        self.assertLess(result['result']['sent_row_count'], 50)
        self.assertFalse(result['result']['complete'])
        pd.testing.assert_frame_equal(frame, before)

    def test_settings_are_configurable_and_invalid_limits_fail(self):
        env = {'GCP_PROJECT_ID': 'test', 'BQ_DEFAULT_DATASET': 'data', 'MAX_DIRECT_ROWS': '30',
               'MAX_PREVIEW_ROWS': '10', 'MAX_DATA_TOKENS': '2000', 'TARGET_DATA_TOKENS': '1000'}
        with patch.dict(os.environ, env, clear=True):
            settings = load_settings()
            policy = BudgetPolicy.from_settings(settings)
            self.assertEqual(policy.max_direct_rows, 30)
            self.assertEqual(policy.max_preview_rows, 10)
            self.assertEqual(policy.max_data_tokens, 2000)
        with patch.dict(os.environ, {**env, 'TARGET_DATA_TOKENS': '9999'}, clear=True):
            with self.assertRaises(ConfigError):
                load_settings()

    def test_wide_schema_and_whole_analysis_response_fit_budget(self):
        policy = BudgetPolicy(max_data_tokens=1500, target_data_tokens=1000)
        frame = pd.DataFrame({('長い列名' * 30) + str(i): range(500) for i in range(50)})
        schema = TokenBudgetManager(policy).initial_result(frame)
        self.assertLessEqual(estimate_tokens(schema), policy.max_data_tokens)
        self.assertGreater(schema['columns_omitted'], 0)
        result = analyze(pd.DataFrame({'x': range(500), 'text': ['日本語' * 60] * 500}),
                         {'operation': 'top_k', 'sort_by': ['x']}, policy)
        self.assertLessEqual(estimate_tokens(result), policy.max_data_tokens)

    def test_history_budget_preserves_latest_request_signature_and_archive(self):
        settings = SimpleNamespace(max_history_tokens=2000)
        ctx = ToolContext(tools=Mock(), settings=settings)
        signed = types.Part(function_call=types.FunctionCall(name='analyze_result', args={'result_id': 'saved', 'operation': 'describe_data'}), thought_signature=b'exact')
        archive = [content('user', '古い質問' * 2000), content('model', '古い回答' * 2000), content('user', '最新の質問'),
                   types.Content(role='model', parts=[signed]), types.Content(role='user', parts=[types.Part.from_function_response(name='analyze_result', response={'result_id': 'saved', 'result': {'csv': 'value\n' + 'x' * 1000, 'row_count': 1000}})])]
        before = [c.model_dump() for c in archive]
        request = bound_history(archive, ctx)
        self.assertLessEqual(estimate_tokens([c.model_dump(exclude_none=True) for c in request]), 2000)
        self.assertEqual([c.model_dump() for c in archive], before)
        self.assertIs(request[-2].parts[0], signed)
        self.assertEqual(request[-2].parts[0].thought_signature, b'exact')
        self.assertIs(request[-1], archive[-1])
        self.assertIn('最新の質問', json.dumps([c.model_dump() for c in request], default=str, ensure_ascii=False))
        with self.assertRaises(ValueError):
            bound_history([content('user', '最新の巨大質問' * 10000)], ctx)

    def test_history_budget_never_discards_unread_latest_tool_result(self):
        ctx = ToolContext(tools=Mock(), settings=SimpleNamespace(max_history_tokens=2000))
        response = types.Content(role='user', parts=[types.Part.from_function_response(
            name='run_query', response={'result_id': 'saved', 'result': {'csv': 'x' * 30000}})])
        archive = [content('user', '質問'), types.Content(role='model', parts=[
            types.Part(function_call=types.FunctionCall(name='run_query', args={'sql': 'SELECT 1'}))]), response]
        before = [c.model_dump() for c in archive]
        with self.assertRaises(ValueError):
            bound_history(archive, ctx)
        self.assertEqual([c.model_dump() for c in archive], before)

    def test_history_budget_counts_binary_signatures_as_sent_base64(self):
        ctx = ToolContext(tools=Mock(), settings=SimpleNamespace(max_history_tokens=2000))
        signed = types.Part(function_call=types.FunctionCall(name='list_datasets', args={}),
                            thought_signature=b'\xff' * 1800)
        archive = [content('user', '質問'), types.Content(role='model', parts=[signed]),
                   types.Content(role='user', parts=[types.Part.from_function_response(
                       name='list_datasets', response={'datasets': ['data']})])]
        self.assertGreater(estimate_tokens([c.model_dump(exclude_none=True) for c in archive]), 2000)
        request = bound_history(archive, ctx)
        self.assertEqual(request, archive)
        self.assertIs(request[1].parts[0], signed)
        self.assertEqual(ctx.context_state['last_request']['estimated_tokens_after'],
                         estimate_tokens([c.model_dump(mode='json', exclude_none=True) for c in archive]))

    def test_history_budget_stops_compacting_once_oldest_result_suffices(self):
        ctx = ToolContext(tools=Mock(), settings=SimpleNamespace(max_history_tokens=2000))
        def result(value):
            return types.Content(role='user', parts=[types.Part.from_function_response(
                name='run_query', response={'result_id': value, 'result': {'csv': value * 5000}})])
        first, second = result('a'), result('b')
        archive = [first, second, content('user', '最新の質問')]
        request = bound_history(archive, ctx)
        self.assertTrue(request[0].parts[0].function_response.response['history_data_omitted'])
        self.assertIs(request[1], second)
        self.assertLessEqual(estimate_tokens([c.model_dump(exclude_none=True) for c in request]), 2000)


class AgentAnalysisTests(unittest.TestCase):
    def setUp(self):
        self.settings = SimpleNamespace(gcp_project='test', allowed_datasets=('data',), gemini_model='test', thinking_budget=None, max_tool_iterations=12)
        self.ctx = ToolContext(tools=Mock(), settings=self.settings)
        self.agent = GeminiAgent(self.settings, client=Mock())
        self.frame = pd.DataFrame({'g': ['a'] * 2500 + ['b'] * 2500, 'x': range(5000)})
        self.sql = 'SELECT saved'
        self.run = QueryRun(self.sql, self.sql, 10, 10, self.frame, {})
        self.ctx.tools.run_query.return_value = self.run
        self.rid = memory.result_id_for('run_query', {'sql': self.sql})

    def test_d_progressive_schema_describe_then_group_analysis_and_e_specific_group(self):
        overview_args = {'result_id': self.rid, 'operation': 'describe_data', 'columns': ['x']}
        group_args = {'result_id': self.rid, 'operation': 'aggregate', 'group_by': ['g'],
                      'aggregations': [{'column': 'x', 'function': 'mean', 'alias': 'avg_x'}]}
        filtered_args = {'result_id': self.rid, 'operation': 'filter_data', 'columns': ['g', 'x'],
                         'filters': [{'column': 'g', 'op': 'eq', 'values': ['b']}], 'limit': 3}
        self.agent._generate = Mock(side_effect=[response('run_query', {'sql': self.sql}),
            response('analyze_result', overview_args), response('analyze_result', group_args),
            response('analyze_result', filtered_args), response()])
        contents = [content('user', 'このデータから面白い傾向を探して')]
        with self.assertLogs('src.agent', level='INFO') as logs:
            self.agent.run(contents, self.ctx)
        first_result = contents[2].parts[0].function_response.response['result']
        self.assertEqual(first_result['delivery'], 'schema')
        self.assertNotIn('csv', first_result)
        self.assertEqual(first_result['row_count'], 5000)
        detailed = contents[8].parts[0].function_response.response
        self.assertEqual([row['x'] for row in rows(detailed)], ['2500', '2501', '2502'])
        self.assertEqual(detailed['matched_row_count'], 2500)
        self.ctx.tools.run_query.assert_called_once()
        self.assertEqual(len(self.ctx.turn_runs[0].dataframe), 5000)
        self.assertIn('estimated_tokens_before', '\n'.join(logs.output))
        prompt = build_system_instruction(['analyze_result'])
        for phrase in ('schema → describe_data', 'compare_groups', '外れ値', '全列・全組み合わせを無差別に計算せず'):
            self.assertIn(phrase, prompt)

    def test_derived_results_survive_history_restore_and_can_be_analyzed_again(self):
        self.ctx.query_cache[_sql_key(self.sql)] = self.run
        memory.register_result(self.ctx, 'run_query', {'sql': self.sql}, self.agent._query_payload(self.run))
        args = {'result_id': self.rid, 'operation': 'aggregate', 'group_by': ['g'],
                'aggregations': [{'column': 'x', 'function': 'sum', 'alias': 'total'}]}
        payload = self.agent._call_tool('analyze_result', args, self.ctx)
        registered = memory.register_result(self.ctx, 'analyze_result', args, payload)
        restored = ToolContext(tools=Mock(), settings=self.settings)
        unpack(pack([], [], self.ctx, None), restored)
        result = self.agent._call_tool('analyze_result', {'result_id': registered['result_id'],
            'operation': 'top_k', 'sort_by': ['total'], 'limit': 1}, restored)
        self.assertEqual(rows(result)[0]['g'], 'b')
        self.assertEqual(len(restored.analysis_cache), 2)
        other = ToolContext(tools=Mock(), settings=self.settings)
        with self.assertRaises(BQToolError):
            self.agent._call_tool('analyze_result', args, other)

    def test_explicit_display_only_small_query_has_no_row_data(self):
        run = replace(self.run, dataframe=self.frame.head(20))
        self.assertEqual(self.agent._query_payload(run, result_mode='schema')['result']['delivery'], 'schema')
        self.assertEqual(self.agent._query_payload(run)['result']['delivery'], 'csv')


class BigQueryTimeoutTests(unittest.TestCase):
    def test_timeout_requests_cancel_without_losing_execution_control(self):
        from src.bq_tools import BigQueryTools
        from src.sql_guard import check_sql
        settings = SimpleNamespace(max_bytes_billed=1024, max_result_rows=5000, sample_rows=500,
                                   bq_location=None, bq_query_timeout_seconds=7,
                                   gcp_project='proj', allowed_datasets=('data',), default_row_limit=1000)
        guard = check_sql('SELECT 1 AS n', allowed_datasets=('data',), default_row_limit=1000, project='proj')
        client = Mock()
        job = client.query.return_value
        job.result.side_effect = TimeoutError('deadline')
        control = ExecutionControl()
        tools = BigQueryTools(settings, client=client, execution_control=control)
        with self.assertRaises(BQToolError):
            tools.execute(guard, 1)
        job.result.assert_called_once_with(max_results=5001, timeout=7)
        job.cancel.assert_called_once_with(timeout=10)
        self.assertIsNone(control._query_job)


if __name__ == '__main__':
    unittest.main()
