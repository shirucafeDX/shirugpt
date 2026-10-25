"""保存済み結果の選択参照・表示・原本保全を検証する。"""
import copy
import csv
import io
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
import pandas as pd
from src import context_memory as memory
from src.agent import GeminiAgent, ToolContext, _sql_key
from src.bq_tools import BQToolError, QueryRun
from src.result_access import filter_saved_result
from src.summarize import summarize_dataframe


def csv_rows(result):
    return list(csv.DictReader(io.StringIO(result['csv'])))


class SavedAccessTests(unittest.TestCase):
    def setUp(self):
        self.frame = pd.DataFrame({'日付': pd.date_range('2024-01-01', periods=120),
                                   '区分': ['a', 'b', None] * 40, '件数': range(120)})
        self.settings = SimpleNamespace(gcp_project='test', allowed_datasets=('data',), sample_rows=500)
        self.ctx = ToolContext(tools=Mock(), settings=self.settings)
        self.agent = GeminiAgent(self.settings, client=Mock())
        self.run = QueryRun('SELECT guarded', 'SELECT original', 10, 10, self.frame,
                            summarize_dataframe(self.frame, sample_rows=500), truncated=True)
        self.ctx.query_cache[_sql_key(self.run.original_sql)] = self.run
        self.rid = 'saved-result'
        memory.state_for(self.ctx)['results'][self.rid] = {
            'query_key': _sql_key(self.run.original_sql), 'payload': {'result': self.run.summary}}

    def test_typed_filters_are_and_before_projection_and_pagination(self):
        filters = [{'column': '日付', 'op': 'gte', 'values': ['2024-02-01']},
                   {'column': '件数', 'op': 'lte', 'values': ['40']},
                   {'column': '区分', 'op': 'in', 'values': ['a', 'b']}]
        result = memory.recall(self.ctx, {'result_id': self.rid, 'filters': filters,
                                        'columns': ['件数'], 'offset': 1, 'limit': 2})
        self.assertEqual(result['matched_row_count'], 7)
        self.assertEqual(csv_rows(result['result']), [{'件数': '33'}, {'件数': '34'}])
        self.assertEqual(result['total_row_count'], 120)
        self.assertTrue(result['more_rows'])
        self.assertTrue(result['source_truncated'])
        self.assertEqual(result['statistics_scope'], 'returned_rows_only')
        self.ctx.tools.run_query.assert_not_called()

    def test_nulls_precision_invalid_values_and_original_preserved(self):
        before = self.frame.copy(deep=True)
        self.assertEqual(len(filter_saved_result(self.frame, [{'column': '区分', 'op': 'is_null'}])), 40)
        exact = pd.DataFrame({'n': [2**53, 2**53+1]})
        self.assertEqual(len(filter_saved_result(exact, [{'column': 'n', 'op': 'eq', 'values': [str(2**53+1)]}])), 1)
        for condition in [{'column': '不明', 'op': 'eq', 'values': ['a']},
                          {'column': '件数', 'op': 'eq', 'values': ['1.5']},
                          {'column': '件数', 'op': 'gte', 'values': ['nan']},
                          {'column': '日付', 'op': 'eq', 'values': ['2024-01-01T00:00:00+09:00']},
                          {'column': '区分', 'op': 'eval', 'values': ['a']}]:
            with self.assertRaises(ValueError):
                filter_saved_result(self.frame, [condition])
        pd.testing.assert_frame_equal(self.frame, before)
        self.ctx.query_cache.clear()
        with self.assertRaises(ValueError):
            memory.recall(self.ctx, {'result_id': self.rid, 'filters': [{'column': '件数', 'op': 'gte', 'values': ['1']}]})

    def test_present_reuses_all_rows_without_query_payload_or_history_mutation(self):
        chart = {'kind': 'line', 'x': '日付', 'y': ['件数']}
        result = self.agent._call_tool('recall_context', {'action': 'present', 'result_id': self.rid, 'chart': chart}, self.ctx)
        self.assertTrue(result['presented'])
        self.assertNotIn('sample_rows', result)
        self.assertEqual(len(self.ctx.turn_runs[0].dataframe), 120)
        self.assertEqual(self.ctx.turn_runs[0].chart, chart)
        self.assertIsNone(self.run.chart)
        self.assertEqual(self.ctx.session_billed_bytes, 0)
        self.ctx.tools.run_query.assert_not_called()
        before = list(self.ctx.turn_runs)
        with self.assertRaises(BQToolError):
            self.agent._call_tool('recall_context', {'action': 'present', 'result_id': self.rid,
                'chart': {'kind': 'line', 'x': '不明', 'y': ['件数']}}, self.ctx)
        self.assertEqual(self.ctx.turn_runs, before)

    def test_recalled_result_id_remains_linked_to_full_saved_frame(self):
        args = {'result_id': self.rid, 'columns': ['件数'], 'limit': 1}
        payload = memory.recall(self.ctx, args)
        registered = memory.register_result(self.ctx, 'recall_context', args, payload)
        result = memory.recall(self.ctx, {'result_id': registered['result_id'], 'offset': 110, 'limit': 1})
        self.assertEqual(csv_rows(result['result'])[0]['件数'], '110')
        self.assertEqual(result['total_row_count'], 120)


    def test_cached_results_are_summarized_without_changing_saved_full_results(self):
        original = copy.deepcopy(self.run.summary)
        result = self.agent._run_query({'sql': self.run.original_sql}, self.ctx)
        self.assertEqual(result['result']['delivery'], 'schema')
        self.assertNotIn('csv', result['result'])
        self.assertEqual(self.run.summary, original)
        self.assertEqual(len(self.run.dataframe), 120)
        self.ctx.tools.run_query.assert_not_called()


if __name__ == '__main__':
    unittest.main()
