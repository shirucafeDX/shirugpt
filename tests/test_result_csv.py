"""CSV転送の情報保全・上限判定・詳細ページの継続を外部APIなしで検証。"""
import csv
import datetime as dt
from decimal import Decimal
import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pandas as pd

from src import context_memory as memory
from src.agent import ToolContext, _sql_key
from src.bq_tools import QueryRun
from src.result_csv import csv_result
from src.token_budget import TokenBudgetManager, estimate_tokens


def rows(result):
    return list(csv.DictReader(io.StringIO(result['csv'], newline='')))


class CsvDeliveryTests(unittest.TestCase):
    def test_quotes_commas_newlines_long_cells_and_all_columns_round_trip(self):
        values = ['カンマ,引用符"改行\n次の行', '長文' * 800, '末尾\r\n', '']
        frame = pd.DataFrame({'名前,"\n': values, **{f'列{i}': [i] * 4 for i in range(35)}})
        original = frame.copy(deep=True)
        result = csv_result(frame)
        self.assertEqual(result['delivery'], 'csv')
        parsed = rows(result)
        self.assertEqual([r['名前,"\n'] for r in parsed], values)
        self.assertEqual(len(parsed[0]), 36)
        self.assertEqual(len(result['columns']), 36)
        self.assertNotIn('sample_rows', result)
        self.assertNotIn('overview', result)
        pd.testing.assert_frame_equal(frame, original)

    def test_nulls_empty_string_and_literal_null_marker_remain_distinct(self):
        frame = pd.DataFrame({'値': [None, '', r'\N', r'\N_', pd.NA, float('nan'), pd.NaT]})
        result = csv_result(frame)
        parsed = rows(result)
        self.assertEqual(result['null_marker'], r'\N__')
        self.assertEqual([r['値'] for r in parsed], [r'\N__', '', r'\N', r'\N_', r'\N__', r'\N__', r'\N__'])
        self.assertEqual(result['columns'][0]['null_count'], 4)

    def test_precision_dates_nested_values_bytes_and_special_numbers(self):
        timestamp = dt.datetime(2026, 10, 25, 13, 30, 45, 123456, tzinfo=dt.timezone(dt.timedelta(hours=9)))
        frame = pd.DataFrame({'整数': [2**63 - 1], '小数': [0.1],
                              '金額': [Decimal('12345678901234567890.123456789')],
                              '日時': [timestamp], '日付': [dt.date(2026, 10, 25)],
                              '配列': [np.array([1, 2, 3])],
                              '構造体': [{'値': Decimal('0.123456789123456789'), '空': None}],
                              '真偽': [True], 'バイト': [b'\x00\xff'], '無限大': [float('inf')]})
        result = csv_result(frame)
        parsed = rows(result)[0]
        self.assertEqual(parsed['整数'], str(2**63 - 1))
        self.assertEqual(parsed['金額'], '12345678901234567890.123456789')
        self.assertEqual(parsed['日時'], timestamp.isoformat())
        self.assertEqual(parsed['日付'], '2026-10-25')
        self.assertEqual(json.loads(parsed['配列']), [1, 2, 3])
        self.assertEqual(json.loads(parsed['構造体'])['値'], '0.123456789123456789')
        self.assertEqual(parsed['真偽'], 'true')
        self.assertEqual(parsed['バイト'], 'base64:AP8=')
        self.assertEqual(parsed['無限大'], 'Infinity')
        json.dumps(result, allow_nan=False)


    def test_empty_and_source_truncated_have_explicit_independent_coverage(self):
        result = csv_result(pd.DataFrame({'値': pd.Series([], dtype='Int64')}))
        self.assertEqual(result['delivery'], 'csv')
        self.assertEqual(result['sent_row_count'], 0)
        self.assertEqual(rows(result), [])
        self.assertEqual(result['columns'][0]['dtype'], 'Int64')
        result = csv_result(pd.DataFrame({'値': [1, 2]}), truncated=True)
        self.assertTrue(result['complete'])
        self.assertTrue(result['source_truncated'])
        self.assertTrue(result['truncated'])

    def test_char_budget_counts_json_escaping_and_does_not_split_cells(self):
        value = '\\"\n' * 350
        frame = pd.DataFrame({'値': [value] * 100})
        result = TokenBudgetManager().table(frame)
        self.assertLessEqual(estimate_tokens(result), 8000)
        self.assertGreater(result['sent_row_count'], 0)
        self.assertLess(result['sent_row_count'], 100)
        self.assertFalse(result['complete'])
        self.assertEqual(result['rows_omitted'], 100 - result['sent_row_count'])
        self.assertTrue(all(r['値'] == value for r in rows(result)))
        blocked = TokenBudgetManager().table(pd.DataFrame({'巨大': ['x' * 30000]}))
        self.assertEqual(blocked['sent_row_count'], 0)
        self.assertIn('列', blocked['note'])
        self.assertLessEqual(estimate_tokens(blocked), 8000)


    def test_csv_page_continuation_has_no_lost_or_repeated_rows_and_no_query(self):
        frame = pd.DataFrame({'番号': range(100), '説明': ['長文' * 400] * 100})
        settings = SimpleNamespace()
        ctx = ToolContext(tools=Mock(), settings=settings)
        run = QueryRun('SELECT guarded', 'SELECT original', 10, 10, frame, {})
        key = _sql_key(run.original_sql)
        ctx.query_cache[key] = run
        memory.state_for(ctx)['results']['saved'] = {'query_key': key}
        offset, seen = 0, []
        while True:
            result = memory.recall(ctx, {'result_id': 'saved', 'offset': offset, 'limit': 100})
            seen.extend(int(r['番号']) for r in rows(result['result']))
            self.assertEqual(result['returned_row_count'], result['result']['sent_row_count'])
            if not result['more_rows']:
                break
            self.assertGreater(result['next_offset'], offset)
            offset = result['next_offset']
        self.assertEqual(seen, list(range(100)))
        ctx.tools.run_query.assert_not_called()

    def test_saved_csv_without_dataframe_is_paginated_and_marked_limited(self):
        frame = pd.DataFrame({'番号': range(10), '区分': [None, '', r'\N', 'abc', 'def'] * 2})
        ctx = ToolContext(tools=Mock(), settings=SimpleNamespace())
        memory.state_for(ctx)['results']['saved'] = {'payload': {'result': csv_result(frame), 'executed_sql': 'SELECT original', 'notes': ['LIMITによる切り捨ての可能性']}}
        result = memory.recall(ctx, {'result_id': 'saved', 'columns': ['区分'], 'offset': 1, 'limit': 3})
        self.assertEqual([r['区分'] for r in rows(result['result'])], ['', r'\N', 'abc'])
        self.assertTrue(result['only_saved_samples_available'])
        self.assertEqual(result['available_sample_count'], 10)
        self.assertEqual(result['notes'], ['LIMITによる切り捨ての可能性'])
        self.assertEqual(result['next_offset'], 4)
        self.assertNotIn('sql', result)
        self.assertTrue(result['more_rows'])


if __name__ == '__main__':
    unittest.main()
