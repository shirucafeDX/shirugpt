"""保存用要約で数値精度を維持し、構造体・配列の列にも対応する。"""
import unittest
import pandas as pd
from src.summarize import summarize_dataframe


class SummaryTests(unittest.TestCase):
    def test_large_integer_statistics_and_mixed_type_samples_remain_exact(self):
        value = 2**53 + 1
        frame = pd.DataFrame({'integer': [value, value], 'float': [0.5, 1.5]})
        result = summarize_dataframe(frame, sample_rows=2)
        self.assertEqual(result['columns'][0]['stats']['min'], value)
        self.assertEqual(result['columns'][0]['stats']['max'], value)
        self.assertEqual(result['columns'][0]['stats']['sum'], value * 2)
        self.assertEqual(result['sample_rows'][0]['integer'], value)

    def test_nested_columns_do_not_break_summary(self):
        frame = pd.DataFrame({'array': [[1, 2]], 'struct': [{'x': 1}]})
        result = summarize_dataframe(frame, sample_rows=1)
        self.assertTrue(result['columns'][0]['distinct_count_unavailable'])
        self.assertTrue(result['columns'][1]['distinct_count_unavailable'])
        self.assertEqual(result['sample_rows'][0]['array'], [1, 2])
