"""分析の実行詳細・履歴復元・effortの実イベントを検証する。"""
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import pandas as pd
from google.genai import types
from streamlit.testing.v1 import AppTest

from src import context_memory as memory
from src.agent import GeminiAgent, ToolContext, _sql_key
from src.bq_tools import QueryRun
from src.chat_history import pack, unpack
from src.prompts import COMMON_INSTRUCTION, build_system_instruction


class AnalysisDetailTests(unittest.TestCase):
    def test_analysis_parameters_and_full_result_persist_with_chat(self):
        settings = SimpleNamespace(gcp_project='test', allowed_datasets=('data',), gemini_model='test')
        ctx = ToolContext(tools=Mock(), settings=settings)
        agent = GeminiAgent(settings, client=Mock())
        frame = pd.DataFrame({'店舗': ['a', 'a', 'b'], '件数': [1, 2, 5]})
        run = QueryRun('SELECT saved', 'SELECT saved', 1, 1, frame, {})
        ctx.query_cache[_sql_key(run.original_sql)] = run
        original = memory.register_result(ctx, 'run_query', {'sql': run.original_sql}, agent._query_payload(run))
        args = {'result_id': original['result_id'], 'operation': 'aggregate', 'purpose': '店舗ごとの件数を比較',
                'group_by': ['店舗'], 'aggregations': [{'column': '件数', 'function': 'sum', 'alias': '合計'}],
                'context_note': {'summary': '確定した事実の引き継ぎ', 'references': []}}
        agent._dispatch(types.FunctionCall(name='analyze_result', args=args), ctx)
        self.assertEqual(len(ctx.turn_analyses), 1)
        record = ctx.turn_analyses[0]
        self.assertEqual(record['purpose'], args['purpose'])
        self.assertNotIn('context_note', record['parameters'])
        self.assertEqual(record['dataframe']['合計'].tolist(), [3, 5])
        self.assertEqual(record['source_row_count'], 3)
        message = {'role': 'assistant', 'text': '結果', 'analyses': list(ctx.turn_analyses)}
        raw = pack([message], [], ctx, None)
        ctx.start_turn()
        self.assertEqual(ctx.turn_analyses, [])
        self.assertEqual(len(message['analyses']), 1)
        restored = ToolContext(tools=Mock(), settings=settings)
        messages, _, _ = unpack(raw, restored)
        pd.testing.assert_frame_equal(messages[0]['analyses'][0]['dataframe'], record['dataframe'])
        self.assertEqual(restored.turn_analyses[0]['purpose'], args['purpose'])
        ctx.tools.run_query.assert_not_called()

    def test_analysis_error_is_recorded_without_claiming_success(self):
        ctx = ToolContext(tools=Mock(), settings=SimpleNamespace())
        agent = GeminiAgent(ctx.settings, client=Mock())
        part = agent._dispatch(types.FunctionCall(name='analyze_result', args={
            'operation': 'top_k', 'result_id': 'missing', 'sort_by': ['件数']}), ctx)
        self.assertIn('error', part.function_response.response)
        self.assertEqual(len(ctx.turn_analyses), 1)
        self.assertIn('error', ctx.turn_analyses[0])
        self.assertNotIn('dataframe', ctx.turn_analyses[0])

    def test_analysis_only_message_renders_details_table_conditions_and_local_sql(self):
        app = AppTest.from_string('''
import pandas as pd
from app import render_results
render_results({'question':'店舗の比較', 'analyses': [{
    'purpose':'店舗ごとの合計を比較', 'operation':'sql',
    'parameters':{'sql':'SELECT store, SUM(n) FROM data GROUP BY store', 'limit':2},
    'dataframe':pd.DataFrame({'店舗':['a','b'],'合計':[3,5]}),
    'source_row_count':100, 'matched_row_count':80, 'sent_row_count':2,
    'analysis_seconds':0.01, 'source_truncated':True,
}]}, 0, set())
''').run(timeout=20)
        self.assertFalse(app.exception)
        self.assertEqual([e.label for e in app.expander], ['データ・実行詳細', '分析：店舗ごとの合計を比較'])
        self.assertEqual(app.dataframe[0].value['合計'].tolist(), [3, 5])
        self.assertIn('GROUP BY', app.code[0].value)
        captions = '\n'.join(c.value for c in app.caption)
        self.assertIn('条件一致: 80', captions)
        self.assertIn('AIへの返却: 2', captions)
        self.assertIn('切り捨て', captions)

    def test_shared_policy_is_short_and_advice_only_used_for_final_answer(self):
        selector = build_system_instruction()
        analysis = build_system_instruction(['analyze_result'])
        final = build_system_instruction(['finish_answer'])
        self.assertTrue(analysis.startswith(COMMON_INSTRUCTION))
        for text in ('読み取り専用', '課金上限を回避しない', 'current_question_state.open_questions', '事実'):
            self.assertIn(text, selector)
        for text in ('施策検討・壁打ちの説明', '## 回答スタイル', '## SQL を書くときの規則', '## 図表と回答用クエリ'):
            self.assertNotIn(text, analysis)
        self.assertIn('## 施策検討・壁打ちの説明', final)
        self.assertIn('purpose', analysis)
        self.assertEqual(analysis.count('## 定義'), 1)



if __name__ == '__main__':
    unittest.main()
