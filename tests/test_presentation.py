"""表示方針と、実際の Streamlit 描画・キャッシュの回帰確認。GCP 接続不要。"""
import unittest
import os
from unittest.mock import Mock
from unittest.mock import patch

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.agent import GeminiAgent, ToolContext
from src.bq_tools import QueryRun
from src.config import load_settings
from src.presentation import choose_presentation


def run(df, role='answer', sql='SELECT 1'):
    return QueryRun(sql, sql, 0, 0, df, {}, result_role=role)


def categories(n=5):
    return pd.DataFrame({'店舗': [f'店舗{i}' for i in range(n)], '来店数': range(n)})


class PresentationTests(unittest.TestCase):
    def test_scalar_and_two_values_stay_in_prose(self):
        for df in [pd.DataFrame({'件数': [123]}), categories(2)]:
            self.assertEqual(choose_presentation('来店数を比較して', [run(df)]).mode, 'none')

    def test_trends_comparisons_and_distributions_get_one_chart(self):
        for question in ['店舗別の来店数', '来店数の分布', '来店数を比較して']:
            plan = choose_presentation(question, [run(categories()), run(categories(6))])
            self.assertEqual((plan.mode, plan.index), ('chart', 1))

    def test_explanation_does_not_add_automatic_chart(self):
        self.assertEqual(choose_presentation('なぜ来店数が減った？', [run(categories())]).mode, 'none')

    def test_explicit_format_overrides_automatic_rules(self):
        for question, mode in [('一覧にして', 'table'), ('表で比較して', 'table'),
                               ('グラフではなく表で', 'table'), ('グラフで説明して', 'chart'),
                               ('図表なしで比較して', 'none'), ('グラフは不要', 'none')]:
            self.assertEqual(choose_presentation(question, [run(categories())]).mode, mode, question)

    def test_explicit_chart_of_scalar_falls_back_to_table(self):
        plan = choose_presentation('グラフで', [run(pd.DataFrame({'件数': [5]}))])
        self.assertEqual(plan.mode, 'table')
        self.assertTrue(plan.note)

    def test_supporting_queries_never_displace_answer(self):
        runs = [run(categories()), run(categories(8), 'supporting')]
        self.assertEqual(choose_presentation('店舗別の比較', runs).index, 0)
        self.assertIsNone(choose_presentation('表で', [runs[1]]).index)

    def test_empty_final_result_does_not_show_old_chart(self):
        plan = choose_presentation('グラフで', [run(categories()), run(categories(0))])
        self.assertEqual((plan.index, plan.mode), (1, 'none'))

    def test_repeat_suppressed_but_explicit_request_and_changed_data_shown(self):
        df = categories()
        first = choose_presentation('店舗別の来店数', [run(df)])
        seen = {first.signature}
        self.assertEqual(choose_presentation('店舗別の来店数', [run(df)], seen).mode, 'none')
        self.assertEqual(choose_presentation('グラフで', [run(df)], seen).mode, 'chart')
        changed = df.copy()
        changed.loc[0, '来店数'] = 100
        self.assertEqual(choose_presentation('店舗別の来店数', [run(changed)], seen).mode, 'chart')

    def test_top_ten_keeps_original_data_intact(self):
        df = categories(25)
        plan = choose_presentation('店舗別の来店数', [run(df)])
        self.assertEqual(len(plan.dataframe), 10)
        self.assertEqual(plan.dataframe['来店数'].tolist(), list(range(24, 14, -1)))
        self.assertEqual(len(df), 25)
        self.assertIn('上位10件', plan.note)

    def test_single_column_distribution_preserves_observations(self):
        df = pd.DataFrame({'思考トークン': [0, 10, 20, 300, 4000, None]})
        plan = choose_presentation('思考トークンの分布', [run(df)])
        self.assertEqual(plan.mode, 'chart')
        self.assertEqual(int(plan.dataframe['件数'].sum()), 5)

    def test_distribution_of_count_column_does_not_overwrite_bins(self):
        df = pd.DataFrame({'件数': [1, 2, 3, float('inf')]})
        plan = choose_presentation('件数の分布', [run(df)])
        self.assertEqual(plan.mode, 'chart')
        self.assertEqual(list(plan.dataframe.columns), ['件数の範囲', '件数'])
        self.assertEqual(int(plan.dataframe['件数'].sum()), 3)

    def test_requested_chart_kind_is_respected(self):
        df = pd.DataFrame({'日付': pd.date_range('2026-01-01', periods=5), '件数': range(5)})
        self.assertEqual(choose_presentation('棒グラフで', [run(df)]).chart.kind, 'bar')

    def test_cached_chart_followup_does_not_query_or_rebill_or_mutate_history(self):
        with patch.dict(os.environ, {
            'GCP_PROJECT_ID': 'test-project', 'BQ_DEFAULT_DATASET': 'test_data',
            'BQ_ALLOWED_DATASETS': 'test_data', 'BQ_LOG_DATASET': '',
            'GOOGLE_APPLICATION_CREDENTIALS': '',
        }):
            settings = load_settings()
        tools = Mock()
        original = run(categories(), 'supporting')
        original.billed_bytes = 100
        tools.run_query.return_value = original
        ctx = ToolContext(tools=tools, settings=settings)
        agent = GeminiAgent(settings, client=Mock())
        agent._run_query({'sql': original.sql, 'result_role': 'supporting'}, ctx)
        previous = list(ctx.turn_runs)
        ctx.start_turn()
        payload = agent._run_query({'sql': original.sql, 'result_role': 'answer'}, ctx)
        self.assertTrue(payload['cached'])
        self.assertEqual(tools.run_query.call_count, 1)
        self.assertEqual(ctx.session_billed_bytes, 100)
        self.assertEqual(previous[0].result_role, 'supporting')
        self.assertEqual(choose_presentation('グラフで', ctx.turn_runs).mode, 'chart')
        agent._run_query({'sql': original.sql}, ctx)
        self.assertEqual(len(ctx.turn_runs), 1)


class StreamlitRenderingTests(unittest.TestCase):
    def render(self, question, scalar=False, error=False):
        source = f'''
import pandas as pd
from app import render_message
from src.bq_tools import QueryRun
frame = pd.DataFrame({{'件数': [123]}}) if {scalar!r} else pd.DataFrame({{'店舗': ['A','B','C'], '件数': [1,2,3]}})
r = QueryRun('SELECT 1','SELECT 1',0,0,frame,{{}})
render_message({{'role':'assistant','text':'回答です','question':{question!r},'runs':[r], 'is_error':{error!r}}}, 0)
'''
        app = AppTest.from_string(source).run(timeout=20)
        self.assertEqual(len(app.exception), 0, [str(e) for e in app.exception])
        return app

    def test_scalar_has_only_collapsed_data_and_details(self):
        app = self.render('件数は？', scalar=True)
        self.assertEqual([e.label for e in app.expander], ['データを見る', '実行詳細'])
        self.assertTrue(all(not e.proto.expanded for e in app.expander))
        self.assertEqual(len(app.selectbox), 0)

    def test_chart_and_axis_interaction(self):
        app = self.render('店舗別の比較')
        self.assertEqual(len(app.selectbox), 2)
        app.selectbox[0].select('line').run()
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(app.selectbox[0].value, 'line')

    def test_explicit_table_is_visible_without_data_expander(self):
        app = self.render('一覧にして')
        self.assertEqual(len(app.dataframe), 1)
        self.assertEqual([e.label for e in app.expander], ['実行詳細'])
        self.assertEqual(len(app.selectbox), 0)

    def test_error_never_promotes_a_chart(self):
        app = self.render('グラフで', error=True)
        self.assertEqual(len(app.selectbox), 0)
        self.assertEqual(len(app.warning), 1)


if __name__ == '__main__':
    unittest.main()
