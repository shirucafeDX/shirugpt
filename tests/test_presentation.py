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
        self.assertIn('上位10カテゴリ', plan.note)

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

    def test_ai_can_choose_time_axis_and_optional_breakdown(self):
        df = pd.DataFrame({'時間': [9, 9, 10, 10], '店舗': ['A', 'B', 'A', 'B'], '来店数': [1, 2, 3, 4]})
        result = run(df)
        result.chart = {'kind': 'line', 'x': '時間', 'y': ['来店数'], 'color': '店舗'}
        plan = choose_presentation('時間帯の特徴は？', [result])
        self.assertEqual(plan.mode, 'chart')
        self.assertEqual((plan.chart.kind, plan.chart.x, plan.chart.y, plan.chart.color),
                         ('line', '時間', ['来店数'], '店舗'))
        self.assertEqual(len(plan.dataframe), 4)
        result.chart = {'kind': 'none'}
        self.assertEqual(choose_presentation('時間帯を比較', [result]).mode, 'none')
        result.chart = {'kind': 'line', 'x': '時間', 'y': ['来店数']}
        self.assertIsNone(choose_presentation('時間帯の特徴は？', [result]).chart.color)
        self.assertEqual(choose_presentation('表で', [result]).mode, 'table')

    def test_new_ai_view_is_not_suppressed_as_a_duplicate_of_the_same_data(self):
        result = run(categories())
        result.chart = {'kind': 'bar', 'x': '店舗', 'y': ['来店数']}
        first = choose_presentation('店舗比較', [result])
        seen = {first.signature}
        self.assertEqual(choose_presentation('店舗比較', [result], seen).mode, 'none')
        result.chart = {'kind': 'line', 'x': '店舗', 'y': ['来店数']}
        self.assertEqual(choose_presentation('別の表示', [result], seen).mode, 'chart')

    def test_invalid_ai_columns_and_duplicate_groups_fall_back_to_table(self):
        result = run(categories())
        for chart in ({'kind': 'line', 'x': '存在しない列', 'y': ['来店数']},
                      {'kind': 'bar', 'x': '店舗', 'y': ['来店数'], 'color': '店舗'},
                      {'kind': 'bar', 'x': '来店数', 'y': ['店舗']},
                      {'kind': []}):
            result.chart = chart
            self.assertEqual(choose_presentation('比較', [result]).mode, 'table')
        result = run(pd.DataFrame({'月': [1, 1], '店舗': ['A', 'A'], '件数': [2, 3]}))
        result.chart = {'kind': 'bar', 'x': '月', 'y': ['件数'], 'color': '店舗'}
        self.assertEqual(choose_presentation('比較', [result]).mode, 'table')

    def test_breakdown_top_ten_retains_every_series_for_each_selected_category(self):
        df = pd.DataFrame({'店舗': [f'店{i}' for i in range(12) for _ in range(2)],
                           '学年': ['1年', '2年'] * 12, '件数': [v for i in range(12) for v in [i, i * 2]]})
        result = run(df)
        result.chart = {'kind': 'bar', 'x': '店舗', 'y': ['件数'], 'color': '学年'}
        plan = choose_presentation('店舗別の内訳', [result])
        self.assertEqual(len(plan.dataframe), 20)
        self.assertEqual(plan.dataframe['店舗'].nunique(), 10)
        self.assertEqual(plan.dataframe.groupby('店舗')['学年'].nunique().tolist(), [2] * 10)
        self.assertEqual(len(result.dataframe), 24)

    def test_cached_chart_spec_can_change_without_mutating_previous_answer_or_requerying(self):
        from src.token_budget import BudgetPolicy
        settings = Mock(sample_rows=500, **vars(BudgetPolicy()))
        ctx = ToolContext(tools=Mock(), settings=settings)
        result = run(categories())
        ctx.tools.run_query.return_value = result
        agent = GeminiAgent(settings, client=Mock())
        original_chart = {'kind': 'bar', 'x': '店舗', 'y': ['来店数']}
        agent._run_query({'sql': result.sql, 'chart': original_chart}, ctx)
        previous = list(ctx.turn_runs)
        ctx.start_turn()
        agent._run_query({'sql': result.sql, 'chart': {'kind': 'none'}}, ctx)
        self.assertEqual(previous[0].chart, original_chart)
        self.assertEqual(ctx.turn_runs[0].chart, {'kind': 'none'})
        ctx.tools.run_query.assert_called_once()
        from src.chat_history import pack, unpack
        settings.gcp_project = 'test'
        settings.allowed_datasets = ('data',)
        settings.gemini_model = 'gemini-3.8-flash'
        restored, _, _ = unpack(pack([{'role': 'assistant', 'runs': previous}], [], ctx, None),
                                ToolContext(tools=Mock(), settings=settings))
        self.assertEqual(restored[0]['runs'][0].chart, original_chart)


class StreamlitRenderingTests(unittest.TestCase):
    def test_turn_updates_status_with_query_purpose(self):
        source = '''
from types import SimpleNamespace
from unittest.mock import patch
import streamlit as st
from app import process_turn
from src.agent import AnswerResult

class Agent:
    def run(self, contents, ctx, on_progress):
        on_progress('データを取得・集計しています：店舗別の来店数を集計')
        return AnswerResult('回答です')

st.session_state.messages = [{'role': 'user', 'text': '店舗別の来店数は？'}]
st.session_state.contents = []
ctx = SimpleNamespace(turn_runs=[], turn_errors=[], usage_logger=None)
with patch('app.save_conversation'), patch('app.st.rerun'):
    process_turn(Agent(), ctx)
'''
        app = AppTest.from_string(source).run(timeout=20)
        self.assertEqual(len(app.exception), 0, [str(e) for e in app.exception])
        self.assertEqual(app.status[0].label, 'データを取得・集計しています：店舗別の来店数を集計')

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
        self.assertEqual([e.label for e in app.expander], ['データ・実行詳細', 'クエリ 1'])
        self.assertTrue(all(not e.proto.expanded for e in app.expander))
        self.assertEqual(len(app.expander[0].dataframe), 1)
        self.assertEqual(app.expander[0].code[0].value, 'SELECT 1')
        self.assertEqual(len(app.selectbox), 0)

    def test_query_descriptions_are_separate_collapsed_details(self):
        app = AppTest.from_string("""
import pandas as pd
from src.bq_tools import QueryRun
from app import render_results
frame = pd.DataFrame({'件数': [5]})
runs = [QueryRun('SELECT 1','SELECT 1',0,0,frame,{},purpose='件数を確認'),
        QueryRun('SELECT 2','SELECT 2',0,0,frame,{},purpose='条件を検証')]
render_results({'question':'件数は？', 'runs':runs}, 0, set())
""").run(timeout=20)
        self.assertFalse(app.exception)
        self.assertEqual([e.label for e in app.expander], ['データ・実行詳細', '件数を確認', '条件を検証'])
        self.assertTrue(all(not e.proto.expanded for e in app.expander))
        self.assertEqual(app.expander[1].code[0].value, 'SELECT 1')
        self.assertEqual(app.expander[2].code[0].value, 'SELECT 2')

    def test_chart_has_no_manual_type_or_axis_controls(self):
        app = self.render('店舗別の比較')
        self.assertEqual(len(app.selectbox), 0)
        self.assertEqual(len(app.multiselect), 0)
        self.assertEqual(len(app.get('vega_lite_chart')), 1)

    def test_explicit_table_is_visible_without_data_expander(self):
        app = self.render('一覧にして')
        self.assertEqual(len(app.dataframe), 1)
        self.assertEqual([e.label for e in app.expander], ['データ・実行詳細', 'クエリ 1'])
        self.assertEqual(len(app.expander[0].dataframe), 0)
        self.assertEqual(app.expander[0].code[0].value, 'SELECT 1')
        self.assertEqual(len(app.selectbox), 0)

    def test_error_never_promotes_a_chart(self):
        app = self.render('グラフで', error=True)
        self.assertEqual(len(app.selectbox), 0)
        self.assertEqual(len(app.warning), 1)


if __name__ == '__main__':
    unittest.main()


class BreakdownRenderingTests(unittest.TestCase):
    def test_line_and_stacked_bar_include_color_encoding(self):
        import json
        for kind in ('line', 'bar'):
            app = AppTest.from_string("""
import pandas as pd
from app import render_chart
from src.charts import ChartSpec
frame = pd.DataFrame({'月': ['2026-01', '2026-01', '2026-02', '2026-02'],
    '店舗': ['A', 'B', 'A', 'B'], '件数': [1, 2, 3, 4]})
render_chart(frame, ChartSpec(KIND, '月', ['件数'], color='店舗'), 'test')
""".replace('KIND', repr(kind))).run(timeout=20)
            self.assertFalse(app.exception)
            self.assertEqual(len(app.warning), 0)
            charts = app.get('vega_lite_chart')
            self.assertEqual(len(charts), 1)
            spec = json.loads(charts[0].proto.spec)
            encoding = spec.get('encoding') or spec['layer'][0]['encoding']
            self.assertEqual(encoding['color']['field'], '店舗')
            self.assertEqual(encoding['x']['field'], '月')
            if kind == 'bar':
                self.assertTrue(encoding['y']['stack'])


class AdditionalChartTests(unittest.TestCase):
    def test_pie_uses_all_categories_without_top_ten_cutoff(self):
        result = run(categories(12))
        result.chart = {'kind': 'pie', 'x': '店舗', 'y': ['来店数']}
        plan = choose_presentation('店舗の構成比を円グラフで', [result])
        self.assertEqual(plan.mode, 'chart')
        self.assertEqual(plan.chart.kind, 'pie')
        self.assertEqual(len(plan.dataframe), 12)
        result.truncated = True
        self.assertEqual(choose_presentation('構成比', [result]).mode, 'table')

    def test_pie_rejects_negative_zero_missing_or_duplicate_category_values(self):
        for frame in (pd.DataFrame({'店舗': ['A', 'B'], '件数': [1, -1]}),
                      pd.DataFrame({'店舗': ['A', 'B'], '件数': [0, 0]}),
                      pd.DataFrame({'店舗': ['A', 'B'], '件数': [1, None]}),
                      pd.DataFrame({'店舗': ['A', 'A'], '件数': [1, 2]}),
                      pd.DataFrame({'店舗': ['A', 'B'], '件数': [1, float('inf')]})):
            result = run(frame)
            result.chart = {'kind': 'pie', 'x': '店舗', 'y': ['件数']}
            self.assertEqual(choose_presentation('円グラフで', [result]).mode, 'table')

    def test_heatmap_keeps_two_axes_and_does_not_fill_missing_cells(self):
        df = pd.DataFrame({'時間帯': [9, 10, 10], '曜日': ['1 月', '1 月', '2 火'], '件数': [2, 3, 5]})
        result = run(df)
        result.chart = {'kind': 'heatmap', 'x': '時間帯', 'row': '曜日', 'y': ['件数']}
        plan = choose_presentation('来店の多い曜日と時間帯をヒートマップで', [result])
        self.assertEqual(plan.mode, 'chart')
        self.assertEqual(plan.chart.row, '曜日')
        pd.testing.assert_frame_equal(plan.dataframe, df)
        result.chart['row'] = '存在しない'
        self.assertEqual(choose_presentation('ヒートマップ', [result]).mode, 'table')
        result.chart['row'] = '曜日'
        result.dataframe = pd.concat([df, df.iloc[:1]], ignore_index=True)
        self.assertEqual(choose_presentation('ヒートマップ', [result]).mode, 'table')

    def test_special_charts_render_real_vega_specs_without_manual_controls(self):
        import json
        for kind in ('pie', 'heatmap'):
            app = AppTest.from_string("""
import pandas as pd
from app import render_results
from src.bq_tools import QueryRun
frame = pd.DataFrame({'曜日': ['1 月', '2 火'], '時間帯': [9, 10], '来店数': [2, 3]})
choice = {'kind': KIND, 'x': '曜日', 'y': ['来店数']}
if KIND == 'heatmap':
    choice['row'] = '時間帯'
run = QueryRun('SELECT 1', 'SELECT 1', 0, 0, frame, {}, chart=choice)
render_results({'question': 'グラフで', 'runs': [run]}, 0, set())
""".replace('KIND', repr(kind))).run(timeout=20)
            self.assertFalse(app.exception)
            self.assertEqual(len(app.warning), 0)
            self.assertEqual(len(app.selectbox), 0)
            charts = app.get('vega_lite_chart')
            self.assertEqual(len(charts), 1)
            spec = json.loads(charts[0].proto.spec)
            self.assertEqual(spec['mark']['type'], 'arc' if kind == 'pie' else 'rect')
            self.assertEqual(spec['encoding']['color']['type'], 'nominal' if kind == 'pie' else 'quantitative')
            if kind == 'heatmap':
                self.assertEqual(spec['encoding']['y']['field'], '時間帯')
                self.assertEqual(spec['encoding']['x']['field'], '曜日')
            else:
                self.assertEqual(spec['encoding']['theta']['field'], '来店数')

    def test_multiple_numeric_series_share_one_line_chart(self):
        import json
        app = AppTest.from_string("""
import pandas as pd
from app import render_results
from src.bq_tools import QueryRun
frame = pd.DataFrame({'月': ['2026-01', '2026-02'], 'A店': [2, 3], 'B店': [5, 4]})
choice = {'kind': 'line', 'x': '月', 'y': ['A店', 'B店']}
run = QueryRun('SELECT 1', 'SELECT 1', 0, 0, frame, {}, chart=choice)
render_results({'question': '店舗別推移', 'runs': [run]}, 0, set())
""").run(timeout=20)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.warning), 0)
        charts = app.get('vega_lite_chart')
        self.assertEqual(len(charts), 1)
        spec = json.loads(charts[0].proto.spec)
        encoding = spec.get('encoding') or spec['layer'][0]['encoding']
        self.assertIn('color', encoding)

    def test_literal_column_names_are_escaped_in_special_chart_specs(self):
        from src.charts import ChartSpec, special_chart_spec
        spec = special_chart_spec(ChartSpec('heatmap', '月.日', ['来店[数]'], row='店名'))
        self.assertEqual(spec['encoding']['x']['field'], r'月\.日')
        self.assertEqual(spec['encoding']['color']['field'], r'来店\[数\]')
