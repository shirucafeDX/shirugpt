"""当月利用量の本人境界、日本時間の月境界、表示と旧ログの互換を確認。"""
import unittest
from datetime import date, datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
from streamlit.testing.v1 import AppTest

from src.commands import current_month_usage_text
from src.usage_log import UsageLogger


class UsageDetailTests(unittest.TestCase):
    def logger(self, frame):
        settings = SimpleNamespace(logging_enabled=True, gcp_project='test', log_dataset='logs', log_table='events',
                                   vertex_location='global', max_bytes_billed=10 * 1024**3, bq_location=None)
        client = Mock()
        client.query.return_value.result.return_value.to_dataframe.return_value = frame
        return UsageLogger(settings, client=client), client

    def test_report_is_parameterized_for_only_current_user_and_uses_jst_month_boundary(self):
        frame = pd.DataFrame([dict(month=date(2026, 11, 1), input_tokens=1000, thinking_tokens=300,
                                   output_tokens=500, gemini_calls=2, bq_queries=3, billed_bytes=1024**3, gemini_usd=0.12)])
        logger, client = self.logger(frame)
        email = "me@example.com' OR TRUE"
        report = logger.current_month_details(email, datetime(2026, 10, 31, 15, tzinfo=timezone.utc))
        call = client.query.call_args
        sql = call.args[0]
        self.assertIn('WHERE user_email = @user_email', sql)
        self.assertNotIn(email, sql)
        self.assertIn("DATE(event_time, 'Asia/Tokyo')", sql)
        self.assertNotIn('IFNULL(output_tokens, 0) -', sql)
        self.assertNotIn('thinking_tokens', report.columns)
        params = {p.name: p.value for p in call.kwargs['job_config'].query_parameters}
        self.assertEqual(params, dict(user_email=email, start_date=date(2026, 11, 1), end_date=date(2026, 12, 1)))
        self.assertEqual(len(report), 1)
        self.assertEqual(report.iloc[0]['month'], date(2026, 11, 1))
        self.assertEqual(report['input_tokens'].sum(), 1000)
        text = current_month_usage_text(report)
        for value in ('Input', 'Output', '1,000', '500', '1.00 GB', 'USD 0.1200', '2026-11'):
            self.assertIn(value, text)

    def test_empty_current_month_is_zero_and_anonymous_report_never_queries(self):
        logger, client = self.logger(pd.DataFrame())
        report = logger.current_month_details('me@example.com', datetime(2026, 1, 1, tzinfo=timezone.utc))
        self.assertEqual(len(report), 1)
        self.assertEqual(report.iloc[-1]['month'], date(2026, 1, 1))
        self.assertEqual(report['billed_bytes'].sum(), 0)
        client.reset_mock()
        self.assertTrue(logger.current_month_details('').empty)
        client.query.assert_not_called()

    def test_usage_command_includes_current_month_details_without_changing_chat(self):
        app = AppTest.from_string('''
from types import SimpleNamespace
from unittest.mock import Mock, patch
import pandas as pd
import streamlit as st
from src.agent import ToolContext
from app import handle_command
ctx = ToolContext(tools=Mock(), settings=SimpleNamespace(logging_enabled=True))
usage = Mock()
usage.current_month_details.return_value = pd.DataFrame([dict(month='2026-10-01', input_tokens=1200,
    thinking_tokens=100, output_tokens=50, gemini_calls=1, bq_queries=2, billed_bytes=1024**3, gemini_usd=0.1)])
st.session_state.messages = [{'role':'user', 'text':'前の質問'}]
st.session_state.contents = ['会話は維持']
with patch('app.st.rerun'):
    handle_command('/usage', ctx, ctx.settings, ctx.tools, usage, 'me@example.com', 0.2, 5)
usage.current_month_details.assert_called_once_with('me@example.com')
st.session_state['tools_called'] = bool(ctx.tools.mock_calls)
''').run(timeout=20)
        self.assertFalse(app.exception)
        self.assertIn('2026-10', app.session_state.command_output)
        self.assertIn('1,200', app.session_state.command_output)
        self.assertEqual(app.session_state.contents, ['会話は維持'])
        self.assertFalse(app.session_state['tools_called'])
