"""実際に適用中の値と管理者アクセス境界の確認。ネットワーク接続不要。"""
import unittest
from streamlit.testing.v1 import AppTest


class AdminSettingsTests(unittest.TestCase):
    def render(self, viewer):
        source = f'''
import os
from unittest.mock import Mock, patch
from src.config import load_settings
from src.admin_page import render_admin_page
with patch.dict(os.environ, {{
    'GCP_PROJECT_ID':'test-project', 'BQ_DEFAULT_DATASET':'allowed',
    'BQ_ALLOWED_DATASETS':'allowed', 'GOOGLE_APPLICATION_CREDENTIALS':'',
    'BQ_LOG_DATASET':'', 'ADMIN_EMAILS':'admin@example.com', 'ADMIN_ALLOW_LOCAL':'false',
    'SAMPLE_ROWS':'100', 'GEMINI_THINKING_BUDGET':'10000',
}}):
    settings = load_settings()
logger = Mock()
render_admin_page(settings, logger, {viewer!r})
logger.query_usage.assert_not_called()
logger.get_limits.assert_not_called()
'''
        app = AppTest.from_string(source).run(timeout=20)
        self.assertEqual(len(app.exception), 0, [str(e) for e in app.exception])
        return app

    def test_effective_values_visible_even_when_logging_disabled(self):
        app = self.render('admin@example.com')
        self.assertEqual([tab.label for tab in app.tabs], ['利用状況', '使用制限', '実行設定'])
        values = app.dataframe[0].value.set_index('設定項目')['適用値']
        self.assertEqual(values['Gemini に渡すサンプル行数'], '100 行')
        self.assertIn('Medium（/effortでLow・Medium・Highを選択）', values['思考設定'])
        rendered = app.dataframe[0].value.to_string()
        self.assertNotIn('admin@example.com', rendered)
        self.assertNotIn('credentials', rendered)

    def test_non_admin_cannot_read_effective_settings(self):
        app = self.render('viewer@example.com')
        self.assertEqual(len(app.error), 1)
        self.assertEqual(len(app.dataframe), 0)
        self.assertEqual(len(app.tabs), 0)


if __name__ == '__main__':
    unittest.main()
