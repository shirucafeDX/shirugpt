"""履歴の型保持・所有者境界・競合と Streamlit 操作の回帰確認。"""
import os
import unittest
from datetime import date, datetime, timezone
from decimal import Decimal
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
from google.genai import types
from google.api_core.exceptions import PreconditionFailed
from streamlit.testing.v1 import AppTest

from src.agent import ToolContext, ConfirmationPending, GeminiAgent, _sql_key
from src.bq_tools import QueryRun
from src.chat_history import HistoryStore, pack, unpack
from src.usage_log import MonthlyUsage


def settings():
    return SimpleNamespace(gcp_project='test', allowed_datasets=('data',),
                           gemini_model='test-model', chat_history_bucket='test-bucket')


class SnapshotTests(unittest.TestCase):
    def setUp(self):
        self.ctx = ToolContext(tools=Mock(), settings=settings(), user_email='a@example.com')
        self.frame = pd.DataFrame({
            '日付': [date(2026, 1, 1), date(2026, 1, 2)],
            '金額': [Decimal('1.25'), None],
            '件数': pd.Series([1, None], dtype='Int64'),
            '日時': pd.date_range('2026-01-01', periods=2, tz='UTC'),
        })
        self.run = QueryRun('SELECT 1', 'SELECT 1', 1, 2, self.frame, {'rows': 2})
        self.ctx.query_cache[_sql_key('SELECT 1')] = self.run
        self.ctx.turn_runs = [self.run]
        self.ctx.session_prompt_tokens = 10
        self.ctx.session_billed_bytes = 2
        self.ctx.schema_cache = {'data.table': {'name': 'table'}}
        self.messages = [{'role': 'user', 'text': '表で'},
                         {'role': 'assistant', 'text': '回答', 'question': '表で', 'runs': [self.run]}]
        self.contents = [types.Content(role='user', parts=[types.Part.from_text(text='表で')]),
                         types.Content(role='model', parts=[types.Part(text='回答', thought_signature=b'\xff\x00signature')])]

    def restore(self, pending=None, status='ready'):
        new = ToolContext(tools=Mock(), settings=settings(), user_email='a@example.com')
        result = unpack(pack(self.messages, self.contents, self.ctx, pending, status), new)
        return new, result

    def test_roundtrip_preserves_frames_signatures_and_runtime_boundary(self):
        new, (messages, contents, pending) = self.restore()
        pd.testing.assert_frame_equal(messages[1]['runs'][0].dataframe, self.frame)
        self.assertEqual(contents[1].parts[0].thought_signature, b'\xff\x00signature')
        self.assertEqual(new.schema_cache, self.ctx.schema_cache)
        self.assertEqual(new.session_prompt_tokens, 10)
        self.assertEqual(new.session_id, self.ctx.session_id)
        self.assertIsNone(new.limit_baseline_month_start)
        self.assertIsNone(pending)
        self.assertIsNot(new.tools, self.ctx.tools)

    def test_restored_query_cache_does_not_requery_or_rebill(self):
        new, _ = self.restore()
        agent = GeminiAgent(new.settings, client=Mock())
        new.start_turn()
        result = agent._run_query({'sql': 'SELECT 1'}, new)
        self.assertTrue(result['cached'])
        new.tools.run_query.assert_not_called()
        self.assertEqual(new.session_billed_bytes, 2)

    def test_pending_function_calls_restore_without_automatic_execution(self):
        self.ctx.resume_calls = [types.FunctionCall(name='run_query', args={'sql': 'SELECT 2'})]
        self.contents.append(types.Content(role='model', parts=[types.Part(function_call=self.ctx.resume_calls[0])]))
        pending = ConfirmationPending('key', 'SELECT 2', 20, 10, ['note'])
        new, (_, contents, restored) = self.restore(pending)
        self.assertEqual(restored, pending)
        self.assertEqual(new.resume_calls[0].args['sql'], 'SELECT 2')
        self.assertEqual(contents[-1].parts[0].function_call.name, 'run_query')
        new.tools.run_query.assert_not_called()

    def test_interrupted_prompt_gets_notice_and_no_automatic_rerun(self):
        self.contents = self.contents[:1]
        self.messages = self.messages[:1]
        _, (messages, contents, pending) = self.restore(status='running')
        self.assertTrue(messages[-1]['is_error'])
        self.assertEqual(contents[-1].role, 'model')
        self.assertIsNone(pending)

    def test_changed_access_policy_rejected(self):
        data = pack(self.messages, self.contents, self.ctx, None)
        new = ToolContext(tools=Mock(), settings=settings())
        new.settings.allowed_datasets = ('other',)
        with self.assertRaises(ValueError):
            unpack(data, new)

    def test_restored_totals_are_not_added_to_monthly_usage_again(self):
        from app import _effective_monthly_usage_usd
        from src.config import gemini_cost_usd
        new, _ = self.restore()
        usage = Mock()
        usage.monthly_usage.return_value = MonthlyUsage(100, 50, 1000)
        expected = usage.monthly_usage.return_value.cost_usd()
        self.assertEqual(_effective_monthly_usage_usd(new, usage, new.user_email), expected)
        new.session_prompt_tokens += 5
        self.assertAlmostEqual(_effective_monthly_usage_usd(new, usage, new.user_email),
                               expected + gemini_cost_usd(5, 0))
        self.assertEqual(usage.monthly_usage.call_count, 1)


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.client = Mock()
        self.store = HistoryStore('bucket', client=self.client)
        self.blob = self.client.bucket.return_value.blob.return_value
        self.blob.generation = 8
        self.id = 'a' * 32

    def test_owner_prefix_and_ids_cannot_escape(self):
        self.assertEqual(self.store._prefix('A@example.com'), self.store._prefix('a@example.com'))
        self.assertNotEqual(self.store._prefix('a@example.com'), self.store._prefix('b@example.com'))
        for email, cid in [('', self.id), ('a@example.com', '../other')]:
            with self.assertRaises(ValueError):
                self.store.load(email, cid)

    def test_conditional_create_update_download_delete(self):
        self.assertEqual(self.store.save('a@example.com', self.id, b'data', 'タイトル'), 8)
        self.assertEqual(self.blob.upload_from_string.call_args.kwargs['if_generation_match'], 0)
        self.store.save('a@example.com', self.id, b'new', 'タイトル', 8)
        self.assertEqual(self.blob.upload_from_string.call_args.kwargs['if_generation_match'], 8)
        self.store.load('a@example.com', self.id)
        self.assertEqual(self.blob.download_as_bytes.call_args.kwargs['if_generation_match'], 8)
        self.store.delete('a@example.com', self.id, 8)
        self.assertEqual(self.blob.delete.call_args.kwargs['if_generation_match'], 8)

    def test_conflict_is_reported_without_overwriting(self):
        self.blob.upload_from_string.side_effect = PreconditionFailed('changed')
        with self.assertRaises(PreconditionFailed):
            self.store.save('a@example.com', self.id, b'data', 'title', 7)
        self.assertEqual(self.blob.upload_from_string.call_count, 1)


UI_SOURCE = '''
from datetime import datetime, timezone
from unittest.mock import Mock, patch
import streamlit as st
from src.config import load_settings
from src.chat_history import pack
from app import init_state, render_history, save_conversation
s = load_settings()
s = __import__('dataclasses').replace(s, chat_history_bucket='test')
tools, usage = Mock(), Mock()
ctx = init_state(s, tools, usage, 'a@example.com')
store = Mock()
store.list.return_value = [{'id': 'a'*32, 'title': '保存した質問', 'updated': datetime.now(timezone.utc), 'generation': 5}]
store.save.return_value = 6
store.load.return_value = (pack([{'role':'user','text':'保存した質問'}], [], ctx, None), 5)
with patch('app.history_store', return_value=store):
    if st.button('保存テスト'):
        st.session_state.messages = [{'role':'user','text':'新しい質問'}]
        save_conversation(ctx)
    if st.button('保存失敗テスト'):
        store.save.side_effect = RuntimeError('offline')
        st.session_state.messages = [{'role':'user','text':'未保存の質問'}]
        save_conversation(ctx)
    render_history(s, tools, usage, 'a@example.com', ctx)
st.write([m['text'] for m in st.session_state.messages])
'''


class HistoryUITests(unittest.TestCase):
    def render(self):
        app = AppTest.from_string(UI_SOURCE).run(timeout=20)
        self.assertFalse(app.exception)
        return app

    def test_load_save_and_delete_ui(self):
        app = self.render()
        next(b for b in app.button if b.label == 'この会話を開く').click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['messages'][0]['text'], '保存した質問')
        app.button[0].click().run()
        self.assertFalse(app.session_state['history_dirty'])
        self.assertEqual(app.session_state['history_generation'], 6)
        app.checkbox[0].check().run()
        app.button[-1].click().run()
        self.assertFalse(app.exception)

    def test_failed_save_preserves_content_and_blocks_switching(self):
        app = self.render()
        app.button[1].click().run()
        self.assertFalse(app.exception)
        self.assertTrue(app.session_state['history_dirty'])
        self.assertEqual(app.session_state['messages'][0]['text'], '未保存の質問')
        self.assertTrue(next(b for b in app.button if b.label == 'この会話を開く').disabled)
        next(b for b in app.button if b.label == '再保存').click().run()
        self.assertFalse(app.session_state['history_dirty'])

    def test_identity_change_clears_previous_users_conversation(self):
        source = '''
from unittest.mock import Mock
import streamlit as st
from app import init_state
from src.config import load_settings
s = load_settings()
email = st.selectbox('user', ['a@example.com', 'b@example.com'])
ctx = init_state(s, Mock(), Mock(), email)
if st.button('seed'):
    st.session_state.messages = [{'role':'user', 'text':'private'}]
'''
        app = AppTest.from_string(source).run()
        app.button[0].click().run()
        self.assertTrue(app.session_state['messages'])
        app.selectbox[0].select('b@example.com').run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['messages'], [])
        self.assertEqual(app.session_state['ctx'].user_email, 'b@example.com')

    def test_anonymous_session_never_uses_storage(self):
        source = '''
from unittest.mock import Mock, patch
import streamlit as st
from app import init_state, save_conversation, render_history
from src.config import load_settings
s = __import__('dataclasses').replace(load_settings(), chat_history_bucket='test')
ctx = init_state(s, Mock(), Mock(), '')
st.session_state.messages = [{'role':'user', 'text':'local'}]
with patch('app.history_store', side_effect=AssertionError('anonymous storage access')):
    save_conversation(ctx)
    render_history(s, Mock(), Mock(), '', ctx)
'''
        app = AppTest.from_string(source).run()
        self.assertFalse(app.exception)
        self.assertIn('ログイン', app.caption[0].value)


if __name__ == '__main__':
    unittest.main()
