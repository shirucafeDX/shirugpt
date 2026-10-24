"""コマンドが課金・会話の処理へ流れず、ローカルだけで扱われることを確認。"""
import unittest
from datetime import datetime
from types import SimpleNamespace
from zoneinfo import ZoneInfo

from streamlit.testing.v1 import AppTest

from src.commands import parse_command, usage_text


class CommandTests(unittest.TestCase):
    def test_only_leading_slash_is_command_and_arguments_are_preserved(self):
        self.assertIsNone(parse_command("売上 / 月を教えて"))
        command = parse_command("  /MODEL\tgemini-3.8-flash  ")
        self.assertEqual((command.name, command.argument), ("/model", "gemini-3.8-flash"))

    def test_usage_distinguishes_monthly_and_conversation_and_thinking_subset(self):
        ctx = SimpleNamespace(session_prompt_tokens=1000, session_output_tokens=2000,
                              session_thinking_tokens=1500, session_billed_bytes=1024**3)
        text = usage_text(ctx, 2, 5, datetime(2026, 11, 1, tzinfo=ZoneInfo("Asia/Tokyo")))
        for expected in ("40.0%", "USD 3.0000", "2026年11月01日 00:00", "2,000", "USD 1.50", "USD 7.50", "BigQuery推定費用", "1.00 GB"):
            self.assertIn(expected, text)
        ctx.session_thinking_tokens = None
        self.assertNotIn("思考トークン（出力の内数）", usage_text(ctx, None, None, None))
        self.assertNotIn("$", text)

    def test_local_commands_do_not_append_messages_call_model_or_record_prompts(self):
        for command in ("/help", "/usage", "/unknown", "/help extra", "/model", "/thinking"):
            with self.subTest(command=command):
                app = AppTest.from_string('''
import streamlit as st
from unittest.mock import Mock, patch
from types import SimpleNamespace
from app import handle_command
from src.agent import ToolContext
ctx = ToolContext(tools=Mock(), settings=SimpleNamespace(gemini_model='gemini-3.8-flash'))
st.session_state.messages = [{'role':'user', 'text':'前の質問'}]
st.session_state.contents = ['元の会話']
usage_logger = Mock()
with patch('app.st.rerun'):
    handled = handle_command(COMMAND, ctx, ctx.settings, ctx.tools, usage_logger, '', None, 0)
st.session_state['handled'] = handled
st.session_state['logged'] = usage_logger.log_prompt.called
st.session_state['called_tools'] = bool(ctx.tools.mock_calls)
'''.replace("COMMAND", repr(command))).run(timeout=20)
                self.assertFalse(app.exception)
                self.assertTrue(app.session_state['handled'])
                self.assertFalse(app.session_state['logged'])
                self.assertFalse(app.session_state['called_tools'])
                self.assertEqual(app.session_state.contents, ['元の会話'])
                self.assertEqual(len(app.session_state.messages), 1)

    def test_direct_command_submission_is_rejected_before_logging(self):
        from app import submit_prompt
        from unittest.mock import Mock
        agent, ctx = Mock(), Mock()
        with self.assertRaises(ValueError):
            submit_prompt('/new', agent, ctx)
        agent.run.assert_not_called()
        ctx.usage_logger.log_prompt.assert_not_called()


class CommandOutputDismissTests(unittest.TestCase):
    def test_enter_callback_closes_output_without_touching_conversation(self):
        app = AppTest.from_string('''
import streamlit as st
from src.command_menu import _keyboard_event
st.session_state.command_output = 'モデルの説明'
st.session_state.messages = [{'role': 'user', 'text': '前の質問'}]
st.session_state.contents = ['会話の文脈']
st.session_state['slash_command_menu'] = {'action': {'kind': 'close_command_output'}}
_keyboard_event()
''').run(timeout=20)
        self.assertFalse(app.exception)
        self.assertNotIn('command_output', app.session_state)
        self.assertEqual(app.session_state.contents, ['会話の文脈'])
        self.assertEqual(len(app.session_state.messages), 1)

    def test_question_submission_closes_output_before_analysis_starts(self):
        app = AppTest.from_string('''
from types import SimpleNamespace
from unittest.mock import Mock, patch
import streamlit as st
from app import submit_prompt
from src.agent import ToolContext
ctx = ToolContext(tools=Mock(), settings=SimpleNamespace(chat_history_bucket=''))
st.session_state.command_output = '使用量の説明'
st.session_state.messages = []
st.session_state.contents = []
with patch('app.BackgroundTurn') as turn, patch('app.st.rerun'):
    submit_prompt('来店数は？', Mock(), ctx)
    turn.return_value.start.assert_called_once()
''').run(timeout=20)
        self.assertFalse(app.exception)
        self.assertNotIn('command_output', app.session_state)
        self.assertEqual(app.session_state.messages[0]['text'], '来店数は？')
