"""/effortの選択・解除、ローカル処理、Highの入力欄表示を確認する。"""
import unittest
from pathlib import Path
import shutil
import subprocess

from streamlit.testing.v1 import AppTest


SOURCE = '''
from types import SimpleNamespace
from unittest.mock import Mock
import streamlit as st
from app import handle_command, render_prompt_input
from src.agent import ToolContext
if 'ctx' not in st.session_state:
    st.session_state.ctx = ToolContext(tools=Mock(), settings=SimpleNamespace(gemini_model='gemini-3.8-flash'))
    st.session_state.messages = []
    st.session_state.contents = []
    st.session_state.pending = None
ctx = st.session_state.ctx
prompt = render_prompt_input(ctx)
if prompt:
    handle_command(prompt, ctx, ctx.settings, ctx.tools, Mock(), '', None, 0)
'''


class EffortTests(unittest.TestCase):
    def test_effort_popup_switches_without_buttons_or_success_message(self):
        app = AppTest.from_string(SOURCE).run(timeout=20)
        self.assertEqual(app.session_state.ctx.thinking_mode, 'low')
        self.assertTrue(any('#0d9488' in m.value for m in app.markdown))
        app.chat_input[0].set_value('/effort').run()
        self.assertFalse(app.exception)
        self.assertTrue(app.session_state['effort_panel'])
        self.assertEqual(len(app.button), 0)
        for mode, color in [('high', '#8b5cf6'), ('medium', '#3b82f6'), ('low', '#0d9488')]:
            app.session_state['keyboard_effort'] = mode
            app.run()
            self.assertFalse(app.exception)
            self.assertEqual(app.session_state.ctx.thinking_mode, mode)
            self.assertTrue(any(color in m.value for m in app.markdown))
            self.assertNotIn('command_output', app.session_state)
        self.assertEqual(app.session_state.contents, [])
        self.assertFalse(app.session_state.ctx.tools.mock_calls)

    def test_arguments_rejected_and_confirmation_blocks_popup(self):
        app = AppTest.from_string(SOURCE).run(timeout=20)
        app.chat_input[0].set_value('/effort HIGH').run()
        self.assertIn('引数なし', app.session_state.command_output)
        app.session_state.pending = True
        app.chat_input[0].set_value('/effort').run()
        self.assertNotIn('effort_panel', app.session_state)
        self.assertIn('実行確認待ち', app.session_state.command_output)
        app.session_state['keyboard_effort'] = 'high'
        app.run()
        self.assertEqual(app.session_state.ctx.thinking_mode, 'low')

    def test_thinking_restores_original_effort_and_keeps_chat(self):
        app = AppTest.from_string(SOURCE).run(timeout=20)
        app.session_state.contents = ['続きの会話']
        app.session_state.ctx.thinking_mode = 'low'
        app.chat_input[0].set_value('/thinking').run()
        self.assertTrue(app.session_state.ctx.thinking_enabled)
        self.assertEqual(app.session_state.ctx.thinking_mode, 'high')
        app.chat_input[0].set_value('/effort').run()
        app.session_state['keyboard_effort'] = 'medium'
        app.run()
        self.assertTrue(app.session_state.ctx.thinking_enabled)
        app.chat_input[0].set_value('/thinking').run()
        self.assertFalse(app.session_state.ctx.thinking_enabled)
        self.assertEqual(app.session_state.ctx.thinking_mode, 'low')
        self.assertEqual(app.session_state.contents, ['続きの会話'])
        self.assertFalse(app.session_state.ctx.tools.mock_calls)

    def test_command_menu_trigger_queues_only_supported_local_commands(self):
        app = AppTest.from_string("""
import streamlit as st
from src.command_menu import _keyboard_event
st.session_state['slash_command_menu'] = {'action': {'kind': 'command', 'command': '/thinking'}}
_keyboard_event()
st.session_state['queued'] = st.session_state.pop('keyboard_command', None)
st.session_state['slash_command_menu'] = {'action': {'kind': 'command', 'command': '/unknown'}}
_keyboard_event()
""").run(timeout=20)
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['queued'], '/thinking')
        self.assertNotIn('keyboard_command', app.session_state)

    def test_effort_callback_accepts_named_modes_and_escape_closes_popup(self):
        app = AppTest.from_string("""
import streamlit as st
from src.command_menu import _keyboard_event
st.session_state['effort_panel'] = True
for mode in ('3', 'high'):
    st.session_state['slash_command_menu'] = {'action': {'kind': 'effort', 'mode': mode}}
    _keyboard_event()
    if mode == '3':
        assert 'keyboard_effort' not in st.session_state
st.session_state['slash_command_menu'] = {'action': {'kind': 'close_effort'}}
_keyboard_event()
""").run(timeout=20)
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state['keyboard_effort'], 'high')
        self.assertNotIn('effort_panel', app.session_state)

    @unittest.skipUnless(shutil.which('node'), 'Node.jsがないためキーボードの単体テストをスキップ')
    def test_effort_digits_modifiers_ime_arrow_enter_and_escape(self):
        result = subprocess.run([shutil.which('node'), str(Path(__file__).with_name('command_menu_keyboard.mjs'))],
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('effort keyboard checks passed', result.stdout)
