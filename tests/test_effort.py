"""/effortの選択・解除、ローカル処理、Highの入力欄表示を確認する。"""
import unittest

from streamlit.testing.v1 import AppTest


SOURCE = '''
from types import SimpleNamespace
from unittest.mock import Mock
import streamlit as st
from app import handle_command, render_effort_panel, render_prompt_input
from src.agent import ToolContext
if 'ctx' not in st.session_state:
    st.session_state.ctx = ToolContext(tools=Mock(), settings=SimpleNamespace(gemini_model='gemini-3.8-flash'))
    st.session_state.messages = []
    st.session_state.contents = []
    st.session_state.pending = None
ctx = st.session_state.ctx
if st.session_state.get('effort_panel'):
    render_effort_panel(ctx)
prompt = render_prompt_input(ctx)
if prompt:
    handle_command(prompt, ctx, ctx.settings, ctx.tools, Mock(), '', None, 0)
'''


class EffortTests(unittest.TestCase):
    def test_effort_opens_choices_and_switches_high_then_medium_without_model_calls(self):
        app = AppTest.from_string(SOURCE).run(timeout=20)
        app.chat_input[0].set_value('/effort').run()
        self.assertFalse(app.exception)
        self.assertEqual([b.label for b in app.button], ['1. Low', '2. Medium', '3. High', '閉じる'])
        self.assertEqual([app.button(key=f'effort_{mode}').proto.shortcut for mode in ('low', 'medium', 'high')], ['1', '2', '3'])
        app.button(key='effort_high').click().run()
        self.assertFalse(app.exception)
        self.assertEqual(app.session_state.ctx.thinking_mode, 'high')
        self.assertTrue(any('border: 2px solid #8b5cf6' in m.value for m in app.markdown))
        app.button(key='effort_medium').click().run()
        self.assertTrue(any('border: 2px solid #3b82f6' in m.value for m in app.markdown))
        self.assertEqual(app.session_state.ctx.thinking_mode, 'medium')
        self.assertFalse(any('border: 2px solid #8b5cf6' in m.value for m in app.markdown))
        app.button(key='effort_low').click().run()
        self.assertEqual(app.session_state.ctx.thinking_mode, 'low')
        self.assertTrue(any('border: 2px solid #0d9488' in m.value for m in app.markdown))
        self.assertEqual(app.session_state.messages, [])
        self.assertEqual(app.session_state.contents, [])
        self.assertFalse(app.session_state.ctx.tools.mock_calls)

    def test_arguments_are_rejected_and_pending_confirmation_disables_selection(self):
        app = AppTest.from_string(SOURCE).run(timeout=20)
        app.chat_input[0].set_value('/effort HIGH').run()
        self.assertEqual(app.session_state.ctx.thinking_mode, 'medium')
        self.assertIn('引数なし', app.session_state.command_output)
        app.session_state.pending = True
        app.chat_input[0].set_value('/effort').run()
        self.assertTrue(app.button(key='effort_medium').disabled)
        self.assertTrue(app.button(key='effort_high').disabled)

    def test_closing_choices_keeps_current_effort(self):
        app = AppTest.from_string(SOURCE).run(timeout=20)
        app.chat_input[0].set_value('/effort').run()
        app.button(key='effort_high').click().run()
        app.button(key='close_effort_panel').click().run()
        self.assertNotIn('effort_panel', app.session_state)
        self.assertEqual(app.session_state.ctx.thinking_mode, 'high')

    def test_thinking_restores_original_effort_and_keeps_chat(self):
        app = AppTest.from_string(SOURCE).run(timeout=20)
        app.session_state.contents = ['続きの会話']
        app.session_state.ctx.thinking_mode = 'low'
        app.chat_input[0].set_value('/thinking').run()
        self.assertTrue(app.session_state.ctx.thinking_enabled)
        self.assertEqual(app.session_state.ctx.thinking_mode, 'high')
        app.chat_input[0].set_value('/effort').run()
        app.button(key='effort_medium').click().run()
        self.assertTrue(app.session_state.ctx.thinking_enabled)
        app.chat_input[0].set_value('/thinking').run()
        self.assertFalse(app.session_state.ctx.thinking_enabled)
        self.assertEqual(app.session_state.ctx.thinking_mode, 'low')
        self.assertEqual(app.session_state.contents, ['続きの会話'])
        self.assertFalse(app.session_state.ctx.tools.mock_calls)
