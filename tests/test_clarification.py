"""確認質問の中断・回答・履歴保存と、選択／自由記述の操作を確認する。"""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from google.genai import types
from streamlit.testing.v1 import AppTest

from src.agent import AnswerResult, ClarificationResult, GeminiAgent, ToolContext
from src.chat_history import pack, unpack


def reply(*calls, text=None):
    parts = [types.Part(function_call=call) for call in calls]
    if text:
        parts.append(types.Part.from_text(text=text))
    return types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role='model', parts=parts), finish_reason='STOP',
    )])


class ClarificationTests(unittest.TestCase):
    def setUp(self):
        self.settings = SimpleNamespace(max_tool_iterations=3, gcp_project='test',
                                        allowed_datasets=('data',), gemini_model='test')
        self.ctx = ToolContext(tools=Mock(), settings=self.settings)
        self.ctx.schema_cache['data.来店'] = {'columns': [
            {'name': 'ordered_at', 'type': 'DATETIME'}, {'name': 'store', 'type': 'STRING'},
        ]}
        self.agent = GeminiAgent(self.settings, client=Mock())
        self.contents = [types.Content(role='user', parts=[types.Part.from_text(text='来店数を比較して')])]
        self.call = types.FunctionCall(name='ask_clarification', args={
            'question': '比較する期間を教えてください。', 'options': ['今月と先月', '今年と昨年'],
            'selection_mode': 'single',
        })

    def test_clarification_waits_until_schema_is_known(self):
        self.ctx.schema_cache.clear()
        self.ctx.tools.get_table_schema.return_value = {'columns': [{'name': 'store', 'type': 'STRING'}]}
        schema_call = types.FunctionCall(name='get_table_schema', args={'dataset': 'data', 'table': '来店'})
        self.agent._generate = Mock(side_effect=[reply(self.call), reply(schema_call), reply(self.call)])
        result = self.agent.run(self.contents, self.ctx)
        self.assertIsInstance(result, ClarificationResult)
        self.assertIn('get_table_schema', self.contents[2].parts[0].function_response.response['error'])
        self.ctx.tools.get_table_schema.assert_called_once_with('data', '来店')
        self.assertIn('data.来店', self.ctx.schema_cache)
        self.assertEqual(self.agent._generate.call_count, 3)

    def test_multiple_selection_mode_is_returned_and_validated(self):
        self.call.args['selection_mode'] = 'multi'
        self.agent._generate = Mock(return_value=reply(self.call))
        self.assertEqual(self.agent.run(self.contents, self.ctx).selection_mode, 'multi')
        self.call.args['selection_mode'] = 'invalid'
        self.assertIn('error', self.agent._dispatch(self.call, self.ctx).function_response.response)

    def test_general_discussion_can_clarify_goals_without_querying_data(self):
        self.ctx.schema_cache.clear()
        self.call.args.update(scope='general', question='まず検討したい観点はどれですか？',
                              options=['課題を整理する', '施策を考える'])
        self.agent._generate = Mock(return_value=reply(self.call))
        result = self.agent.run(self.contents, self.ctx)
        self.assertIsInstance(result, ClarificationResult)
        self.ctx.tools.get_table_schema.assert_not_called()
        self.ctx.tools.run_query.assert_not_called()

    def test_waits_for_answer_without_executing_other_calls_and_can_resume(self):
        generated = reply(types.FunctionCall(name='run_query', args={'sql': 'SELECT 1'}), self.call)
        generated.candidates[0].content.parts[0].thought_signature = b'signature'
        self.agent._generate = Mock(side_effect=[generated, reply(text='前年との比較です。')])
        result = self.agent.run(self.contents, self.ctx)
        self.assertIsInstance(result, ClarificationResult)
        self.assertEqual(result.options, ['今月と先月', '今年と昨年'])
        self.assertEqual(self.agent._generate.call_count, 1)
        self.ctx.tools.run_query.assert_not_called()
        self.assertEqual([c.role for c in self.contents], ['user', 'model', 'user', 'model'])
        self.assertEqual(self.contents[1].parts[0].thought_signature, b'signature')
        self.assertEqual(len(self.contents[2].parts), 2)
        self.assertIn('未実行', self.contents[2].parts[0].function_response.response['error'])
        self.assertIn('今年と昨年', self.contents[-1].parts[0].text)
        self.contents.append(types.Content(role='user', parts=[types.Part.from_text(text='今年と昨年')]))
        self.ctx.start_turn()
        self.assertIsInstance(self.agent.run(self.contents, self.ctx), AnswerResult)
        self.assertEqual(self.contents[-2].parts[0].text, '今年と昨年')

    def test_invalid_choices_are_returned_to_model_for_correction(self):
        for options in [[], ['一つだけ'], ['同じ', ' 同じ '], ['有効', ' '], ['有効', 1]]:
            with self.subTest(options=options):
                call = types.FunctionCall(name='ask_clarification', args={'question': '期間は？', 'options': options})
                result = self.agent._dispatch(call, self.ctx)
                self.assertIn('error', result.function_response.response)
        self.agent._generate = Mock(side_effect=[
            reply(types.FunctionCall(name='ask_clarification', args={'question': '期間は？', 'options': []})),
            reply(self.call),
        ])
        self.assertIsInstance(self.agent.run(self.contents, self.ctx), ClarificationResult)

    def test_waiting_question_survives_history_roundtrip(self):
        self.call.args['selection_mode'] = 'multi'
        self.agent._generate = Mock(return_value=reply(self.call))
        result = self.agent.run(self.contents, self.ctx)
        messages = [{'role': 'assistant', 'text': result.text, 'question': '来店数を比較して',
                     'clarification': {'options': result.options, 'selection_mode': result.selection_mode, 'answered': False}}]
        restored_ctx = ToolContext(tools=Mock(), settings=self.settings)
        restored, contents, pending = unpack(pack(messages, self.contents, self.ctx, None), restored_ctx)
        self.assertEqual(restored, messages)
        self.assertEqual(contents[-1].parts[0].text, self.contents[-1].parts[0].text)
        self.assertIsNone(pending)
        restored_ctx.tools.run_query.assert_not_called()


UI_SOURCE = '''
from types import SimpleNamespace
from unittest.mock import Mock
import streamlit as st
from google.genai import types
from app import process_turn, render_clarification, render_message
from src.agent import AnswerResult, ClarificationResult, ToolContext

class Agent:
    def __init__(self, selection_mode):
        self.selection_mode = selection_mode

    def run(self, contents, ctx, on_progress):
        if contents[-1].parts[0].text.startswith('確認への回答'):
            return AnswerResult('分析を続けました。')
        return ClarificationResult('比較する期間を教えてください。', ['今月と先月', '今年と昨年'],
                                   self.selection_mode)

agent = Agent(st.session_state.get('selection_mode', 'single'))
if 'ctx' not in st.session_state:
    st.session_state.ctx = ToolContext(tools=Mock(), settings=SimpleNamespace(chat_history_bucket=''))
    st.session_state.messages = [{'role': 'user', 'text': '来店数をグラフで比較して'}]
    st.session_state.contents = [types.Content(role='user', parts=[types.Part.from_text(text='来店数をグラフで比較して')])]
    process_turn(agent, st.session_state.ctx)
if st.session_state.get('active_turn') is not None:
    process_turn(agent, st.session_state.ctx)
for i, message in enumerate(st.session_state.messages):
    render_message(message, i)
last = st.session_state.messages[-1]
if last.get('clarification') and not last['clarification']['answered']:
    render_clarification(last, len(st.session_state.messages)-1, agent, st.session_state.ctx,
                         blocked=st.session_state.get('blocked', False))
'''


class ClarificationUITests(unittest.TestCase):
    def choices(self, app):
        return [b for b in app.button if b.key and '_option_' in b.key]

    def selected(self, app):
        return [b.label.split('. ', 1)[1] for b in self.choices(app) if b.proto.type == 'primary']

    def choose(self, app, number):
        self.choices(app)[number - 1].click().run()
        self.assertFalse(app.exception)

    def submit(self, app):
        next(b for b in app.button if b.label == '回答して続ける').click().run()
        self.assertFalse(app.exception)

    def open(self, selection_mode='single'):
        app = AppTest.from_string(UI_SOURCE)
        app.session_state['selection_mode'] = selection_mode
        app.run(timeout=20)
        self.assertFalse(app.exception)
        self.assertEqual(self.selected(app), [])
        return app

    def test_number_shortcuts_are_assigned_to_choices(self):
        app = self.open()
        self.assertEqual([b.label for b in self.choices(app)], ['1. 今月と先月', '2. 今年と昨年'])
        self.assertEqual([b.proto.shortcut for b in self.choices(app)], ['', ''])

    def test_selection_and_supplement_continue_analysis_and_preserve_original_request(self):
        app = self.open()
        app.text_area[0].input('東京の店舗だけ').run()
        self.choose(app, 1)
        self.assertEqual(app.text_area[0].value, '東京の店舗だけ')
        self.submit(app)
        messages = app.session_state.messages
        self.assertTrue(messages[1]['clarification']['answered'])
        self.assertIn('今月と先月', messages[2]['text'])
        self.assertIn('東京の店舗だけ', messages[2]['text'])
        self.assertEqual(messages[-1]['question'], '来店数をグラフで比較して')
        self.assertEqual(messages[-1]['text'], '分析を続けました。')
        self.assertEqual(self.choices(app), [])
        app.run()
        self.assertEqual(len(app.session_state.messages), 4)

    def test_free_text_only_is_accepted(self):
        app = self.open()
        app.text_area[0].input('2025年4月と2026年4月')
        self.submit(app)
        self.assertIn('2025年4月と2026年4月', app.session_state.messages[2]['text'])
        self.assertEqual(self.choices(app), [])

    def test_empty_answer_does_not_continue(self):
        app = self.open()
        app.text_area[0].input('  ')
        self.submit(app)
        self.assertEqual(len(app.warning), 1)
        self.assertEqual(len(app.session_state.messages), 2)
        self.assertFalse(app.session_state.messages[-1]['clarification']['answered'])

    def test_usage_limit_disables_submission(self):
        app = self.open()
        app.session_state['blocked'] = True
        app.run()
        self.assertTrue(all(b.disabled for b in app.button))
        self.assertTrue(app.text_area[0].disabled)
        self.assertEqual(len(app.session_state.messages), 2)

    def test_single_choice_can_be_changed_and_cleared(self):
        app = self.open()
        self.choose(app, 1)
        self.assertEqual(self.selected(app), ['今月と先月'])
        self.choose(app, 2)
        self.assertEqual(self.selected(app), ['今年と昨年'])
        self.choose(app, 2)
        self.assertEqual(self.selected(app), [])
        self.submit(app)
        self.assertEqual(len(app.warning), 1)
        self.assertEqual(len(app.session_state.messages), 2)

    def test_multiple_choices_and_supplement_are_submitted(self):
        app = self.open('multi')
        self.choose(app, 1)
        self.choose(app, 2)
        self.assertEqual(self.selected(app), ['今月と先月', '今年と昨年'])
        self.choose(app, 1)
        self.assertEqual(self.selected(app), ['今年と昨年'])
        self.choose(app, 1)
        app.text_area[0].input('東京の店舗だけ')
        self.submit(app)
        answer = app.session_state.messages[2]['text']
        self.assertIn('今月と先月', answer)
        self.assertIn('今年と昨年', answer)
        self.assertIn('東京の店舗だけ', answer)
        self.assertEqual(app.session_state.messages[-1]['question'], '来店数をグラフで比較して')

    def test_older_history_defaults_to_single_choice(self):
        app = self.open()
        app.session_state.messages[-1]['clarification'].pop('selection_mode')
        app.run()
        self.assertFalse(app.exception)
        self.choose(app, 1)
        self.submit(app)
        self.assertIn('今月と先月', app.session_state.messages[2]['text'])

    def keyboard(self, app, index=None, submit=False, panel=None):
        ctx = app.session_state.ctx
        mode = app.session_state['selection_mode']
        app.session_state['clarification_keyboard'] = dict(
            panel=panel or f'clarification_picker_{mode}_{ctx.session_id}_1', index=index, submit=submit)
        app.run()
        self.assertFalse(app.exception)

    def test_keyboard_single_submits_and_includes_supplement(self):
        app = self.open()
        app.text_area[0].input('東京の店舗だけ').run()
        self.keyboard(app, index=1)
        self.assertTrue(app.session_state.messages[1]['clarification']['answered'])
        self.assertIn('今年と昨年', app.session_state.messages[2]['text'])
        self.assertIn('東京の店舗だけ', app.session_state.messages[2]['text'])

    def test_keyboard_multi_toggles_then_enter_submits(self):
        app = self.open('multi')
        self.keyboard(app, index=0)
        self.keyboard(app, index=1)
        self.assertEqual(self.selected(app), ['今月と先月', '今年と昨年'])
        self.keyboard(app, index=0)
        self.assertEqual(self.selected(app), ['今年と昨年'])
        self.keyboard(app, submit=True)
        self.assertIn('今年と昨年', app.session_state.messages[2]['text'])
        self.assertNotIn('今月と先月', app.session_state.messages[2]['text'])

    def test_stale_or_blocked_keyboard_selection_cannot_submit(self):
        app = self.open()
        self.keyboard(app, index=0, panel='old_question')
        self.assertFalse(app.session_state.messages[1]['clarification']['answered'])
        app.session_state['blocked'] = True
        self.keyboard(app, index=0)
        self.assertFalse(app.session_state.messages[1]['clarification']['answered'])
