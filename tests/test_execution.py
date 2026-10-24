"""停止要求の伝播、課金記録、会話の復旧をGCP接続なしで確認する。"""
import unittest
from threading import Event
from types import SimpleNamespace
from unittest.mock import Mock

from google.genai import types
from streamlit.testing.v1 import AppTest

from src.agent import GeminiAgent, ToolContext
from src.execution import BackgroundTurn, ExecutionControl, ExecutionStopped
from test_agent_and_guard import response


class ExecutionTests(unittest.TestCase):
    def test_stop_cancels_only_registered_query(self):
        control = ExecutionControl()
        cancelled = Event()
        job = Mock()
        job.cancel.side_effect = lambda **kwargs: cancelled.set()
        control.set_query_job(job)
        control.request_stop()
        self.assertTrue(cancelled.wait(1))
        job.cancel.assert_called_once_with(timeout=10)
        with self.assertRaises(ExecutionStopped):
            control.check()

    def test_query_started_after_stop_is_cancelled(self):
        control = ExecutionControl()
        control.request_stop()
        cancelled = Event()
        job = Mock()
        job.cancel.side_effect = lambda **kwargs: cancelled.set()
        control.set_query_job(job)
        self.assertTrue(cancelled.wait(1))

    def test_completed_query_is_not_cancelled(self):
        control = ExecutionControl()
        job = Mock()
        control.set_query_job(job)
        control.clear_query_job()
        control.request_stop()
        job.cancel.assert_not_called()

    def test_background_turn_stops_at_next_progress_and_restores_runtime(self):
        entered = Event()
        release = Event()
        ctx = ToolContext(tools=Mock(), settings=SimpleNamespace())
        original_tools = ctx.tools

        def run(contents, context, on_progress):
            entered.set()
            release.wait(1)
            on_progress('次の処理')
            self.fail('停止後の処理は実行しない')

        task = BackgroundTurn(SimpleNamespace(run=run), [], ctx)
        task.start()
        self.assertTrue(entered.wait(1))
        task.control.request_stop()
        release.set()
        self.assertTrue(task.done.wait(1))
        self.assertIsInstance(task.error, ExecutionStopped)
        self.assertIs(ctx.tools, original_tools)
        self.assertIsNone(ctx.execution_control)

    def test_stop_after_generation_records_usage_but_does_not_execute_tools(self):
        settings = SimpleNamespace(max_tool_iterations=3)
        ctx = ToolContext(tools=Mock(), settings=settings, usage_logger=Mock())
        ctx.execution_control = ExecutionControl()
        agent = GeminiAgent(settings, client=Mock())

        def generate(contents):
            ctx.execution_control.request_stop()
            return response('', call='run_query')

        agent._generate = generate
        contents = [types.Content(role='user', parts=[types.Part.from_text(text='集計して')])]
        with self.assertRaises(ExecutionStopped):
            agent.run(contents, ctx)
        ctx.tools.run_query.assert_not_called()
        ctx.usage_logger.log_gemini_call.assert_called_once()
        ctx.usage_logger.log_error.assert_not_called()
        self.assertEqual(ctx.session_prompt_tokens, 10)
        self.assertEqual(len(contents), 1)

    def test_stopped_turn_repairs_tool_history_and_clears_pending(self):
        source = '''
from types import SimpleNamespace
from threading import Event
from unittest.mock import Mock, patch
import streamlit as st
from google.genai import types
from app import process_turn
from src.agent import ToolContext
from src.execution import ExecutionControl

ctx = ToolContext(tools=Mock(), settings=SimpleNamespace())
call = types.FunctionCall(name='run_query', args={'sql':'SELECT 1'}, id='call-1')
ctx.resume_calls = [call]
st.session_state.messages = [{'role':'user', 'text':'集計して'}]
st.session_state.contents = [
    types.Content(role='user', parts=[types.Part.from_text(text='集計して')]),
    types.Content(role='model', parts=[types.Part(function_call=call)]),
]
control = ExecutionControl()
control.request_stop()
done = Event()
done.set()
task = SimpleNamespace(control=control, done=done, result=None, error=None,
                       start=lambda:None, message='処理中')
with patch('app.BackgroundTurn', return_value=task), patch('app.save_conversation'), patch('app.st.rerun'):
    process_turn(Mock(), ctx)
st.session_state['resume_cleared'] = ctx.resume_calls is None
'''
        app = AppTest.from_string(source).run(timeout=20)
        self.assertFalse(app.exception)
        self.assertTrue(app.session_state.messages[-1]['is_stopped'])
        self.assertEqual(app.session_state.messages[-1]['errors'], [])
        self.assertTrue(app.session_state['resume_cleared'])
        self.assertIsNone(app.session_state.pending)
        self.assertNotIn('active_turn', app.session_state)
        contents = app.session_state.contents
        self.assertEqual(contents[-2].parts[0].function_response.id, 'call-1')
        self.assertEqual(contents[-1].parts[0].text, '処理を停止しました。')


class PromptExecutionUITests(unittest.TestCase):
    def test_normal_submission_replaces_input_without_duplicate_widgets(self):
        source = '''
from types import SimpleNamespace
from unittest.mock import Mock
import streamlit as st
from app import render_prompt_input, submit_prompt, process_turn
from src.agent import ToolContext, AnswerResult
class Agent:
    def run(self, contents, ctx, on_progress):
        return AnswerResult('確認完了')
if 'ctx' not in st.session_state:
    st.session_state.ctx = ToolContext(tools=Mock(), settings=SimpleNamespace(gemini_model='gemini-3.8-flash', chat_history_bucket=''))
    st.session_state.messages=[]
    st.session_state.contents=[]
    st.session_state.pending=None
ctx=st.session_state.ctx
if st.session_state.get('active_turn') is not None:
    process_turn(Agent(), ctx)
else:
    prompt=render_prompt_input(ctx)
    if prompt:
        submit_prompt(prompt, Agent(), ctx)
'''
        app = AppTest.from_string(source).run(timeout=20)
        app.chat_input[0].set_value('質問です').run()
        self.assertFalse(app.exception)
        self.assertEqual([m['text'] for m in app.session_state.messages], ['質問です', '確認完了'])
        self.assertEqual(len(app.chat_input), 1)
        self.assertFalse(app.chat_input[0].disabled)
        self.assertNotIn('active_turn', app.session_state)
