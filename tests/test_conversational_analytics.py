"""New backend contracts: stateful delta requests, returned artifacts, unknown usage."""
import unittest
from datetime import datetime, timezone, timedelta
from dataclasses import replace
from unittest.mock import Mock

from google.api_core.exceptions import NotFound
from google.cloud import geminidataanalytics as ca
from google.genai import types

from src.agent import ToolContext
from src.chat_history import pack, unpack
from src.config import load_settings
from src.conversational_analytics import ConversationalAnalyticsAgent, INSTRUCTION
from src.bq_tools import BQToolError


def message(system):
    return ca.Message(system_message=system)


class AnalyticsTests(unittest.TestCase):
    def setUp(self):
        self.settings = replace(load_settings(), gcp_project='test-project', default_dataset='data',
            allowed_datasets=('data',), analytics_tables=('visits',), analysis_backend='conversational_analytics')
        self.tools = Mock()
        self.tools.list_tables.return_value = {'tables': [{'table': 'visits'}]}
        self.tools.get_table_schema.return_value = {'columns': [{'name': 'connected_id', 'type': 'STRING'}]}
        self.agents, self.chat = Mock(), Mock()
        self.chat.create_conversation.return_value = ca.Conversation(name='projects/test-project/locations/global/conversations/c1')
        self.backend = ConversationalAnalyticsAgent(self.settings, agent_client=self.agents, chat_client=self.chat)
        self.ctx = ToolContext(tools=self.tools, settings=self.settings, usage_logger=Mock())
        self.contents = [types.Content(role='user', parts=[types.Part.from_text(text='最初の質問')])]

    def run_turn(self, events):
        self.ctx.start_turn()
        self.chat.chat.return_value = iter(events)
        return self.backend.run(self.contents, self.ctx)

    def test_second_question_sends_one_new_message_without_old_transcript_or_csv(self):
        self.run_turn([message({'text': {'text_type': 'FINAL_RESPONSE', 'parts': ['回答1']}})])
        self.contents.append(types.Content(role='user', parts=[types.Part.from_text(text='次の質問')]))
        self.run_turn([message({'text': {'text_type': 'FINAL_RESPONSE', 'parts': ['回答2']}})])
        request = self.chat.chat.call_args.kwargs['request']
        self.assertEqual(len(request.messages), 1)
        self.assertTrue(request.messages[0].user_message.text.startswith('次の質問'))
        self.assertNotIn('回答1', ca.ChatRequest.to_json(request))
        self.assertNotIn('system_instruction', ca.ChatRequest.to_json(request))
        self.chat.create_conversation.assert_called_once()
        self.tools.list_tables.assert_called_once()
        self.tools.get_table_schema.assert_called_once()
        self.assertTrue(self.ctx.turn_events[0]['continued'])
        self.assertIsNone(self.ctx.turn_events[0]['input_tokens'])
        self.ctx.usage_logger.log_gemini_call.assert_not_called()

    def test_confirmed_table_only_and_python_enabled_in_created_agent(self):
        self.agents.get_data_agent.side_effect = NotFound('missing')
        self.run_turn([message({'text': {'text_type': 'FINAL_RESPONSE', 'parts': ['回答']}})])
        agent = self.agents.create_data_agent_sync.call_args.kwargs['data_agent']
        context = agent.data_analytics_agent.published_context
        self.assertEqual(context.system_instruction, INSTRUCTION)
        self.assertEqual(context.datasource_references.bq.table_references[0].table_id, 'visits')
        self.assertTrue(context.options.analysis.python.enabled)
        self.assertEqual(context.options.datasource.big_query_max_billed_bytes, self.settings.max_bytes_billed)

    def test_unknown_table_fails_before_creating_agent_or_querying(self):
        self.tools.list_tables.return_value = {'tables': [{'table': 'different'}]}
        with self.assertRaises(BQToolError):
            self.run_turn([])
        self.agents.create_data_agent_sync.assert_not_called()
        self.chat.chat.assert_not_called()

    def test_results_after_final_response_are_retained_and_sql_is_grouped(self):
        events = [message({'text': {'text_type': 'THOUGHT', 'parts': ['進行説明']}}),
            message({'text': {'text_type': 'FINAL_RESPONSE', 'parts': ['582人']}}),
            message({'group_id': 1, 'data': {'generated_sql': 'SELECT 582 AS users'}}),
            message({'group_id': 2, 'data': {'generated_sql': 'SELECT 1586 AS visits'}}),
            message({'group_id': 1, 'data': {'result': {'name': 'users', 'data': [{'n': 582}],
                'schema': {'fields': [{'name': 'n', 'type_': 'INTEGER'}]}}}}),
            message({'group_id': 2, 'data': {'result': {'name': 'visits', 'data': [{'n': 1586}]}}})]
        answer = self.run_turn(events)
        self.assertEqual(answer.text, '582人')
        self.assertEqual([run.sql for run in self.ctx.turn_runs], ['SELECT 582 AS users', 'SELECT 1586 AS visits'])
        self.assertEqual(self.ctx.turn_runs[0].dataframe.iloc[0]['n'], 582)
        self.assertEqual(len(self.ctx.analytics_state['results']), 2)
        self.assertEqual(self.ctx.turn_events[0]['stream_events'], 6)
        self.assertEqual(self.ctx.turn_prompt_tokens, 0)  # legacy counter; no fabricated billing record
        self.assertIsNone(self.ctx.turn_events[0]['thinking_tokens'])

    def test_saved_history_preserves_conversation_and_original_results(self):
        self.run_turn([message({'text': {'text_type': 'FINAL_RESPONSE', 'parts': ['回答']}})])
        new = ToolContext(tools=self.tools, settings=self.settings, usage_logger=Mock())
        _, restored, _ = unpack(pack([], self.contents, self.ctx, None), new)
        self.assertEqual(new.analytics_state['conversation'], self.ctx.analytics_state['conversation'])
        new.start_turn()
        restored.append(types.Content(role='user', parts=[types.Part.from_text(text='復元後の質問')]))
        self.chat.chat.return_value = iter([message({'text': {'text_type': 'FINAL_RESPONSE', 'parts': ['続き']}})])
        self.backend.run(restored, new)
        self.chat.create_conversation.assert_called_once()

    def test_partial_stream_without_final_response_fails_instead_of_success(self):
        with self.assertRaises(BQToolError):
            self.run_turn([message({'text': {'text_type': 'THOUGHT', 'parts': ['まだ分析中']}})])
        self.assertTrue(self.ctx.turn_events[0]['error'])
        self.ctx.usage_logger.log_analytics_call.assert_called_once()

    def test_all_jobs_without_group_ids_are_recorded_and_charged_once(self):
        started = datetime(2026, 1, 1, tzinfo=timezone.utc)
        self.tools.client.get_job.return_value = Mock(query='SELECT 1', total_bytes_billed=10,
            total_bytes_processed=20, cache_hit=False, started=started, ended=started + timedelta(seconds=1))
        events = [message({'data': {'big_query_job': {'project_id': 'test-project', 'job_id': job}}})
                  for job in ('j1', 'j2', 'j1')]
        events.append(message({'text': {'text_type': 'FINAL_RESPONSE', 'parts': ['回答']}}))
        self.run_turn(events)
        self.assertEqual(self.tools.client.get_job.call_count, 2)
        self.assertEqual(self.ctx.session_billed_bytes, 20)
        self.assertEqual(len([e for e in self.ctx.turn_events if e['event'] == 'analytics_query_job']), 2)

    def test_unrelated_gemini_model_setting_does_not_reset_managed_conversation(self):
        self.run_turn([message({'text': {'text_type': 'FINAL_RESPONSE', 'parts': ['回答']}})])
        new = ToolContext(tools=self.tools, settings=replace(self.settings, gemini_model='different'), usage_logger=Mock())
        unpack(pack([], self.contents, self.ctx, None), new)
        self.assertEqual(new.analytics_state['conversation'], self.ctx.analytics_state['conversation'])

    def test_usage_does_not_display_unreported_tokens_as_zero(self):
        from src.commands import usage_text
        text = usage_text(self.ctx, None, None, None)
        self.assertIn('未取得', text)
        self.assertNotIn('Gemini推定費用', text)

    def test_native_chart_is_rendered_once_with_its_data(self):
        from streamlit.testing.v1 import AppTest
        source = '''
import pandas as pd
from app import render_results
from src.bq_tools import QueryRun
frame = pd.DataFrame({'month':['1','2','3'], 'n':[1,2,3]})
spec = {'mark':'line','data':{'values':frame.to_dict(orient='records')},
        'encoding':{'x':{'field':'month','type':'ordinal'},'y':{'field':'n','type':'quantitative'}}}
events = [{'event':'analytics_call','input_tokens':None},
          {'event':'analytics_message','message':{'system_message':{'chart':{'result':{'vega_config':spec}}}}}]
render_results({'question':'月別推移をグラフにして','execution_events':events,
                'runs':[QueryRun('SELECT 1','SELECT 1',0,0,frame,{})]},0,set())
'''
        app = AppTest.from_string(source).run(timeout=20)
        self.assertFalse(app.exception)
        self.assertEqual(len(app.get('vega_lite_chart')), 1)
        self.assertTrue(any('未取得' in c.value for c in app.caption))


if __name__ == '__main__':
    unittest.main()
