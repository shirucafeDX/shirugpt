"""数値を含む回答を拒否・書き換えず、結果参照と会話を継続する。"""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
from google.genai import types

from src import analysis_safety as safety, context_memory as memory, result_references
from src.agent import GeminiAgent, ToolContext
from src.bq_tools import QueryRun
from src.chat_history import pack, unpack


def reply(name, args):
    return types.GenerateContentResponse(candidates=[types.Candidate(finish_reason='STOP',
        content=types.Content(role='model', parts=[types.Part(function_call=types.FunctionCall(name=name, args=args))]))])


class PlainAnswerTests(unittest.TestCase):
    def setUp(self):
        self.settings = SimpleNamespace(gcp_project='test', allowed_datasets=('data',),
            gemini_model='test', thinking_budget=None, max_tool_iterations=8)
        self.ctx = ToolContext(settings=self.settings, tools=Mock())
        self.ctx.start_turn()
        self.agent = GeminiAgent(self.settings, client=Mock())

    def finish(self, answer, refs=None):
        return {'answer': answer, 'conversation_summary': answer,
                'context_note': {'summary': answer, 'references': refs or []}}

    def test_numbers_dates_labels_approximations_and_headings_are_not_rejected(self):
        safety.state(self.ctx)['data_requested'] = True
        for answer in ['一人あたりの利用回数です。', '### 1. 課題\n学部3年は2026年7月に60人です。',
                       '29.8%、約3割。五万件。', '2週間、定員5人で試す提案です。',
                       '、'.join(str(n) for n in range(30))]:
            with self.subTest(answer=answer):
                payload = self.agent._dispatch(types.FunctionCall(name='finish_answer', args=self.finish(answer)), self.ctx).function_response.response
                self.assertEqual(payload['answer'], answer)
        self.assertFalse(self.ctx.turn_errors)
        self.assertNotIn('numeric_grounding', safety.snapshot(self.ctx))

    def test_single_numeric_cell_still_requires_the_models_answer(self):
        sql = 'SELECT 123 AS n'
        self.ctx.tools.run_query.return_value = QueryRun(sql, sql, 1, 1, pd.DataFrame({'n': [123]}), {})
        self.agent._generate = Mock(side_effect=[reply('run_query', {'sql': sql}),
                                                 reply('finish_answer', self.finish('123', ['R1']))])
        contents = [types.Content(role='user', parts=[types.Part.from_text(text='数値だけで答えてください。')])]
        self.assertEqual(self.agent.run(contents, self.ctx).text, '123')
        self.assertEqual(self.agent._generate.call_count, 2)
        self.assertNotIn('numeric_grounding', safety.snapshot(self.ctx))

    def test_finish_schema_and_result_payload_have_no_numeric_validation_contract(self):
        self.ctx.tools.run_query.return_value = QueryRun('SELECT 1', 'SELECT 1', 1, 1, pd.DataFrame({'n': [1]}), {})
        payload = self.agent._dispatch(types.FunctionCall(name='run_query', args={'sql': 'SELECT 1'}), self.ctx).function_response.response
        config = self.agent._config(['finish_answer'], self.ctx)
        finish = next(d for t in config.tools for d in t.function_declarations if d.name == 'finish_answer')
        self.assertNotIn('numeric_claims', finish.parameters.properties)
        self.assertNotIn('numeric_claims', config.system_instruction)
        self.assertNotIn('numeric_grounding', payload)
        self.assertEqual(payload['result_ref'], 'R1')

    def test_existing_aliases_survive_saved_conversation_and_resolve_for_recall(self):
        self.ctx.tools.run_query.return_value = QueryRun('SELECT 1', 'SELECT 1', 1, 1, pd.DataFrame({'n': [1]}), {})
        payload = self.agent._dispatch(types.FunctionCall(name='run_query', args={'sql': 'SELECT 1'}), self.ctx).function_response.response
        raw = pack([], [], self.ctx, None)
        restored = ToolContext(settings=self.settings, tools=Mock())
        unpack(raw, restored)
        self.assertEqual(result_references.resolve_ref(restored, 'R1'), payload['result_id'])
        note = result_references.normalize_note(restored, {'summary': '1件', 'references': ['R1']})
        self.assertEqual(note['references'], [payload['result_id']])
        self.assertEqual(memory.saved_result(restored, note['references'][0]).dataframe['n'].iloc[0], 1)
