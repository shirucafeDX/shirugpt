"""思考設定の分離・API指定・会話の継続と費用計算を外部接続なしで確認。"""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from google.genai import types
from streamlit.testing.v1 import AppTest

from src.agent import GeminiAgent, ToolContext, ConfirmationPending, SYSTEM_INSTRUCTION
from src.chat_history import pack, unpack
from src.config import gemini_cost_usd
from src.usage_log import MonthlyUsage, UsageLogger
from test_agent_and_guard import response


def settings():
    return SimpleNamespace(gemini_model='gemini-3.8-flash', vertex_location='global',
                           thinking_budget=10000, max_tool_iterations=3,
                           max_bytes_billed=10 * 1024**3,
                           gcp_project='test', allowed_datasets=('data',),
                           logging_enabled=False, chat_history_bucket='')


def user(text):
    return types.Content(role='user', parts=[types.Part.from_text(text=text)])


class ThinkingTests(unittest.TestCase):
    def test_medium_is_fixed_without_router_and_high_is_isolated_per_conversation(self):
        client = Mock()
        client.models.generate_content.side_effect = [response('件数です'), response('詳しい分析です'), response('比較です')]
        agent = GeminiAgent(settings(), client=client)
        normal = ToolContext(tools=Mock(), settings=agent.settings)
        high = ToolContext(tools=Mock(), settings=agent.settings, thinking_mode='high')
        contents = [user('今月の来店数は')]
        agent.run(contents, normal)
        agent.run([user('今月の来店数は')], high)
        contents.append(user('前年比較もお願い'))
        agent.run(contents, normal)
        calls = client.models.generate_content.call_args_list
        self.assertEqual(len(calls), 3)
        for call, expected in zip(calls, ('MEDIUM', 'HIGH', 'MEDIUM')):
            config = call.kwargs['config']
            self.assertEqual(config.thinking_config.thinking_level.value, expected)
            self.assertIsNone(config.thinking_config.thinking_budget)
            self.assertTrue(config.system_instruction.startswith(SYSTEM_INSTRUCTION))
            self.assertIn(f'今回の分析方針（{expected.lower()}）', config.system_instruction)
            self.assertEqual(call.kwargs['model'], 'gemini-3.8-flash')
        self.assertIsNone(agent.thinking_level)
        self.assertEqual(normal.thinking_mode, 'medium')
        self.assertIn('件数です', calls[-1].kwargs['contents'][1].parts[0].text)
        self.assertAlmostEqual(normal.gemini_cost_usd(), 2 * gemini_cost_usd(10, 10, 'gemini-3.8-flash'))

    def test_pending_query_keeps_the_level_used_before_confirmation(self):
        client = Mock()
        client.models.generate_content.return_value = response('前の調査を継続しました')
        agent = GeminiAgent(settings(), client=client)
        ctx = ToolContext(tools=Mock(), settings=agent.settings, active_thinking_level='medium')
        ctx.resume_calls = [types.FunctionCall(name='list_datasets', args={})]
        ctx.tools.list_datasets.return_value = {'datasets': []}
        agent.run([user('来店数の合計')], ctx)
        config = client.models.generate_content.call_args.kwargs['config']
        self.assertEqual(config.thinking_config.thinking_level.value, 'MEDIUM')
        client.models.generate_content.assert_called_once()

    def test_history_restores_mode_and_model_change_rebuilds_context_without_old_signatures(self):
        ctx = ToolContext(tools=Mock(), settings=settings(), thinking_mode='high', session_gemini_cost_usd=0.2)
        contents = [user('来店数は'), types.Content(role='model', parts=[types.Part(text='10件です', thought_signature=b'old')])]
        messages = [{'role': 'user', 'text': '来店数は'}, {'role': 'assistant', 'text': '10件です'}]
        saved = pack(messages, contents, ctx, None)
        restored = ToolContext(tools=Mock(), settings=settings())
        unpack(saved, restored)
        self.assertEqual(restored.thinking_mode, 'high')
        self.assertEqual(restored.gemini_cost_usd(), 0.2)
        ctx.settings.gemini_model = 'gemini-2.5-flash'
        ctx.resume_calls = [types.FunctionCall(name='run_query', args={'sql': 'SELECT 1'})]
        saved = pack(messages, contents, ctx, ConfirmationPending('key', 'SELECT 1', 2, 1))
        _, converted, pending = unpack(saved, restored)
        self.assertIsNone(pending)
        self.assertIsNone(restored.resume_calls)
        self.assertEqual(converted[1].parts[0].text, '10件です')
        self.assertIsNone(converted[1].parts[0].thought_signature)

    def test_old_auto_history_is_migrated_to_medium(self):
        ctx = ToolContext(tools=Mock(), settings=settings(), thinking_mode='auto', active_thinking_level='low')
        restored = ToolContext(tools=Mock(), settings=settings())
        unpack(pack([], [], ctx, None), restored)
        self.assertEqual(restored.thinking_mode, 'medium')
        self.assertEqual(restored.active_thinking_level, 'medium')

    def test_usage_queries_and_monthly_cost_keep_model_specific_prices(self):
        self.assertAlmostEqual(gemini_cost_usd(1_000_000, 1_000_000), 2.8)
        self.assertAlmostEqual(gemini_cost_usd(1_000_000, 1_000_000, 'gemini-3.8-flash'), 9)
        s = settings()
        s.logging_enabled, s.log_dataset, s.log_table, s.bq_location = True, 'logs', 'events', None
        client = Mock()
        client.query.return_value.result.return_value = [SimpleNamespace(prompt_tokens=100, output_tokens=20,
                                                                         billed_bytes=0, gemini_usd=0.12)]
        usage = UsageLogger(s, client=client).monthly_usage('me@example.com', None, None)
        self.assertEqual(usage.cost_usd(), 0.12)
        sql = client.query.call_args.args[0]
        self.assertIn("model = 'gemini-3.8-flash'", sql)
        self.assertIn('7.5', sql)
        self.assertAlmostEqual(MonthlyUsage(100, 20, 0).cost_usd(), gemini_cost_usd(100, 20))
