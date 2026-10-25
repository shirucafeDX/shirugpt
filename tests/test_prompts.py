"""ツールの切替に合わせて必要な指示だけを送る。業務上の定義と制約の脱落も確認する。"""
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

from src.agent import GeminiAgent, SYSTEM_INSTRUCTION
from src.prompts import build_system_instruction, TOOL_INSTRUCTION_KEYS
from src.tool_catalog import TOOL_SUMMARIES


class PromptTests(unittest.TestCase):
    def setUp(self):
        self.agent = GeminiAgent(SimpleNamespace(gemini_model='test', thinking_budget=None), client=Mock())

    def prompt(self, selected=None):
        return self.agent._config(selected).system_instruction

    def test_selector_only_keeps_shared_policy_and_no_tool_specific_details(self):
        prompt = self.prompt()
        self.assertEqual(prompt, SYSTEM_INSTRUCTION)
        for phrase in ('読み取り専用', '課金上限', '事実', '日本語', 'Asia/Tokyo',
                       '9時間を加算しない', 'current_question_state.open_questions',
                       '条件確認が必要ならask_clarification', '最終回答はfinish_answer'):
            self.assertIn(phrase, prompt)
        for detail in ('## SQL を書くときの規則', '## 図表と回答用クエリ', 'selection_mode',
                       'conversation_summary', 'scope', 'include_sql', '## 会話の引き継ぎメモ'):
            self.assertNotIn(detail, prompt)
        self.assertEqual(set(TOOL_INSTRUCTION_KEYS), set(TOOL_SUMMARIES))

    def test_query_gets_sql_chart_definitions_and_datetime_rules(self):
        prompt = self.prompt(['run_query'])
        for phrase in ('## SQL を書くときの規則', 'バッククォート', 'GROUP BY', 'DU_id',
                       'planned_attendance', '予約数', '00:00:00', 'TIMESTAMP',
                       '9時間を加算しない', "CURRENT_DATE('Asia/Tokyo')", '最大 2 回',
                       'pie', 'heatmap', 'result_role', 'purpose', 'context_note', '1200文字以内'):
            self.assertIn(phrase, prompt)
        for detail in ('## 確認質問の作成', 'selection_mode', 'conversation_summary', '## 保存済み結果の参照'):
            self.assertNotIn(detail, prompt)

    def test_clarification_gets_form_rules_and_schema_prerequisite(self):
        prompt = self.prompt(['ask_clarification'])
        for phrase in ('## 確認質問の作成', 'selection_mode', 'single', 'multi', '2〜5個',
                       'scope', 'general', '必要はない', '列・型・説明', 'その他', '自由記述'):
            self.assertIn(phrase, prompt)
        self.assertNotIn('## SQL を書くときの規則', prompt)
        self.assertNotIn('## 図表と回答用クエリ', prompt)
        self.assertNotIn('conversation_summary', prompt)

    def test_finish_keeps_answer_rules_and_business_meaning_without_query_mechanics(self):
        prompt = self.prompt(['finish_answer'])
        for phrase in ('conversation_summary', '4000文字以内', '主要な事実', '未確認事項',
                       '前置きや余計な解説', '分からない', '単位', 'DU_id', '予約数',
                       '00:00:00', '9時間を加算しない', 'Markdownの表を重ねて書かない'):
            self.assertIn(phrase, prompt)
        for detail in ('## SQL を書くときの規則', '## 図表と回答用クエリ', 'selection_mode'):
            self.assertNotIn(detail, prompt)

    def test_recall_has_saved_scope_and_pagination_rules(self):
        prompt = self.prompt(['recall_context'])
        for phrase in ('## 保存済み結果の参照', 'include_sql=true', 'offset・limit', '最大100行',
                       'source_truncated', '新しいBigQuery照会ではない', 'DU_id', '9時間を加算しない'):
            self.assertIn(phrase, prompt)
        self.assertNotIn('## SQL を書くときの規則', prompt)
        self.assertIn('action="present"', prompt)
        self.assertIn('filters', prompt)
        self.assertIn('## 図表と回答用クエリ', prompt)

    def test_schema_and_listing_have_only_their_lookup_guidance(self):
        listing = self.prompt(['list_datasets', 'list_tables', 'get_table_schema'])
        self.assertIn('datasetに必ず指定', listing)
        self.assertIn('3〜4テーブルに絞り', listing)
        for detail in ('## SQL を書くときの規則', '## 図表と回答用クエリ', 'selection_mode',
                       'conversation_summary', 'planned_attendance'):
            self.assertNotIn(detail, listing)

    def test_shared_blocks_are_deduplicated_and_order_is_stable(self):
        selected = ['run_query', 'finish_answer', 'ask_clarification']
        prompt = self.prompt(selected)
        for heading in ('## 定義', '## 日時・タイムゾーンの扱い', '## 会話の引き継ぎメモ'):
            self.assertEqual(prompt.count(heading), 1)
        self.assertEqual(prompt, self.prompt(list(reversed(selected))))
        self.assertTrue(prompt.startswith(SYSTEM_INSTRUCTION))

    def test_switching_and_invalid_saved_selection_do_not_leak_old_prompts(self):
        query = self.prompt(['run_query'])
        clarification = self.prompt(['ask_clarification'])
        again = self.prompt(['run_query'])
        self.assertIn('## SQL を書くときの規則', query)
        self.assertNotIn('## SQL を書くときの規則', clarification)
        self.assertEqual(again, query)
        for invalid in (None, [], ['unknown'], ['run_query'] * 2, list(TOOL_SUMMARIES)):
            self.assertEqual(self.prompt(invalid), SYSTEM_INSTRUCTION)

    def test_deep_analysis_is_independent_and_shared_safety_is_always_present(self):
        high = GeminiAgent(SimpleNamespace(gemini_model='test', thinking_budget=None), client=Mock(), deep_analysis=True)
        for selected in (None, ['run_query'], ['ask_clarification'], ['finish_answer']):
            prompt = high._config(selected).system_instruction
            self.assertEqual(prompt.count('## 詳細分析モード'), 1)
            self.assertIn('課金上限を回避しない', prompt)
            self.assertTrue(prompt.startswith(SYSTEM_INSTRUCTION))
            self.assertNotIn('## 詳細分析モード', self.prompt(selected))


if __name__ == '__main__':
    unittest.main()
