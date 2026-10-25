"""引き継ぎメモ・原本保全・選択的参照。外部APIは使わない。"""
import io
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock
from zipfile import ZipFile, ZIP_DEFLATED

import pandas as pd
from google.genai import types

from src import context_memory as memory
from src.agent import GeminiAgent, ToolContext, _sql_key, build_declarations, ConfirmationPending
from src.bq_tools import QueryRun, ConfirmationRequired
from src.chat_history import pack, unpack
from src.summarize import summarize_dataframe


def text(role, value):
    return types.Content(role=role, parts=[types.Part.from_text(text=value)])


def reply(name=None, args=None, value='回答', call_id='call-1'):
    parts = [types.Part(function_call=types.FunctionCall(name=name, args=args or {}, id=call_id),
                        thought_signature=b'original-signature')] if name else [types.Part.from_text(text=value)]
    return types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role='model', parts=parts), finish_reason='STOP')])


class MemoryTests(unittest.TestCase):
    def setUp(self):
        self.settings = SimpleNamespace(max_tool_iterations=8, gemini_model='test-model',
                                        gcp_project='test', allowed_datasets=('data',))
        self.ctx = ToolContext(tools=Mock(), settings=self.settings)
        self.agent = GeminiAgent(self.settings, client=Mock())
        self.contents = [text('user', '今月の店舗別の売上を教えて')]
        self.frame = pd.DataFrame({'店舗': [f'店舗{i}' for i in range(600)], '売上': range(600)})
        self.run = QueryRun('SELECT * FROM table LIMIT 5000', 'SELECT * FROM table', 10, 10,
                            self.frame, summarize_dataframe(self.frame, sample_rows=500), notes=['取得済み範囲のみ'], truncated=True)
        self.ctx.tools.run_query.return_value = self.run
        self.sql = 'SELECT * FROM table'
        self.rid = memory.result_id_for('run_query', {'sql': self.sql})

    def note(self, refs=None, summary='今月の店舗別売上。店舗599は599円。取得上限による切り捨てに注意。'):
        return {'summary': summary, 'references': refs if refs is not None else [self.rid]}

    def query_then_analyze(self, note=None):
        replies = [reply('run_query', {'sql': self.sql, 'purpose': '店舗別売上'}),
                   reply('list_datasets', {'context_note': note or self.note()}), reply(value='回答')]
        self.ctx.tools.list_datasets.return_value = {'datasets': ['data']}
        self.agent._generate = Mock(side_effect=replies)
        self.agent.run(self.contents, self.ctx)
        return self.agent._generate.call_args_list

    def test_only_read_results_shrink_and_archive_and_signatures_remain_exact(self):
        calls = self.query_then_analyze()
        before = calls[1].args[0]
        after = calls[2].args[0]
        self.assertEqual(before[2].parts[0].function_response.response['result']['sample_rows'][499]['売上'], 499)
        short = after[2].parts[0].function_response
        self.assertEqual(short.id, 'call-1')
        self.assertEqual(short.response['result_id'], self.rid)
        self.assertEqual(short.response['warnings'], ['取得済み範囲のみ'])
        self.assertNotIn('sample_rows', short.response)
        self.assertLess(len(memory.encoded([c.model_dump() for c in after])), len(memory.encoded([c.model_dump() for c in before])) / 2)
        self.assertEqual(self.contents[2].parts[0].function_response.response['result']['sample_rows'][499]['売上'], 499)
        self.assertIs(after[1], self.contents[1])
        self.assertEqual(after[1].parts[0].thought_signature, b'original-signature')
        self.assertEqual(after[1].parts[0].function_call.args['sql'], self.sql)
        self.assertEqual(self.ctx.context_state['last_request']['compacted_results'], 1)

    def test_bad_note_does_not_discard_or_block_tool(self):
        calls = self.query_then_analyze(self.note(['outside-this-chat']))
        self.assertIn('sample_rows', calls[2].args[0][2].parts[0].function_response.response['result'])
        self.assertIn('context_note_warning', self.contents[4].parts[0].function_response.response)
        self.ctx.tools.list_datasets.assert_called_once()

    def test_finish_returns_answer_and_memos_in_existing_generation(self):
        self.agent._generate = Mock(side_effect=[reply('run_query', {'sql': self.sql}), reply('finish_answer', {
            'answer': '店舗599は599円です。', 'context_note': self.note(),
            'conversation_summary': '目的は店舗別売上。今月・円単位。結果ID: ' + self.rid,
        })])
        result = self.agent.run(self.contents, self.ctx)
        self.assertEqual(result.text, '店舗599は599円です。')
        self.assertEqual(self.agent._generate.call_count, 2)
        self.ctx.tools.run_query.assert_called_once()
        turn = self.ctx.context_state['turns'][0]
        self.assertEqual(turn['summary'], self.note()['summary'])
        self.assertIn(self.rid, turn['overview'])
        self.assertEqual(turn['result_ids'], [self.rid])
        # 後続質問に過去のSQLと明細を再送しない。
        self.contents.append(text('user', '最大の店舗は？'))
        request = memory.encoded([c.model_dump() for c in memory.build_request(self.contents, self.ctx)])
        self.assertIn('599円', request)
        self.assertNotIn(self.sql, request)
        self.assertNotIn('sample_rows', request)

    def test_finish_rejects_missing_or_unknown_memo_and_can_recover(self):
        self.agent._generate = Mock(side_effect=[reply('finish_answer', {'answer': '不完全', 'conversation_summary': '概要'}),
                                                reply(value='通常の回答にフォールバック')])
        self.assertIn('フォールバック', self.agent.run(self.contents, self.ctx).text)
        self.assertIn('error', self.contents[2].parts[0].function_response.response)
        self.assertFalse(self.ctx.context_state['turns'][0]['summary'])

    def test_two_recent_turns_and_older_cumulative_summary(self):
        for i in range(5):
            if i:
                self.contents.append(text('user', f'質問{i}'))
            self.contents.append(text('model', f'回答{i}'))
            memory.finish_turn(self.contents, self.ctx, f'要点{i}', f'累積要約0から{i}')
        self.contents.append(text('user', '最新の質問'))
        request = memory.build_request(self.contents, self.ctx)
        first = json.loads(request[0].parts[0].text)
        self.assertEqual(first['older_conversation_summary'], '累積要約0から2')
        self.assertEqual(first['older_turn_count'], 3)
        self.assertIn('要点3', request[3].parts[0].text)
        self.assertIn('要点4', request[5].parts[0].text)
        self.assertEqual(request[-1].parts[0].text, '最新の質問')

    def test_legacy_or_plain_answer_without_memo_keeps_original(self):
        self.query_then_analyze(self.note(['unknown']))
        old = list(self.contents)
        self.contents.append(text('user', '追加質問'))
        request = memory.build_request(self.contents, self.ctx)
        self.assertEqual(request[:len(old)], old)

    def test_recall_selects_columns_and_rows_beyond_original_sample_without_queries(self):
        self.query_then_analyze()
        result = memory.recall(self.ctx, {'result_id': self.rid, 'columns': ['売上'], 'offset': 550, 'limit': 2})
        self.assertEqual(result['result']['sample_rows'], [{'売上': 550}, {'売上': 551}])
        self.assertTrue(result['source_truncated'])
        self.assertEqual(result['total_row_count'], 600)
        self.assertNotIn('sql', result)
        with_sql = memory.recall(self.ctx, {'result_id': self.rid, 'include_sql': True, 'limit': 1})
        self.assertEqual(with_sql['sql'], self.run.sql)
        self.ctx.tools.run_query.assert_called_once()
        for args in [{'result_id': 'foreign'}, {'result_id': self.rid, 'limit': 101},
                     {'result_id': self.rid, 'columns': ['不明']}, {'result_id': self.rid, 'offset': -1},
                     {'result_id': self.rid, 'search': '二つの指定'}]:
            with self.subTest(args=args), self.assertRaises(ValueError):
                memory.recall(self.ctx, args)

    def test_search_and_turn_lookup_are_scoped_to_current_conversation(self):
        self.query_then_analyze()
        self.contents.append(text('user', '前回の話'))
        memory.sync_turns(self.contents, self.ctx)
        matches = memory.recall(self.ctx, {'search': '売上'})['matches']
        turn_id = matches[0]['turn_id']
        self.assertIn(self.rid, [r['result_id'] for r in matches[0]['results']])
        self.assertIn('店舗別', memory.recall(self.ctx, {'turn_id': turn_id})['question'])
        other = ToolContext(tools=Mock(), settings=self.settings)
        with self.assertRaises(ValueError):
            memory.recall(other, {'result_id': self.rid})

    def test_old_result_migrates_to_query_cache_and_samples_are_paginated_if_cache_missing(self):
        self.ctx.query_cache[_sql_key(self.sql)] = self.run
        model = reply('run_query', {'sql': self.sql}).candidates[0].content
        self.contents.extend([model, types.Content(role='user', parts=[types.Part.from_function_response(
            name='run_query', response={'executed_sql': self.run.sql, 'result': self.run.summary})])])
        memory.sync_turns(self.contents, self.ctx)
        result = memory.recall(self.ctx, {'result_id': 'legacy_2_0', 'offset': 550, 'limit': 1})
        self.assertEqual(result['result']['sample_rows'][0]['売上'], 550)
        self.ctx.query_cache.clear()
        result = memory.recall(self.ctx, {'result_id': 'legacy_2_0', 'limit': 1})
        self.assertEqual(len(result['result']['sample_rows']), 1)
        self.assertTrue(result['only_saved_samples_available'])
        self.assertNotIn('executed_sql', result)

    def test_confirmation_resume_keeps_signed_call_and_uses_only_one_generation_for_finish(self):
        self.ctx.tools.run_query.side_effect = [ConfirmationRequired(self.sql, 20, 10, []), self.run]
        self.agent._generate = Mock(side_effect=[reply('run_query', {'sql': self.sql}), reply('finish_answer', {
            'answer': '回答', 'conversation_summary': '全体', 'context_note': self.note()})])
        pending = self.agent.run(self.contents, self.ctx)
        self.assertIsInstance(pending, ConfirmationPending)
        self.assertEqual(self.contents[-1].parts[0].thought_signature, b'original-signature')
        self.ctx.decisions[pending.key] = 'approved'
        self.assertEqual(self.agent.run(self.contents, self.ctx).text, '回答')
        self.assertEqual(self.agent._generate.call_count, 2)
        self.assertTrue(self.ctx.tools.run_query.call_args.kwargs['approved'])
        self.assertEqual(len(self.ctx.context_state['turns']), 1)

    def test_memory_persists_and_old_saves_and_changed_model_reset_indices(self):
        self.query_then_analyze()
        messages = [{'role': 'user', 'text': '質問'}, {'role': 'assistant', 'text': '回答'}]
        raw = pack(messages, self.contents, self.ctx, None)
        restored = ToolContext(tools=Mock(), settings=self.settings)
        unpack(raw, restored)
        self.assertEqual(restored.context_state, self.ctx.context_state)
        self.assertEqual(memory.recall(restored, {'result_id': self.rid, 'offset': 550, 'limit': 1})['result']['sample_rows'][0]['売上'], 550)
        changed = ToolContext(tools=Mock(), settings=SimpleNamespace(**{**vars(self.settings), 'gemini_model': 'different'}))
        unpack(raw, changed)
        self.assertEqual(changed.context_state, {})
        output = io.BytesIO()
        with ZipFile(io.BytesIO(raw)) as source, ZipFile(output, 'w', ZIP_DEFLATED) as target:
            for name in source.namelist():
                data = source.read(name)
                if name == 'conversation.json':
                    payload = json.loads(data)
                    del payload['context']['context_state']
                    data = json.dumps(payload).encode()
                target.writestr(name, data)
        unpack(output.getvalue(), restored)
        self.assertEqual(restored.context_state, {})

    def test_recall_through_agent_reads_saved_rows_and_finishes_without_requery(self):
        args = {'result_id': self.rid, 'columns': ['売上'], 'offset': 550, 'limit': 1,
                'context_note': self.note()}
        recall_id = memory.result_id_for('recall_context', args)
        self.agent._generate = Mock(side_effect=[
            reply('run_query', {'sql': self.sql}), reply('recall_context', args),
            reply('finish_answer', {'answer': '店舗550の売上は550円です。',
                                   'context_note': self.note([self.rid, recall_id]),
                                   'conversation_summary': '今月の店舗別売上。店舗550は550円。'})])
        self.assertIn('550円', self.agent.run(self.contents, self.ctx).text)
        self.ctx.tools.run_query.assert_called_once()
        self.assertEqual(self.agent._generate.call_count, 3)
        request = self.agent._generate.call_args_list[2].args[0]
        recalled = request[4].parts[0].function_response.response
        self.assertEqual(recalled['result']['sample_rows'], [{'売上': 550}])
        self.assertEqual(recalled['statistics_scope'], 'returned_rows_only')

    def test_open_questions_close_and_only_latest_state_is_sent(self):
        question = '対象期間はいつか？'
        note = self.note([], '- 店舗Aの売上は100万円。')
        note['open_questions'] = [{'id': 'period', 'question': question}]
        self.assertIsNone(memory.accept_note(self.ctx, note))
        request = memory.build_request(self.contents, self.ctx)
        snapshot = json.loads(request[-1].parts[-1].text)
        self.assertEqual(snapshot['open_questions'], [{'id': 'period', 'question': question}])
        closing = self.note([], '- 対象店舗はA。')
        closing['resolved_questions'] = [{'id': 'period', 'conclusion': '対象期間は今月。ユーザーが指定。'}]
        self.assertIsNone(memory.accept_note(self.ctx, closing))
        request = memory.build_request(self.contents, self.ctx)
        snapshot = json.loads(request[-1].parts[-1].text)
        self.assertEqual(snapshot['open_questions'], [])
        self.assertNotIn(question, memory.encoded(snapshot))
        self.assertIn('対象期間は今月', memory.factual_summary(closing))
        self.assertEqual(self.ctx.context_state['questions']['period']['status'], 'resolved')
        self.assertEqual(self.ctx.context_state['questions']['period']['question'], question)

    def test_invalid_updates_are_atomic_and_closed_questions_cannot_reopen(self):
        opened = self.note([], '- 確定事実。')
        opened['open_questions'] = [{'id': 'period', 'question': '対象期間はいつか？'}]
        self.assertIsNone(memory.accept_note(self.ctx, opened))
        invalid = self.note([])
        invalid['open_questions'] = [{'id': 'new', 'question': '店舗はどこか？'}]
        invalid['resolved_questions'] = [{'id': 'unknown', 'conclusion': '判明した'}]
        self.assertIsNotNone(memory.accept_note(self.ctx, invalid))
        self.assertNotIn('new', self.ctx.context_state['questions'])
        closed = self.note([])
        closed['resolved_questions'] = [{'id': 'period', 'conclusion': '今月と確認。'}]
        self.assertIsNone(memory.accept_note(self.ctx, closed))
        self.assertIsNotNone(memory.accept_note(self.ctx, opened))
        opened['open_questions'][0]['id'] = 'another_id'
        self.assertIsNotNone(memory.accept_note(self.ctx, opened))
        duplicate = self.note([])
        duplicate['open_questions'] = [{'id': 'new', 'question': '店舗はどこか？'}] * 2
        self.assertIsNotNone(memory.accept_note(self.ctx, duplicate))

    def test_omitted_or_empty_updates_keep_unresolved_questions_and_bounds_apply(self):
        note = self.note([])
        note['open_questions'] = [{'id': f'q{i}', 'question': f'条件{i}は何か？'} for i in range(12)]
        self.assertIsNone(memory.accept_note(self.ctx, note))
        self.assertIsNone(memory.accept_note(self.ctx, self.note([])))
        empty = self.note([])
        empty.update(open_questions=[], resolved_questions=[])
        self.assertIsNone(memory.accept_note(self.ctx, empty))
        self.assertEqual(len(memory.active_questions(self.ctx.context_state)['open_questions']), 12)
        extra = self.note([])
        extra['open_questions'] = [{'id': 'too_many', 'question': '追加条件は何か？'}]
        self.assertIsNotNone(memory.accept_note(self.ctx, extra))
        for update in [None, ['invalid'], [{'id': 'bad id', 'question': '質問'}],
                       [{'id': 'new', 'question': 'a' * 301}]]:
            note = self.note([])
            note['open_questions'] = update
            with self.subTest(update=update):
                self.assertIsNotNone(memory.accept_note(self.ctx, note))

    def test_resolved_question_removed_from_recent_older_and_tool_memos(self):
        question = '原因は来客数か？'
        initial = self.note([], '- 売上は100万円。')
        initial['open_questions'] = [{'id': 'cause', 'question': question}]
        memory.accept_note(self.ctx, initial)
        state = memory.state_for(self.ctx)
        # 旧形式のメモがある場合も独立した疑問の行を引き継がない。
        for i in range(4):
            if i:
                self.contents.append(text('user', f'質問{i}'))
            self.contents.append(text('model', '売上は100万円。'))
            memory.finish_turn(self.contents, self.ctx, f'- 売上は100万円。\n- 未確認点: {question}',
                               f'- 売上は100万円。\n- {question}')
        self.contents.append(text('user', '続けて'))
        rid = memory.register_result(self.ctx, 'list_tables', {}, {'tables': []})['result_id']
        state['results'][rid]['note'] = f'- 売上は100万円。\n- 未確認点: {question}'
        state['turns'][2]['result_ids'].append(rid)
        closing = self.note([], '- 来客数は前月比20%減。')
        closing['resolved_questions'] = [{'id': 'cause', 'conclusion': '来客数の減少と確認。'}]
        memory.accept_note(self.ctx, closing)
        wire = memory.encoded([c.model_dump() for c in memory.build_request(self.contents, self.ctx)])
        self.assertNotIn(question, wire)
        self.assertIn('売上は100万円', wire)
        self.assertIn(question, state['turns'][0]['overview'])
        historical = memory.recall(self.ctx, {'turn_id': state['turns'][0]['id']})
        self.assertTrue(historical['historical_record'])
        self.assertEqual(historical['current_question_state']['open_questions'], [])
        # 単語が重なっていても確定事実の行は除外しない。
        self.assertEqual(memory.facts_only(state, '- 原因は来客数か？を検証し、減少と確定。'),
                         '- 原因は来客数か？を検証し、減少と確定。')

    def test_new_turn_can_ask_same_wording_for_different_analysis_with_new_id(self):
        memory.sync_turns(self.contents, self.ctx)
        opened = self.note([])
        opened['open_questions'] = [{'id': 'period', 'question': '対象期間はいつか？'}]
        memory.accept_note(self.ctx, opened)
        closing = self.note([])
        closing['resolved_questions'] = [{'id': 'period', 'conclusion': '今月と確認。'}]
        memory.accept_note(self.ctx, closing)
        self.contents.extend([text('model', '今月の売上です。'), text('user', '別の指標も分析したい')])
        memory.sync_turns(self.contents, self.ctx)
        opened['open_questions'][0]['id'] = 'new_analysis_period'
        self.assertIsNone(memory.accept_note(self.ctx, opened))
        current = memory.active_questions(self.ctx.context_state)['open_questions']
        self.assertEqual([q['id'] for q in current], ['new_analysis_period'])

    def test_same_turn_signed_question_history_kept_but_latest_snapshot_is_empty(self):
        self.ctx.tools.list_datasets.return_value = {'datasets': ['data']}
        opened = self.note([], '- 今月を分析。')
        opened['open_questions'] = [{'id': 'count', 'question': '店舗数は何件か？'}]
        closed = self.note([], '- 店舗一覧を確認。')
        closed['resolved_questions'] = [{'id': 'count', 'conclusion': '店舗一覧は1件。'}]
        self.agent._generate = Mock(side_effect=[
            reply('list_datasets', {'context_note': opened}),
            reply('list_datasets', {'context_note': closed}), reply(value='回答')])
        self.agent.run(self.contents, self.ctx)
        last = self.agent._generate.call_args_list[2].args[0]
        self.assertEqual(last[1].parts[0].thought_signature, b'original-signature')
        self.assertEqual(last[1].parts[0].function_call.args['context_note']['open_questions'], opened['open_questions'])
        self.assertEqual(json.loads(last[-1].parts[-1].text)['open_questions'], [])

    def test_closed_state_and_conclusion_survive_finish_and_history_roundtrip(self):
        opened = self.note([], '- 期間を確認。')
        opened['open_questions'] = [{'id': 'period', 'question': '期間は何月か？'}]
        memory.accept_note(self.ctx, opened)
        closing = self.note([], '- 対象は店舗A。')
        closing['resolved_questions'] = [{'id': 'period', 'conclusion': '期間は今月。'}]
        self.agent._generate = Mock(return_value=reply('finish_answer', {
            'answer': '今月について回答します。', 'context_note': closing,
            'conversation_summary': '- 対象店舗はA。期間は今月。'}))
        self.agent.run(self.contents, self.ctx)
        self.assertIn('期間は今月', self.ctx.context_state['turns'][0]['summary'])
        raw = pack([], self.contents, self.ctx, None)
        restored = ToolContext(tools=Mock(), settings=self.settings)
        unpack(raw, restored)
        self.assertEqual(memory.active_questions(restored.context_state)['open_questions'], [])
        self.assertEqual(restored.context_state['questions']['period']['conclusion'], '期間は今月。')
        self.assertIsNotNone(memory.accept_note(restored, opened))

    def test_declarations_include_notes_and_no_extra_model_tool_needed_for_notes(self):
        declarations = {d.name: d for d in build_declarations()}
        self.assertIn('finish_answer', declarations)
        self.assertIn('recall_context', declarations)
        for declaration in declarations.values():
            self.assertIn('context_note', declaration.parameters.properties)
        self.assertIn('context_note', declarations['finish_answer'].parameters.required)
        for declaration in declarations.values():
            note = declaration.parameters.properties['context_note']
            self.assertIn('open_questions', note.required)
            self.assertIn('resolved_questions', note.required)


if __name__ == '__main__':
    unittest.main()
