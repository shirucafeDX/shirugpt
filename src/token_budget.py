"""Do not move data to the LLM when computation can be moved to the data.

推定値はUTF-8バイト数/3のヒューリスティック。モデル実測とは区別して記録する。
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math

import pandas as pd

from .result_csv import csv_result


def estimate_tokens(value) -> int:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, default=str, separators=(',', ':'))
    return math.ceil(len(text.encode('utf-8')) / 3)


@dataclass(frozen=True)
class BudgetPolicy:
    max_direct_rows: int = 100
    max_preview_rows: int = 50
    medium_result_rows: int = 1000
    max_data_tokens: int = 8000
    target_data_tokens: int = 4000
    max_history_tokens: int = 16000
    analysis_timeout_seconds: int = 3

    def __post_init__(self):
        if (self.max_direct_rows < 0 or self.max_preview_rows < 1
                or self.medium_result_rows < self.max_direct_rows
                or not 256 <= self.target_data_tokens <= self.max_data_tokens
                or self.max_history_tokens < 1024 or self.analysis_timeout_seconds < 1):
            raise ValueError('分析の行数・トークン上限・タイムアウト設定が不正です。')

    @classmethod
    def from_settings(cls, settings):
        return cls(**{name: getattr(settings, name, field.default) for name, field in cls.__dataclass_fields__.items()})


class TokenBudgetManager:
    def __init__(self, policy: BudgetPolicy | None = None):
        self.policy = policy or BudgetPolicy()

    def schema(self, frame: pd.DataFrame, *, truncated=False, reason='schema_first'):
        result = {'delivery': 'schema', 'row_count': len(frame), 'column_count': len(frame.columns),
                  'columns': [{'name': str(c), 'dtype': str(frame[c].dtype)} for c in frame.columns],
                  'source_truncated': bool(truncated), 'complete': False, 'sent_row_count': 0,
                  'rows_omitted': len(frame), 'reduction_reason': reason,
                  'size_class': ('small' if len(frame) <= self.policy.max_direct_rows else
                                 'medium' if len(frame) <= self.policy.medium_result_rows else 'large'),
                  'details_available_via': 'analyze_result。まずdescribe_data、次に必要な集計・比較・絞り込みを指定する。',
                  'scope': '取得済み範囲のみ。LIMITや取得上限で除かれた行は分析できない。'}
        while estimate_tokens(result) > self.policy.target_data_tokens and result['columns']:
            result['columns'].pop()
            result['columns_omitted'] = len(frame.columns) - len(result['columns'])
        return result

    def initial_result(self, frame, *, truncated=False, schema_only=False):
        if schema_only or len(frame) > self.policy.max_direct_rows:
            return self.schema(frame, truncated=truncated)
        result = csv_result(frame, truncated=truncated)
        if estimate_tokens(result) <= self.policy.max_data_tokens:
            return result
        return self.schema(frame, truncated=truncated, reason='token_budget')

    def table(self, frame, *, truncated=False):
        """分析済み表の返却。セルは切らず、返す行数をトークン予算まで減らす。"""
        cap = min(len(frame), self.policy.max_preview_rows)
        before = estimate_frame_tokens(frame)
        full = csv_result(frame.iloc[:cap], truncated=truncated)
        limit = self.policy.max_data_tokens
        target = self.policy.target_data_tokens if estimate_tokens(full) > limit else limit
        while cap and estimate_tokens(full) > target:
            cap //= 2
            full = csv_result(frame.iloc[:cap], truncated=truncated)
        if estimate_tokens(full) > limit or (not cap and len(frame)):
            result = self.schema(frame, truncated=truncated, reason='token_budget')
            result['note'] = '1行が予算に収まりません。列・条件を絞ってください。'
            return result
        full.update(row_count=len(frame), sent_row_count=cap, rows_omitted=len(frame) - cap,
                    complete=cap == len(frame))
        if cap < len(frame):
            full['reduction_reason'] = 'preview_row_limit_or_token_budget'
        full['estimated_tokens_before'] = before
        # メタデータ追加後も上限を超えない。予算の小さい設定にも対応。
        if estimate_tokens(full) > limit:
            return self.schema(frame, truncated=truncated, reason='token_budget')
        return full


def estimate_frame_tokens(frame):
    # 巨大な転送JSONを組み立てず、データ量の概算をログに使う。
    return max(estimate_tokens([str(c) for c in frame.columns]),
               math.ceil(int(frame.memory_usage(index=False, deep=True).sum()) / 3))


def bound_history(request, ctx):
    """原本と署名付きmodel Partを変えず、送信履歴を予算内にする。"""
    from google.genai import types
    from . import context_memory
    policy = BudgetPolicy.from_settings(ctx.settings)
    def size(contents):
        return estimate_tokens([c.model_dump(exclude_none=True) for c in contents])
    original = size(request)
    reduced = list(request)
    if original > policy.max_history_tokens:
        # 旧形式・未読の巨大結果も、まずメタデータに縮小。原本はcontext_state/cacheに残る。
        for index, content in enumerate(reduced):
            parts = []
            changed = False
            for part in content.parts or []:
                fr = part.function_response
                payload = fr.response if fr else None
                if isinstance(payload, dict) and estimate_tokens(payload) > 300:
                    result = payload.get('result', {})
                    compact = {key: payload[key] for key in ('result_id', 'source_result_id', 'source_truncated', 'current_question_state', 'context_note_warning') if key in payload}
                    if payload.get('error'):
                        compact['error'] = str(payload['error'])[:1000]
                    if isinstance(result, dict):
                        compact['result'] = {key: result[key] for key in ('row_count', 'column_count', 'source_truncated', 'size_class') if key in result}
                        compact['result']['columns'] = [{'name': c.get('name'), 'dtype': c.get('dtype')} for c in result.get('columns', [])[:20]]
                    compact['history_data_omitted'] = True
                    compact['details_available_via'] = 'analyze_result / recall_context。省略データを読んだとみなさない。'
                    parts.append(part.model_copy(update={'function_response': fr.model_copy(update={'response': compact})}))
                    changed = True
                else:
                    parts.append(part)
            if changed:
                reduced[index] = content.model_copy(update={'parts': parts})
    omitted = 0
    if size(reduced) > policy.max_history_tokens:
        starts = [i for i, c in enumerate(reduced) if context_memory.is_question(c)]
        if len(starts) > 1:
            # 古いターンから除き、可能な範囲で直近の原文を維持。最新の依頼は切らない。
            state = context_memory.state_for(ctx)
            past = state['turns'][:-1]
            memories = [{'turn_id': t['id'], 'question': t['question'][:200],
                         'summary': context_memory.facts_only(state, t.get('summary', '')[:1000])}
                        for t in past[-3:]]
            prefix = types.Content(role='user', parts=[types.Part.from_text(text=json.dumps({
                'history_budget_applied': True, 'recent_memories': memories,
                'current_question_state': context_memory.active_questions(state),
                'details_available_via': 'recall_context。省略された過去の事実は確認してから使う。'}, ensure_ascii=False))])
            for start in starts[1:]:
                candidate = [prefix, types.Content(role='model', parts=[types.Part.from_text(text='必要な過去の詳細は保存済み履歴から参照します。')]), *reduced[start:]]
                omitted = start
                if size(candidate) <= policy.max_history_tokens:
                    reduced = candidate
                    break
    after = size(reduced)
    if after > policy.max_history_tokens:
        raise ValueError('現在の依頼または署名付きツール呼び出しが会話の入力予算を超えています。質問・SQLを絞るか新しいチャットで続けてください。')
    ctx.context_state.setdefault('last_request', {}).update(
        estimated_tokens_before=original, estimated_tokens_after=after,
        history_contents_omitted=omitted, history_budget_applied=original > policy.max_history_tokens,
        sent_chars=len(json.dumps([c.model_dump(exclude_none=True) for c in reduced], ensure_ascii=False, default=str, separators=(',', ':'))))
    return reduced
