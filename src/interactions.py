"""Interactions APIの状態IDと差分送信。原本・SQL・画面用表はローカルに残す。"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import time
from email.utils import parsedate_to_datetime

from google import genai
from google.genai import types

from . import context_memory as memory
from .token_budget import estimate_tokens
from .bq_tools import error_code
from .execution import ExecutionStopped
from .interaction_stream import consume
from . import request_audit, tool_wire

logger = logging.getLogger(__name__)
RETENTION_SECONDS = 7 * 86400
TOOL_WIRE_FORMAT = "object-values-v1"
RATE_LIMIT_RETRIES = 3
MAX_RETRY_WAIT_SECONDS = 30


def enabled(settings):
    # 旧クライアントを明示して使用する呼び出し元との互換も維持する。
    return getattr(settings, "gemini_api", "generate_content") == "interactions"


def _dump(contents):
    return [c.model_dump(mode='json', exclude_none=True) for c in contents]


def _digest(contents):
    return hashlib.sha256(memory.encoded(_dump(contents)).encode()).hexdigest()


def _schema(schema):
    """GenAI SchemaをJSON Schemaへ変換。引数名自体は変換しない。"""
    raw = schema.model_dump(mode='json', exclude_none=True)
    def convert(value):
        result = {}
        for key, item in value.items():
            key = {'min_items': 'minItems', 'max_items': 'maxItems', 'any_of': 'anyOf',
                   'min_length': 'minLength', 'max_length': 'maxLength',
                   'additional_properties': 'additionalProperties'}.get(key, key)
            if key == 'type':
                item = item.lower()
            elif key == 'properties':
                item = {name: convert(child) for name, child in item.items()}
            elif key in ('items', 'additionalProperties') and isinstance(item, dict):
                item = convert(item)
            elif key == 'anyOf':
                item = [convert(child) for child in item]
            if key not in ('nullable', 'property_ordering'):
                result[key] = item
        if value.get('nullable') and 'type' in result:
            result['type'] = [result['type'], 'null']
        return result
    return convert(raw)


def tools_from_config(config):
    return [{'type': 'function', 'name': d.name, 'description': d.description or '',
             'parameters': _schema(d.parameters)}
            for tool in config.tools or [] for d in tool.function_declarations or []]


def _input(contents):
    steps = []
    for content in contents:
        text = []
        for part in content.parts or []:
            if part.function_response:
                fr = part.function_response
                if not fr.id:
                    raise ValueError('状態に対応するツール呼び出しIDがありません。')
                steps.append({'type': 'function_result', 'call_id': fr.id, 'name': fr.name,
                              'result': dict(fr.response or {}),
                              'is_error': bool((fr.response or {}).get('error'))})
            elif part.function_call:
                raise ValueError('送信済みのモデル呼び出しを差分に再送できません。')
            elif part.text and not part.thought:
                text.append({'type': 'text', 'text': part.text})
            elif part.inline_data or part.file_data:
                raise ValueError('このチャットの状態送信はテキストとツール結果に対応しています。')
        if text:
            if (content.role == 'model' and steps and steps[-1]['type'] == 'function_result'
                    and steps[-1].get('name') in ('finish_answer', 'ask_clarification')):
                # 画面用の回答・確認文は直前のツール結果に含まれるため二重送信しない。
                continue
            steps.append({'type': 'model_output' if content.role == 'model' else 'user_input', 'content': text})
    return steps


def _wire_input(steps):
    """数値・IDをオブジェクトとして渡す。空配列等でVertexの型が消える問題を回避。"""
    return [{**step, 'result': tool_wire.encode(json.loads(memory.encoded(step['result'])))}
            if step['type'] == 'function_result' else step for step in steps]


def _bootstrap(contents, ctx):
    """期限切れ・旧会話は原本を再送せず、既知の要点と保存結果目録から復元。"""
    memory.sync_turns(contents, ctx)
    state = memory.state_for(ctx)
    starts = [i for i, c in enumerate(contents) if memory.is_question(c)]
    start = starts[-1] if starts else 0
    current = contents[start:]
    summaries = [{'turn_id': turn['id'], 'question': turn['question'][:300],
                  'summary': memory.facts_only(state, (turn.get('summary') or turn.get('answer', ''))[:1200]),
                  'summary_may_be_incomplete': not bool(turn.get('summary'))}
                 for turn in state['turns'][:-1][-3:]]
    ids = list(state['results'])[-24:]
    context = {'restored_from_saved_context': True, 'recent_turns': summaries,
               'summaries_are_model_interpretations': True,
               'current_question_state': memory.active_questions(state),
               'saved_results': memory.catalogue(state, ids),
               'details_available_via': 'recall_context / analyze_result。要点にない数値は保存済み結果で確認する。'}
    # 現在の質問を切らない。期限切れがツール応答待ちでも孤立したcall_idは作らない。
    question = '\n'.join(memory.text_of(c) for c in current if memory.is_question(c))
    latest = next((c for c in reversed(current) if any(p.function_response for p in c.parts or [])), None)
    fresh = [{'tool': p.function_response.name, 'payload': p.function_response.response}
             for p in latest.parts or [] if p.function_response] if latest else []
    context['current_question'] = question
    context['latest_tool_results'] = fresh
    note = state.get('current_note', {})
    if state['turns'] and note.get('turn_id') == state['turns'][-1]['id']:
        context['current_working_note'] = {**note, 'summary': memory.facts_only(state, note['summary'])}
    past = state['turns'][:-1]
    if past and past[-1].get('overview'):
        context['conversation_summary'] = memory.facts_only(state, past[-1]['overview'][:4000])
    # メモや目録のみ減らす。未読の直近結果とユーザー質問は省略しない。
    limit = getattr(ctx.settings, 'max_history_tokens', 16000)
    while estimate_tokens(context) > limit and context['saved_results']:
        context['saved_results'].pop(0)
        context['older_results_available_via_search'] = True
    while estimate_tokens(context) > limit and context['recent_turns']:
        context['recent_turns'].pop(0)
    if estimate_tokens(context) > limit:
        context.pop('conversation_summary', None)
    if estimate_tokens(context) > limit:
        raise ValueError('現在の質問または直近の結果が入力予算を超えています。対象・列を絞ってください。')
    return [{'type': 'user_input', 'content': [{'type': 'text', 'text': memory.encoded(context)}]}]


def _get(value, key, default=None):
    return value.get(key, default) if isinstance(value, dict) else getattr(value, key, default)


def _code(exc):
    return error_code(exc)


def _retry_delay(exc, attempt):
    delay = 2 ** (attempt + 1) + random.uniform(0, 1)
    headers = getattr(getattr(exc, 'response', None), 'headers', {})
    retry_after = headers.get('retry-after') if headers else None
    if retry_after is not None:
        try:
            requested = float(retry_after)
        except (TypeError, ValueError):
            try:
                requested = parsedate_to_datetime(retry_after).timestamp() - time.time()
            except (TypeError, ValueError, OverflowError):
                requested = 0
        if math.isfinite(requested):
            delay = max(delay, requested)
    # 長いRetry-Afterは無視して早期再送せず、利用者へ待機を案内する。
    return delay if delay <= MAX_RETRY_WAIT_SECONDS else None


def _wait_for_retry(ctx, seconds):
    control = ctx.execution_control
    if control is None:
        time.sleep(seconds)
    else:
        control.check()
        control.stopped.wait(seconds)
        control.check()


def _create(api, body, ctx, on_progress, metrics=None):
    """明示的に拒否された429だけを再試行。SQLや結果処理は繰り返さない。"""
    metrics = metrics if metrics is not None else {}
    started = time.monotonic()
    for attempt in range(RATE_LIMIT_RETRIES + 1):
        if ctx.execution_control is not None:
            ctx.execution_control.check()
        try:
            metrics['sent_tool_results'] = request_audit.results(body['input'])
            metrics['request_input_sha256'] = request_audit.digest(body['input'])
            audit_token = request_audit.active_request.set(metrics)
            try:
                result = consume(api.create(**body), ctx, on_progress, metrics, started)
            finally:
                request_audit.active_request.reset(audit_token)
        except Exception as exc:
            if _code(exc) != 429 or attempt == RATE_LIMIT_RETRIES or metrics.get('stream_events', 0):
                raise
            delay = _retry_delay(exc, attempt)
            if delay is None:
                raise
            logger.warning('analysis_metrics %s', memory.encoded({
                'event': 'interaction_rate_limit_retry', 'session_id': ctx.session_id,
                'turn_id': ctx.turn_id, 'attempt': attempt + 1, 'wait_seconds': round(delay, 3)}))
            if on_progress is not None:
                on_progress(f'AIが混み合っています。{math.ceil(delay)}秒待って再試行します（{attempt + 1}/{RATE_LIMIT_RETRIES}）…')
            retry_started = time.monotonic()
            try:
                _wait_for_retry(ctx, delay)
            finally:
                ctx.turn_events.append({'event': 'retry_wait', 'attempt': attempt + 1,
                                        'seconds': round(time.monotonic() - retry_started, 6)})
        else:
            if attempt and on_progress is not None:
                try:
                    on_progress('調査を続けています…')
                except ExecutionStopped:
                    # 応答済みの使用量を記録してから、エージェント側で停止する。
                    pass
            return result


def _response(interaction):
    """既存エージェントの実行・確認・エラー処理へ結果を渡す。思考本文は表示しない。"""
    status = _get(interaction, 'status')
    parts = []
    for step in _get(interaction, 'steps', []) or []:
        kind = _get(step, 'type')
        if kind == 'function_call':
            parts.append(types.Part(function_call=types.FunctionCall(
                name=_get(step, 'name'), args=_get(step, 'arguments', {}), id=_get(step, 'id'))))
        elif kind == 'model_output':
            for item in _get(step, 'content', []) or []:
                if _get(item, 'type') == 'text' and _get(item, 'text'):
                    parts.append(types.Part.from_text(text=_get(item, 'text')))
    finish = 'STOP' if status in ('completed', 'requires_action') else 'OTHER'
    if status in ('incomplete', 'budget_exceeded'):
        finish = 'MAX_TOKENS'
    usage = _get(interaction, 'usage')
    metadata = None
    if usage is not None:
        # Interactionsも出力本文とthoughtsを別に返す。既存経路で合算する。
        output = _get(usage, 'total_output_tokens', 0) or 0
        thoughts = _get(usage, 'total_thought_tokens', None)
        metadata = types.GenerateContentResponseUsageMetadata(
            prompt_token_count=_get(usage, 'total_input_tokens', 0) or 0,
            candidates_token_count=output, thoughts_token_count=thoughts,
            cached_content_token_count=_get(usage, 'total_cached_tokens', 0) or 0,
            total_token_count=_get(usage, 'total_tokens', 0) or 0)
    return types.GenerateContentResponse(candidates=[types.Candidate(
        content=types.Content(role='model', parts=parts), finish_reason=finish,
        finish_message=status if finish != 'STOP' else None)], usage_metadata=metadata)


def generate(client, contents, ctx, config, *, on_progress=None, metrics=None):
    metrics = metrics if metrics is not None else {}
    api = client.interactions
    if isinstance(client, genai.Client):
        # SDK 2.18.1はattempts=0を1へ正規化した後、Interactions側で
        # 「再試行1回」と扱う。生成済みクライアントのretry設定も明示的に無効化する。
        api.sdk_configuration.retry_config = None
    state = ctx.interaction_state
    identity = [ctx.settings.gcp_project, getattr(ctx.settings, 'vertex_location', 'global'), ctx.settings.gemini_model]
    now = time.time()
    cursor = state.get('cursor', 0)
    consistent = (state.get('version') == 1 and state.get('identity') == identity
                  and isinstance(cursor, int) and 0 <= cursor <= len(contents)
                  and state.get('prefix_digest') == _digest(contents[:cursor]))
    expired = now - state.get('chain_started_at', 0) >= RETENTION_SECONDS - 60
    limit = getattr(ctx.settings, 'interaction_max_input_tokens', 32000)
    delta = _input(contents[cursor:]) if consistent else []
    projected = state.get('input_tokens', 0) + state.get('output_tokens', 0) + estimate_tokens(delta)
    near_limit = projected >= int(limit * .8)
    wire_changed = bool(state.get('id')) and state.get('tool_wire_format') != TOOL_WIRE_FORMAT
    restart = not state.get('id') or not consistent or expired or near_limit or state.get('reset_required') or wire_changed
    if restart:
        reason = ('initial_or_legacy' if not state.get('id') else 'history_changed' if not consistent
                  else 'expired' if expired else 'input_budget' if near_limit else 'wire_format_changed' if wire_changed else 'interrupted')
        steps = (_input(contents) if len(contents) == 1 and memory.is_question(contents[0])
                 else _bootstrap(contents, ctx))
        previous = None
    else:
        reason = None
        steps = delta
        previous = state['id']
    if not steps:
        raise ValueError('APIへ追加する質問・ツール結果がありません。')
    if memory.state_for(ctx)['questions']:
        snapshot = memory.active_questions(memory.state_for(ctx))
        # 状態の最新スナップショットを既存のツール結果内に入れる。
        latest_result = next((step for step in reversed(steps) if step['type'] == 'function_result'), None)
        if latest_result is not None:
            data = dict(latest_result['result'])
            data['current_question_state'] = snapshot
            latest_result['result'] = data
    tools = tools_from_config(config)
    thinking = getattr(config.thinking_config, 'thinking_level', None)
    thinking = getattr(thinking, 'value', thinking)
    body = {'model': ctx.settings.gemini_model, 'input': _wire_input(steps), 'store': True, 'stream': True,
            'tools': tools, 'system_instruction': config.system_instruction,
            'generation_config': {'thinking_level': str(thinking or 'low').lower()}}
    if config.temperature is not None:
        body['generation_config']['temperature'] = config.temperature
    if previous:
        body['previous_interaction_id'] = previous
    try:
        interaction = _create(api, body, ctx, on_progress, metrics)
    except Exception as exc:
        if previous and _code(exc) == 404 and not metrics.get('stream_events'):
            # モデルやエンドポイントの404を期限切れと誤認しない。
            try:
                api.get(id=previous)
            except Exception as lookup:
                if _code(lookup) != 404:
                    raise exc
            else:
                raise
            body.pop('previous_interaction_id')
            body['input'] = _wire_input(_bootstrap(contents, ctx))
            interaction = _create(api, body, ctx, on_progress, metrics)
            reason, previous = 'expired_on_server', None
        else:
            raise
    response = _response(interaction)
    interaction_id = _get(interaction, 'id')
    if not interaction_id:
        raise ValueError('Interactions APIの応答に状態IDがありません。')
    expected = [*contents, response.candidates[0].content]
    measured_input = response.usage_metadata.prompt_token_count if response.usage_metadata else None
    measured_output = (((response.usage_metadata.candidates_token_count or 0)
                        + (response.usage_metadata.thoughts_token_count or 0))
                       if response.usage_metadata else None)
    config_estimate = estimate_tokens(config.system_instruction) + estimate_tokens(tools)
    fallback_input = (state.get('input_tokens', 0) + state.get('output_tokens', 0)
                      + max(0, config_estimate - state.get('config_tokens_estimate', 0)) if previous else config_estimate)
    fallback_input += estimate_tokens(body['input'])
    ctx.interaction_state = {'version': 1, 'tool_wire_format': TOOL_WIRE_FORMAT, 'id': interaction_id, 'identity': identity,
                             'cursor': len(expected), 'prefix_digest': _digest(expected),
                             'chain_started_at': state.get('chain_started_at', now) if previous else now,
                             'input_tokens': measured_input if measured_input is not None else fallback_input,
                             'input_tokens_estimated': measured_input is None,
                             'output_tokens': measured_output if measured_output is not None else estimate_tokens(_dump([response.candidates[0].content])),
                             'config_tokens_estimate': config_estimate,
                             'last_reset_reason': reason,
                             'last_sent_tokens_estimate': estimate_tokens(body['input'])}
    memory.state_for(ctx)['last_request'] = {'api': 'interactions', 'delta_only': previous is not None,
        'reset_reason': reason, 'estimated_tokens_after': estimate_tokens(body['input']),
        'estimated_system_tokens': estimate_tokens(config.system_instruction),
        'estimated_tool_tokens': estimate_tokens(tools)}
    logger.info('analysis_metrics %s', memory.encoded({
        'event': 'interaction_request', 'session_id': ctx.session_id, 'turn_id': ctx.turn_id,
        'delta_only': previous is not None, 'reset_reason': reason,
        'sent_input_tokens_estimate': estimate_tokens(body['input']),
        'estimated_system_tokens': estimate_tokens(config.system_instruction),
        'estimated_tool_tokens': estimate_tokens(tools)}))
    return response
