"""InteractionsのSSEを組み立てる。未完了の文章・ツール引数は実行側へ渡さない。"""
from __future__ import annotations

import json
import logging
import time


logger = logging.getLogger(__name__)


class StreamResponseError(RuntimeError):
    def __init__(self, message, code=None):
        super().__init__(message)
        self.status_code = {'resource_exhausted': 429, 'too_many_requests': 429}.get(code, code if isinstance(code, int) else None)


def _dict(value):
    if isinstance(value, dict):
        return value
    return value.model_dump(mode='json', exclude_none=True)


def consume(stream, ctx, on_progress, metrics, started):
    # JSON応答を返すサーバーとの互換。通常はstream=TrueのSSEを受信する。
    if isinstance(stream, dict) or isinstance(getattr(stream, 'status', None), str):
        metrics['streamed'] = False
        return stream
    metrics['streamed'] = True
    control = ctx.execution_control
    steps, fragments, stopped = {}, {}, set()
    completed = None
    last_progress = -1.0
    try:
        if control is not None:
            control.set_response_stream(stream)
            control.check()
        for native in stream:
            event = _dict(native)
            kind = event.get('event_type')
            elapsed = round(time.monotonic() - started, 6)
            metrics.setdefault('first_event_seconds', elapsed)
            metrics['stream_events'] = metrics.get('stream_events', 0) + 1
            metrics['last_event_seconds'] = elapsed
            metrics['last_event_type'] = kind
            if event.get('event_id'):
                metrics['last_event_id'] = event['event_id']
            interaction = event.get('interaction') or {}
            if interaction.get('id'):
                metrics['interaction_id'] = interaction['id']
            if kind == 'interaction.completed':
                completed = interaction
                metrics['completion_status'] = interaction.get('status')
                if interaction.get('usage') is not None:
                    metrics['reported_usage'] = interaction['usage']
                # [DONE]/EOFまで読み切り、サーバー側の状態確定前に接続を閉じない。
                continue
            if control is not None:
                control.check()
            if kind == 'error':
                error = event.get('error') or {}
                raise StreamResponseError(f"AI応答の受信中にエラーが発生しました: {error.get('message', '詳細なし')}", error.get('code'))
            if kind == 'step.start':
                index = event['index']
                if index in steps:
                    raise StreamResponseError('AI応答のステップが重複しています。')
                steps[index] = dict(event['step'])
                # 初期contentは独立した辞書にしてイベント原本を変更しない。
                if steps[index].get('content') is not None:
                    steps[index]['content'] = [dict(c) for c in steps[index]['content']]
            elif kind == 'step.delta':
                index, delta = event['index'], event['delta']
                if index not in steps or index in stopped:
                    raise StreamResponseError('AI応答のステップ順序が不正です。')
                step = steps[index]
                delta_kind = delta.get('type')
                if delta_kind == 'arguments_delta':
                    if step.get('type') != 'function_call':
                        raise StreamResponseError('ツール引数の受信先が不正です。')
                    text = delta.get('arguments') or ''
                    fragments.setdefault(index, []).append(text)
                elif delta_kind == 'text':
                    if step.get('type') != 'model_output':
                        raise StreamResponseError('文章の受信先が不正です。')
                    text = delta.get('text') or ''
                    content = step.setdefault('content', [])
                    if content and content[-1].get('type') == 'text':
                        content[-1]['text'] += text
                    else:
                        content.append({'type': 'text', 'text': text})
                else:
                    # 思考要約・署名などは表示せず、状態はサーバー側で引き継ぐ。
                    text = ''
                if text:
                    metrics.setdefault('first_content_seconds', elapsed)
                    metrics['received_chars'] = metrics.get('received_chars', 0) + len(text)
            elif kind == 'step.stop':
                index = event['index']
                if index not in steps or index in stopped:
                    raise StreamResponseError('AI応答の終了ステップが不正です。')
                stopped.add(index)
                if event.get('usage') is not None:
                    metrics['partial_usage'] = event['usage']
            if on_progress is not None and elapsed - last_progress >= 1:
                chars = metrics.get('received_chars', 0)
                on_progress(f'AI応答を受信しています（{chars:,}文字・検証後に表示します）…' if chars else 'AIからの応答が始まりました。処理を続けています…')
                last_progress = elapsed
        metrics['stream_end_seconds'] = round(time.monotonic() - started, 6)
        if completed is None:
            raise StreamResponseError('AI応答の受信が完了前に途切れました。途中のSQL・回答は実行していません。')
        if steps and set(steps) != stopped:
            raise StreamResponseError('AI応答に未完了のステップがあります。途中のSQL・回答は実行していません。')
        assembled = []
        for index in sorted(steps):
            step = steps[index]
            if index in fragments:
                try:
                    arguments = json.loads(''.join(fragments[index]))
                except (ValueError, TypeError) as exc:
                    raise StreamResponseError('AIが返したツール引数のJSONが不完全です。実行していません。') from exc
                if not isinstance(arguments, dict):
                    raise StreamResponseError('AIが返したツール引数がオブジェクトではありません。実行していません。')
                step['arguments'] = arguments
            if step.get('type') == 'function_call' and (not step.get('id') or not step.get('name') or not isinstance(step.get('arguments'), dict)):
                raise StreamResponseError('AIのツール呼び出しが不完全です。実行していません。')
            assembled.append(step)
        # 完了イベントが出力を省略する仕様に対応。完全な出力があればそちらを使用。
        result = {**completed, 'steps': completed.get('steps') or assembled}
        metrics['stream_completed'] = True
        return result
    finally:
        response = getattr(stream, 'response', None)
        if response is not None:
            headers = getattr(response, 'headers', {})
            for key in ('x-request-id', 'x-goog-request-id'):
                value = headers.get(key)
                if isinstance(value, str):
                    metrics[key.replace('-', '_')] = value[:300]
        if control is not None:
            control.clear_response_stream(stream)
        try:
            stream.close()
        except Exception:
            logger.warning("AI応答ストリームの終了処理に失敗しました。", exc_info=True)
