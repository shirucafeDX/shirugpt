"""SDKが送るツール結果を照合する。本文・認証ヘッダーはログへ保存しない。"""
from contextvars import ContextVar
import hashlib
import json

from . import tool_wire

active_request = ContextVar('shirugpt_request_audit', default=None)


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), default=str).encode()).hexdigest()


def results(steps):
    items = []
    for step in steps:
        if step.get('type') != 'function_result':
            continue
        raw = step.get('result')
        try:
            value = json.loads(raw) if isinstance(raw, str) else raw
        except (ValueError, TypeError):
            value = raw
        value = tool_wire.decode(value)
        data = value if isinstance(value, dict) else {}
        result = data.get('result') or {}
        csv = result.get('csv') if isinstance(result, dict) else None
        items.append({'call_id': step.get('call_id'), 'tool': step.get('name'),
                      'result_id': data.get('result_id'), 'result_ref': data.get('result_ref'),
                      'payload_sha256': digest(value), 'wire_type': type(raw).__name__,
                      'csv_sha256': hashlib.sha256(csv.encode()).hexdigest() if isinstance(csv, str) else None,
                      'csv_chars': len(csv) if isinstance(csv, str) else None})
    return items


def capture_http_request(request):
    metrics = active_request.get()
    if metrics is None or request.method != 'POST':
        return
    body = json.loads(request.content)
    if 'input' not in body:
        return
    actual = results(body['input'])
    expected = metrics.get('sent_tool_results', [])
    metrics['wire_attempts'] = metrics.get('wire_attempts', 0) + 1
    metrics['wire_input_sha256'] = digest(body['input'])
    metrics['wire_tool_results'] = actual
    # JSONのキー順・文字列/オブジェクトの表現差は除き、ツール結果そのものを照合。
    same = [(x['call_id'], x['payload_sha256']) for x in actual] == [(x['call_id'], x['payload_sha256']) for x in expected]
    metrics['wire_results_verified'] = same
    if not same:
        raise ValueError('SDKへ渡したツール結果と実際の送信内容が一致しません。送信を停止しました。')
