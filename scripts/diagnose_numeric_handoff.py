"""既知の固定CSVを使い、指示量とツール結果の送信形式を比較する手動診断。"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import tempfile
import time
from unittest.mock import patch

from google import genai
from google.genai import types

from src import interactions, request_audit, context_memory
from src.agent import ToolContext
from src.config import load_settings
from src.prompts import build_system_instruction

# 合成データのみ。実際のBigQueryデータや利用者の会話を読み出さない。
FIXTURE = {'result_id': 'diagnostic_known_count',
           'result': {'row_count': 1, 'csv': 'count\n987654321\n'}, 'notes': [], 'missing': None}


def run_case(client, settings, style, wire):
    ctx = ToolContext(settings=settings, tools=None)
    ctx.start_turn()
    instruction = ('数値の転記を確認する診断です。この試験ではread_fixtureをkind=countで一度だけ呼び、'
                   '結果を受け取った後はCSVのcountとresult_idを変更せず、'
                   '{"count":数値,"result_id":"取得したID"}というJSONだけを返してください。'
                   '追加のツール選択・SQL・要約は不要です。')
    if style == 'app_instructions':
        instruction = build_system_instruction(['run_query', 'finish_answer']) + '\n\n' + instruction
    config = types.GenerateContentConfig(system_instruction=instruction, temperature=0.1,
        thinking_config=types.ThinkingConfig(thinking_level='LOW'), tools=[types.Tool(function_declarations=[
            types.FunctionDeclaration(name='read_fixture', description='固定CSVと結果IDを返す接続試験用ツール。',
                parameters=types.Schema(type='OBJECT', properties={'kind': types.Schema(type='STRING', enum=['count'])}, required=['kind']))])])
    contents = [types.Content(role='user', parts=[types.Part.from_text(text='read_fixtureの結果にある数値と結果IDを正確に転記してください。')])]
    transform = interactions._wire_input
    if wire == 'legacy_string':
        transform = lambda steps: [{**step, 'result': context_memory.encoded(step['result'])}
            if step['type'] == 'function_result' else step for step in steps]
    record = {'style': style, 'wire': wire, 'metrics': []}
    started = time.monotonic()
    try:
        # 旧形式はこの単独診断プロセス内でだけ再現する。
        with patch.object(interactions, '_wire_input', transform):
            metrics = {}
            first = interactions.generate(client, contents, ctx, config, metrics=metrics)
            record['metrics'].append(metrics)
            content = first.candidates[0].content
            calls = [part.function_call for part in content.parts if part.function_call]
            if len(calls) != 1 or calls[0].name != 'read_fixture':
                raise ValueError('期待したread_fixture呼び出しがありません。')
            contents.append(content)
            result = types.Part.from_function_response(name='read_fixture', response=FIXTURE)
            result.function_response.id = calls[0].id
            contents.append(types.Content(role='user', parts=[result]))
            metrics = {}
            second = interactions.generate(client, contents, ctx, config, metrics=metrics)
            record['metrics'].append(metrics)
            answer = ''.join(part.text or '' for part in second.candidates[0].content.parts)
            record['answer'] = answer
            parsed = json.loads(answer.strip().removeprefix('```json').removesuffix('```').strip())
            record.update(number_matches=parsed.get('count') == 987654321,
                          id_matches=parsed.get('result_id') == FIXTURE['result_id'])
    except Exception as exc:  # 診断を続行し、失敗条件も保存する。
        record['error'] = f'{type(exc).__name__}: {exc}'
    record['seconds'] = round(time.monotonic() - started, 3)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=None)
    args = parser.parse_args()
    output = args.output or Path(tempfile.mkdtemp(prefix='shirugpt-handoff-')) / 'results.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    settings = load_settings()
    client = genai.Client(enterprise=True, project=settings.gcp_project, location=settings.vertex_location,
        http_options=types.HttpOptions(api_version='v1beta1', timeout=settings.gemini_request_timeout_seconds * 1000,
                                      client_args={'event_hooks': {'request': [request_audit.capture_http_request]}}))
    records = []
    try:
        for style in ('minimal', 'app_instructions'):
            for wire in ('legacy_string', 'object'):
                record = run_case(client, settings, style, wire)
                records.append(record)
                output.write_text(json.dumps(records, ensure_ascii=False, indent=2))
                print(json.dumps({key: record.get(key) for key in ('style', 'wire', 'number_matches', 'id_matches', 'seconds', 'error')}, ensure_ascii=False), flush=True)
    finally:
        client.close()
    print(output)


if __name__ == '__main__':
    main()
