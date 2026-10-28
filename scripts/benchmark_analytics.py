"""Real API smoke tests. Run with PYTHONPATH=. and an unused --output directory."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

from google.genai import types
from src.agent import ToolContext
from src.bq_tools import BigQueryTools, technical_detail
from src.chat_history import pack, unpack
from src.config import load_settings
from src.conversational_analytics import build_agent, INSTRUCTION
from src.execution import BackgroundTurn
from src.usage_log import UsageLogger

CASES = [('a1', 'general', '37×24の結果を、数字だけで答えてください。'),
 ('a2', 'general', 'その結果を3で割ってください。数字だけで答えてください。'),
 ('a3',
  'general',
  '大学生向け無料カフェで、短期の来店数と長期の関係づくりを両立する施策を3案挙げ、利点と不利点を比較してください。一般的な仮説で400文字以内。実データの取得は不要です。'),
 ('b1', 'data', '神大店の2026年7月の来店者数と来店数を教えてください。'),
 ('b2', 'data', 'その店舗の2025年と2026年の月別の来店者数の推移をグラフにしてください。'),
 ('b3',
  'data',
  '同じ店舗で、2026年1〜9月と2025年1〜9月の来店実績を比較し、運営上の課題と検証施策を2つ挙げてください。来店者数はconnected_idの重複を期間全体で除いた人数、来店数はDU_idの重複を除いた件数とします。月別と学年別の違いも調べ、事実と仮説を分けて600文字以内で説明してください。'),
 ('b4',
  'data',
  '2026年7月の一人あたり来店回数を計算し、2025年7月と比較してください。小数2桁で、見出しは「### 1. '
  '比較」とし、最後に2週間の検証施策を1案、実績と区別して提案してください。新しいSQLは実行せず、保存結果を使ってください。'),
 ('c1', 'clarify', 'データを使って課題を見つけてください。いまはSQLを実行せず、最初に調査目的と対象を確認する質問を1つしてください。')]

def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str))


def count_context(settings, backend, ctx):
    # This measures a standalone JSON representation, not managed API usage.
    from google import genai
    from google.cloud import geminidataanalytics as ca
    context = backend._context(ctx)
    text = json.dumps(ca.Context.to_dict(context, use_integers_for_enums=False,
        always_print_fields_with_no_presence=False), ensure_ascii=False, separators=(',', ':'))
    client = genai.Client(vertexai=True, project=settings.gcp_project, location=settings.vertex_location)
    try:
        result = client.models.count_tokens(model=settings.gemini_model, contents=text)
        return {'tokenizer_model': settings.gemini_model, 'chars': len(text),
                'tokens': result.total_tokens,
                'scope': '登録コンテキストJSON単独。分析API内部の使用量・課金トークンではない。'}
    finally:
        client.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count-tokens', action='store_true', help='登録コンテキスト単独を別APIで参考計測')
    parser.add_argument('--resume', type=Path, help='保存ZIPを開いて--questionを1問追加する')
    parser.add_argument('--question', help='単独で試す質問。--resumeと併用可能')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    settings = replace(load_settings(), analysis_backend='conversational_analytics')
    backend = build_agent(settings)
    logger = UsageLogger(settings)
    identity = '(diagnostic-conversational-analytics)'
    manifest = {'instruction': INSTRUCTION, 'backend': settings.analysis_backend,
                'tables': settings.analytics_tables,
                'source_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                  for p in [Path('app.py'), *Path('src').glob('*.py')]}}
    write(args.output / 'manifest.json', manifest)
    sessions, results = {}, []
    cases = [('custom', 'custom', args.question)] if args.question else CASES
    if args.resume and not args.question:
        parser.error('--resumeには--questionを指定してください。')
    for key, group, prompt in cases:
        if group not in sessions:
            ctx = ToolContext(tools=BigQueryTools(settings), settings=settings,
                              usage_logger=logger, user_email=identity)
            messages, contents = [], []
            if args.resume:
                messages, contents, _ = unpack(args.resume.read_bytes(), ctx)
            sessions[group] = [ctx, contents, messages]
        ctx, contents, messages = sessions[group]
        if key == 'b4':
            snapshot = pack(messages, contents, ctx, None)
            (args.output / 'b4-before-restore.zip').write_bytes(snapshot)
            ctx = ToolContext(tools=BigQueryTools(settings), settings=settings,
                              usage_logger=logger, user_email=identity)
            messages, contents, _ = unpack(snapshot, ctx)
            sessions[group] = [ctx, contents, messages]
        if args.count_tokens and not results:
            write(args.output / 'count-tokens.json', count_context(settings, backend, ctx))
        ctx.start_turn()
        contents.append(types.Content(role='user', parts=[types.Part.from_text(text=prompt)]))
        messages.append({'role': 'user', 'text': prompt})
        logger.log_prompt(user_email=identity, session_id=ctx.session_id, turn_id=ctx.turn_id, prompt_text=prompt)
        print(json.dumps({'phase': 'started', 'key': key, 'turn_id': ctx.turn_id}), flush=True)
        started = time.monotonic()
        task = BackgroundTurn(backend, contents, ctx)
        task.start()
        last = started
        while not task.done.wait(1):
            if time.monotonic() - last > 20:
                print(json.dumps({'phase': 'progress', 'key': key,
                    'seconds': time.monotonic() - started, 'message': task.message}, ensure_ascii=False), flush=True)
                last = time.monotonic()
            if time.monotonic() - started > 400:
                task.control.request_stop()
        record = {'key': key, 'prompt': prompt, 'turn_id': ctx.turn_id,
                  'session_id': ctx.session_id, 'seconds': round(time.monotonic() - started, 3),
                  'answer': getattr(task.result, 'text', None),
                  'error': technical_detail(task.error, limit=10000) if task.error else None,
                  'events': ctx.turn_events, 'errors': ctx.turn_errors,
                  'results': ctx.analytics_state.get('results', {}),
                  'conversation': ctx.analytics_state.get('conversation')}
        write(args.output / (key + '.json'), record)
        results.append(record)
        messages.append({'role': 'assistant', 'text': record['answer'] or record['error'],
                         'turn_id': ctx.turn_id, 'execution_events': ctx.turn_events, 'runs': ctx.turn_runs})
        (args.output / (key + '.zip')).write_bytes(pack(messages, contents, ctx, None))
        write(args.output / 'results.json', results)
        print(json.dumps({'phase': 'finished', **{k: record[k] for k in
              ('key', 'seconds', 'answer', 'error')}}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
