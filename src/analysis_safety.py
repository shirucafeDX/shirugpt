"""SQLの列参照を検査し、取得結果と処理の成否を記録する。"""
from __future__ import annotations

import hashlib
import re

import sqlglot
from sqlglot import exp
from sqlglot.errors import OptimizeError
from sqlglot.optimizer.qualify import qualify

from . import context_memory as memory, result_references
from .sql_guard import SQLGuardError



def state(ctx):
    root = memory.state_for(ctx)
    current = root.get('execution_summary')
    if not current or current['turn_id'] != ctx.turn_id:
        current = {'turn_id': ctx.turn_id, 'data_requested': False, 'failures': {},
                   'steps': [], 'schema_keys': [], 'question': '', 'next_action': '質問に必要なデータと確認済みスキーマを確認する。'}
        root['execution_summary'] = current
    return current


def query_key(sql):
    # 空白やコメントだけ変えた同じSQLの再試行も防ぐ。値の大小文字は保つ。
    try:
        sql = sqlglot.parse_one(sql, read='bigquery').sql(dialect='bigquery')
    except sqlglot.errors.SqlglotError:
        pass
    return hashlib.sha256(sql.encode()).hexdigest()


def likely_tables(question, listing):
    """名前が確認済みで、質問との対応が明確なものだけ最大3件選ぶ。"""
    if not re.search(r'推移|比較|傾向|集計|ランキング|調べ|分析|何人|件数|人数|数は|数を', question):
        return []
    names = [item['table'] for item in listing.get('tables', [])]
    # 固有テーブル名の一致を優先。ドメイン別名も一覧に実在するときだけ使う。
    matched = [name for name in names if name in question]
    aliases = {'来店': ('来店', 'DU'), 'Meetup参加': ('Meetup', 'ミートアップ', '参加率', '予約数'),
               'イベント': ('イベント',)}
    for name, words in aliases.items():
        if name in names and any(word in question for word in words) and name not in matched:
            matched.append(name)
    return matched[:3]


def is_data_question(question):
    if re.search(r'一般的|データを使わず|データ不要', question):
        return False
    return bool(re.search(r'来店|Meetup|ミートアップ|イベント|売上|店舗|顧客', question)
                and re.search(r'推移|比較|傾向|集計|ランキング|調べ|分析|何人|件数|人数|数は|数を', question))


def verify_columns(sql, references, schemas):
    """単純な表・CTE・別名・結合は構文木で照合。複雑な型はBigQueryのdry runに委ねる。"""
    if not references:
        return
    tree = sqlglot.parse_one(sql, read='bigquery')
    mapping, available = {}, {}
    for reference in references:
        project, dataset, table = reference.split('.', 2)
        columns = schemas[f'{dataset}.{table}'].get('columns', [])
        if not columns or any('.' in c['name'] or c.get('type') in ('RECORD', 'STRUCT') for c in columns):
            return
        # 型推論はBigQueryに任せ、ここでは列の存在・スコープだけ検査する。
        mapping.setdefault(project, {}).setdefault(dataset, {})[table] = {c['name']: 'UNKNOWN' for c in columns}
        available[reference] = [c['name'] for c in columns]
    if tree.find(exp.Unnest) or tree.find(exp.Pivot):
        return
    try:
        qualify(tree, dialect='bigquery', schema=mapping, identify=False, quote_identifiers=False)
    except OptimizeError as exc:
        raise SQLGuardError(f"SQLの列参照を確認できません: {exc}。確認済みの列: {memory.encoded(available)}。存在する列名・別名でSQLを修正してください。") from exc


def has_data_reference(ctx, refs):
    records = memory.state_for(ctx)['results']
    for ref in refs:
        record = records.get(ref, {})
        if record.get('payload', {}).get('error'):
            continue
        try:
            memory.saved_result(ctx, ref)
            return True
        except ValueError:
            pass
    return False


def needs_evidence(ctx):
    current = state(ctx)
    if current['data_requested'] or any(e.get('tool') in ('run_query', 'analyze_result') for e in ctx.turn_events):
        return True
    # 省略された追質問でも、保持中の表を使った回答には参照を求める。
    # データを必要としない相談・挨拶まで、過去の表への参照を強制しない。
    return bool((ctx.query_cache or ctx.analysis_cache) and re.search(
        r'前年|昨年|前月|先月|今月|今年|比較|内訳|説明|なぜ|グラフ|図|数字|数値|それ|その|この|増減|店舗|来店|売上|参加|予約',
        current['question']) and not re.search(r'一般的|データを使わず|データ不要', current['question']))


def record(ctx, name, args, payload):
    current = state(ctx)
    if name in ('run_query', 'analyze_result'):
        current['data_requested'] = True
    if name == 'recall_context' and has_data_reference(ctx, [args.get('result_id')]):
        current['data_requested'] = True
    entry = {'tool': name, 'status': 'failed' if payload.get('error') else 'succeeded'}
    if name == 'get_table_schema' and not payload.get('error'):
        key = f"{args.get('dataset') or getattr(ctx.settings, 'default_dataset', '')}.{args.get('table')}"
        current['schema_keys'] = [item for item in current['schema_keys'] if item != key] + [key]
    if payload.get('result_id'):
        entry['result_id'] = payload['result_id']
    if payload.get('error'):
        entry['error'] = (payload.get('technical_detail') or payload['error'])[:1000]
        if name == 'run_query':
            current['failures'][query_key(str(args.get('sql', '')))] = entry['error']
        current['next_action'] = 'エラーと確認済みスキーマを照合し、同じ失敗を繰り返さず条件・SQLを修正する。成功するまで数値は未確認。'
        if name in ('llm_call', 'response_validation'):
            current['next_action'] = '保存済み結果を再利用し、AI応答の取得を再開する。受信途中のSQL・回答は未実行。'
    elif name in ('run_query', 'analyze_result', 'recall_context'):
        current['next_action'] = '保存結果を確認し、実在する結果IDを根拠としてfinish_answerに渡す。'
    if name == 'finish_answer' and not payload.get('error'):
        current['next_action'] = '回答済み。次の質問に応じて保存結果を再利用する。'
    current['steps'].append(entry)


def snapshot(ctx):
    analytics = getattr(ctx, 'analytics_state', {})
    calls = [e for e in ctx.turn_events if e.get('event') == 'analytics_call']
    if calls:
        return {'turn_id': ctx.turn_id,
            'execution_note': f"Conversational Analytics API。取得結果: {len(ctx.turn_runs)}件。",
            'summary': '会話IDで継続。アプリが作る要約・作業メモは送信していません。',
            'conversation_summary': '内部の履歴選択・圧縮内容はAPIから公開されません。',
            'context_note': {}, 'steps': [],
            'analytics_context': {k: analytics.get(k) for k in ('agent', 'conversation', 'context_sha256', 'authored_context')}}
    current = state(ctx)
    root = memory.state_for(ctx)
    turn = root['turns'][-1] if root['turns'] else {}
    note = root.get('current_note', {})
    successful = [step['result_id'] for step in current['steps'] if step.get('result_id') and has_data_reference(ctx, [step['result_id']])]
    failed_count = sum(step['status'] == 'failed' for step in current['steps'])
    execution_note = f"成功したデータ結果: {len(set(successful))}件。失敗した処理: {failed_count}件。"
    if not successful and current['data_requested']:
        execution_note += "このターンでは有効なデータ結果をまだ取得・参照していません。数値は未確認です。"
    return {'turn_id': ctx.turn_id, 'execution_note': execution_note, 'steps': list(current['steps']),
            'next_action': current['next_action'],
            'summary': turn.get('summary', ''), 'conversation_summary': turn.get('overview', ''),
            'context_note': dict(note) if note.get('turn_id') in (ctx.turn_id, turn.get('id')) else {},
            'verified_schemas': {key: [dict(c) for c in value.get('columns', [])] for key, value in ctx.schema_cache.items() if key in current['schema_keys']}}


def request_context(ctx, tools):
    current = state(ctx)
    value = {'execution_status': {'data_requested': current['data_requested'], 'recent_steps': current['steps'][-4:],
                                  'next_action': current['next_action']}}
    value['verified_tables'] = current['schema_keys'][-3:]
    if 'run_query' in (tools or []):
        keys = (current['schema_keys'] or list(ctx.schema_cache))[-3:]
        value['verified_schemas'] = {key: [{'name': c['name'], 'type': c.get('type', c.get('dtype', ''))}
                                            for c in ctx.schema_cache[key].get('columns', [])] for key in keys}
    if needs_evidence(ctx):
        refs = [ref for ref in memory.state_for(ctx)['results'] if has_data_reference(ctx, [ref])][-6:]
        value['available_data_result_ids'] = refs
        value['available_result_refs'] = {alias: rid for alias, rid in result_references.data_refs(ctx).items() if rid in refs}
    return '\nアプリが確認した実行状態（過去の要約より優先。失敗は成功扱いしない）: ' + memory.encoded(value)
