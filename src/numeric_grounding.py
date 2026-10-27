"""数値を保存済みのセル・計算へ結び付け、モデルに値の再入力をさせない。"""
from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP, localcontext
import hashlib
import re
import unicodedata

import pandas as pd

from . import context_memory as memory

TOKEN = re.compile(r'\{\{([A-Za-z][A-Za-z0-9_]{0,31})\}\}')
MAX_CLAIMS = 24


def resolve_ref(ctx, ref):
    return memory.state_for(ctx).get('result_aliases', {}).get(ref, ref) if isinstance(ref, str) else ref


def normalize_note(ctx, note):
    if not isinstance(note, dict) or not isinstance(note.get('references'), list):
        return note
    return {**note, 'references': [resolve_ref(ctx, ref) for ref in note['references']]}


def data_refs(ctx):
    root = memory.state_for(ctx)
    aliases = root.setdefault('result_aliases', {})
    for result_id, record in root['results'].items():
        if record.get('payload', {}).get('error') or result_id in aliases.values():
            continue
        try:
            memory.saved_result(ctx, result_id)
        except ValueError:
            continue
        aliases[f'R{len(aliases) + 1}'] = result_id
    valid = {}
    for alias, result_id in aliases.items():
        try:
            memory.saved_result(ctx, result_id)
        except ValueError:
            continue
        valid[alias] = result_id
    return valid


def _number(value):
    if isinstance(value, bool) or not pd.api.types.is_number(value):
        raise ValueError('数値以外のセルは計算できません。')
    try:
        number = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError('セルを数値として扱えません。') from exc
    if not number.is_finite():
        raise ValueError('欠損値・無限大は数値の根拠にできません。')
    return number


def _text(value):
    if value is None or value is pd.NaT or value is pd.NA:
        raise ValueError('欠損値は根拠にできません。')
    if isinstance(value, str):
        if not value or len(value) > 300:
            raise ValueError('空文字・長文のセルは数値の根拠にできません。')
        return value
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return value.isoformat()
    return str(_number(value))


def _source(ctx, source):
    if not isinstance(source, dict):
        raise ValueError('sourcesは保存結果の参照オブジェクトで指定してください。')
    ref, column = source.get('result_ref'), source.get('column')
    result_id = resolve_ref(ctx, ref)
    run = memory.saved_result(ctx, result_id)
    frame = run.dataframe
    if not isinstance(column, str) or column not in frame.columns or list(frame.columns).count(column) != 1:
        raise ValueError(f'列が存在しないか重複しています: {column!r}')
    modes = [key for key in ('row_index', 'aggregate', 'row_filters') if key in source]
    if len(modes) != 1:
        raise ValueError('参照にはrow_index・aggregate・row_filtersのいずれか1つを指定してください。')
    audit = {'result_ref': ref, 'result_id': result_id, 'column': column}
    if modes[0] == 'aggregate':
        if run.truncated:
            raise ValueError('取得範囲が切り捨てられた表の全体集計はできません。SQLで集計してください。')
        operation = source['aggregate']
        values = frame[column]
        if operation == 'count':
            value = Decimal(int(values.notna().sum()))
        elif operation == 'sum':
            # SQL SUMと同じくNULLは除外。全てNULLの場合は不明とする。
            nonnull = values.dropna()
            if nonnull.empty:
                raise ValueError('全行が欠損値の列は集計できません。')
            value = sum((_number(v) for v in nonnull), Decimal(0))
        else:
            raise ValueError('aggregateはsumまたはcountです。')
        audit.update(aggregate=operation, row_count=len(frame), value=str(value))
        return value, audit
    if modes[0] == 'row_filters':
        filters = source['row_filters']
        if not isinstance(filters, list) or not 1 <= len(filters) <= 4:
            raise ValueError('row_filtersは1〜4個の列・値で指定してください。')
        def matches(value, expected):
            try:
                return _text(value) == expected
            except ValueError:
                return False
        positions = list(range(len(frame)))
        for condition in filters:
            if not isinstance(condition, dict) or condition.get('column') not in frame.columns or not isinstance(condition.get('value'), str):
                raise ValueError('行の条件には実在する列と文字列の値を指定してください。')
            positions = [i for i in positions if matches(frame[condition['column']].iloc[i], condition['value'])]
        if len(positions) != 1:
            raise ValueError(f'行の条件は1行に一致させてください（現在{len(positions)}行）。')
        index = positions[0]
        audit['row_filters'] = filters
    else:
        index = source['row_index']
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(frame):
            raise ValueError('row_indexには原表の0始まりの有効な行番号を指定してください。')
    value = frame[column].iloc[index]
    text = _text(value)
    audit.update(row_index=index, value=text)
    return value if isinstance(value, (str, date, datetime)) else _number(value), audit


def _evaluate(ctx, claim):
    if not isinstance(claim, dict) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,31}', str(claim.get('key', ''))):
        raise ValueError('数値のkeyは英字で始まる32文字以内の英数字・アンダースコアです。')
    sources = claim.get('sources')
    if not isinstance(sources, list) or not 1 <= len(sources) <= 8:
        raise ValueError('数値には1〜8個のsourcesを指定してください。')
    values, audit = [], []
    for source in sources:
        value, entry = _source(ctx, source)
        values.append(value); audit.append(entry)
    operation = claim.get('operation', 'cell')
    if operation == 'cell':
        if len(values) != 1:
            raise ValueError('cellの参照は1個です。')
        value = values[0]
    else:
        numbers = [_number(v) for v in values]
        if operation == 'sum':
            value = sum(numbers, Decimal(0))
        elif operation in ('ratio', 'difference', 'percent_change'):
            if len(numbers) != 2:
                raise ValueError(f'{operation}の参照は2個です。')
            left, right = numbers
            if operation == 'percent_change' and left == 0:
                raise ValueError('増減率の基準値が0のため計算できません。')
            if operation == 'ratio' and right == 0:
                raise ValueError('分母が0のため比率を計算できません。')
            value = (left / right if operation == 'ratio' else right - left)
            if operation == 'percent_change':
                value = (right - left) / left * 100
        else:
            raise ValueError('未知の計算方法です。')
    if 'as_percent' in claim and not isinstance(claim['as_percent'], bool):
        raise ValueError('as_percentは真偽値で指定してください。')
    if claim.get('as_percent'):
        if operation != 'ratio':
            raise ValueError('as_percentはratioでのみ指定できます。')
        value *= 100
    decimals = claim.get('decimals', 2 if operation in ('ratio', 'percent_change') else None)
    if decimals is not None and (isinstance(decimals, bool) or not isinstance(decimals, int) or not 0 <= decimals <= 6):
        raise ValueError('decimalsは0〜6の整数です。')
    if isinstance(value, Decimal):
        if decimals is not None:
            value = value.quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP)
        rendered = format(value, ',f' if claim.get('format', 'plain') == 'comma' else 'f')
        if decimals is None and '.' in rendered:
            rendered = rendered.rstrip('0').rstrip('.')
    else:
        if decimals is not None or claim.get('format', 'plain') != 'plain':
            raise ValueError('文字・日時セルの数値書式は指定できません。')
        rendered = _text(value)
    if claim.get('format', 'plain') not in ('plain', 'comma'):
        raise ValueError('formatはplainまたはcommaです。')
    terms = [f"{item['result_ref']}[{item['row_index']}].{item['column']}" if 'row_index' in item
             else f"{item['aggregate'].upper()}({item['result_ref']}.{item['column']})" for item in audit]
    formula = (terms[0] if operation == 'cell' else ' + '.join(terms) if operation == 'sum'
               else f"({terms[1]}) - ({terms[0]})" if operation == 'difference'
               else f"({terms[0]}) / ({terms[1]})" if operation == 'ratio'
               else f"(({terms[1]}) - ({terms[0]})) / ({terms[0]}) × 100")
    if claim.get('as_percent'):
        formula += ' × 100'
    return {'key': claim['key'], 'operation': operation, 'formula': formula, 'sources': audit, 'as_percent': bool(claim.get('as_percent')),
            'decimals': decimals, 'value': _text(value), 'rendered': rendered}


def render(ctx, answer, claims, *, require_grounding):
    """数値の直書きを拒否し、セル・式から値を埋め込む。"""
    claims = claims if claims is not None else []
    if not isinstance(claims, list) or len(claims) > MAX_CLAIMS:
        raise ValueError(f'numeric_claimsは最大{MAX_CLAIMS}件の配列です。')
    remaining = TOKEN.sub('', answer)
    if '{{' in remaining or '}}' in remaining:
        raise ValueError('数値の差し込みは{{key}}形式で指定してください。')
    if require_grounding:
        # 箇条書きの段階番号は実績値ではない。本文の数値は必ず差し込みにする。
        plain = unicodedata.normalize('NFKC', remaining)
        plain = re.sub(r'(?m)^[ \t]*(?:[-*]\s+)?\d{1,2}[.)、]\s*', '', plain)
        if re.search(r'\d|[零〇一二三四五六七八九十百千万億兆]+(?:件|人|回|円|割|倍|社|店舗|日|月|年)|[一二三四五六七八九十百千]+(?:万|億|兆)', plain):
            raise ValueError('分析文の数値は直書きせず{{key}}で示し、numeric_claimsに結果・列・行・計算を指定してください。')
    bindings = {}
    with localcontext() as context:
        context.prec = 128
        for claim in claims:
            binding = _evaluate(ctx, claim)
            if binding['key'] in bindings:
                raise ValueError('numeric_claimsのkeyが重複しています。')
            bindings[binding['key']] = binding
    used = TOKEN.findall(answer)
    if set(used) != set(bindings):
        raise ValueError('本文の差し込みとnumeric_claimsのkeyを一致させてください。')
    rendered = TOKEN.sub(lambda match: bindings[match[1]]['rendered'], answer)
    root = memory.state_for(ctx)
    root['numeric_grounding'] = {'turn_id': ctx.turn_id, 'mode': 'template', 'status': 'values_verified',
                                 'bindings': list(bindings.values()), 'answer_sha256': hashlib.sha256(rendered.encode()).hexdigest()}
    return rendered, list(bindings.values())


def attach_ref(ctx, payload):
    result_id = payload.get('result_id')
    alias = next((a for a, r in data_refs(ctx).items() if r == result_id), None)
    if alias is None:
        return payload
    return {**payload, 'result_ref': alias,
            'numeric_grounding': '参照はresult_refを使用。行は原表の0始まり。絞り込んだ表示ではrow_filtersで行を特定。数値は{{key}}とnumeric_claimsで回答。'}


def scalar_answer(ctx, payloads):
    question = memory.state_for(ctx).get('execution_summary', {}).get('question', '')
    if not re.search(r'(?:数値|数字)(?:だけ|のみ)|(?:1|一)つの数値|単一の数値', question):
        return None
    candidates = []
    for name, payload in payloads:
        if name != 'run_query' or payload.get('error') or not payload.get('result_id'):
            continue
        record = memory.state_for(ctx)['results'].get(payload['result_id'], {})
        run = ctx.query_cache.get(record.get('query_key'))
        if run is None or run.result_role != 'answer' or run.truncated or run.dataframe.shape != (1, 1):
            continue
        value = run.dataframe.iloc[0, 0]
        try:
            _number(value)
        except ValueError:
            continue
        candidates.append((payload['result_id'], str(run.dataframe.columns[0])))
    if len(candidates) != 1:
        return None
    result_id, column = candidates[0]
    answer, bindings = render(ctx, '{{value}}', [{'key': 'value', 'operation': 'cell',
        'sources': [{'result_ref': result_id, 'column': column, 'row_index': 0}]}], require_grounding=True)
    memory.state_for(ctx)['numeric_grounding'].update(mode='scalar', status='verified')
    return answer, bindings


def claims_schema():
    from google.genai import types
    string = lambda description: types.Schema(type='STRING', description=description)
    source = types.Schema(type='OBJECT', properties={
        'result_ref': string('受け取ったR1等の短い参照。推測でIDを生成しない。'),
        'column': string('原表の正確な列名。'),
        'row_index': types.Schema(type='INTEGER', minimum=0, description='原表の0始まりの行番号。'),
        'aggregate': types.Schema(type='STRING', enum=['sum', 'count'], description='列全体の合計または非NULL件数。'),
        'row_filters': types.Schema(type='ARRAY', max_items=4, items=types.Schema(type='OBJECT', properties={
            'column': string('行を特定する列名。'), 'value': string('その行の値。文字列で正確に指定。')}, required=['column', 'value']),
            description='1行だけに一致する条件。row_index/aggregate/row_filtersの1つを指定。'),
    }, required=['result_ref', 'column'])
    return types.Schema(type='ARRAY', max_items=MAX_CLAIMS, items=types.Schema(type='OBJECT', properties={
        'key': string('本文の{{key}}に対応する英数字のキー。'),
        'operation': types.Schema(type='STRING', enum=['cell', 'sum', 'difference', 'ratio', 'percent_change'],
            description='cell=1参照、sum=参照の合計、difference=後-前、ratio=前/後、percent_change=(後-前)/前×100。'),
        'sources': types.Schema(type='ARRAY', min_items=1, max_items=8, items=source),
        'as_percent': types.Schema(type='BOOLEAN', description='ratioを百分率にする場合true。'),
        'decimals': types.Schema(type='INTEGER', minimum=0, maximum=6, description='小数桁数。比率・増減率の既定は2桁。'),
        'format': types.Schema(type='STRING', enum=['plain', 'comma']),
    }, required=['key', 'operation', 'sources']))


def verified_note(ctx, answer, note, bindings):
    """未検証のAI要約の数値を引き継がず、確定した回答から作り直す。"""
    prefix = '検証済みの保存結果を参照して回答した。'
    if len(answer) <= 1000:
        summary = prefix + '\n' + answer
    else:
        parts = [prefix]
        for binding in bindings:
            line = f"{binding['key']}={binding['rendered']}"
            if len('\n'.join([*parts, line])) <= 1000:
                parts.append(line)
        summary = '\n'.join(parts)
    updated = {**note, 'summary': summary,
               'resolved_questions': [{**item, 'conclusion': prefix} for item in note.get('resolved_questions', [])]}
    previous = memory.state_for(ctx).get('verified_conversation_summary', '')
    question = memory.state_for(ctx).get('execution_summary', {}).get('question', '')
    current = f"ユーザーの質問: {question[:300]}\n{summary}\n参照結果: {', '.join(note.get('references', []))}"
    overview = previous + '\n\n' + current if previous and len(previous) + len(current) + 2 <= memory.OVERVIEW_CHARS else current
    memory.state_for(ctx)['verified_conversation_summary'] = overview
    return updated, overview
