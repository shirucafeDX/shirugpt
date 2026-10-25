"""モデル向けのCSV転送。文字列・整数・Decimal・日時を省略せず保持する。"""
from __future__ import annotations

import base64
import csv
import datetime as dt
from decimal import Decimal
import io
import json
import math

import numpy as np
import pandas as pd

def _value(value):
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        return _value(value.item())
    if isinstance(value, float):
        if math.isnan(value):
            return None
        return value if math.isfinite(value) else ('Infinity' if value > 0 else '-Infinity')
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, Decimal):
        return None if value.is_nan() else str(value)
    if isinstance(value, (dt.datetime, dt.date, dt.time)):
        return value.isoformat()
    if isinstance(value, bytes):
        return 'base64:' + base64.b64encode(value).decode('ascii')
    if isinstance(value, np.ndarray):
        return [_value(v) for v in value.tolist()]
    if isinstance(value, (list, tuple)):
        return [_value(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _value(v) for k, v in value.items()}
    return str(value)


def _cell(value):
    value = _value(value)
    if value is None:
        return None
    if isinstance(value, (dict, list, bool)):
        return json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    return str(value)


def csv_result(frame: pd.DataFrame, *, truncated: bool = False) -> dict:
    """全列・全行のCSVと型情報を返す。セル単位の切り捨てや丸めはしない。"""
    rows = [[_cell(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    occupied = {value for row in rows for value in row if value is not None}
    null_marker = r'\N'
    while null_marker in occupied:
        null_marker += '_'
    output = io.StringIO(newline='')
    writer = csv.writer(output, lineterminator='\n')
    writer.writerow([str(c) for c in frame.columns])
    writer.writerows([[null_marker if value is None else value for value in row] for row in rows])
    result = {
        'delivery': 'csv', 'row_count': len(frame), 'column_count': len(frame.columns),
        'columns': [{'name': str(c), 'dtype': str(frame[c].dtype), 'null_count': int(frame[c].isna().sum())}
                    for c in frame.columns],
        'csv': output.getvalue(), 'sent_row_count': len(frame), 'rows_omitted': 0,
        'complete': True, 'source_truncated': bool(truncated), 'null_marker': null_marker,
        'scope': '取得済み結果の範囲。元テーブル全体とは限らない。',
        'encoding_note': 'CSVはヘッダー付き・カンマ区切り・引用符は二重化。空文字とnull_markerは別。日時はISO形式で時差変換なし。Decimalは正確な文字列、bytesはbase64:、配列・構造体はJSON。',
    }
    if truncated:
        result['truncated'] = True
        result['truncation_note'] = 'アプリの取得上限による切り捨てあり。CSVが完全でも元のクエリ結果全体ではない。'
    return result


def recall_csv_snapshot(summary: dict, *, columns=None, offset=0, limit=20, policy=None) -> tuple[dict, int]:
    """原DataFrameのない履歴では、保存されたCSV内だけをページ参照する。"""
    reader = csv.DictReader(io.StringIO(summary.get('csv', ''), newline=''))
    available = reader.fieldnames or [col['name'] for col in summary.get('columns', [])]
    if columns is None:
        columns = available
    if (not isinstance(columns, list) or not columns
            or any(not isinstance(c, str) or c not in available for c in columns)
            or len(set(columns)) != len(columns)):
        raise ValueError('columnsには保存済みCSVにある列名を指定してください。')
    rows = list(reader)
    marker = summary.get('null_marker')
    selected = [{c: None if row[c] == marker else row[c] for c in columns}
                for row in rows[offset:offset + limit]]
    from .token_budget import TokenBudgetManager
    page = TokenBudgetManager(policy).table(pd.DataFrame(selected, columns=columns, dtype=object),
                                           truncated=summary.get('source_truncated', False))
    dtypes = {column['name']: column.get('dtype', 'object') for column in summary.get('columns', [])}
    for column in page['columns']:
        column['dtype'] = dtypes.get(column['name'], 'object')
    return page, len(rows)
