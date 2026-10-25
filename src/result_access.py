"""保存済みDataFrameへの型付き条件指定。SQLや式の評価は行わない。"""
from __future__ import annotations

import datetime as dt
from decimal import Decimal, InvalidOperation
import math

import pandas as pd


def _typed_value(series: pd.Series, value: str):
    if not isinstance(value, str) or len(value) > 300:
        raise ValueError('条件の値は300文字以内の文字列で指定してください。')
    non_null = series.dropna()
    example = non_null.iloc[0] if len(non_null) else None
    if pd.api.types.is_bool_dtype(series) or isinstance(example, bool):
        if value.casefold() not in ('true', 'false'):
            raise ValueError('真偽値はtrueまたはfalseで指定してください。')
        return value.casefold() == 'true'
    if pd.api.types.is_numeric_dtype(series) or isinstance(example, Decimal):
        try:
            number = Decimal(value)
            if not number.is_finite():
                raise ValueError('条件に非有限の数値は指定できません。')
            if isinstance(example, Decimal):
                return number
            if pd.api.types.is_integer_dtype(series):
                if number != number.to_integral_value():
                    raise ValueError('整数列には整数を指定してください。')
                return int(number)
            converted = float(number)
            if not math.isfinite(converted):
                raise ValueError('条件の数値が大きすぎます。')
            return converted
        except InvalidOperation as exc:
            raise ValueError('数値列の条件には数値を指定してください。') from exc
    if pd.api.types.is_datetime64_any_dtype(series) or isinstance(example, (dt.date, dt.datetime)):
        try:
            # ISOのみ。タイムゾーン付き/なしを暗黙に変換しない。
            stamp = pd.Timestamp(dt.datetime.fromisoformat(value))
            timezone = getattr(series.dtype, 'tz', None) or getattr(example, 'tzinfo', None)
            if bool(stamp.tzinfo) != bool(timezone):
                raise ValueError('日時条件のタイムゾーンの有無を列に合わせてください。')
            if isinstance(example, dt.date) and not isinstance(example, dt.datetime):
                if stamp.time() != dt.time():
                    raise ValueError('日付列の条件に時刻を指定できません。')
                return stamp.date()
            return stamp
        except (ValueError, TypeError) as exc:
            raise ValueError('日時条件は列に合うISO形式で指定してください。') from exc
    return value


def filter_saved_result(frame: pd.DataFrame, filters: object = None) -> pd.DataFrame:
    if filters is None:
        return frame
    if not isinstance(filters, list) or len(filters) > 5:
        raise ValueError('filtersは最大5件の条件配列で指定してください。')
    mask = pd.Series(True, index=frame.index)
    for condition in filters:
        if not isinstance(condition, dict):
            raise ValueError('条件はcolumn、op、valuesで指定してください。')
        column, op = condition.get('column'), condition.get('op')
        if not isinstance(column, str) or column not in frame.columns:
            raise ValueError('条件には保存済み結果にある列名を指定してください。')
        if op not in ('eq', 'in', 'gte', 'lte', 'is_null', 'not_null'):
            raise ValueError('未対応の比較演算です。')
        series = frame[column]
        values = condition.get('values', [])
        if not isinstance(values, list):
            raise ValueError('valuesは文字列の配列で指定してください。')
        if op in ('is_null', 'not_null'):
            if values:
                raise ValueError('欠損判定にはvaluesを指定しないでください。')
            selected = series.isna() if op == 'is_null' else series.notna()
        else:
            if not 1 <= len(values) <= (20 if op == 'in' else 1):
                raise ValueError('eq/gte/lteは1値、inは1〜20値を指定してください。')
            typed = [_typed_value(series, value) for value in values]
            try:
                if op == 'in':
                    selected = series.isin(typed)
                elif op == 'eq':
                    selected = series.eq(typed[0])
                elif op == 'gte':
                    selected = series.ge(typed[0])
                else:
                    selected = series.le(typed[0])
            except (TypeError, ValueError) as exc:
                raise ValueError('列の型に合う比較条件を指定してください。') from exc
        mask &= selected.fillna(False)
    return frame.loc[mask]
