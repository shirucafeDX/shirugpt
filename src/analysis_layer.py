"""保存済み表の分析。LLMは方法を選び、計算はデータ側で行う。

Do not move data to the LLM when computation can be moved to the data.
任意Python/evalは実行しない。自由分析SQLはローカルのdata表への読み取りに限定。
"""
from __future__ import annotations

from dataclasses import replace
from itertools import combinations
import sqlite3
import time

import numpy as np
import pandas as pd
import sqlglot
from sqlglot import exp

from .result_access import filter_saved_result, _typed_value
from .result_csv import _cell
from .token_budget import BudgetPolicy, TokenBudgetManager, estimate_tokens

OPERATIONS = ('get_schema', 'get_row_count', 'describe_data', 'aggregate', 'top_k',
              'filter_data', 'compare_groups', 'correlation', 'detect_outliers',
              'analyze_timeseries', 'get_distinct_values', 'sample_data', 'sql')
AGGREGATIONS = ('count', 'nunique', 'sum', 'mean', 'median', 'min', 'max', 'std', 'var')


def _columns(frame, value, *, maximum=12, empty=False):
    if (not isinstance(value, list) or (not value and not empty) or len(value) > maximum
            or any(not isinstance(c, str) or c not in frame.columns for c in value)
            or len(set(value)) != len(value)):
        raise ValueError(f'保存済み結果の列名を重複なく最大{maximum}列指定してください。')
    return value


def _number(frame, name):
    _columns(frame, [name])
    if not pd.api.types.is_numeric_dtype(frame[name]) or pd.api.types.is_bool_dtype(frame[name]):
        raise ValueError(f'{name}は数値列ではありません。精密なDecimal計算はBigQueryで集計してください。')
    return frame[name].replace([float('inf'), float('-inf')], np.nan)


def _limit(args, policy):
    value = args.get('limit', policy.max_preview_rows)
    if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= policy.max_preview_rows:
        raise ValueError(f'limitは1〜{policy.max_preview_rows}の整数で指定してください。')
    return value


def _group_by(frame, args):
    return _columns(frame, args.get('group_by', []), maximum=6, empty=True)


def _aggregate_spec(frame, specs):
    if not isinstance(specs, list) or not 1 <= len(specs) <= 8:
        raise ValueError('aggregationsはcolumn、function、aliasを指定する1〜8件の配列です。')
    result = {}
    for item in specs:
        if not isinstance(item, dict):
            raise ValueError('集計指定はオブジェクトで指定してください。')
        col, function, alias = item.get('column'), item.get('function'), item.get('alias')
        _columns(frame, [col])
        if function not in AGGREGATIONS or not isinstance(alias, str) or not alias or len(alias) > 100 or alias in result:
            raise ValueError('集計関数と重複しないaliasを指定してください。')
        if function in ('sum', 'mean', 'median', 'std', 'var'):
            _number(frame, col)
        # 全欠損を0と扱わない。整数合計はint64オーバーフローを避ける。
        method = function
        if function == 'sum':
            method = lambda values: (None if values.dropna().empty else
                sum(map(int, values.dropna())) if pd.api.types.is_integer_dtype(values)
                else values.sum(min_count=1))
        result[alias] = pd.NamedAgg(column=col, aggfunc=method)
    return result


def get_schema(frame, args):
    columns = _columns(frame, args['columns']) if args.get('columns') else list(frame.columns)
    return pd.DataFrame([{'column': c, 'dtype': str(frame[c].dtype), 'missing': int(frame[c].isna().sum())}
                         for c in columns])


def get_row_count(frame, args):
    return pd.DataFrame([{'row_count': len(frame), 'column_count': len(frame.columns)}])


def describe_data(frame, args):
    columns = _columns(frame, args['columns']) if args.get('columns') else list(frame.columns[:12])
    results = []
    for col in columns:
        series = frame[col]
        item = {'column': col, 'dtype': str(series.dtype), 'count': int(series.notna().sum()),
                'missing': int(series.isna().sum())}
        if pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series):
            values = _number(frame, col).dropna()
            item['nonfinite_or_missing'] = len(series) - len(values)
            if len(values):
                item.update(min=values.min(), max=values.max(), mean=values.mean(),
                            median=values.median(), std=values.std(), variance=values.var(),
                            q25=values.quantile(.25), q75=values.quantile(.75))
        else:
            try:
                item['distinct_count'] = series.nunique()
            except TypeError:
                item['distinct_count_unavailable'] = True
        results.append(item)
    # object型で組み立て、大整数のmin/maxを欠損混在によるfloat変換から守る。
    return pd.DataFrame(results, dtype=object)


def aggregate(frame, args):
    groups = _group_by(frame, args)
    spec = _aggregate_spec(frame, args.get('aggregations'))
    if set(groups) & set(spec):
        raise ValueError('aliasはgroup_byと異なる名前にしてください。')
    if groups:
        return frame.groupby(groups, dropna=False, observed=True, sort=False).agg(**spec).reset_index()
    return pd.DataFrame([{name: (specification.aggfunc(frame[specification.column])
                                  if callable(specification.aggfunc) else frame[specification.column].agg(specification.aggfunc))
                          for name, specification in spec.items()}], dtype=object)


def top_k(frame, args):
    order = _columns(frame, args.get('sort_by'), maximum=3)
    columns = _columns(frame, args['columns']) if args.get('columns') else list(frame.columns)
    values = frame
    if args.get('absolute', False):
        values = frame.copy()
        for col in order:
            values[col] = _number(frame, col).abs()
    indices = values.sort_values(order, ascending=args.get('ascending', False), na_position='last', kind='stable').index
    return frame.loc[indices, columns]


def filter_data(frame, args):
    columns = _columns(frame, args['columns']) if args.get('columns') else list(frame.columns)
    offset = args.get('offset', 0)
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < 0:
        raise ValueError('offsetは0以上の整数です。')
    result = top_k(frame, {**args, 'ascending': args.get('ascending', True)}) if args.get('sort_by') else frame[columns]
    return result.iloc[offset:]


def compare_groups(frame, args):
    groups = _group_by(frame, args)
    if len(groups) != 1:
        raise ValueError('compare_groupsはgroup_byを1列指定してください。')
    metric = args.get('metric')
    _number(frame, metric)
    function = args.get('aggregation', 'mean')
    result = aggregate(frame, {'group_by': groups, 'aggregations': [
        {'column': metric, 'function': function, 'alias': 'value'}]})
    if args.get('baseline') is None:
        baseline = aggregate(frame, {'aggregations': [{'column': metric, 'function': function, 'alias': 'value'}]})['value'].iloc[0]
        label = '取得済み対象全体の同じ集計値'
    else:
        target = _typed_value(frame[groups[0]], args['baseline'])
        selected = result[result[groups[0]].eq(target)]
        if len(selected) != 1:
            raise ValueError('baselineに一致する比較対象がありません。')
        baseline = selected['value'].iloc[0]
        label = str(args['baseline'])
    result['baseline'] = baseline
    result['baseline_label'] = label
    result['delta'] = result['value'] - baseline
    result['change_pct'] = result['delta'] / baseline * 100 if pd.notna(baseline) and baseline > 0 else np.nan
    return result.loc[result['delta'].abs().sort_values(ascending=False, kind='stable').index]


def correlation(frame, args):
    columns = _columns(frame, args.get('columns'), maximum=6)
    if len(columns) < 2:
        raise ValueError('相関は2〜6列を指定してください。')
    method = args.get('method', 'pearson')
    if method not in ('pearson', 'spearman'):
        raise ValueError('methodはpearsonまたはspearmanです。')
    result = []
    for a, b in combinations(columns, 2):
        pair = pd.DataFrame({a: _number(frame, a), b: _number(frame, b)}).dropna()
        if method == 'spearman':
            values = pair.rank(method='average')
        else:
            values = pair
        r = values[a].corr(values[b]) if len(pair) >= 3 and values[a].nunique() > 1 and values[b].nunique() > 1 else np.nan
        result.append({'x': a, 'y': b, 'paired_rows': len(pair), 'correlation': r, 'method': method})
    result = pd.DataFrame(result)
    return result.loc[result['correlation'].abs().sort_values(ascending=False, na_position='last').index]


def detect_outliers(frame, args):
    metric = args.get('metric')
    values = _number(frame, metric)
    method = args.get('method', 'iqr')
    if method == 'iqr':
        q1, q3 = values.quantile(.25), values.quantile(.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        selected = values.lt(lower) | values.gt(upper)
        score = (values - values.median()).abs() / iqr if iqr > 0 else (values - values.median()).abs()
    elif method == 'z_score':
        std = values.std(ddof=0)
        score = (values - values.mean()).abs() / std if std > 0 else values * 0
        selected = score.gt(3)
        lower, upper = values.mean() - 3 * std, values.mean() + 3 * std
    else:
        raise ValueError('methodはiqrまたはz_scoreです。')
    columns = _columns(frame, args['columns']) if args.get('columns') else list(frame.columns)
    result = frame.loc[selected, columns].copy()
    for name in ('outlier_score', 'lower_bound', 'upper_bound'):
        if name in result:
            raise ValueError('外れ値の出力列名と元の列名が重なります。columnsを絞ってください。')
    result['outlier_score'] = score.loc[selected]
    result['lower_bound'], result['upper_bound'] = lower, upper
    return result.sort_values('outlier_score', ascending=False, kind='stable')


def analyze_timeseries(frame, args):
    time_column, metric = args.get('time_column'), args.get('metric')
    _columns(frame, [time_column])
    values = _number(frame, metric)
    groups = _group_by(frame, args)
    if time_column in groups or metric in groups:
        raise ValueError('group_byに時間列や指標列を指定しないでください。')
    parsed = pd.to_datetime(frame[time_column], format='mixed', errors='coerce')
    if not pd.api.types.is_datetime64_any_dtype(parsed):
        raise ValueError('日時の形式・タイムゾーンをSQLで統一してください。')
    freq = args.get('frequency', 'month')
    codes = {'day': 'D', 'week': 'W', 'month': 'M', 'year': 'Y'}
    if freq not in codes:
        raise ValueError('frequencyはday/week/month/yearです。')
    if parsed.dt.tz is not None:
        parsed = parsed.dt.tz_localize(None)  # 表示されている業務時刻を保持。UTCへの変換はしない。
    period = parsed.dt.to_period(codes[freq])
    work = frame[groups].copy()
    for reserved in ('period', 'value', 'previous_period', 'delta', 'change_pct', 'moving_average', 'gap'):
        if reserved in groups:
            raise ValueError('group_byが時系列の出力列名と重複します。列をSQLで別名にしてください。')
    work['period'], work['value'] = period, values
    work = work[work['period'].notna()]
    keys = [*groups, 'period']
    aggregation = args.get('aggregation')
    if work.duplicated(keys).any():
        if aggregation not in AGGREGATIONS:
            raise ValueError('同じ期間・内訳が重複しています。aggregationを明示するかSQLで粒度を整えてください。')
        work = aggregate(work, {'group_by': keys, 'aggregations': [
            {'column': 'value', 'function': aggregation, 'alias': 'value'}]})
    work = work.sort_values(keys, kind='stable')
    window = args.get('window', 3)
    if isinstance(window, bool) or not isinstance(window, int) or not 1 <= window <= 100:
        raise ValueError('windowは1〜100の整数です。')
    pieces = []
    grouped = work.groupby(groups, dropna=False, observed=True, sort=False) if groups else [(None, work)]
    for _, series in grouped:
        series = series.copy()
        previous = series['value'].shift(1)
        previous_period = series['period'].shift(1)
        gap = pd.Series([False if pd.isna(before) else after.ordinal - before.ordinal != 1
                         for before, after in zip(previous_period, series['period'])], index=series.index)
        series['gap'] = gap
        series['previous_period'] = previous_period.astype(str).replace('NaT', None)
        series['delta'] = (series['value'] - previous).where(~gap)
        series['change_pct'] = (series['delta'] / previous * 100).where(previous.gt(0) & ~gap)
        series['moving_average'] = series['value'].rolling(window, min_periods=window).mean().where(
            gap.rolling(max(1, window - 1), min_periods=1).sum().eq(0))
        series['period'] = series['period'].astype(str)
        pieces.append(series)
    result = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame(columns=[*keys, 'value', 'delta', 'change_pct', 'moving_average', 'gap'])
    view = args.get('view', 'largest_change')
    if view == 'chronological':
        return result
    orders = {'largest_change': ('delta', False, True), 'largest_increase': ('delta', False, False),
              'largest_decrease': ('delta', True, False), 'largest_rate': ('change_pct', False, True)}
    if view not in orders:
        raise ValueError('viewはchronological/largest_change/largest_increase/largest_decrease/largest_rateです。')
    column, ascending, absolute = orders[view]
    rank = result[column].abs() if absolute else result[column]
    return result.loc[rank.sort_values(ascending=ascending, na_position='last', kind='stable').index]


def get_distinct_values(frame, args):
    column = _columns(frame, args.get('columns'), maximum=1)[0]
    return frame.groupby(column, dropna=False, observed=True).size().reset_index(name='result_rows').sort_values('result_rows', ascending=False, kind='stable')


def sample_data(frame, args):
    columns = _columns(frame, args['columns']) if args.get('columns') else list(frame.columns)
    # 再現可能な無作為順。代表性や全体傾向を保証しない。
    return frame[columns].sample(frac=1, random_state=0)


def local_sql(frame, args, policy, control=None):
    sql = args.get('sql')
    if not isinstance(sql, str) or not sql.strip() or len(sql) > 50_000:
        raise ValueError('sqlは50,000文字以内のSELECTを指定してください。')
    try:
        statements = sqlglot.parse(sql, read='sqlite')
    except sqlglot.errors.ParseError as exc:
        raise ValueError('ローカルSQLの構文を確認してください。') from exc
    if len(statements) != 1 or not isinstance(statements[0], exp.Query):
        raise ValueError('ローカルSQLは1つのSELECTだけ実行できます。')
    tree = statements[0]
    if any(with_.args.get('recursive') for with_ in tree.find_all(exp.With)):
        raise ValueError('再帰CTEは使用できません。')
    ctes = {cte.alias for cte in tree.find_all(exp.CTE)}
    for table in tree.find_all(exp.Table):
        if table.db or table.catalog or table.name not in {'data', *ctes}:
            raise ValueError('ローカルSQLで参照できるのはこの結果のdata表だけです。')
    allowed_functions = {'count', 'sum', 'avg', 'min', 'max', 'abs', 'round', 'coalesce', 'ifnull', 'nullif',
                         'cast', 'length', 'lower', 'upper', 'substr', 'substring', 'replace', 'trim',
                         'date', 'datetime', 'strftime', 'julianday', 'lag', 'lead', 'row_number', 'rank',
                         'dense_rank', 'ntile', 'first_value', 'last_value', 'percent_rank', 'cume_dist',
                         'total', 'typeof', 'iif', 'instr'}
    deadline = time.monotonic() + policy.analysis_timeout_seconds
    connection = sqlite3.connect(':memory:')
    try:
        connection.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 1_000_000)
        connection.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 50_000)
        connection.setlimit(sqlite3.SQLITE_LIMIT_EXPR_DEPTH, 50)
        converted = frame.copy()
        for column in converted.columns:
            if converted[column].dtype == object:
                converted[column] = converted[column].map(_cell)
        converted.to_sql('data', connection, index=False)
        connection.execute('PRAGMA query_only = ON')
        def authorize(action, first, second, database, source):
            if action == sqlite3.SQLITE_READ:
                return sqlite3.SQLITE_OK if first == 'data' else sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_FUNCTION:
                return sqlite3.SQLITE_OK if str(second).lower() in allowed_functions else sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK if action == sqlite3.SQLITE_SELECT else sqlite3.SQLITE_DENY
        connection.set_authorizer(authorize)
        def interrupted():
            if control is not None and control.stopped.is_set():
                return 1
            return int(time.monotonic() > deadline)
        connection.set_progress_handler(interrupted, 1000)
        cursor = connection.execute(sql)
        fetched = cursor.fetchmany(policy.max_preview_rows + 1)
        names = [description[0] for description in cursor.description]
        if len(set(names)) != len(names):
            raise ValueError('SELECTの出力列に重複しない別名を付けてください。')
        return pd.DataFrame(fetched[:policy.max_preview_rows], columns=names), len(fetched) > policy.max_preview_rows
    except sqlite3.Error as exc:
        if control is not None:
            control.check()
        raise ValueError(f'ローカルSQLを実行できません（許可外の処理・型・タイムアウトを確認）: {exc}') from exc
    finally:
        connection.close()


FUNCTIONS = {name: globals()[name] for name in OPERATIONS if name != 'sql'}


def analyze(frame, args, policy=None, *, truncated=False, control=None, return_frame=False):
    policy = policy or BudgetPolicy()
    operation = args.get('operation')
    if operation not in OPERATIONS:
        raise ValueError('未対応のanalysis operationです。')
    if control is not None:
        control.check()
    started = time.monotonic()
    frame = frame.reset_index(drop=True)
    selected = filter_saved_result(frame, args.get('filters'))
    more = False
    if operation == 'sql':
        output, more = local_sql(selected, args, policy, control)
    else:
        output = FUNCTIONS[operation](selected, args)
    if control is not None:
        control.check()
    if time.monotonic() - started > policy.analysis_timeout_seconds:
        raise ValueError('分析がタイムアウトしました。列・対象・処理を絞ってください。')
    limit = _limit(args, policy)
    # limitを小さく指定しても元の分析結果の件数は残す。
    # 外側のscope/limitation/結果IDにも余白を取り、表だけで予算を使い切らない。
    inner_limit = max(256, policy.max_data_tokens - 700)
    inner_policy = replace(policy, max_data_tokens=inner_limit,
                           target_data_tokens=min(policy.target_data_tokens, inner_limit))
    result = TokenBudgetManager(inner_policy).table(output, truncated=truncated)
    if result.get('sent_row_count', 0) > limit:
        result = TokenBudgetManager(inner_policy).table(output.iloc[:limit], truncated=truncated)
        result.update(row_count=len(output), rows_omitted=len(output) - result['sent_row_count'], complete=False)
    if more:
        result.update(complete=False, row_count=len(output) + 1, rows_omitted=None,
                      row_count_is_lower_bound=True, more_rows=True,
                      note='ローカルSQLの結果が返却上限を超えました。WHERE/GROUP BY/ORDER BY/LIMITで絞ってください。')
    payload = {'operation': operation, 'source_row_count': len(frame), 'matched_row_count': len(selected),
            'source_truncated': bool(truncated), 'result': result,
            'analysis_seconds': round(time.monotonic() - started, 6),
            'scope': '保存済み範囲のみ。平均・率・重複するユニーク数の再集計には定義の確認が必要。',
            'limitations': '相関は因果ではない。外れ値は誤りと断定しない。時系列の欠けた期間は0補完しない。SQLiteのDecimal数値演算は近似になる場合があり、精密計算はBigQueryを使う。'}

    if estimate_tokens(payload) > policy.max_data_tokens:
        payload['result'] = TokenBudgetManager(inner_policy).schema(output, truncated=truncated, reason='token_budget')
        if estimate_tokens(payload) > policy.max_data_tokens:
            raise ValueError('メタデータがトークン予算を超えています。列数を絞ってください。')
    return (payload, output) if return_frame else payload
