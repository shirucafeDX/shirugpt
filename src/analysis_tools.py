"""Analysis Layerの選択式ツール定義。選択された時だけ詳細を送る。"""
from google.genai import types
from .analysis_layer import OPERATIONS, AGGREGATIONS


def declaration(filters):
    def string(description='', enum=None):
        return types.Schema(type=types.Type.STRING, description=description or None, enum=enum)
    def strings(description='', maximum=12):
        return types.Schema(type=types.Type.ARRAY, items=string(), max_items=maximum, description=description or None)
    return types.FunctionDeclaration(
        name='analyze_result',
        description='保存済み結果をローカル分析。行データを全部読む必要はない。まずschema/describe、次に必要な集計・比較・絞り込み。BigQuery再実行なし。',
        parameters=types.Schema(type=types.Type.OBJECT, properties={
            'result_id': string('この会話の取得済み結果ID'),
            'operation': string(enum=list(OPERATIONS)),
            'columns': strings('出力列またはdescribe/correlationの対象列。相関は2〜6列'),
            'filters': filters,
            'group_by': strings('集計・比較・時系列の内訳列', 6),
            'aggregations': types.Schema(type=types.Type.ARRAY, max_items=8,
                items=types.Schema(type=types.Type.OBJECT, properties={
                    'column': string(), 'function': string(enum=list(AGGREGATIONS)),
                    'alias': string('重複しない出力名'),
                }, required=['column', 'function', 'alias'])),
            'sort_by': strings('top_kの並べ替え列。最大3列', 3),
            'ascending': types.Schema(type=types.Type.BOOLEAN, description='trueなら下位。既定false'),
            'absolute': types.Schema(type=types.Type.BOOLEAN, description='top_kで絶対値順にする場合true'),
            'metric': string('比較・外れ値・時系列で計算する数値列'),
            'aggregation': string('同じ期間の重複を集計する方法。compare_groupsの既定mean', list(AGGREGATIONS)),
            'baseline': string('比較元カテゴリの値。未指定なら取得済み全体の同じ集計値'),
            'method': string('correlation: pearson/spearman、detect_outliers: iqr/z_score'),
            'time_column': string('時系列の日時列'),
            'frequency': string(enum=['day', 'week', 'month', 'year']),
            'window': types.Schema(type=types.Type.INTEGER, description='移動平均の期間数。既定3'),
            'view': string('時系列の選択方法', ['chronological', 'largest_change', 'largest_increase', 'largest_decrease', 'largest_rate']),
            'limit': types.Schema(type=types.Type.INTEGER, description='返却行数。既定/上限はMAX_PREVIEW_ROWS（既定50）'),
            'offset': types.Schema(type=types.Type.INTEGER, description='filter_dataの開始行。既定0'),
            'sql': string('operation=sql専用。保存済みdata表に対するSQLite SELECT。CTE・window可、書き込み・外部参照・再帰不可。元データの精密Decimal集計はBigQueryで行う。'),
        }, required=['result_id', 'operation']))
