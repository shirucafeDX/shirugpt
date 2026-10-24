"""AIのグラフ指定の検証と、指定のない旧履歴向けの推定。追加のAI呼び出しは行わない。"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

# 棒グラフはカテゴリが多すぎると読めないので上限を設ける
MAX_BAR_CATEGORIES = 60

KIND_LABELS = {
    "bar": "棒グラフ",
    "line": "折れ線",
    "area": "面グラフ",
    "scatter": "散布図",
    "pie": "円グラフ",
    "heatmap": "ヒートマップ",
}


@dataclass
class ChartSpec:
    kind: str
    x: str
    y: list[str]
    reason: str = ""
    candidates_x: list[str] = field(default_factory=list)
    candidates_y: list[str] = field(default_factory=list)
    color: str | None = None
    row: str | None = None


def _classify(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """列を (日時, 数値, カテゴリ) に分類する。"""
    datetimes: list[str] = []
    numerics: list[str] = []
    categoricals: list[str] = []

    for col in df.columns:
        series = df[col]
        if pd.api.types.is_datetime64_any_dtype(series):
            datetimes.append(str(col))
        elif pd.api.types.is_bool_dtype(series):
            categoricals.append(str(col))
        elif pd.api.types.is_numeric_dtype(series):
            numerics.append(str(col))
        else:
            # DATE / TIME は object 型で来ることがあるので中身で判定する
            valid = series.dropna()
            if not valid.empty and hasattr(valid.iloc[0], "isoformat"):
                datetimes.append(str(col))
            else:
                categoricals.append(str(col))

    return datetimes, numerics, categoricals


def suggest_chart(df: pd.DataFrame) -> ChartSpec | None:
    """DataFrame の列構成からグラフを推定する。描けなければ None。"""
    if df is None or df.empty or len(df) < 2 or df.shape[1] < 2:
        return None

    datetimes, numerics, categoricals = _classify(df)
    if not numerics:
        return None

    candidates_x = datetimes + categoricals + numerics

    # 時系列があれば折れ線
    if datetimes:
        return ChartSpec(
            kind="line",
            x=datetimes[0],
            y=numerics[:3],
            reason=f"日時列「{datetimes[0]}」があるため時系列として表示しています。",
            candidates_x=candidates_x,
            candidates_y=numerics,
        )

    # カテゴリ × 数値 は棒グラフ
    if categoricals and len(df) <= MAX_BAR_CATEGORIES:
        return ChartSpec(
            kind="bar",
            x=categoricals[0],
            y=numerics[:3],
            reason=f"カテゴリ列「{categoricals[0]}」と数値列の組み合わせです。",
            candidates_x=candidates_x,
            candidates_y=numerics,
        )

    # 数値同士は散布図
    if len(numerics) >= 2:
        return ChartSpec(
            kind="scatter",
            x=numerics[0],
            y=numerics[1:2],
            reason="数値列が複数あるため散布図として表示しています。",
            candidates_x=candidates_x,
            candidates_y=numerics,
        )

    if categoricals:
        return ChartSpec(
            kind="bar",
            x=categoricals[0],
            y=numerics[:1],
            reason=(
                f"カテゴリが {len(df)} 件あります。"
                f"棒グラフは {MAX_BAR_CATEGORIES} 件程度までが見やすいため、"
                "必要に応じて SQL 側で上位に絞り込んでください。"
            ),
            candidates_x=candidates_x,
            candidates_y=numerics,
        )

    return None


def validate_chart(df: pd.DataFrame, value: object) -> ChartSpec | None:
    """AIが指定した列を実結果で検証する。不正な指定でデータを誤表示しない。"""
    if not isinstance(value, dict) or not isinstance(value.get("kind"), str) or value["kind"] not in KIND_LABELS:
        return None
    x, y, color = value.get("x"), value.get("y"), value.get("color") or None
    if not isinstance(x, str) or x not in df.columns or not isinstance(y, list) or not y:
        return None
    if any(not isinstance(col, str) or col not in df.columns or col == x for col in y):
        return None
    if len(y) != len(set(y)) or any(not pd.api.types.is_numeric_dtype(df[col]) or pd.api.types.is_bool_dtype(df[col]) for col in y):
        return None
    if color is not None and (not isinstance(color, str) or color not in df.columns or color == x or color in y or len(y) != 1):
        return None
    row = value.get("row") or None
    if value["kind"] in ("pie", "heatmap"):
        if len(y) != 1 or color is not None:
            return None
        values = df[y[0]]
        if values.isin([float("inf"), float("-inf")]).any() or values.dropna().empty:
            return None
        if value["kind"] == "pie":
            # 負数・合計0や同じカテゴリの重複から、誤った構成比を作らない。
            if (values.dropna() < 0).any() or values.isna().any() or values.sum() <= 0 or df.duplicated([x]).any():
                return None
        else:
            if not isinstance(row, str) or row not in df.columns or row == x or row in y:
                return None
            if df.duplicated([x, row]).any():
                return None
    elif row is not None:
        return None
    if value["kind"] != "heatmap" and row is not None:
        return None
    return ChartSpec(kind=value["kind"], x=x, y=list(y), color=color, row=row)


def special_chart_spec(spec: ChartSpec) -> dict:
    """円・ヒートマップを描くVega-Lite指定。列名はネストしたフィールドとして解釈させない。"""
    def field(column: str, kind: str) -> dict:
        escaped = column.replace("\\", "\\\\").replace(".", "\\.").replace("[", "\\[").replace("]", "\\]")
        return {"field": escaped, "type": kind, "title": column}

    metric = field(spec.y[0], "quantitative")
    if spec.kind == "pie":
        category = field(spec.x, "nominal")
        return {
            "mark": {"type": "arc", "outerRadius": 130},
            "encoding": {
                "theta": {**metric, "stack": True},
                "color": category,
                "tooltip": [category, metric],
            },
        }
    if spec.kind == "heatmap" and spec.row:
        horizontal, vertical = field(spec.x, "ordinal"), field(spec.row, "ordinal")
        return {
            "mark": {"type": "rect"},
            "encoding": {
                "x": {**horizontal, "sort": "ascending"},
                "y": {**vertical, "sort": "ascending"},
                "color": {**metric, "scale": {"scheme": "blues"}},
                "tooltip": [horizontal, vertical, metric],
            },
        }
    raise ValueError("円グラフまたはヒートマップを指定してください。")
