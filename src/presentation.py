"""回答単位の表示方針。生データの判定と加工はサーバー内で完結する。"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace

import pandas as pd

from .bq_tools import QueryRun
from .charts import ChartSpec, suggest_chart, validate_chart


@dataclass
class Presentation:
    index: int | None = None
    mode: str = "none"
    chart: ChartSpec | None = None
    dataframe: pd.DataFrame | None = None
    note: str = ""
    signature: str = ""


def explicit_format(question: str) -> str | None:
    if re.search(r"文章だけ|文章のみ|図表.*(?:不要|いらない|なし)|図表なし", question):
        return "none"
    chart = re.search(r"グラフ|チャート|可視化|ヒストグラム|散布図|折れ線|棒グラフ", question)
    table = re.search(r"表(?:で|に|形式|を(?:出|見|表示|作))|一覧|明細|リスト", question)
    no_chart = re.search(r"(?:グラフ|チャート)(?:は|を|も|が)?(?:不要|いらない|なし|出さない|表示しない)|グラフではなく", question)
    no_table = re.search(r"表(?:は|を|も|が)?(?:不要|いらない|なし|出さない|表示しない)|表ではなく", question)
    if table and not no_table and not (chart and not no_chart):
        return "table"
    if chart and not no_chart:
        return "chart"
    if no_chart:
        return "none"
    return None


def choose_presentation(
    question: str, runs: list[QueryRun], shown: set[str] | None = None,
) -> Presentation:
    """最後の回答用結果だけを選ぶ。空結果から過去の成功結果へは戻らない。"""
    candidates = [i for i, run in enumerate(runs) if getattr(run, "result_role", "answer") == "answer"]
    if not candidates:
        return Presentation()
    index = candidates[-1]
    df = runs[index].dataframe
    if df.empty:
        return Presentation(index=index)
    requested = explicit_format(question)
    signature = hashlib.sha256(
        (str(list(zip(df.columns, map(str, df.dtypes)))) + df.to_json(date_format="iso", default_handler=str)
         + json.dumps(runs[index].chart, sort_keys=True, default=str)).encode()
    ).hexdigest()
    if requested == "none" or (requested is None and signature in (shown or set())):
        return Presentation(index=index)
    if requested == "table":
        return Presentation(index, "table", signature=signature)

    chart_choice = runs[index].chart
    if isinstance(chart_choice, dict) and chart_choice.get("kind") == "none":
        if requested == "chart":
            return Presentation(index, "table", note="グラフに適した結果がないため、表で表示しています。", signature=signature)
        return Presentation(index=index)
    spec = validate_chart(df, chart_choice) if chart_choice is not None else suggest_chart(df)
    if chart_choice is not None and spec is None:
        return Presentation(index, "table", note="グラフ指定・数値・集計粒度が適切でないため、表で表示しています。", signature=signature)
    chart_df = df
    # 単一数値列の明細も、分布の依頼ならサーバー内で度数分布にする。
    distribution = re.search(r"分布|ヒストグラム", question)
    if distribution and len(df) >= 3 and df.shape[1] == 1 and pd.api.types.is_numeric_dtype(df.iloc[:, 0]):
        values = df.iloc[:, 0].replace([float("inf"), float("-inf")], float("nan")).dropna()
        if not values.empty:
            counts = pd.cut(values, bins=10, duplicates="drop").value_counts(sort=False)
            chart_df = pd.DataFrame({f"{df.columns[0]}の範囲": counts.index.astype(str), "件数": counts.values})
            spec = suggest_chart(chart_df)

    if requested != "chart" and chart_choice is None:
        explanation = re.search(r"なぜ|理由|原因|どうして|説明|考察", question)
        visual_intent = re.search(r"推移|月別|日別|週別|年別|時系列|比較|店舗別|ランキング|上位|分布|傾向|内訳|相関", question)
        if len(df) < 3 or explanation or not (visual_intent or (spec and spec.kind == "line")):
            return Presentation(index=index)
    if spec is None:
        if requested == "chart":
            return Presentation(index, "table", note="グラフに適した列がないため、表で表示しています。", signature=signature)
        return Presentation(index=index)

    requested_kind = next((kind for word, kind in [("棒グラフ", "bar"), ("折れ線", "line"), ("散布図", "scatter"), ("面グラフ", "area")] if word in question), None)
    if requested_kind:
        spec = replace(spec, kind=requested_kind)

    note = ""
    if spec.color:
        # 同じ軸・内訳が重複すると描画時の暗黙の集約で値が変わるので、表へ戻す。
        if chart_df.duplicated([spec.x, spec.color]).any():
            return Presentation(index, "table", note="同じ軸・内訳の組に複数行があるため、表で表示しています。", signature=signature)
        chart_df = chart_df.copy()
        chart_df[spec.color] = chart_df[spec.color].astype("string").fillna("未設定")
    if spec.kind == "bar" and chart_df[spec.x].nunique() > 10:
        y = spec.y[0]
        totals = chart_df.groupby(spec.x, observed=True)[y].sum(min_count=1)
        top = totals.nlargest(10).index
        chart_df = chart_df[chart_df[spec.x].isin(top)].copy()
        chart_df = chart_df.sort_values(spec.x, key=lambda series: series.map({value: i for i, value in enumerate(top)}), kind="stable")
        note = f"「{y}」の合計による上位10カテゴリを、内訳を含めて表示しています。全件は「データ・実行詳細」で確認できます。"
    spec = replace(spec, reason="")
    return Presentation(index, "chart", spec, chart_df, note, signature)
