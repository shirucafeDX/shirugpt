"""Vertex AI (Gemini) の Function Calling ループ。

設計上のポイント:
  * スキーマは list_tables → get_table_schema の 2 段階でしか渡さない
  * クエリ結果は要約だけを tool response として返す（生データは渡さない）
  * 取得済みスキーマ・実行済みクエリはコンテキスト側でキャッシュし再取得させない
  * 高コストクエリは ConfirmationPending を返して一旦ループを中断し、
    ユーザーの承認後に同じ地点から再開できる
"""

from __future__ import annotations

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Any, Callable

from google import genai
from google.genai import types

from .bq_tools import (
    BigQueryTools,
    BQToolError,
    ConfirmationRequired,
    QueryRun,
    error_code,
    friendly_error,
    technical_detail,
)
from .config import Settings, human_bytes
from .sql_guard import SQLGuardError
from .usage_log import UsageLogger
from .execution import ExecutionControl, ExecutionStopped

logger = logging.getLogger(__name__)

SYSTEM_INSTRUCTION = """あなたは BigQuery のデータ分析アシスタントです。
ユーザーの質問に対し、提供されたツールを使って BigQuery を調べ、日本語で回答します。

## 進め方
0. 確認質問の前に、質問に関連するテーブルと列・型・説明をツールで確認し、
   どのデータで何を分析できるかを把握する。確認済みの関連スキーマは再利用してよい。
   店舗名・商品名などの値も、必要なら少量の集計や DISTINCT で実在を確認する。
   売上・顧客単価など、必要な列が確認できていない指標を選択肢に含めない。
   そのうえで期間・対象店舗・指標など、回答を左右する条件が曖昧なら
   ask_clarification でユーザーに確認する。質問と選択肢はその都度、日本語で作る。
   ユーザーの回答を求める確認質問は、必ず ask_clarification を呼び出して提示する。
   通常の回答本文に質問と箇条書きの選択肢を書くだけで終了してはいけない。
   例: 「京大店の強み」のように分析の観点を確認する場合も、question に確認内容、
   options に選択肢を指定して呼ぶ。画面が選択式フォームと自由記述欄を表示する。
   一度に確認するのは1項目とし、選びやすい2〜5個の選択肢を用意する。
   selection_mode は、排他的な条件を1つ選ぶ場合（集計単位など）は single、
   複数を同時に選べる場合（比較したい店舗や分析指標など）は multi にする。
   質問文にも「1つ」「複数選択可」など回答方法を明記する。
   自由記述欄は画面が付けるため、選択肢に「その他」「自由記述」を含めない。
   確認が不要な明確な質問には、そのまま回答する。既に回答された条件を再確認しない。
   確認したデータで分析できる選択肢だけを提示する。
   ask_clarification は単独で呼び、ユーザーの回答を待ってから分析を続ける。
1. どのテーブルを見るか不明なときは list_datasets → list_tables でテーブル名を確認する。
2. 使いそうなテーブルにだけ get_table_schema を呼び、列名と型を確認する。
   全テーブルのスキーマを片端から取得してはいけない。多くても 3〜4 テーブルに絞る。
3. 確認した実在の列名だけを使って SQL を組み立て、run_query で実行する。
4. 返ってくるのは行数・列情報・統計・先頭数行のサンプルだけです。
   全行データは画面側でユーザーに表示されるので、あなたが全件を見る必要はありません。
5. 結果をもとに回答する。数値は単位を添え、必要なら次に見るとよい観点を一言添える。

## 定義
- connected_id はユーザを識別するID。
- 来店テーブルのカラム DU_id のユニークカウントを来店数またはDUとして扱う。
- Meetup参加テーブルのカラム attendance は参加すると1、そうでないと0。合計値を 参加数として扱う。
- Meetup参加テーブルのカラム planned_attendance は参加予定だと1、そうでないと0。合計値を 参加予定数として扱う。
- Meetup参加テーブルのカラム cancell は事前キャンセルをすると1、そうでないとは0。合計値を 事前キャンセル数として扱う。
- Meetup参加テーブルのカラム no_show は無断欠席をすると1、そうでないとは0。合計値を 無断欠席数として扱う。
- キャンセル数は 事前キャンセル数 + 無断欠席数 と定義する。
- 予約数は 参加数 + 参加予定数 + キャンセル数 と定義する。
- 参加率やキャンセル率は 参加数 / 予約数、キャンセル数 / 予約数 のように計算する。

## 日時・タイムゾーンの扱い
- 業務上の日付・時間帯の集計と回答は日本時間（Asia/Tokyo）を基準にする。
  必ず get_table_schema で DATE / DATETIME / TIMESTAMP の型を確認する。
- 来店.ordered_at、Meetup参加.start_at / end_at、イベント.start_at / end_at の
  DATETIME は日本時間の業務日時として扱う。そのまま集計・表示し、9時間を加算しない。
  DATETIME を CAST(... AS TIMESTAMP) で UTC と解釈してから Asia/Tokyo に
  変換してはいけない。過去の会話にこの変換をした SQL があっても踏襲しない。
  例: 来店の時間帯は EXTRACT(HOUR FROM ordered_at)、
      日付は DATE(ordered_at)、曜日は EXTRACT(DAYOFWEEK FROM ordered_at) を使う。
- 日次データ.date、月次データ.month の DATETIME は日付・月を表す値として扱う。
  00:00:00 を来店・開催などの実際の発生時刻として説明しない。
  DATE 型にも時刻の情報はないため、実際の時刻を推測しない。
- TIMESTAMP は時点を表す型なので、日本時間の日付・時間帯の集計では
  DATE(timestamp列, 'Asia/Tokyo') や DATETIME(timestamp列, 'Asia/Tokyo') を使う。
  タイムゾーン変換と手動の9時間加算を重ねない。
- 上記以外の DATETIME や文字列の日時は、日本時間・UTC のどちらかを型だけで
  決めつけない。列の説明などで意味を確認し、回答を左右する場合はユーザーに確認する。
  不自然な深夜の値があっても、根拠なく時刻を補正・除外しない。

## SQL を書くときの規則
- テーブルは必ず `プロジェクトID.データセット.テーブル名` の形でバッククォートで囲む。
- 日本語・空白・記号を含む識別子は、テーブル名・列名だけでなく
  **AS で付けるエイリアスも必ずバッククォートで囲む**。
  BigQuery は裸の日本語識別子を構文エラーにする。
    正: SELECT COUNT(*) AS `件数` FROM `proj.202506.来店`
    誤: SELECT COUNT(*) AS 件数   FROM `proj.202506.来店`
  迷う場合は英数字のエイリアス（例: AS cnt）を使ってもよい。
- 読み取り専用の SELECT 文のみ。INSERT / UPDATE / DELETE / CREATE 等は実行できません。
- SELECT * は避け、必要な列だけを指定する（スキャン量の削減）。
- 件数を数えるだけなら COUNT(*)、傾向を見るなら GROUP BY と集計関数を使う。
  生データを大量に取り出すのではなく、まず集計する。
- パーティション列がある場合は WHERE で期間を絞る。
- 「今日」「今月」「先月」などの相対日付は CURRENT_DATE('Asia/Tokyo') を基準に
  計算する。現在の業務日時が必要なら CURRENT_DATETIME('Asia/Tokyo') を使う。
- 推測した列名を使ってはいけない。必ず get_table_schema で確認した列名を使う。

## エラーへの対処
run_query が error を返したら、内容を読んで SQL を修正し、最大 2 回まで再試行する。
それでも解決しない場合は、何が分からなかったかをユーザーに日本語で説明する。
スキャン量が大きすぎると言われた場合は、期間を絞る・列を減らす・集計に変えるなどして作り直す。

## 回答スタイル
- 結論を文章で先に示す。単一の数値・少数の比較は数値と差分を文章で説明する。
- 図表は画面側が質問と結果から選ぶ。表を明示的に求められていない限り、
  回答本文に Markdown の表を重ねて書かない。SQL や調査途中の結果も本文に列挙しない。
- run_query の result_role は、探索・検証・途中計算なら supporting、
  質問の結論に使う結果なら answer を指定する。画面は最後の answer のみを表示候補にする。
  推移・分布・比較の回答用クエリは、必要な系列を1つの結果にまとめる。
- run_query の purpose には「店舗別の来店数を集計」のように、何を確認するかを
  短い日本語で必ず指定する。実行中の進捗と実行詳細に表示される。
- 日本語で、データにない内容を推測で補わない。
- 質問の種類によって長さを変える。
  - 件数・合計・単純な集計結果など「事実を聞かれただけ」の質問には、
    数値と単位を添えて簡潔に答える。前置きや余計な解説は付けない。
  - 「傾向はどうか」「なぜそうなっているか」「何が言えるか」のように、
    データから単純に読み取れない考察を求められた質問には、簡潔さを優先しない。
    比較対象・時系列の変化・考えられる要因など、データから言える範囲で
    根拠を示しながら踏み込んで説明する。分からない部分は「分からない」と
    明言し、憶測で埋めない。
- 結果が 0 件だったときは「該当なし」と明言し、条件の見直し案を提示する。
- SQL 全文を回答本文に貼り付ける必要はありません（画面に別途表示されます）。
"""


def _sql_key(sql: str) -> str:
    return hashlib.sha256(sql.strip().encode("utf-8")).hexdigest()


def tool_progress(call: Any) -> str:
    """ツールの実行内容を、SQL を出さず短い日本語で伝える。"""
    args = dict(call.args or {})
    if call.name == "list_datasets":
        return "利用できるデータセットを確認しています…"
    if call.name == "list_tables":
        return f"「{args.get('dataset', '')}」のテーブル一覧を確認しています…"
    if call.name == "get_table_schema":
        return f"「{args.get('dataset', '')}.{args.get('table', '')}」の列名・データ型を確認しています…"
    if call.name == "run_query":
        purpose = str(args.get("purpose") or "").strip()
        return f"データを取得・集計しています：{purpose}" if purpose else "データを取得・集計しています…"
    if call.name == "ask_clarification":
        return "確認する質問と選択肢を用意しています…"
    return "調査に必要な情報を確認しています…"


@dataclass
class ToolContext:
    """1 会話分のツール実行状態。Streamlit の session_state に載せる。"""

    tools: BigQueryTools
    settings: Settings
    schema_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    listing_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    query_cache: dict[str, QueryRun] = field(default_factory=dict)
    decisions: dict[str, str] = field(default_factory=dict)
    resume_calls: list[Any] | None = None
    execution_control: ExecutionControl | None = None

    # 直近ターンの記録（UI 表示用）
    turn_runs: list[QueryRun] = field(default_factory=list)
    turn_errors: list[str] = field(default_factory=list)
    turn_prompt_tokens: int = 0
    turn_output_tokens: int = 0
    turn_thinking_tokens: int = 0
    turn_total_tokens: int = 0

    # セッション全体の累計（サイドバー表示用）
    session_prompt_tokens: int = 0
    session_output_tokens: int = 0
    session_total_tokens: int = 0
    session_billed_bytes: int = 0

    # 利用ログ（Gemini 境界の外。UsageLogger 側で例外は握られる）
    usage_logger: UsageLogger | None = None
    user_email: str = ""
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    turn_id: str = ""

    # 月次の使用制限判定用ベースライン（app.py 側で更新する。ロジックは
    # ここには置かない。BigQuery への問い合わせを毎ターン避けるため、
    # 月の開始時点の使用量 + このセッションでの増分、で近似する）
    limit_baseline_usd: float = 0.0
    limit_baseline_month_start: datetime | None = None
    limit_baseline_session_prompt_tokens: int = 0
    limit_baseline_session_output_tokens: int = 0
    limit_baseline_session_billed_bytes: int = 0

    def start_turn(self) -> None:
        self.turn_runs = []
        self.turn_errors = []
        self.turn_prompt_tokens = 0
        self.turn_output_tokens = 0
        self.turn_thinking_tokens = 0
        self.turn_total_tokens = 0
        self.turn_id = uuid.uuid4().hex


@dataclass
class AnswerResult:
    text: str


@dataclass
class ClarificationResult:
    text: str
    options: list[str]
    selection_mode: str = "single"


@dataclass
class ConfirmationPending:
    key: str
    sql: str
    estimated_bytes: int
    threshold_bytes: int
    notes: list[str] = field(default_factory=list)


class _Pending(Exception):
    def __init__(self, key: str, need: ConfirmationRequired):
        super().__init__("confirmation required")
        self.key = key
        self.need = need


def build_declarations() -> list[types.FunctionDeclaration]:
    return [
        types.FunctionDeclaration(
            name="ask_clarification",
            description=(
                "分析に必要な条件・観点をユーザーに確認し、選択式フォームで回答を待つ。"
                "先に関連するテーブルのスキーマと必要な値を確認し、実データに基づく質問・選択肢を作る。"
                "確認質問は必ずこのツールで提示し、通常の回答本文に質問・選択肢を書くだけで終了しない。"
                "単独で呼ぶ。自由記述欄は画面側が付けるため、その他・自由記述を選択肢に含めない。"
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "question": types.Schema(type=types.Type.STRING, description="確認したい1項目の具体的な質問（日本語）"),
                    "options": types.Schema(
                        type=types.Type.ARRAY,
                        items=types.Schema(type=types.Type.STRING),
                        description="回答の選択肢2〜5個。その他・自由記述は含めない。",
                        min_items=2, max_items=5,
                    ),
                    "selection_mode": types.Schema(
                        type=types.Type.STRING, enum=["single", "multi"],
                        description="排他的な条件を1つ選ぶなら single、店舗・指標など複数選べるなら multi。",
                    ),
                },
                required=["question", "options", "selection_mode"],
            ),
        ),
        types.FunctionDeclaration(
            name="list_datasets",
            description=(
                "アクセスが許可されている BigQuery データセットの一覧を返す。"
                "どのデータセットを見ればよいか分からないときに最初に呼ぶ。"
            ),
            parameters=types.Schema(type=types.Type.OBJECT, properties={}),
        ),
        types.FunctionDeclaration(
            name="list_tables",
            description=(
                "指定したデータセット内のテーブル名一覧を返す。スキーマ（列情報）は含まれない。"
                "どのテーブルが存在するかを確認するために使う。"
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "dataset": types.Schema(
                        type=types.Type.STRING,
                        description="データセット ID（例: 202506）",
                    )
                },
                required=["dataset"],
            ),
        ),
        types.FunctionDeclaration(
            name="get_table_schema",
            description=(
                "指定した 1 テーブルのスキーマ（列名・型・パーティション情報）を返す。"
                "SQL を書く前に、使う予定のテーブルについてのみ呼ぶこと。"
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "dataset": types.Schema(
                        type=types.Type.STRING, description="データセット ID"
                    ),
                    "table": types.Schema(
                        type=types.Type.STRING,
                        description="テーブル名（list_tables で得た正確な名前）",
                    ),
                },
                required=["dataset", "table"],
            ),
        ),
        types.FunctionDeclaration(
            name="run_query",
            description=(
                "読み取り専用の SELECT クエリを BigQuery で実行し、結果の要約"
                "（行数・列と型・数値統計・先頭数行のサンプル）を返す。"
                "全行データは返らない（ユーザーの画面には別途表示される）。"
                "テーブルは必ずバッククォートで完全修飾すること。"
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "sql": types.Schema(
                        type=types.Type.STRING,
                        description="実行する SELECT 文（1 ステートメントのみ）",
                    ),
                    "purpose": types.Schema(
                        type=types.Type.STRING,
                        description="このクエリで何を確認したいかの短い説明（日本語）",
                    ),
                    "result_role": types.Schema(
                        type=types.Type.STRING,
                        enum=["supporting", "answer"],
                        description="探索・検証・途中計算は supporting、結論に使う最終結果は answer。省略時は answer。",
                    ),
                },
                required=["sql"],
            ),
        ),
    ]


class GeminiAgent:
    def __init__(self, settings: Settings, client: genai.Client | None = None):
        self.settings = settings
        self.client = client or genai.Client(
            vertexai=True,
            project=settings.gcp_project,
            location=settings.vertex_location,
        )
        self.declarations = build_declarations()

    # -- 生成 ---------------------------------------------------------
    def _config(self) -> types.GenerateContentConfig:
        kwargs: dict[str, Any] = {
            "system_instruction": SYSTEM_INSTRUCTION,
            "tools": [types.Tool(function_declarations=self.declarations)],
            "temperature": 0.1,
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        }
        if self.settings.thinking_budget is not None:
            kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=self.settings.thinking_budget
            )
        return types.GenerateContentConfig(**kwargs)

    def _generate(self, contents: list[types.Content]):
        return self.client.models.generate_content(
            model=self.settings.gemini_model,
            contents=contents,
            config=self._config(),
        )

    # -- ツール実行 ---------------------------------------------------
    def _call_tool(
        self, name: str, args: dict[str, Any], ctx: ToolContext
    ) -> dict[str, Any]:
        if name == "ask_clarification":
            if not ctx.schema_cache:
                raise BQToolError(
                    "確認質問の前に、関連するテーブルの get_table_schema を呼んでください。"
                    "取得できるデータを把握し、分析できる選択肢だけを提示してください。"
                )
            question = args.get("question")
            options = args.get("options")
            selection_mode = args.get("selection_mode", "single")
            if selection_mode not in ("single", "multi"):
                raise BQToolError("selection_mode は single または multi を指定してください。")
            if not isinstance(question, str) or not question.strip():
                raise BQToolError("question に確認したい質問を指定してください。")
            if (not isinstance(options, list) or not 2 <= len(options) <= 5
                    or any(not isinstance(option, str) or not option.strip() for option in options)):
                raise BQToolError("options に空でない選択肢を2〜5個指定してください。")
            options = [option.strip() for option in options]
            if len(set(options)) != len(options):
                raise BQToolError("options の選択肢は重複しないようにしてください。")
            return {
                "status": "awaiting_user", "question": question.strip(),
                "options": options, "selection_mode": selection_mode,
            }

        if name == "list_datasets":
            if "datasets" not in ctx.listing_cache:
                ctx.listing_cache["datasets"] = ctx.tools.list_datasets()
            return ctx.listing_cache["datasets"]

        if name == "list_tables":
            dataset = str(args.get("dataset", "")).strip()
            cache_key = f"tables:{dataset}"
            if cache_key not in ctx.listing_cache:
                ctx.listing_cache[cache_key] = ctx.tools.list_tables(dataset)
            return ctx.listing_cache[cache_key]

        if name == "get_table_schema":
            dataset = str(args.get("dataset", "")).strip()
            table = str(args.get("table", "")).strip()
            cache_key = f"{dataset}.{table}"
            if cache_key in ctx.schema_cache:
                cached = dict(ctx.schema_cache[cache_key])
                cached["cached"] = True
                return cached
            schema = ctx.tools.get_table_schema(dataset, table)
            ctx.schema_cache[cache_key] = schema
            return schema

        if name == "run_query":
            return self._run_query(args, ctx)

        raise BQToolError(f"未知のツール「{name}」が呼び出されました。")

    def _run_query(self, args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        sql = str(args.get("sql", "")).strip()
        if not sql:
            raise BQToolError("sql が指定されていません。")

        key = _sql_key(sql)
        result_role = str(args.get("result_role", "answer"))
        if result_role not in ("supporting", "answer"):
            raise BQToolError("result_role は supporting または answer を指定してください。")

        # 同一 SQL の再実行・再課金を防ぐ。再表示の要否は UI で決める。
        if key in ctx.query_cache:
            run = ctx.query_cache[key]
            # 図表だけを求める追質問にも対応する。履歴内の結果は書き換えない。
            ctx.turn_runs = [r for r in ctx.turn_runs if _sql_key(r.original_sql) != key]
            ctx.turn_runs.append(replace(run, result_role=result_role))
            payload = self._query_payload(run)
            payload["cached"] = True
            return payload

        if ctx.decisions.get(key) == "denied":
            return {
                "error": (
                    "スキャン量が大きいため、ユーザーがこのクエリの実行を承認しませんでした。"
                    "対象期間を絞る、列を減らす、集計に変えるなどしてスキャン量を減らした"
                    "別のクエリを作り直してください。"
                )
            }

        purpose = str(args.get("purpose", "")).strip()
        approved = ctx.decisions.get(key) == "approved"
        try:
            run = ctx.tools.run_query(sql, approved=approved, purpose=purpose)
        except ConfirmationRequired as need:
            raise _Pending(key, need) from None

        run.result_role = result_role
        ctx.query_cache[key] = run
        ctx.turn_runs.append(run)
        ctx.session_billed_bytes += run.billed_bytes
        # キャッシュヒット時は上で早期 return しており課金も発生しないため、
        # ログもここ（新規実行時）だけに置く。二重計上を避ける。
        if ctx.usage_logger is not None:
            ctx.usage_logger.log_bq_query(
                user_email=ctx.user_email,
                session_id=ctx.session_id,
                turn_id=ctx.turn_id,
                billed_bytes=run.billed_bytes,
            )
        return self._query_payload(run)

    @staticmethod
    def _query_payload(run: QueryRun) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "executed_sql": run.sql,
            "bytes_processed": human_bytes(run.estimated_bytes),
            "result": run.summary,
        }
        if run.notes:
            payload["notes"] = run.notes
        return payload

    def _dispatch(self, call: Any, ctx: ToolContext) -> types.Part:
        name = call.name
        args = dict(call.args or {})

        try:
            payload = self._call_tool(name, args, ctx)
        except ExecutionStopped:
            raise
        except _Pending:
            raise
        except (SQLGuardError, BQToolError) as exc:
            if ctx.execution_control is not None:
                ctx.execution_control.check()
            ctx.turn_errors.append(self._error_report(name, args, str(exc)))
            payload = {"error": str(exc)}
            if ctx.usage_logger is not None:
                ctx.usage_logger.log_error(
                    user_email=ctx.user_email,
                    session_id=ctx.session_id,
                    turn_id=ctx.turn_id,
                    error_source="tool_error",
                    error_code=error_code(exc),
                    error_text=technical_detail(exc, limit=10_000),
                )
        except Exception as exc:  # noqa: BLE001 - Gemini に自己修正させる
            if ctx.execution_control is not None:
                ctx.execution_control.check()
            detail = technical_detail(exc)
            ctx.turn_errors.append(
                self._error_report(name, args, friendly_error(exc), detail)
            )
            payload = {
                "error": friendly_error(exc),
                "technical_detail": detail,
            }
            if ctx.usage_logger is not None:
                ctx.usage_logger.log_error(
                    user_email=ctx.user_email,
                    session_id=ctx.session_id,
                    turn_id=ctx.turn_id,
                    error_source="tool_error",
                    error_code=error_code(exc),
                    error_text=technical_detail(exc, limit=10_000),
                )

        return types.Part.from_function_response(name=name, response=payload)

    @staticmethod
    def _error_report(
        name: str, args: dict[str, Any], message: str, detail: str | None = None
    ) -> str:
        """UI の診断情報向けに、失敗時の SQL・技術的詳細を1つにまとめる。"""
        lines = [message]
        if name == "run_query" and args.get("sql"):
            lines.append(f"SQL: {args['sql']}")
        if detail:
            lines.append(f"詳細: {detail}")
        return "\n".join(lines)

    @staticmethod
    def _answer(contents: list[types.Content], text: str,
                content: types.Content | None = None) -> AnswerResult:
        # UI だけでなく、次の API 呼び出しにも最終回答を引き継ぐ。
        # 元の Content を使うことで thought signature も保持する。
        contents.append(content if content is not None else types.Content(
            role="model", parts=[types.Part.from_text(text=text)]
        ))
        return AnswerResult(text)

    @staticmethod
    def _record_generation_issue(ctx: ToolContext, source: str, detail: str,
                                 response: Any = None, reason: str | None = None) -> None:
        usage = getattr(response, "usage_metadata", None)
        candidate = (getattr(response, "candidates", None) or [None])[0]
        diagnostics = {
            "detail": detail,
            "token_scope": "response",
            "finish_reason": reason,
            "finish_message": getattr(candidate, "finish_message", None),
            "prompt_feedback": repr(getattr(response, "prompt_feedback", None)),
            "prompt_tokens": getattr(usage, "prompt_token_count", None),
            "output_tokens": ((usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)) if usage else None,
            "thinking_tokens": getattr(usage, "thoughts_token_count", None),
            "total_tokens": getattr(usage, "total_token_count", None),
        }
        if response is None:
            diagnostics.update(
                token_scope="turn", prompt_tokens=ctx.turn_prompt_tokens,
                output_tokens=ctx.turn_output_tokens, thinking_tokens=ctx.turn_thinking_tokens,
                total_tokens=ctx.turn_total_tokens,
            )
        encoded = json.dumps(diagnostics, ensure_ascii=False)
        logger.warning("%s: %s", source, encoded)
        ctx.turn_errors.append(encoded)
        if ctx.usage_logger is not None:
            # トークンは診断用 JSON 内のみ。課金用の列には二重計上しない。
            ctx.usage_logger.log_error(
                user_email=ctx.user_email, session_id=ctx.session_id,
                turn_id=ctx.turn_id, error_source=source, error_text=encoded,
            )

    # -- ループ本体 ---------------------------------------------------
    def run(
        self, contents: list[types.Content], ctx: ToolContext,
        on_progress: Callable[[str], None] | None = None,
    ) -> AnswerResult | ConfirmationPending | ClarificationResult:
        """1 ターンを最後まで進める。確認が必要な場合は中断して状態を残す。

        contents は破壊的に更新される（Gemini の応答とツール結果を追記）。
        """
        pending_calls = ctx.resume_calls
        ctx.resume_calls = None
        processed_tools = False

        for _ in range(self.settings.max_tool_iterations):
            if ctx.execution_control is not None:
                ctx.execution_control.check()
            if pending_calls is None:
                if on_progress is not None:
                    on_progress(
                        "取得したデータを確認し、回答をまとめています…" if ctx.turn_runs else
                        "確認した情報をもとに、分析を進めています…" if processed_tools else
                        "質問に合う調べ方を考えています…"
                    )
                try:
                    response = self._generate(contents)
                except Exception as exc:  # noqa: BLE001 - 記録してから同じ例外を上に伝える
                    if ctx.execution_control is not None:
                        ctx.execution_control.check()
                    if ctx.usage_logger is not None:
                        ctx.usage_logger.log_error(
                            user_email=ctx.user_email,
                            session_id=ctx.session_id,
                            turn_id=ctx.turn_id,
                            error_source="gemini_call_error",
                            error_code=error_code(exc),
                            error_text=technical_detail(exc, limit=10_000),
                        )
                    raise
                usage = response.usage_metadata
                if usage is not None:
                    prompt_tokens = usage.prompt_token_count or 0
                    thinking_tokens = usage.thoughts_token_count or 0
                    # thinking トークンも出力と同じ単価で課金されるため出力側に含める
                    # （UI・コスト計算・週次制限はこの合算値のまま。ログにだけ
                    # thinking_tokens で内訳を別途残す）
                    output_tokens = (usage.candidates_token_count or 0) + thinking_tokens
                    total_tokens = usage.total_token_count or 0
                    ctx.turn_prompt_tokens += prompt_tokens
                    ctx.turn_output_tokens += output_tokens
                    ctx.turn_thinking_tokens += thinking_tokens
                    ctx.turn_total_tokens += total_tokens
                    ctx.session_prompt_tokens += prompt_tokens
                    ctx.session_output_tokens += output_tokens
                    ctx.session_total_tokens += total_tokens
                    if ctx.usage_logger is not None:
                        ctx.usage_logger.log_gemini_call(
                            user_email=ctx.user_email,
                            session_id=ctx.session_id,
                            turn_id=ctx.turn_id,
                            prompt_tokens=prompt_tokens,
                            output_tokens=output_tokens,
                            total_tokens=total_tokens,
                            thinking_tokens=thinking_tokens,
                        )
                if ctx.execution_control is not None:
                    ctx.execution_control.check()
                candidate = (response.candidates or [None])[0]
                reason_value = getattr(candidate, "finish_reason", None)
                reason = getattr(reason_value, "value", str(reason_value)) if reason_value else None
                if candidate is None or candidate.content is None:
                    source = "output_limit" if reason == "MAX_TOKENS" else "missing_candidate"
                    self._record_generation_issue(ctx, source, "有効な candidate/content がありません。", response, reason)
                    message = (
                        "出力トークン上限に達して回答が打ち切られました。質問をもう少し絞ってお試しください。"
                        if source == "output_limit" else
                        "モデルから有効な応答が得られませんでした。質問を言い換えてお試しください。"
                    )
                    return self._answer(contents, message)

                parts = candidate.content.parts or []
                calls = [p.function_call for p in parts if p.function_call]
                text = "".join(p.text for p in parts if p.text and not p.thought)
                abnormal = reason not in (None, "STOP", "FINISH_REASON_UNSPECIFIED")
                if abnormal:
                    source = "output_limit" if reason == "MAX_TOKENS" else "model_response_error"
                    self._record_generation_issue(ctx, source, "モデルが正常終了しませんでした。", response, reason)
                    # 不完全な function_call は実行しない。
                    if reason == "MAX_TOKENS" and text:
                        message = text + "\n\n※ 出力トークン上限に達したため、回答は途中で打ち切られています。"
                        content = candidate.content.model_copy(update={
                            "parts": [*[p for p in parts if not p.function_call], types.Part.from_text(text="出力トークン上限に達したため、回答は途中で打ち切られています。")]
                        })
                        return self._answer(contents, message, content)
                    message = (
                        "出力トークン上限に達して回答が打ち切られました。質問をもう少し絞ってお試しください。"
                        if reason == "MAX_TOKENS" else
                        "モデルの応答を正常に受け取れませんでした。質問を言い換えてお試しください。"
                    )
                    return self._answer(contents, message)

                if not calls:
                    if not text:
                        self._record_generation_issue(ctx, "empty_response", "text も function_call も返りませんでした。", response, reason)
                    return self._answer(
                        contents, text or "回答を生成できませんでした。質問を言い換えてお試しください。",
                        candidate.content if text else None,
                    )

                contents.append(candidate.content)
                pending_calls = calls

            try:
                responses = []
                # 確認質問と同時に要求されたデータ取得は、回答が来るまで実行しない。
                clarification_call = next((call for call in pending_calls if call.name == "ask_clarification"), None)
                clarification = None
                for call in pending_calls:
                    if ctx.execution_control is not None:
                        ctx.execution_control.check()
                    if clarification_call is not None and call is not clarification_call:
                        responses.append(types.Part.from_function_response(
                            name=call.name, response={"error": "ユーザーへの確認を優先したため未実行です。回答後に必要なら再度呼んでください。"},
                        ))
                        continue
                    if on_progress is not None:
                        on_progress(tool_progress(call))
                    tool_response = self._dispatch(call, ctx)
                    responses.append(tool_response)
                    if ctx.execution_control is not None:
                        ctx.execution_control.check()
                    payload = tool_response.function_response.response
                    if call.name == "ask_clarification" and payload.get("status") == "awaiting_user":
                        clarification = ClarificationResult(
                            payload["question"], payload["options"], payload["selection_mode"],
                        )
            except _Pending as pending:
                # ここで中断。承認後に同じ pending_calls から再開する。
                ctx.resume_calls = pending_calls
                return ConfirmationPending(
                    key=pending.key,
                    sql=pending.need.sql,
                    estimated_bytes=pending.need.estimated_bytes,
                    threshold_bytes=pending.need.threshold_bytes,
                    notes=list(pending.need.notes),
                )

            contents.append(types.Content(role="user", parts=responses))
            if clarification is not None:
                # ツール呼出への応答を完結させ、次のユーザー回答を通常の会話として受ける。
                mode_label = "複数選択可" if clarification.selection_mode == "multi" else "1つ選択"
                self._answer(contents, clarification.text + f"\n（{mode_label}・自由記述可）\n" + "\n".join(f"- {option}" for option in clarification.options))
                return clarification
            processed_tools = True
            pending_calls = None

        self._record_generation_issue(
            ctx, "iteration_limit",
            f"調査ループが上限 {self.settings.max_tool_iterations} 回に達しました。",
            reason="MAX_TOOL_ITERATIONS",
        )
        return self._answer(contents,
            "調査の手順が想定より多くなったため、途中で打ち切りました。"
            "質問をもう少し具体的にしていただけますか。"
        )
