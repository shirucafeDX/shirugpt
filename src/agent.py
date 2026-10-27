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
import time
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
from .config import Settings, human_bytes, gemini_cost_usd
from .prompts import SYSTEM_INSTRUCTION, build_system_instruction
from .sql_guard import SQLGuardError, check_sql
from .thinking import DEFAULT_EFFORT
from . import interactions, analysis_safety
from .usage_log import UsageLogger
from .execution import ExecutionControl, ExecutionStopped
from . import context_memory, tool_catalog
from .token_budget import BudgetPolicy, TokenBudgetManager, estimate_tokens, estimate_frame_tokens
from .analysis_layer import analyze
from .analysis_tools import declaration as analysis_declaration
from .charts import validate_chart

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def _sql_key(sql: str) -> str:
    return hashlib.sha256(sql.strip().encode("utf-8")).hexdigest()


def tool_progress(call: Any) -> str:
    """ツールの実行内容を、SQL を出さず短い日本語で伝える。"""
    args = dict(call.args or {})
    if call.name == tool_catalog.SELECT_TOOL:
        return "分析の手順を準備しています…"
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
    if call.name == "analyze_result":
        return f"取得済みデータを分析しています：{args.get('operation', '')}"
    if call.name == "recall_context":
        if args.get("action") == "present":
            return "取得済みデータからグラフを準備しています…"
        return "この会話の取得済みデータを確認しています…"
    if call.name == "finish_answer":
        return "回答を作成しています…"
    return "調査に必要な情報を確認しています…"


@dataclass
class ToolContext:
    """1 会話分のツール実行状態。Streamlit の session_state に載せる。"""

    tools: BigQueryTools
    settings: Settings
    schema_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    listing_cache: dict[str, dict[str, Any]] = field(default_factory=dict)
    query_cache: dict[str, QueryRun] = field(default_factory=dict)
    context_state: dict[str, Any] = field(default_factory=dict)
    active_tool_names: list[str] = field(default_factory=list)
    interaction_state: dict[str, Any] = field(default_factory=dict)
    tool_receipts: dict[str, Any] = field(default_factory=dict)
    analysis_cache: dict[str, Any] = field(default_factory=dict)
    decisions: dict[str, str] = field(default_factory=dict)
    resume_calls: list[Any] | None = None
    execution_control: ExecutionControl | None = None
    thinking_enabled: bool = False
    thinking_previous_effort: str = DEFAULT_EFFORT
    thinking_mode: str = DEFAULT_EFFORT
    active_thinking_level: str = DEFAULT_EFFORT

    # 直近ターンの記録（UI 表示用）
    turn_runs: list[QueryRun] = field(default_factory=list)
    turn_analyses: list[dict[str, Any]] = field(default_factory=list)
    turn_errors: list[str] = field(default_factory=list)
    turn_prompt_tokens: int = 0
    turn_output_tokens: int = 0
    turn_thinking_tokens: int = 0
    turn_total_tokens: int = 0

    # セッション全体の累計（サイドバー表示用）
    session_prompt_tokens: int = 0
    session_output_tokens: int = 0
    session_thinking_tokens: int | None = 0
    session_gemini_cost_usd: float | None = None
    session_total_tokens: int = 0
    session_billed_bytes: int = 0

    # 利用ログ（Gemini 境界の外。UsageLogger 側で例外は握られる）
    usage_logger: UsageLogger | None = None
    user_email: str = ""
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    turn_id: str = ""
    turn_events: list[dict[str, Any]] = field(default_factory=list)
    turn_started_at: float = 0.0

    # 月次の使用制限判定用ベースライン（app.py 側で更新する。ロジックは
    # ここには置かない。BigQuery への問い合わせを毎ターン避けるため、
    # 月の開始時点の使用量 + このセッションでの増分、で近似する）
    limit_baseline_usd: float = 0.0
    limit_baseline_month_start: datetime | None = None
    limit_baseline_session_prompt_tokens: int = 0
    limit_baseline_session_output_tokens: int = 0
    limit_baseline_session_billed_bytes: int = 0
    limit_baseline_session_gemini_cost_usd: float = 0.0

    def gemini_cost_usd(self) -> float:
        # 旧履歴には費用の累計がない。旧モデルの単価で引き継ぐ。
        if self.session_gemini_cost_usd is None:
            return gemini_cost_usd(self.session_prompt_tokens, self.session_output_tokens)
        return self.session_gemini_cost_usd

    def start_turn(self) -> None:
        self.turn_events = []
        self.turn_started_at = time.time()
        self.tool_receipts = {}
        self.turn_runs = []
        self.turn_analyses = []
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
    declarations = [
        types.FunctionDeclaration(
            name="ask_clarification",
            description=(
                "分析に必要な条件・観点をユーザーに確認し、選択式フォームで回答を待つ。"
                "実データの分析条件（scope=data）では先にスキーマと必要な値を確認する。"
                "一般的な壁打ちの目的・観点の確認（scope=general）ではデータ照会は不要。"
                "確認質問は必ずこのツールで提示し、通常の回答本文に質問・選択肢を書くだけで終了しない。"
                "単独で呼ぶ。自由記述欄は画面側が付けるため、その他・自由記述を選択肢に含めない。"
            ),
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "scope": types.Schema(
                        type=types.Type.STRING, enum=["data", "general"],
                        description="data:実データの分析条件（既定）。general:データに依存しない一般相談の目的・観点。",
                    ),
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
                "（小さい結果はCSV、中・大規模結果は列情報・行数のみ）を返す。"
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
                    "chart": types.Schema(
                        type=types.Type.OBJECT,
                        description="回答のグラフ指定。不要ならkind=none。種類・軸・内訳を質問に応じて判断する。",
                        properties={
                            "kind": types.Schema(type=types.Type.STRING, enum=["none", "bar", "line", "area", "scatter", "pie", "heatmap"]),
                            "x": types.Schema(type=types.Type.STRING, description="時間・年月やカテゴリの横軸の出力列名"),
                            "y": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING), description="数値の出力列名。color使用時は1列"),
                            "color": types.Schema(type=types.Type.STRING, description="折れ線・棒などで必要な場合だけ指定する内訳列。円・ヒートマップでは指定しない"),
                            "row": types.Schema(type=types.Type.STRING, description="heatmap専用。縦軸のカテゴリ列名（曜日・時間帯など）。xは横軸、yは色の濃淡に使う数値1列"),
                        },
                        required=["kind"],
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

    declarations.extend([
        types.FunctionDeclaration(
            name="finish_answer", description="最終回答と引き継ぎメモを同時に返して終了する。単独で呼ぶ。",
            parameters=types.Schema(type=types.Type.OBJECT, properties={
                "answer": types.Schema(type=types.Type.STRING, description="ユーザーに表示する回答"),
                "conversation_summary": types.Schema(type=types.Type.STRING, description="今回までの確定した事実・合意した条件。未確認事項は混ぜない。4000文字以内"),
            }, required=["answer", "conversation_summary", "context_note"])),
        types.FunctionDeclaration(
            name="recall_context", description="この会話の保存済み原結果を参照。result_id、turn_id、searchのどれか1つを指定。再問い合わせしない。",
            parameters=types.Schema(type=types.Type.OBJECT, properties={
                "result_id": types.Schema(type=types.Type.STRING),
                "turn_id": types.Schema(type=types.Type.STRING),
                "search": types.Schema(type=types.Type.STRING, description="過去の質問・要点を探すキーワード"),
                "columns": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)),
                "offset": types.Schema(type=types.Type.INTEGER, description="開始行。既定0"),
                "limit": types.Schema(type=types.Type.INTEGER, description="取得行数。既定20、最大100"),
                "action": types.Schema(type=types.Type.STRING, enum=["read", "present"], description="既定read。presentは保存済み全結果をグラフ表示し、行データを再送しない。result_idとchartが必要"),
                "filters": types.Schema(type=types.Type.ARRAY, max_items=5,
                    items=types.Schema(type=types.Type.OBJECT, properties={
                        "column": types.Schema(type=types.Type.STRING),
                        "op": types.Schema(type=types.Type.STRING, enum=["eq", "in", "gte", "lte", "is_null", "not_null"]),
                        "values": types.Schema(type=types.Type.ARRAY, max_items=20, items=types.Schema(type=types.Type.STRING), description="列の型に合わせた文字列。日付はISO形式。eq/gte/lteは1値、inは1〜20値。欠損判定は不要"),
                    }, required=["column", "op"])),
                "include_sql": types.Schema(type=types.Type.BOOLEAN, description="SQLも必要な場合のみtrue"),
            })),
    ])
    by_name = {d.name: d for d in declarations}
    by_name["recall_context"].parameters.properties["chart"] = by_name["run_query"].parameters.properties["chart"]
    declarations.append(analysis_declaration(by_name["recall_context"].parameters.properties["filters"]))
    by_name["run_query"].parameters.properties["result_mode"] = types.Schema(
        type=types.Type.STRING, enum=["auto", "schema"], description="表示用で行データを読まない場合schema。autoも大きい結果は列・行数だけ返す")
    for declaration in declarations:
        declaration.parameters.properties["context_note"] = types.Schema(
            type=types.Type.OBJECT, description="これまでに読んだ結果の要点。要約用の別API呼び出しは不要。",
            properties={
                "summary": types.Schema(type=types.Type.STRING, description="確定した事実・条件の箇条書き。未確認事項は混ぜない。1200文字以内"),
                "references": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING), description="読み終えたresult_id。最大12件"),
                "open_questions": types.Schema(type=types.Type.ARRAY, max_items=12,
                    description="未確認事項をID付きで追加・更新。未変更なら空配列。解決済みIDの再利用は禁止。",
                    items=types.Schema(type=types.Type.OBJECT, properties={
                        "id": types.Schema(type=types.Type.STRING, description="英字で始まる40文字以内の英数字・ハイフン・アンダースコア"),
                        "question": types.Schema(type=types.Type.STRING, description="未確認の具体的な事項。300文字以内"),
                    }, required=["id", "question"])),
                "resolved_questions": types.Schema(type=types.Type.ARRAY, max_items=12,
                    description="登録済み事項の解決。根拠を確認してIDと結論を指定。変更なしなら空配列。",
                    items=types.Schema(type=types.Type.OBJECT, properties={
                        "id": types.Schema(type=types.Type.STRING),
                        "conclusion": types.Schema(type=types.Type.STRING, description="確認した結論と根拠。600文字以内"),
                    }, required=["id", "conclusion"])),
            }, required=["summary", "references", "open_questions", "resolved_questions"])
    return declarations


class GeminiAgent:
    def __init__(self, settings: Settings, client: genai.Client | None = None,
                 thinking_level: str | None = None, deep_analysis: bool = False):
        self.settings = settings
        self.thinking_level = thinking_level
        self.deep_analysis = deep_analysis
        self.client = client or genai.Client(
            **({'enterprise': True} if interactions.enabled(settings) else {'vertexai': True}),
            project=settings.gcp_project,
            location=settings.vertex_location,
            **({'http_options': types.HttpOptions(api_version='v1beta1',
                                                  timeout=getattr(settings, 'gemini_request_timeout_seconds', 120) * 1000,
                                                  retry_options=types.HttpRetryOptions(attempts=1))}
               if interactions.enabled(settings) else {}),
        )
        self.declarations = build_declarations()

    # -- 生成 ---------------------------------------------------------
    def _config(self, tool_names: list[str] | None = None, ctx: ToolContext | None = None) -> types.GenerateContentConfig:
        kwargs: dict[str, Any] = {
            "system_instruction": build_system_instruction(tool_names, deep_analysis=self.deep_analysis) + (
                f"\n既定の分析対象はデータセット {self.settings.default_dataset}。別の指定がなければこれを使い、list_datasetsで探し直さない。"
                if getattr(self.settings, "default_dataset", "") else "") + (
                analysis_safety.request_context(ctx, tool_names) if ctx is not None else ""),
            "tools": [types.Tool(function_declarations=tool_catalog.active_declarations(self.declarations, tool_names))],
            "temperature": 0.1,
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(
                disable=True
            ),
        }
        if self.settings.gemini_model == "gemini-3.8-flash":
            kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_level=(self.thinking_level or DEFAULT_EFFORT).upper(),
            )
        elif self.settings.thinking_budget is not None:
            kwargs["thinking_config"] = types.ThinkingConfig(
                thinking_budget=self.settings.thinking_budget
            )
        return types.GenerateContentConfig(**kwargs)

    def _generate(self, contents: list[types.Content], *, tool_names: list[str] | None = None, ctx: ToolContext | None = None):
        request = context_memory.ensure_user_ending(contents)
        config = self._config(tool_names, ctx)
        logger.info("analysis_metrics %s", json.dumps({"event": "llm_request",
            "selected_tools": tool_names or [], "estimated_history_tokens": estimate_tokens([c.model_dump(mode='json', exclude_none=True) for c in request]),
            "estimated_system_tokens": estimate_tokens(config.system_instruction),
            "estimated_tool_tokens": estimate_tokens([t.model_dump(exclude_none=True) for t in config.tools or []])}))
        try:
            return self.client.models.generate_content(
                model=self.settings.gemini_model,
                contents=request,
                config=config,
            )
        except Exception:
            # 原因調査には構造だけを残す。質問・結果・SQL・署名の内容はログへ追加しない。
            part_kinds = [[key for key in ("text", "function_call", "function_response", "inline_data", "file_data")
                           if getattr(part, key, None) is not None] for part in request[-1].parts or []]
            logger.warning("Gemini送信失敗: contents=%d, last_roles=%s, last_part_kinds=%s",
                           len(request), [c.role for c in request[-12:]], part_kinds)
            raise

    def _record_usage(self, ctx, response, model):
        usage = response.usage_metadata
        if usage is not None:
            prompt_tokens = usage.prompt_token_count or 0
            thinking_tokens = usage.thoughts_token_count or 0
            # thinking トークンも出力と同じ単価で課金されるため出力側に含める
            # （UI・コスト計算・月次制限はこの合算値のまま。ログにだけ
            # thinking_tokens で内訳を別途残す）
            output_tokens = (usage.candidates_token_count or 0) + thinking_tokens
            total_tokens = usage.total_token_count or 0
            cached_tokens = getattr(usage, 'cached_content_token_count', None) or 0
            call_cost = gemini_cost_usd(
                prompt_tokens, output_tokens,
                model,
                getattr(self.settings, "vertex_location", "global"),
            )
            ctx.session_gemini_cost_usd = ctx.gemini_cost_usd() + call_cost
            logger.info("analysis_metrics %s", json.dumps({"event": "llm_call", "turn_id": ctx.turn_id,
                "session_id": ctx.session_id, "selected_tools": ctx.active_tool_names, "model": model,
                "input_tokens": prompt_tokens, "cached_input_tokens": cached_tokens,
                "output_tokens": output_tokens, "llm_calls": 1}))
            ctx.turn_prompt_tokens += prompt_tokens
            ctx.turn_output_tokens += output_tokens
            ctx.turn_thinking_tokens += thinking_tokens
            ctx.turn_total_tokens += total_tokens
            ctx.session_prompt_tokens += prompt_tokens
            ctx.session_output_tokens += output_tokens
            if ctx.session_thinking_tokens is not None:
                ctx.session_thinking_tokens += thinking_tokens
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
                    model=model,
                    cached_input_tokens=cached_tokens,
                    api=getattr(ctx.settings, 'gemini_api', 'generate_content'),
                )

    # -- ツール実行 ---------------------------------------------------
    def _call_tool(
        self, name: str, args: dict[str, Any], ctx: ToolContext
    ) -> dict[str, Any]:
        if name == tool_catalog.SELECT_TOOL:
            try:
                names = tool_catalog.validate_selection(args.get("names"))
            except ValueError as exc:
                raise BQToolError(str(exc)) from exc
            ctx.active_tool_names = names
            return {"available_tools": names, "instruction": "次の生成のtoolsに、選択した詳細定義を渡します。使用中の定義を再選択する必要はありません。"}
        if name == "analyze_result":
            try:
                return self._analyze_saved_result(ctx, args)
            except (ValueError, TypeError, KeyError) as exc:
                raise BQToolError(str(exc)) from exc
        if name == "recall_context":
            try:
                action = args.get("action", "read")
                if action == "present":
                    return self._present_saved_result(ctx, args)
                if action != "read":
                    raise ValueError("actionはreadまたはpresentを指定してください。")
                return context_memory.recall(ctx, args)
            except ValueError as exc:
                raise BQToolError(str(exc)) from exc
        if name == "finish_answer":
            answer, overview = args.get("answer"), args.get("conversation_summary")
            note = args.get("context_note")
            if not isinstance(answer, str) or not answer.strip():
                raise BQToolError("answerには空でない最終回答を指定してください。")
            if not isinstance(overview, str) or not overview.strip() or len(overview) > context_memory.OVERVIEW_CHARS:
                raise BQToolError("conversation_summaryには4000文字以内で会話全体の要点を指定してください。")
            warning = context_memory.validate_note(ctx, note)
            if warning or note is None:
                raise BQToolError(warning or "context_noteを指定してください。")
            evidence_error = analysis_safety.validate_answer(ctx, note)
            if evidence_error:
                raise BQToolError(evidence_error)
            return {"status": "final_answer", "answer": answer.strip(),
                    "summary": context_memory.factual_summary(note), "overview": overview.strip()}

        if name == "ask_clarification":
            scope = args.get("scope", "data")
            if scope not in ("data", "general"):
                raise BQToolError("scope は data または general を指定してください。")
            if scope == "data" and not ctx.schema_cache:
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
            dataset = str(args.get("dataset") or getattr(ctx.settings, "default_dataset", "")).strip()
            cache_key = f"tables:{dataset}"
            if cache_key not in ctx.listing_cache:
                ctx.listing_cache[cache_key] = ctx.tools.list_tables(dataset)
            return ctx.listing_cache[cache_key]

        if name == "get_table_schema":
            dataset = str(args.get("dataset") or getattr(ctx.settings, "default_dataset", "")).strip()
            table = str(args.get("table", "")).strip()
            cache_key = f"{dataset}.{table}"
            if cache_key in ctx.schema_cache:
                cached = dict(ctx.schema_cache[cache_key])
                cached["cached"] = True
                return cached
            self._require_known_table(ctx, dataset, table)
            schema = ctx.tools.get_table_schema(dataset, table)
            ctx.schema_cache[cache_key] = schema
            return schema

        if name == "run_query":
            return self._run_query(args, ctx)

        raise BQToolError(f"未知のツール「{name}」が呼び出されました。")

    def _prefetch_schemas(self, contents, ctx):
        question = next((context_memory.text_of(c) for c in reversed(contents) if context_memory.is_question(c)), '')
        current = analysis_safety.state(ctx)
        current['question'] = question
        current['data_requested'] = current['data_requested'] or analysis_safety.is_data_question(question)
        dataset = getattr(ctx.settings, 'default_dataset', '')
        if not isinstance(dataset, str) or not dataset or not current['data_requested']:
            return
        key = f'tables:{dataset}'
        known = {'tables': [{'table': key.split('.', 1)[1]} for key in ctx.schema_cache if key.startswith(dataset + '.')]}
        if key not in ctx.listing_cache and analysis_safety.likely_tables(question, known):
            for table in analysis_safety.likely_tables(question, known):
                schema_key = f'{dataset}.{table}'
                current['schema_keys'] = [item for item in current['schema_keys'] if item != schema_key] + [schema_key]
            return
        if key not in ctx.listing_cache:
            part = self._dispatch(types.FunctionCall(name='list_tables', args={'dataset': dataset}), ctx)
            if part.function_response.response.get('error'):
                return
        for table in analysis_safety.likely_tables(question, ctx.listing_cache[key]):
            if ctx.execution_control is not None:
                ctx.execution_control.check()
            if f'{dataset}.{table}' not in ctx.schema_cache:
                self._dispatch(types.FunctionCall(name='get_table_schema', args={'dataset': dataset, 'table': table}), ctx)

    def _require_known_table(self, ctx, dataset, table):
        # スキーマが保存済みなら実在確認済み。一覧の取り直しは不要。
        if f"{dataset}.{table}" in ctx.schema_cache:
            return
        key = f"tables:{dataset}"
        if key not in ctx.listing_cache:
            part = self._dispatch(types.FunctionCall(name='list_tables', args={'dataset': dataset}), ctx)
            if part.function_response.response.get('error'):
                raise BQToolError(part.function_response.response['error'])
        tables = ctx.listing_cache[key]["tables"]
        if table not in {item["table"] for item in tables}:
            raise BQToolError(f"テーブル {table!r} は確認した一覧にありません。推測せず一覧の正確な名前を使ってください: {tables}")

    def _run_query(self, args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        sql = str(args.get("sql", "")).strip()
        if not sql:
            raise BQToolError("sql が指定されていません。")

        if args.get("result_mode", "auto") not in ("auto", "schema"):
            raise BQToolError("result_modeはautoまたはschemaです。")
        key = _sql_key(sql)
        result_role = str(args.get("result_role", "answer"))
        if result_role not in ("supporting", "answer"):
            raise BQToolError("result_role は supporting または answer を指定してください。")

        # 同一 SQL の再実行・再課金を防ぐ。再表示の要否は UI で決める。
        if key in ctx.query_cache:
            run = ctx.query_cache[key]
            # 図表だけを求める追質問にも対応する。履歴内の結果は書き換えない。
            ctx.turn_runs = [r for r in ctx.turn_runs if _sql_key(r.original_sql) != key]
            ctx.turn_runs.append(replace(run, result_role=result_role, chart=args.get("chart", run.chart)))
            payload = self._query_payload(run, result_mode=args.get("result_mode", "auto"))
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

        failed = analysis_safety.state(ctx)['failures'].get(analysis_safety.query_key(sql))
        if failed:
            raise BQToolError(f"同じSQLは失敗済みのため再実行しません。前回のエラー: {failed}。確認済みスキーマと照合してSQLを修正してください。")

        guard = check_sql(sql, project=getattr(ctx.settings, "gcp_project", ""),
                          allowed_datasets=getattr(ctx.settings, "allowed_datasets", ()),
                          default_row_limit=getattr(ctx.settings, "default_row_limit", 1000))
        for reference in guard.referenced_tables:
            _, dataset, table = reference.split(".", 2)
            self._require_known_table(ctx, dataset, table)
            if f"{dataset}.{table}" not in ctx.schema_cache:
                schema_part = self._dispatch(types.FunctionCall(name='get_table_schema', args={'dataset': dataset, 'table': table}), ctx)
                if schema_part.function_response.response.get('error'):
                    raise BQToolError(schema_part.function_response.response['error'])
        analysis_safety.verify_columns(sql, guard.referenced_tables, ctx.schema_cache)

        purpose = str(args.get("purpose", "")).strip()
        approved = ctx.decisions.get(key) == "approved"
        try:
            run = ctx.tools.run_query(sql, approved=approved, purpose=purpose)
        except ConfirmationRequired as need:
            raise _Pending(key, need) from None

        run.result_role = result_role
        run.chart = args.get("chart")
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
        return self._query_payload(run, result_mode=args.get("result_mode", "auto"))

    def _query_payload(self, run: QueryRun, result_mode="auto") -> dict[str, Any]:
        if result_mode not in ("auto", "schema"):
            raise BQToolError("result_modeはautoまたはschemaです。")
        manager = TokenBudgetManager(BudgetPolicy.from_settings(self.settings))
        result = manager.initial_result(run.dataframe, truncated=run.truncated, schema_only=result_mode == "schema")
        return {"executed_sql": run.sql, "bytes_processed": human_bytes(run.estimated_bytes),
                "result": result, **({"notes": run.notes} if run.notes else {})}

    def _analyze_saved_result(self, ctx, args):
        source = context_memory.saved_result(ctx, args.get("result_id"))
        payload, frame = analyze(source.dataframe, args, BudgetPolicy.from_settings(self.settings),
                                 truncated=source.truncated, control=ctx.execution_control, return_frame=True)
        rid = context_memory.result_id_for("analyze_result", args)
        derived_truncated = source.truncated or bool(payload["result"].get("more_rows"))
        ctx.analysis_cache[rid] = {"dataframe": frame, "source_truncated": derived_truncated,
                                   "source_result_id": args["result_id"], "notes": source.notes}
        payload["estimated_tokens_before"] = estimate_frame_tokens(source.dataframe)
        ctx.turn_analyses.append({
            "result_id": rid, "source_result_id": args["result_id"], "operation": args["operation"],
            "purpose": str(args.get("purpose") or ""),
            "parameters": {key: value for key, value in args.items() if key not in ("result_id", "operation", "context_note", "purpose")},
            "dataframe": frame, "source_row_count": payload["source_row_count"],
            "matched_row_count": payload["matched_row_count"],
            "sent_row_count": payload["result"].get("sent_row_count", 0),
            "source_truncated": derived_truncated, "analysis_seconds": payload["analysis_seconds"],
            "notes": list(source.notes),
        })
        return {"source_result_id": args["result_id"], **payload}

    def _present_saved_result(self, ctx: ToolContext, args: dict) -> dict:
        if not args.get("result_id") or any(args.get(k) for k in ("turn_id", "search", "columns", "filters", "offset", "limit", "include_sql")):
            raise ValueError("presentはresult_idとchartだけで保存済み全結果を表示します。絞り込みはreadで行ってください。")
        record = context_memory.state_for(ctx)["results"].get(args["result_id"], {})
        run = ctx.query_cache.get(record.get("query_key"))
        if run is None:
            raise ValueError("この会話に表示可能な保存済み全結果がありません。")
        chart = args.get("chart")
        if validate_chart(run.dataframe, chart) is None:
            raise ValueError("保存済み結果の列名と型に合うchartを指定してください。")
        key = _sql_key(run.original_sql)
        ctx.turn_runs = [r for r in ctx.turn_runs if _sql_key(r.original_sql) != key]
        ctx.turn_runs.append(replace(run, chart=dict(chart), result_role="answer"))
        return {"source_result_id": args["result_id"], "presented": True, "chart": chart,
                "row_count": len(run.dataframe), "source_truncated": run.truncated,
                "note": "保存済み全結果を画面に表示。BigQuery再実行・行データ再送なし。"}

    def _dispatch(self, call: Any, ctx: ToolContext) -> types.Part:
        name = call.name
        args = dict(call.args or {})
        receipt_key = (ctx.turn_id + ':' + str(call.id) + ':' + hashlib.sha256(
            context_memory.encoded([name, args]).encode()).hexdigest()) if getattr(call, 'id', None) else None
        if receipt_key and receipt_key in ctx.tool_receipts:
            return types.Part.model_validate(ctx.tool_receipts[receipt_key])

        started = time.monotonic()
        started_at = datetime.now().astimezone().isoformat()
        try:
            payload = self._call_tool(name, args, ctx)
        except (ExecutionStopped, _Pending) as exc:
            ctx.turn_events.append({"event": "agent_tool", "tool": name, "started_at": started_at,
                                    "seconds": round(time.monotonic() - started, 6),
                                    "status": "stopped" if isinstance(exc, ExecutionStopped) else "confirmation_required"})
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

        if name == "analyze_result" and payload.get("error"):
            ctx.turn_analyses.append({"operation": args.get("operation", ""), "purpose": str(args.get("purpose") or ""),
                "parameters": {key: value for key, value in args.items() if key not in ("result_id", "operation", "context_note", "purpose")},
                "error": payload["error"]})
        result = payload.get("result", {})
        source_rows = payload.get("source_row_count", result.get("row_count", 0)) if isinstance(result, dict) else 0
        event = {"event": "agent_tool", "session_id": ctx.session_id, "turn_id": ctx.turn_id,
                 "tool": name, "started_at": started_at, "selected_tools": list(ctx.active_tool_names),
                 "original_query": (context_memory.state_for(ctx)["turns"][-1].get("question", "") if context_memory.state_for(ctx)["turns"] else ""),
                 "generated_sql": args.get("sql"), "analysis_operation": args.get("operation"),
                 "seconds": round(time.monotonic() - started, 6), "rows_before": source_rows,
                 "rows_after": result.get("sent_row_count", 0) if isinstance(result, dict) else 0,
                 "estimated_tokens_after": estimate_tokens(payload), "error": bool(payload.get("error"))}
        if name == "run_query":
            source = ctx.query_cache.get(_sql_key(str(args.get("sql", ""))))
            if source is not None:
                event["estimated_tokens_before"] = estimate_frame_tokens(source.dataframe)
                event["sql_execution_seconds"] = source.execution_seconds
                event["executed_sql"] = source.sql
        elif isinstance(result, dict):
            event["estimated_tokens_before"] = payload.get("estimated_tokens_before", result.get("estimated_tokens_before"))
        logger.info("analysis_metrics %s", json.dumps(event, ensure_ascii=False, default=str))
        state = context_memory.state_for(ctx)
        ctx.turn_events.append(event)
        state.setdefault("analysis_events", []).append(event)
        state["analysis_events"] = state["analysis_events"][-100:]
        if not payload.get("error") and name != tool_catalog.SELECT_TOOL:
            warning = context_memory.accept_note(ctx, args.get("context_note"))
            if warning:
                payload = {**payload, "context_note_warning": warning}
            if name != "finish_answer":
                payload = context_memory.register_result(ctx, name, args, payload)
        analysis_safety.record(ctx, name, args, payload)
        part = types.Part.from_function_response(name=name, response=payload)
        part.function_response.id = getattr(call, "id", None)
        if receipt_key:
            ctx.tool_receipts[receipt_key] = part.model_dump(mode='json', exclude_none=True)
        return part

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
                content: types.Content | None = None, ctx: ToolContext | None = None) -> AnswerResult:
        # UI だけでなく、次の API 呼び出しにも最終回答を引き継ぐ。
        # 元の Content を使うことで thought signature も保持する。
        contents.append(content if content is not None else types.Content(
            role="model", parts=[types.Part.from_text(text=text)]
        ))
        if ctx is not None:
            context_memory.finish_turn(contents, ctx)
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
        analysis_safety.record(ctx, 'response_validation', {}, {'error': encoded})
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
        if getattr(self.settings, "gemini_model", "") == "gemini-3.8-flash" and self.thinking_level is None:
            if ctx.resume_calls is None:
                ctx.active_thinking_level = ctx.thinking_mode if ctx.thinking_mode in ("low", "medium", "high") else DEFAULT_EFFORT
            # 共用のエージェントを変更せず、今回のターンにだけ設定を固定する。
            turn_agent = GeminiAgent(self.settings, client=self.client,
                                     thinking_level=ctx.active_thinking_level, deep_analysis=ctx.thinking_enabled)
            return turn_agent.run(contents, ctx, on_progress=on_progress)
        context_memory.sync_turns(contents, ctx)
        self._prefetch_schemas(contents, ctx)
        pending_calls = ctx.resume_calls
        ctx.resume_calls = None
        processed_tools = False

        for _ in range(self.settings.max_tool_iterations):
            if ctx.execution_control is not None:
                ctx.execution_control.check()
            if pending_calls is None:
                if on_progress is not None and not processed_tools and not ctx.turn_runs:
                    on_progress("質問に合う調べ方を考えています…")
                request_event = {"event": "llm_call", "started_at": datetime.now().astimezone().isoformat(), "error": True}
                request_started = time.monotonic()
                try:
                    if interactions.enabled(self.settings):
                        response = interactions.generate(self.client, contents, ctx,
                            self._config(list(ctx.active_tool_names), ctx), on_progress=on_progress, metrics=request_event)
                    else:
                        response = self._generate(context_memory.build_request(contents, ctx), tool_names=list(ctx.active_tool_names), ctx=ctx)
                except Exception as exc:  # noqa: BLE001 - 記録してから同じ例外を上に伝える
                    if isinstance(exc, ExecutionStopped) or (ctx.execution_control is not None and ctx.execution_control.stopped.is_set()):
                        request_event.update(status='stopped', error=False)
                        raise ExecutionStopped() from None
                    analysis_safety.record(ctx, 'llm_call', {}, {'error': technical_detail(exc)})
                    # 完了イベントの使用量は、引数の検証に失敗した場合も一度だけ計上する。
                    if request_event.get('reported_usage') is not None:
                        usage_response = interactions._response({'status': 'failed', 'steps': [], 'usage': request_event['reported_usage']})
                        self._record_usage(ctx, usage_response, getattr(self.settings, 'gemini_model', 'gemini-2.5-flash'))
                        u = usage_response.usage_metadata
                        request_event.update(input_tokens=u.prompt_token_count, output_tokens=u.candidates_token_count,
                                             thinking_tokens=u.thoughts_token_count, cached_input_tokens=u.cached_content_token_count)
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
                finally:
                    request_event["seconds"] = round(time.monotonic() - request_started, 6)
                    ctx.turn_events.append(request_event)
                request_event["error"] = False
                usage = response.usage_metadata
                if usage is not None:
                    request_event.update(input_tokens=usage.prompt_token_count, output_tokens=usage.candidates_token_count,
                                         thinking_tokens=usage.thoughts_token_count, cached_input_tokens=getattr(usage, "cached_content_token_count", None))
                self._record_usage(ctx, response, getattr(self.settings, "gemini_model", "gemini-2.5-flash"))
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
                    return self._answer(contents, message, ctx=ctx)

                parts = candidate.content.parts or []
                calls = [p.function_call for p in parts if p.function_call]
                text = "".join(p.text for p in parts if p.text and not p.thought)
                abnormal = reason not in (None, "STOP", "FINISH_REASON_UNSPECIFIED")
                if abnormal:
                    source = "output_limit" if reason == "MAX_TOKENS" else "model_response_error"
                    self._record_generation_issue(ctx, source, "モデルが正常終了しませんでした。", response, reason)
                    # 不完全な function_call は実行しない。
                    if reason == "MAX_TOKENS" and text and not analysis_safety.needs_evidence(ctx):
                        message = text + "\n\n※ 出力トークン上限に達したため、回答は途中で打ち切られています。"
                        content = candidate.content.model_copy(update={
                            "parts": [*[p for p in parts if not p.function_call], types.Part.from_text(text="出力トークン上限に達したため、回答は途中で打ち切られています。")]
                        })
                        return self._answer(contents, message, content, ctx=ctx)
                    message = (
                        "出力トークン上限に達して回答が打ち切られました。質問をもう少し絞ってお試しください。"
                        if reason == "MAX_TOKENS" else
                        "モデルの応答を正常に受け取れませんでした。質問を言い換えてお試しください。"
                    )
                    return self._answer(contents, message, ctx=ctx)

                if not calls:
                    if text and analysis_safety.needs_evidence(ctx):
                        ctx.turn_errors.append('finish_answerによる実行結果の参照確認を経ていない分析回答を非表示にしました。')
                        analysis_safety.state(ctx)['next_action'] = '成功した保存結果のIDを確認し、根拠を参照した回答を作り直す。'
                        analysis_safety.state(ctx)['steps'].append({'tool': 'answer_validation', 'status': 'failed', 'error': 'finish_answerによる根拠の参照確認がありません。'})
                        # APIが返した元の応答は保持し、状態IDの署名・カーソルを壊さない。
                        contents.append(candidate.content)
                        return self._answer(contents, analysis_safety.BLOCKED_ANSWER, ctx=ctx)
                    if not text:
                        self._record_generation_issue(ctx, "empty_response", "text も function_call も返りませんでした。", response, reason)
                    return self._answer(
                        contents, text or "回答を生成できませんでした。質問を言い換えてお試しください。",
                        candidate.content if text else None, ctx=ctx,
                    )

                contents.append(candidate.content)
                pending_calls = calls

            try:
                responses = []
                # 確認質問と同時に要求されたデータ取得は、回答が来るまで実行しない。
                clarification_call = next((call for call in pending_calls if call.name == "ask_clarification"), None)
                selection_call = next((call for call in pending_calls if call.name == tool_catalog.SELECT_TOOL), None)
                final_call = next((call for call in pending_calls if call.name == "finish_answer"), None)
                final_payload = None
                clarification = None
                for call in pending_calls:
                    if ctx.execution_control is not None:
                        ctx.execution_control.check()
                    if clarification_call is not None and call is not clarification_call:
                        skipped = types.Part.from_function_response(
                            name=call.name, response={"error": "ユーザーへの確認を優先したため未実行です。回答後に必要なら再度呼んでください。"},
                        )
                        skipped.function_response.id = getattr(call, "id", None)
                        responses.append(skipped)
                        continue
                    if clarification_call is None and selection_call is not None and call is not selection_call:
                        skipped = types.Part.from_function_response(name=call.name, response={"error": "ツール定義の選択を優先したため未実行です。選択後に必要なら再度呼んでください。"})
                        skipped.function_response.id = getattr(call, "id", None)
                        responses.append(skipped)
                        continue
                    if clarification_call is None and selection_call is None and final_call is not None and call is not final_call:
                        skipped = types.Part.from_function_response(name=call.name, response={"error": "最終回答を優先したため未実行です。"})
                        skipped.function_response.id = getattr(call, "id", None)
                        responses.append(skipped)
                        continue
                    if on_progress is not None:
                        on_progress(tool_progress(call))
                    tool_response = self._dispatch(call, ctx)
                    responses.append(tool_response)
                    if ctx.execution_control is not None:
                        ctx.execution_control.check()
                    payload = tool_response.function_response.response
                    if call.name == "finish_answer" and payload.get("status") == "final_answer":
                        final_payload = payload
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
            if final_payload is not None:
                result = self._answer(contents, final_payload["answer"], ctx=ctx)
                context_memory.finish_turn(contents, ctx, final_payload["summary"], final_payload["overview"])
                return result
            if clarification is not None:
                # ツール呼出への応答を完結させ、次のユーザー回答を通常の会話として受ける。
                mode_label = "複数選択可" if clarification.selection_mode == "multi" else "1つ選択"
                self._answer(contents, clarification.text + f"\n（{mode_label}・自由記述可）\n" + "\n".join(f"- {option}" for option in clarification.options), ctx=ctx)
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
            "質問をもう少し具体的にしていただけますか。", ctx=ctx
        )
