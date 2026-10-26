"""ShiruGPT — BigQuery を自然言語で分析する Streamlit アプリ（プロトタイプ）。

起動:
    streamlit run app.py
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import time
import uuid
from zoneinfo import ZoneInfo
from typing import Any

import pandas as pd
import streamlit as st
from google.genai import types

from src.admin_page import render_admin_page
from src import analysis_safety
from src.agent import (
    AnswerResult,
    ClarificationResult,
    ConfirmationPending,
    GeminiAgent,
    ToolContext,
)
from src.bq_tools import (
    BigQueryTools,
    QueryRun,
    error_code,
    friendly_error,
    technical_detail,
)
from src.charts import ChartSpec, special_chart_spec
from src.execution import BackgroundTurn, ExecutionStopped
from src.commands import HELP, parse_command, usage_text, usage_details_text
from src.command_menu import render_command_menu
from src.thinking import DEFAULT_EFFORT
from src.presentation import choose_presentation
from src.config import (
    DEFAULT_MONTHLY_LIMIT_USD,
    ConfigError,
    Settings,
    bq_cost_usd,
    credentials_leak_warning,
    human_bytes,
    load_settings,
)
from src.usage_log import UsageLogger, jst_month_bounds_utc
from src.chat_history import HistoryStore, conversation_title, pack, unpack
from src.history_cache import HistoryCache
from google.api_core.exceptions import PreconditionFailed

st.set_page_config(page_title="ShiruGPT", page_icon="📊", layout="wide")

logger = logging.getLogger(__name__)

AUTH_HELP = """\
GCP の認証情報が見つからないか、権限が不足しています。ターミナルで次を実行してください。

```
gcloud auth application-default login
gcloud auth application-default set-quota-project <プロジェクトID>
```
"""

FEEDBACK_FORM_URL = (
    "https://docs.google.com/forms/d/e/"
    "1FAIpQLSdLBnajt42qgcmUkia3h8zpOJFFiWHrKvKSwjHCiyea0bQcog/viewform"
)


# --------------------------------------------------------------------
# 初期化
# --------------------------------------------------------------------
_SECRET_ENV_KEYS = (
    "GCP_PROJECT_ID",
    "BQ_LOCATION",
    "BQ_DEFAULT_DATASET",
    "BQ_ALLOWED_DATASETS",
    "VERTEX_LOCATION",
    "GEMINI_MODEL",
    "GEMINI_API", "GEMINI_STATE_MAX_INPUT_TOKENS", "GEMINI_REQUEST_TIMEOUT_SECONDS",
    "GEMINI_THINKING_BUDGET",
    "DRY_RUN_CONFIRM_BYTES",
    "MAX_BYTES_BILLED",
    "DEFAULT_ROW_LIMIT",
    "MAX_RESULT_ROWS",
    "SAMPLE_ROWS",
    "MAX_DIRECT_ROWS", "MAX_PREVIEW_ROWS", "MEDIUM_RESULT_ROWS",
    "MAX_DATA_TOKENS", "TARGET_DATA_TOKENS", "MAX_HISTORY_TOKENS", "ANALYSIS_TIMEOUT_SECONDS", "BQ_QUERY_TIMEOUT_SECONDS",
    "MAX_TOOL_ITERATIONS",
    "BQ_LOG_DATASET",
    "BQ_LOG_TABLE",
    "BQ_LIMIT_TABLE",
    "ADMIN_EMAILS",
    "CHAT_HISTORY_BUCKET",
    # ADMIN_ALLOW_LOCAL は .env 専用（認証を迂回するフラグなので
    # Streamlit Cloud の Secrets からは意図的に読まない）
)


def auth_configured() -> bool:
    """`st.login()` 用の [auth] 設定が secrets にあるか。

    Streamlit 1.42 以降、Community Cloud の閲覧者制限だけでは
    メールアドレスが取れない（st.user は空になる）。取得するには
    自前で OIDC（Google など）を [auth] に設定し st.login() する必要がある。
    ローカル開発では通常これが無いので、ここが False になり
    ログインゲート自体をスキップする。
    """
    try:
        return "auth" in dict(st.secrets)
    except Exception:
        return False


def resolve_viewer_email() -> str:
    """st.login() でログイン済みなら、そのメールアドレス。それ以外は空文字。

    `is_logged_in` は認証未設定でも常に存在する（未ログイン扱い）。
    `email` は実際にログインし、プロバイダがそのクレームを返した場合のみ入る。
    """
    if getattr(st.user, "is_logged_in", False):
        email = getattr(st.user, "email", None)
        if email:
            return str(email)
    return ""


def _bootstrap_secrets() -> None:
    """Streamlit Cloud の Secrets を環境変数へ写す。

    ローカルでは secrets.toml が存在しないため st.secrets へのアクセスが
    例外になる。その場合は何もせず .env 由来の設定に委ねる。
    サービスアカウントキーは中身を検査せずそのまま一時ファイルへ書き出し、
    パスだけを GOOGLE_APPLICATION_CREDENTIALS に渡す（config.py の
    「パスしか扱わない」設計はそのまま維持する）。
    """
    try:
        secrets = dict(st.secrets)
    except Exception:
        return

    for key in _SECRET_ENV_KEYS:
        if key in secrets and not os.getenv(key):
            os.environ[key] = str(secrets[key])

    if "gcp_service_account" in secrets and not os.getenv(
        "GOOGLE_APPLICATION_CREDENTIALS"
    ):
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(dict(secrets["gcp_service_account"]), f)
            key_path = f.name
        os.chmod(key_path, 0o600)
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = key_path


@st.cache_resource(show_spinner=False)
def build_runtime() -> tuple[Settings, BigQueryTools, GeminiAgent, UsageLogger]:
    _bootstrap_secrets()
    settings = load_settings()
    usage_logger = UsageLogger(settings)
    # ベストエフォート。失敗しても例外は投げず、起動を止めない。
    usage_logger.ensure_table()
    return settings, BigQueryTools(settings), GeminiAgent(settings), usage_logger


def _new_context(
    settings: Settings,
    tools: BigQueryTools,
    usage_logger: UsageLogger,
    user_email: str,
) -> ToolContext:
    return ToolContext(
        tools=tools,
        settings=settings,
        usage_logger=usage_logger,
        user_email=user_email,
    )


def init_state(
    settings: Settings,
    tools: BigQueryTools,
    usage_logger: UsageLogger,
    user_email: str,
) -> ToolContext:
    if st.session_state.get("conversation_owner") != user_email:
        reset_conversation(settings, tools, usage_logger, user_email)
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "contents" not in st.session_state:
        st.session_state.contents = []
    if "pending" not in st.session_state:
        st.session_state.pending = None
    if "ctx" not in st.session_state:
        st.session_state.ctx = _new_context(
            settings, tools, usage_logger, user_email
        )
    if st.session_state.ctx.thinking_mode not in ("low", "medium", "high"):
        st.session_state.ctx.thinking_mode = DEFAULT_EFFORT
    return st.session_state.ctx


def reset_conversation(
    settings: Settings,
    tools: BigQueryTools,
    usage_logger: UsageLogger,
    user_email: str,
) -> None:
    active_turn = st.session_state.pop("active_turn", None)
    if active_turn is not None:
        active_turn.control.request_stop()
    if st.session_state.get("conversation_owner") != user_email:
        st.session_state.pop("history_entries", None)
        st.session_state.pop("history_payload_cache", None)
        st.session_state.pop("history_visible_count", None)
        st.session_state.history_list_error = ""
    # 会話を切り替えたとき、以前の図表ウィジェットの選択を持ち越さない。
    for key in list(st.session_state):
        if key.startswith(("chart_", "dl_", "clarification_")):
            del st.session_state[key]
    st.session_state.messages = []
    st.session_state.contents = []
    st.session_state.pending = None
    st.session_state.ctx = _new_context(settings, tools, usage_logger, user_email)
    st.session_state.pop("command_output", None)
    st.session_state.pop("effort_panel", None)
    st.session_state.conversation_owner = user_email
    st.session_state.history_id = uuid.uuid4().hex
    st.session_state.history_generation = 0
    st.session_state.history_dirty = False
    st.session_state.history_error = ""
    st.session_state.history_status = "ready"


@st.cache_resource(show_spinner=False)
def history_store(bucket: str, project: str) -> HistoryStore:
    return HistoryStore(bucket, project)


def history_cache(settings, user_email):
    owner = (settings.gcp_project, settings.chat_history_bucket, user_email)
    cache = st.session_state.get("history_payload_cache")
    if cache is None or cache.owner != owner:
        cache = HistoryCache(owner)
        st.session_state.history_payload_cache = cache
    return cache


def save_conversation(ctx: ToolContext, status: str = "ready") -> None:
    if not ctx.user_email or not ctx.settings.chat_history_bucket:
        return
    st.session_state.history_dirty = True
    st.session_state.history_status = status
    try:
        data = pack(st.session_state.messages, st.session_state.contents, ctx,
                    st.session_state.pending, status)
        title = conversation_title(st.session_state.messages)
        is_new = st.session_state.history_generation == 0
        generation = history_store(ctx.settings.chat_history_bucket, ctx.settings.gcp_project).save(
            ctx.user_email, st.session_state.history_id, data, title,
            st.session_state.history_generation,
        )
        st.session_state.history_generation = generation
        history_cache(ctx.settings, ctx.user_email).remember(st.session_state.history_id, data, generation)
        st.session_state.history_dirty = False
        st.session_state.history_error = ""
        if is_new:
            # 新しい会話を初めて保存した時だけ、クラウドの一覧を再取得する。
            st.session_state.pop("history_entries", None)
            st.session_state.history_list_error = ""
        else:
            # 一覧の再取得なしで、削除時の競合検出に使う世代番号を追従する。
            for entry in st.session_state.get("history_entries", []):
                if entry["id"] == st.session_state.history_id:
                    entry["generation"] = generation
                    entry["title"] = title
    except PreconditionFailed:
        st.session_state.history_error = "別のタブで履歴が更新または削除されました。「別の会話として保存」で現在の内容を残せます。"
    except Exception:
        logger.warning("会話履歴の保存に失敗しました。", exc_info=True)
        st.session_state.history_error = "履歴を保存できませんでした。この画面の会話は保持されています。再保存してください。"


def render_history(settings, tools, usage_logger, user_email, ctx) -> None:
    st.subheader("チャット履歴")
    if not settings.chat_history_bucket:
        st.caption("履歴保存は未設定です。")
        return
    if not user_email:
        st.caption("履歴の保存・再開にはログインが必要です。")
        return
    store = history_store(settings.chat_history_bucket, settings.gcp_project)
    running = st.session_state.get("active_turn") is not None
    if st.session_state.history_error:
        st.warning(st.session_state.history_error)
        if st.button("再保存", disabled=running):
            save_conversation(ctx, st.session_state.history_status)
            st.rerun()
        if st.button("別の会話として保存", disabled=running):
            st.session_state.history_id = uuid.uuid4().hex
            st.session_state.history_generation = 0
            save_conversation(ctx, st.session_state.history_status)
            st.rerun()
    elif st.session_state.messages:
        st.caption("自動保存済み")
    try:
        if "history_entries" not in st.session_state:
            st.session_state.history_entries = store.list(user_email)
        entries = st.session_state.history_entries
    except Exception:
        logger.warning("履歴一覧の取得に失敗しました。", exc_info=True)
        st.session_state.history_entries = []
        st.session_state.history_list_error = "履歴一覧を取得できませんでした。アプリを開き直すか、新しい会話を始めると再取得します。"
    if st.session_state.get("history_list_error"):
        st.warning(st.session_state.history_list_error)
        return
    entries = st.session_state.history_entries
    if not entries:
        st.caption("保存された会話はありません。")
        return
    cache = history_cache(settings, user_email)
    cache.prefetch(store, user_email, entries[:5])
    dirty = st.session_state.history_dirty or st.session_state.get("active_turn") is not None
    # 履歴内だけを装飾する。標準ウィジェットのキーボード操作は維持する。
    st.markdown("""
<style>
.st-key-history_list [class*="st-key-history_row_"] {
    padding: .35rem .55rem; border-radius: .5rem;
}
.st-key-history_list [class*="st-key-history_row_active_"] {
    background: color-mix(in srgb, currentColor 8%, transparent);
}
.st-key-history_list [class*="st-key-history_row_"]:not([class*="history_row_active_"]):hover {
    background: color-mix(in srgb, currentColor 5%, transparent);
}
.st-key-history_list [class*="st-key-open_history_"] button {
    border: none; background: transparent; box-shadow: none;
    justify-content: flex-start; text-align: left; padding: 0;
    min-height: 1.6rem; color: inherit;
}
.st-key-history_list [class*="st-key-open_history_"] button:focus-visible {
    outline: 2px solid currentColor; outline-offset: 3px;
}
.st-key-history_list [class*="st-key-open_history_"] button > div {
    justify-content: flex-start;
}
.st-key-history_list [data-testid="stVerticalBlock"] { gap: .15rem; }
.st-key-history_list [data-testid="stCaptionContainer"] { font-size: .75rem; }
.st-key-history_more button {
    border: none; background: transparent; box-shadow: none;
    font-size: .8rem; opacity: .65; padding: .15rem .55rem; min-height: 1.6rem;
}
.st-key-history_more button:hover { opacity: 1; }
.st-key-history_list [data-testid="stPopover"] button {
    border: none; background: transparent; padding: 0 .3rem;
    min-height: 1.6rem; box-shadow: none;
}
</style>
""", unsafe_allow_html=True)
    def render_entries(items):
        for entry in items:
            selected = entry["id"]
            active = selected == st.session_state.history_id
            row_key = f"history_row_{'active_' if active else ''}{selected}"
            title = conversation_title([{"role": "user", "text": entry["title"]}])
            with st.container(key=row_key):
                text_column, menu_column = st.columns([9, 1])
                with text_column:
                    clicked = st.button(
                        title, key=f"open_history_{selected}", width="stretch", disabled=dirty,
                        help="表示中の会話" if active else "会話を開く",
                    )
                    st.caption(entry["updated"].astimezone(ZoneInfo("Asia/Tokyo")).strftime("%Y/%m/%d %H:%M"))
                if active:
                    with menu_column:
                        with st.popover("⋯", help="会話の操作", disabled=dirty):
                            confirm_delete = st.checkbox("この会話を削除する", key=f"delete_history_{selected}")
                            delete = st.button("履歴から削除", disabled=not confirm_delete or dirty)
                    if delete:
                        try:
                            store.delete(user_email, selected, entry["generation"])
                            cache.discard(selected)
                            st.session_state.history_entries = [e for e in entries if e["id"] != selected]
                            reset_conversation(settings, tools, usage_logger, user_email)
                            st.rerun()
                        except Exception:
                            logger.warning("履歴の削除に失敗しました。", exc_info=True)
                            st.error("削除できませんでした。アプリを開き直して再試行してください。")
                if clicked:
                    open_conversation(settings, tools, usage_logger, user_email, store, selected)

    with st.container(key="history_list"):
        visible = st.session_state.get("history_visible_count", 5)
        render_entries(entries[:visible])
        if len(entries) > visible:
            if st.button("さらに表示", key="history_more", disabled=running):
                st.session_state.history_visible_count = visible + 5
                st.rerun()


def open_conversation(settings, tools, usage_logger, user_email, store, selected) -> None:
    try:
        entry = next((entry for entry in st.session_state.get("history_entries", []) if entry["id"] == selected), None)
        data, generation = history_cache(settings, user_email).load(
            store, user_email, selected, entry["generation"] if entry else None,
        )
        new_ctx = _new_context(settings, tools, usage_logger, user_email)
        messages, contents, pending = unpack(data, new_ctx)
        reset_conversation(settings, tools, usage_logger, user_email)
        st.session_state.messages = messages
        st.session_state.contents = contents
        st.session_state.pending = pending
        st.session_state.ctx = new_ctx
        st.session_state.history_id = selected
        st.session_state.history_generation = generation
        # 開いた会話の最新世代を反映する。履歴一覧の再取得はしない。
        for entry in st.session_state.get("history_entries", []):
            if entry["id"] == selected:
                entry["generation"] = generation
        st.rerun()
    except Exception as exc:
        logger.warning("履歴の復元に失敗しました。", exc_info=True)
        st.error(str(exc) if isinstance(exc, ValueError) else "会話を開けませんでした。アプリを開き直して再試行してください。")


# --------------------------------------------------------------------
# 月次の使用制限
# --------------------------------------------------------------------
@st.cache_data(ttl=60, show_spinner=False)
def _cached_limits(_usage_logger: UsageLogger) -> Any:
    """制限額テーブルの全件（低頻度更新データなので ttl=60 で十分）。"""
    return _usage_logger.get_limits()


def _limit_for(usage_logger: UsageLogger, user_email: str) -> float:
    try:
        limits = _cached_limits(usage_logger)
    except Exception:  # noqa: BLE001 - 読み取り失敗時は既定値にフォールバック
        return DEFAULT_MONTHLY_LIMIT_USD
    if limits.empty:
        return DEFAULT_MONTHLY_LIMIT_USD
    # BigQuery 側の列名は weekly_limit_usd のまま（週次→月次への切り替え時に
    # 列名までは変更していない。デプロイ済みテーブルの移行を避けるための判断）。
    match = limits.loc[limits["user_email"] == user_email, "weekly_limit_usd"]
    return float(match.iloc[0]) if not match.empty else DEFAULT_MONTHLY_LIMIT_USD


def _effective_monthly_usage_usd(
    ctx: ToolContext, usage_logger: UsageLogger, user_email: str
) -> float | None:
    """今月分の使用金額（USD、Gemini トークン + BigQuery クエリ）。
    読み取り失敗時は None（＝判定不能、ブロックしない）。

    月替わり・セッション開始時だけ BigQuery に問い合わせ、以降は
    その時点を基準にセッション内の増分（トークン・課金バイト）を
    インメモリで加算する。同一セッション内で毎ターン問い合わせないための
    近似で、複数タブ/複数セッションを同時に使った場合にわずかな誤差が
    出ることは許容する。
    """
    start_utc, _end_utc = jst_month_bounds_utc()
    if ctx.limit_baseline_month_start != start_utc:
        try:
            usage = usage_logger.monthly_usage(user_email, start_utc, _end_utc)
        except Exception:  # noqa: BLE001 - フェイルオープン。読めなければブロックしない
            logger.warning("月次使用量の取得に失敗しました。", exc_info=True)
            return None
        ctx.limit_baseline_usd = usage.cost_usd()
        ctx.limit_baseline_month_start = start_utc
        ctx.limit_baseline_session_prompt_tokens = ctx.session_prompt_tokens
        ctx.limit_baseline_session_output_tokens = ctx.session_output_tokens
        ctx.limit_baseline_session_billed_bytes = ctx.session_billed_bytes
        ctx.limit_baseline_session_gemini_cost_usd = ctx.gemini_cost_usd()

    delta = (ctx.gemini_cost_usd() - ctx.limit_baseline_session_gemini_cost_usd
             + bq_cost_usd(ctx.session_billed_bytes - ctx.limit_baseline_session_billed_bytes))
    return ctx.limit_baseline_usd + delta


# --------------------------------------------------------------------
# 表示
# --------------------------------------------------------------------
CHART_FUNCS = {
    "bar": st.bar_chart,
    "line": st.line_chart,
    "area": st.area_chart,
    "scatter": st.scatter_chart,
}


def render_table(run: QueryRun, idx: int, msg_idx: int) -> None:
    rows = len(run.dataframe)
    st.dataframe(run.dataframe, width="stretch", height=min(400, 60 + rows * 35))
    st.download_button(
        "結果を CSV でダウンロード",
        data=run.dataframe.to_csv(index=False).encode("utf-8-sig"),
        file_name=f"result_{msg_idx}_{idx}.csv",
        mime="text/csv",
        key=f"dl_{msg_idx}_{idx}",
    )


def render_chart(df, spec: ChartSpec, key_prefix: str) -> None:
    kwargs = {"x": spec.x, "y": spec.y}
    if spec.color:
        kwargs["color"] = spec.color
    if spec.kind == "bar":
        kwargs["sort"] = False
        kwargs["stack"] = True if spec.color else False
    try:
        if spec.kind in ("pie", "heatmap"):
            st.vega_lite_chart(df, special_chart_spec(spec), width="stretch")
        else:
            CHART_FUNCS[spec.kind](df, **kwargs)
    except Exception:  # noqa: BLE001
        st.warning("グラフを描画できなかったため、表で表示しています。")
        st.dataframe(df, hide_index=True)
        return
    if spec.reason:
        st.caption(spec.reason)


def render_results(message: dict[str, Any], msg_idx: int, shown: set[str]) -> None:
    runs = message.get("runs") or []
    analyses = message.get("analyses") or []
    events = message.get("execution_events") or []
    if not runs and not analyses and not events and not message.get("turn_id") and not message.get("errors"):
        return
    plan = choose_presentation(message.get("question", ""), runs, shown)
    if message.get("is_error") or message.get("is_stopped"):
        plan.mode = "none"
    if plan.mode == "table":
        if plan.note:
            st.caption(plan.note)
        render_table(runs[plan.index], plan.index, msg_idx)
    elif plan.mode == "chart":
        render_chart(plan.dataframe, plan.chart, f"chart_{msg_idx}_{plan.index}")
        if plan.note:
            st.caption(plan.note)
    if plan.mode != "none":
        shown.add(plan.signature)

    # 回答に使った結果と途中結果を、必要なときだけ確認できるようにする。
    with st.expander("データ・実行詳細"):
        if message.get("turn_id"):
            st.caption(f"ターンID: {message['turn_id']}")
        if message.get("elapsed_seconds") is not None:
            st.caption(f"処理時間（確認待ちを含む）: {message['elapsed_seconds']:.2f} 秒")
        if events:
            rows = []
            for event in events:
                rows.append({"処理": event.get("tool") or event.get("event"),
                             "開始日時": event.get("started_at"), "秒": event.get("seconds"), "状態": event.get("status", ""), "エラー": event.get("error", False),
                             "入力トークン": event.get("input_tokens"), "出力トークン": event.get("output_tokens"),
                             "思考トークン": event.get("thinking_tokens"), "キャッシュ入力": event.get("cached_input_tokens")})
            st.dataframe(pd.DataFrame(rows), hide_index=True)
            st.caption("AI呼び出し時間は通信・再試行待機を含みます。再試行待機の行はAI呼び出し時間の内数です。空欄は未取得です。モデル内部の思考時間は取得できません。")
        summary = message.get("execution_summary") or {}
        if summary:
            with st.expander("引き継ぎ要約"):
                if summary.get("execution_note"):
                    st.write("実行状態の要約", summary["execution_note"])
                if summary.get("summary"):
                    st.write("今回の要約", summary["summary"])
                if summary.get("conversation_summary"):
                    st.write("会話全体の要約", summary["conversation_summary"])
                note = summary.get("context_note") or {}
                if note.get("summary"):
                    st.write("AIの作業メモ（解釈）", note["summary"])
                    if note.get("references"):
                        st.caption("参照結果ID: " + ", ".join(note["references"]))
                for step in summary.get("steps", []):
                    st.text(f"{step['tool']}: {step['status']}" + (f" · {step['result_id']}" if step.get('result_id') else ""))
                    if step.get("error"):
                        st.text(step["error"])
                if summary.get("next_action"):
                    st.write("次の行動", summary["next_action"])
        for error in message.get("errors") or []:
            st.text(error)
        for idx, run in enumerate(runs):
            with st.expander(run.purpose or f"クエリ {idx + 1}"):
                if not (plan.mode == "table" and idx == plan.index):
                    if run.dataframe.empty:
                        st.info("このクエリの結果は 0 件でした。")
                    else:
                        render_table(run, idx, msg_idx)
                st.code(run.sql, language="sql")
                st.caption(f"実行クエリ量: {human_bytes(run.billed_bytes)} · 結果件数: {len(run.dataframe):,} 件")
                for note in run.notes:
                    st.caption(f"ℹ️ {note}")
        for idx, analysis in enumerate(analyses):
            render_analysis_detail(analysis, idx, msg_idx)


ANALYSIS_LABELS = {
    "get_schema": "列・型の確認", "get_row_count": "行数の確認", "describe_data": "基本統計",
    "aggregate": "グループ別集計", "top_k": "上位・下位の抽出", "filter_data": "条件による絞り込み",
    "compare_groups": "グループ間の比較", "correlation": "相関分析", "detect_outliers": "外れ値候補の検出",
    "analyze_timeseries": "時系列分析", "get_distinct_values": "値ごとの分布", "sample_data": "サンプル抽出",
    "sql": "保存済み結果へのSQL分析",
}


def render_analysis_detail(analysis: dict, idx: int, msg_idx: int) -> None:
    label = ANALYSIS_LABELS.get(analysis.get("operation"), "データ分析")
    with st.expander(f"分析：{analysis.get('purpose') or label}"):
        st.caption(f"分析方法：{label}")
        parameters = analysis.get("parameters") or {}
        labels = {"columns": "対象列", "group_by": "集計軸", "metric": "指標", "aggregation": "集計方法",
                  "aggregations": "指標・集計関数", "sort_by": "並べ替え列", "ascending": "昇順",
                  "absolute": "絶対値で比較", "baseline": "比較基準", "method": "計算方法",
                  "time_column": "日時列", "frequency": "期間単位", "window": "移動平均の期間数",
                  "view": "抽出する変化", "limit": "AIへの返却行数指定", "offset": "開始行", "filters": "抽出条件"}
        for key, name in labels.items():
            if key in parameters:
                st.write(f"**{name}**", parameters[key])
        if parameters.get("sql"):
            st.code(parameters["sql"], language="sql")
        if analysis.get("error"):
            st.warning(analysis["error"])
            return
        frame = analysis.get("dataframe")
        if frame is not None:
            st.caption(f"分析元: {analysis.get('source_row_count', 0):,} 行 · 条件一致: {analysis.get('matched_row_count', 0):,} 行 · 分析結果: {len(frame):,} 行 · AIへの返却: {analysis.get('sent_row_count', 0):,} 行")
            if frame.empty:
                st.info("該当する分析結果はありませんでした。")
            else:
                st.dataframe(frame, hide_index=True)
                st.download_button("分析結果をCSVでダウンロード", frame.to_csv(index=False).encode("utf-8-sig"),
                    file_name=f"analysis_{msg_idx}_{idx}.csv", mime="text/csv", key=f"analysis_csv_{msg_idx}_{idx}")
        st.caption(f"分析時間: {analysis.get('analysis_seconds', 0):.3f} 秒")
        if analysis.get("source_truncated"):
            st.caption("分析対象は保存済みの範囲に限定されています。元の結果に切り捨てがあります。")
        for note in analysis.get("notes", []):
            st.caption(f"ℹ️ {note}")


def render_message(message: dict[str, Any], msg_idx: int, shown: set[str] | None = None) -> None:
    with st.chat_message(message["role"]):
        if message.get("is_error"):
            st.warning(message["text"])
        else:
            st.markdown(message["text"])

        errors = message.get("errors") or []
        if errors:
            with st.expander(f"診断情報（{len(errors)} 件）"):
                for err in errors:
                    st.code(err)
        render_results(message, msg_idx, shown if shown is not None else set())


def render_sidebar(
    settings: Settings,
    ctx: ToolContext,
    tools: BigQueryTools,
    usage_logger: UsageLogger,
    user_email: str,
    monthly_used_usd: float | None,
    monthly_limit_usd: float,
) -> None:
    with st.sidebar:
        if st.button("新しいチャット", width="stretch", disabled=st.session_state.history_dirty or st.session_state.get("active_turn") is not None):
            reset_conversation(settings, tools, usage_logger, user_email)
            st.rerun()

        render_history(settings, tools, usage_logger, user_email, ctx)

        if user_email and settings.logging_enabled and monthly_used_usd is not None:
            st.divider()
            st.subheader("今月の利用状況")
            ratio = monthly_used_usd / monthly_limit_usd if monthly_limit_usd > 0 else 1.0
            st.progress(min(max(ratio, 0.0), 1.0))
            st.markdown(
                f"- 利用量: {ratio * 100:.0f}%"
            )

            if ratio >= 1.0:
                st.error("🚫 今月の上限に達しました。次の質問はブロックされます。")
            elif ratio >= 0.8:
                st.warning("⚠️ 月の上限の80%を使用しています。")
            else:
                st.caption("毎月1日 0:00（JST）にリセットされます。")
            # st.caption("上限の引き上げは店長提案の承認後、DXにお問い合わせください。")

        st.divider()
        st.link_button(
            "📝 質問・報告フォーム", FEEDBACK_FORM_URL, width="stretch"
        )

        if user_email:
            st.divider()
            st.write(f"ログイン中: {user_email}")
            if st.button("ログアウト", width="stretch"):
                active_turn = st.session_state.get("active_turn")
                if active_turn is not None:
                    active_turn.control.request_stop()
                st.logout()


# --------------------------------------------------------------------
# ターン処理
# --------------------------------------------------------------------
def process_turn(agent: GeminiAgent, ctx: ToolContext) -> None:
    question = next((m.get("question", m["text"]) for m in reversed(st.session_state.messages) if m["role"] == "user"), "")
    task = st.session_state.get("active_turn")
    if task is None:
        task = BackgroundTurn(agent, st.session_state.contents, ctx)
        st.session_state.active_turn = task
        task.start()
    try:
        st.markdown("""
<style>
.st-key-prompt_input [data-testid="stChatInputSubmitButton"] { visibility: hidden; }
.st-key-execution_stop { position: fixed; display: none; z-index: 10001; }
.st-key-execution_stop button {
    border-radius: 50%; width: 2rem; height: 2rem;
    min-height: 2rem; padding: 0; border: none;
    background: transparent; color: inherit;
}
.st-key-execution_stop button::before {
    content: ""; display: block; width: 2rem; height: 2rem;
    background: currentColor;
    mask: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 44 44'%3E%3Cpath fill-rule='evenodd' d='M22 1a21 21 0 1 0 0 42a21 21 0 1 0 0-42M16 16h12v12H16z'/%3E%3C/svg%3E") center / contain no-repeat;
}
.st-key-execution_stop button:focus-visible {
    outline: 2px solid currentColor; outline-offset: 3px;
}
.st-key-execution_stop button:disabled {
    opacity: .4;
}
.st-key-execution_stop button > div {
    position: absolute; width: 1px; height: 1px; overflow: hidden;
    clip-path: inset(50%);
}
</style>
""", unsafe_allow_html=True)
        st.button(
            "実行を停止", key="execution_stop", help="実行を停止",
            on_click=task.control.request_stop,
            disabled=task.done.is_set() or task.control.stopped.is_set(),
        )
        render_prompt_input(ctx, running=True)
        with st.status("質問に合う調べ方を考えています…", expanded=False) as status:
            while not task.done.wait(timeout=0.2):
                # このUI更新でStreamlitに制御を返し、停止クリックによる再実行を受ける。
                status.update(label="停止しています…（実行中のリクエストの終了を待っています）" if task.control.stopped.is_set() else task.message)
            status.update(label=task.message)
            if task.control.stopped.is_set():
                status.update(label="処理を停止しました", state="complete")
        if task.control.stopped.is_set():
            raise ExecutionStopped()
        if task.error is not None:
            raise task.error
        result = task.result
        if isinstance(result, ConfirmationPending):
            st.session_state.pending = result
        elif isinstance(result, (AnswerResult, ClarificationResult)):
            st.session_state.pending = None
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "text": result.text,
                    "question": question,
                    "runs": list(ctx.turn_runs), "analyses": list(getattr(ctx, "turn_analyses", [])),
                    "errors": list(ctx.turn_errors),
                    **({"clarification": {
                        "options": result.options, "selection_mode": result.selection_mode,
                        "answered": False,
                    }}
                       if isinstance(result, ClarificationResult) else {}),
                }
            )
        else:
            # 通常発生しない想定外の戻り値（実行環境側の一時的な異常等）。
            # 他の例外と同じ経路で扱い、アプリ全体を落とさない。
            raise AssertionError(f"想定外の戻り値: {result!r}")
    except ExecutionStopped:
        if ctx.interaction_state:
            ctx.interaction_state['reset_required'] = True
        contents = st.session_state.contents
        # 実行途中のツール要求にも応答を付け、次の質問で会話形式が壊れないようにする。
        if contents and contents[-1].role == "model":
            calls = [part.function_call for part in (contents[-1].parts or []) if part.function_call]
            if calls:
                contents.append(types.Content(role="user", parts=[
                    types.Part(function_response=types.FunctionResponse(
                        id=call.id, name=call.name,
                        response={"status": "stopped", "message": "ユーザーが処理を停止しました。途中の結果がある場合も、この調査を自動で続行しないでください。"},
                    )) for call in calls
                ]))
        ctx.resume_calls = None
        st.session_state.pending = None
        text = "処理を停止しました。"
        if contents and contents[-1].role == "model":
            # 完了直前の停止で生成済みの回答があれば、未表示の回答を停止通知に置き換える。
            contents.pop()
        contents.append(types.Content(role="model", parts=[types.Part.from_text(text=text)]))
        st.session_state.messages.append({
            "role": "assistant", "text": text, "question": question,
            "runs": list(ctx.turn_runs), "analyses": list(getattr(ctx, "turn_analyses", [])), "errors": list(ctx.turn_errors), "is_stopped": True,
        })
    except Exception as exc:  # noqa: BLE001 - ユーザーには整形して見せる
        detail = technical_detail(exc, limit=10_000)
        if ctx.usage_logger is not None:
            ctx.usage_logger.log_error(
                user_email=ctx.user_email,
                session_id=ctx.session_id,
                turn_id=ctx.turn_id,
                error_source="process_turn_error",
                error_code=error_code(exc),
                error_text=detail,
            )
        # 既存のツール実行エラーと同じく、UI の「診断情報」expander にも
        # 技術的詳細を出す（CLAUDE.md の「エラーの二経路」方針に揃える）。
        ctx.turn_errors.append(f"{friendly_error(exc)}\n詳細: {detail}")
        st.session_state.pending = None
        st.session_state.messages.append(
            {
                "role": "assistant",
                "text": friendly_error(exc),
                "question": question,
                "runs": list(ctx.turn_runs), "analyses": list(getattr(ctx, "turn_analyses", [])),
                "errors": list(ctx.turn_errors),
                "is_error": True,
            }
        )
    if st.session_state.messages and st.session_state.messages[-1].get("role") == "assistant":
        st.session_state.messages[-1].update(
            turn_id=ctx.turn_id, execution_events=list(ctx.turn_events),
            elapsed_seconds=max(0, time.time() - ctx.turn_started_at) if ctx.turn_started_at else None,
            execution_summary=analysis_safety.snapshot(ctx),
        )
    st.session_state.pop("active_turn", None)
    save_conversation(ctx)
    st.rerun()


def submit_prompt(
    prompt: str, agent: GeminiAgent, ctx: ToolContext, question: str | None = None,
) -> None:
    """通常の質問と確認への回答を、同じ記録・実行経路に送る。"""
    # コマンドはUI側の専用経路だけで扱う。直接呼び出されてもモデルへ送らない。
    if parse_command(prompt) is not None:
        raise ValueError("コマンドは質問として送信できません。")
    st.session_state.pop("command_output", None)
    message = {"role": "user", "text": prompt}
    if question is not None:
        message["question"] = question
    st.session_state.messages.append(message)
    st.session_state.contents.append(
        types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
    )
    ctx.start_turn()
    if ctx.usage_logger is not None:
        ctx.usage_logger.log_prompt(
            user_email=ctx.user_email,
            session_id=ctx.session_id,
            turn_id=ctx.turn_id,
            prompt_text=prompt,
        )
    save_conversation(ctx, "running")
    # 入力ウィジェットを実行中の表示へ切り替えてから処理を描画する。
    # 同じ再実行内で同じchat_inputを二度作らない。
    task = BackgroundTurn(agent, st.session_state.contents, ctx)
    st.session_state.active_turn = task
    task.start()
    st.rerun()


def toggle_clarification_choice(choice_key: str, option: str, selection_mode: str) -> None:
    selected = list(st.session_state.get(choice_key, []))
    if option in selected:
        selected.remove(option)
    elif selection_mode == "multi":
        selected.append(option)
    else:
        selected = [option]
    st.session_state[choice_key] = selected


def render_clarification(
    message: dict[str, Any], msg_idx: int, agent: GeminiAgent,
    ctx: ToolContext, blocked: bool = False,
) -> None:
    selection_mode = message["clarification"].get("selection_mode", "single")
    prefix = f"clarification_{ctx.session_id}_{msg_idx}"
    choice_key = f"{prefix}_choices"
    panel_key = f"clarification_picker_{selection_mode}_{ctx.session_id}_{msg_idx}"
    intent = st.session_state.pop("clarification_keyboard", None)
    keyboard_submit = False
    if not blocked and isinstance(intent, dict) and intent.get("panel") == panel_key:
        index = intent.get("index")
        options = message["clarification"]["options"]
        if isinstance(index, int) and 0 <= index < len(options):
            if selection_mode == "single":
                st.session_state[choice_key] = [options[index]]
            else:
                toggle_clarification_choice(choice_key, options[index], selection_mode)
            keyboard_submit = selection_mode == "single"
        elif intent.get("submit") is True:
            keyboard_submit = True
    selected = st.session_state.get(choice_key, [])
    with st.container(border=True, key=panel_key):
        st.markdown("選択肢から回答（複数選択可）" if selection_mode == "multi" else "選択肢から回答（1つ選択）")
        with st.container(horizontal=True):
            for index, option in enumerate(message["clarification"]["options"], start=1):
                st.button(
                    f"{index}. {option}", key=f"{prefix}_option_{index}",
                    type="primary" if option in selected else "secondary",
                    disabled=blocked,
                    on_click=toggle_clarification_choice,
                    args=(choice_key, option, selection_mode),
                )
        st.caption("数字キーで選択、↑↓で移動。単回答は数字またはEnterで即送信、複数回答は数字で選択・解除してEnterで送信します。文字入力中は選択されません。")
        free_text = st.text_area(
            "自由記述・補足（選択せずに入力することもできます）", disabled=blocked,
            key=f"{prefix}_free_text",
        )
        submitted = st.button("回答して続ける", type="primary", disabled=blocked, key=f"{prefix}_submit")
    if not (submitted or keyboard_submit) or blocked:
        return
    free_text = free_text.strip()
    if not selected and not free_text:
        st.warning("選択肢を選ぶか、自由記述を入力してください。")
        return
    answer = "\n".join(part for part in [*selected, free_text] if part)
    message["clarification"]["answered"] = True
    submit_prompt(
        f"確認への回答（{message['text']}）：\n{answer}", agent, ctx,
        question=message.get("question", ""),
    )


def render_confirmation(
    pending: ConfirmationPending, agent: GeminiAgent, ctx: ToolContext
) -> None:
    with st.chat_message("assistant"):
        st.warning(
            f"このクエリは推定 **{human_bytes(pending.estimated_bytes)}** をスキャンします"
            f"（確認閾値 {human_bytes(pending.threshold_bytes)}）。実行してよろしいですか？"
        )
        st.code(pending.sql, language="sql")
        for note in pending.notes:
            st.caption(f"ℹ️ {note}")

        left, right, _ = st.columns([1, 1, 3])
        approve = left.button("実行する", type="primary", key="approve")
        deny = right.button("実行しない", key="deny")

    if approve or deny:
        ctx.decisions[pending.key] = "approved" if approve else "denied"
        st.session_state.pending = None
        process_turn(agent, ctx)


# --------------------------------------------------------------------
# ページ
# --------------------------------------------------------------------
def set_effort(ctx: ToolContext, mode: str) -> str | None:
    if mode not in ("low", "medium", "high"):
        return "`/effort` の選択画面でLow、Medium、Highを選んでください。"
    if ctx.settings.gemini_model != "gemini-3.8-flash":
        return "この思考設定はGemini 3.8 Flashに対応しています。GEMINI_MODELの設定を確認してください。"
    if st.session_state.get("pending") is not None:
        return "実行確認待ちです。現在のクエリを実行または中止してから思考設定を変更してください。"
    ctx.thinking_mode = mode
    if st.session_state.messages:
        save_conversation(ctx)
    return None


def render_prompt_input(ctx: ToolContext, *, running: bool = False) -> str | None:
    mode = st.session_state.pop("keyboard_effort", None)
    if mode is not None and not running:
        error = set_effort(ctx, mode)
        st.session_state.pop("effort_panel", None)
        if error:
            st.session_state.command_output = error
        else:
            st.session_state.pop("command_output", None)
        st.rerun()
    color = {"low": "#0d9488", "medium": "#3b82f6", "high": "#8b5cf6"}.get(getattr(ctx, "thinking_mode", DEFAULT_EFFORT), "#0d9488")
    st.markdown("""
<style>
.st-key-prompt_input [data-testid="stChatInput"],
.st-key-prompt_input [data-testid="stChatInput"]:focus-within {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}
.st-key-prompt_input [data-testid="stChatInput"] > div {
    border: 2px solid __COLOR__ !important;
    border-radius: .5rem;
}
.st-key-prompt_input [data-testid="stChatInput"]:focus-within > div {
    border-color: __COLOR__ !important;
    box-shadow: 0 0 0 2px __COLOR__33;
}
</style>
""".replace("__COLOR__", color), unsafe_allow_html=True)
    prompt = st.chat_input("処理中…" if running else "ShiruGPT に聞く", key="prompt_input", disabled=running)
    render_command_menu(effort_open=bool(st.session_state.get("effort_panel")) and not running,
                        effort=getattr(ctx, "thinking_mode", DEFAULT_EFFORT), running=running)
    return st.session_state.pop("keyboard_command", None) or prompt


def handle_command(prompt, ctx, settings, tools, usage_logger, user_email,
                   monthly_used_usd, monthly_limit_usd) -> bool:
    command = parse_command(prompt)
    if command is None:
        return False
    if command.name == "/help" and not command.argument:
        output = HELP
    elif command.name == "/usage" and not command.argument:
        _, end_utc = jst_month_bounds_utc()
        output = usage_text(
            ctx, monthly_used_usd, monthly_limit_usd if user_email else None,
            end_utc.astimezone(ZoneInfo("Asia/Tokyo")),
        )
        if user_email and getattr(settings, "logging_enabled", False):
            try:
                output += usage_details_text(usage_logger.usage_details(user_email))
            except Exception as exc:
                logger.warning("当月・全期間使用量の取得に失敗しました。", exc_info=True)
                output += "\n\n当月・全期間の詳細を取得できませんでした。時間をおいて `/usage` を実行してください。"
                usage_logger.log_error(user_email=user_email, session_id=ctx.session_id, turn_id=ctx.turn_id,
                                       error_source="usage_report_error", error_text=technical_detail(exc, limit=10_000))
    elif command.name == "/new" and not command.argument:
        if st.session_state.get("history_dirty"):
            save_conversation(ctx, st.session_state.get("history_status", "ready"))
        if st.session_state.get("history_dirty"):
            output = "現在の会話を保存できませんでした。履歴の再保存後に、新しいチャットを開始してください。"
        else:
            reset_conversation(settings, tools, usage_logger, user_email)
            st.rerun()
            return True
    elif command.name == "/model" and not command.argument:
        output = f"使用中のモデル：`{ctx.settings.gemini_model}`"
    elif command.name == "/thinking" and not command.argument:
        if st.session_state.get("pending") is not None:
            output = "実行確認待ちです。現在のクエリを実行または中止してから変更してください。"
        elif ctx.settings.gemini_model != "gemini-3.8-flash":
            output = "このモードはGemini 3.8 Flashに対応しています。"
        else:
            if ctx.thinking_enabled:
                ctx.thinking_enabled = False
                previous = ctx.thinking_previous_effort
                ctx.thinking_mode = previous if previous in ("low", "medium", "high") else DEFAULT_EFFORT
                output = f"詳細分析モードを解除しました。effortは **{ctx.thinking_mode.title()}** に戻りました。"
            else:
                ctx.thinking_previous_effort = ctx.thinking_mode
                ctx.thinking_enabled = True
                ctx.thinking_mode = "high"
                output = "詳細分析モードを有効にしました。effortは **High** です。多角的な検証と詳しい説明の指示を追加します。"
            if st.session_state.messages:
                save_conversation(ctx)
    elif command.name == "/effort":
        if not command.argument:
            if st.session_state.get("pending") is not None or ctx.settings.gemini_model != "gemini-3.8-flash":
                st.session_state.command_output = "実行確認待ち、または対応外モデルのため変更できません。"
                st.rerun()
                return True
            st.session_state.effort_panel = True
            st.session_state.pop("command_output", None)
            st.rerun()
            return True
        else:
            output = "`/effort` を引数なしで入力し、選択画面からLow、Medium、Highを選んでください。"
    else:
        output = "コマンドまたは引数を確認してください。利用できるコマンドは `/help` で確認できます。"
    st.session_state.command_output = output
    st.rerun()
    return True


def render_chat_page(
    settings: Settings,
    tools: BigQueryTools,
    agent: GeminiAgent,
    usage_logger: UsageLogger,
    user_email: str,
) -> None:
    st.title("📊 ShiruGPT")
    st.caption("SHIRUCAFEデータの分析・課題発見の相談AI")

    leak = credentials_leak_warning(settings)
    if leak:
        st.error(f"⚠️ {leak}")

    ctx = init_state(settings, tools, usage_logger, user_email)

    monthly_used_usd = None
    if user_email and settings.logging_enabled:
        monthly_used_usd = _effective_monthly_usage_usd(ctx, usage_logger, user_email)
    monthly_limit_usd = _limit_for(usage_logger, user_email) if user_email else 0.0
    blocked = monthly_used_usd is not None and monthly_used_usd >= monthly_limit_usd

    render_sidebar(
        settings, ctx, tools, usage_logger, user_email, monthly_used_usd, monthly_limit_usd
    )

    shown: set[str] = set()
    for msg_idx, message in enumerate(st.session_state.messages):
        render_message(message, msg_idx, shown)

    if st.session_state.get("active_turn") is not None:
        process_turn(agent, ctx)
        return

    last_message = st.session_state.messages[-1] if st.session_state.messages else {}
    awaiting_clarification = bool(
        last_message.get("clarification") and not last_message["clarification"].get("answered")
    )
    if awaiting_clarification:
        render_clarification(last_message, len(st.session_state.messages) - 1, agent, ctx,
                             blocked or bool(st.session_state.get("effort_panel")))

    pending = st.session_state.pending
    if pending is not None:
        # 確認待ちのターンは、開始時点で上限未満だったターンの続きなので、
        # 承認/拒否ボタン自体はブロックしない（「回答は完了させる」仕様通り）。
        render_confirmation(pending, agent, ctx)

    if blocked and pending is None:
        st.error(
            "🚫 今月の利用上限に達したため、"
            "今月はこれ以上ご利用いただけません。毎月1日 0:00（JST）にリセットされます。"
        )

    if st.session_state.get("command_output"):
        with st.container(border=True):
            st.markdown(st.session_state.command_output)
            if st.button("閉じる", key="close_command_output"):
                st.session_state.pop("command_output", None)
                st.rerun()

    prompt = render_prompt_input(ctx)
    if not prompt:
        return

    if handle_command(prompt, ctx, settings, tools, usage_logger, user_email,
                      monthly_used_usd, monthly_limit_usd):
        return
    if pending is not None or blocked or awaiting_clarification:
        st.warning("現在はコマンドのみ利用できます。確認待ちの場合は、上のフォームから回答してください。")
        return
    submit_prompt(prompt, agent, ctx)


# --------------------------------------------------------------------
# メイン
# --------------------------------------------------------------------
def main() -> None:
    try:
        settings, tools, agent, usage_logger = build_runtime()
    except ConfigError as exc:
        st.error(str(exc))
        st.stop()
        return
    except Exception as exc:  # noqa: BLE001
        st.error(friendly_error(exc))
        st.markdown(AUTH_HELP)
        st.stop()
        return

    # [auth] が設定されている（= Streamlit Cloud で OIDC ログインを使う）場合のみ、
    # ログインを必須にする。ローカル開発では auth 未設定が通常なのでスキップされ、
    # 従来どおり ADMIN_ALLOW_LOCAL によるバイパスに委ねる。
    if auth_configured() and not getattr(st.user, "is_logged_in", False):
        st.title("📊 ShiruGPT")
        st.info("続けるには Google でログインしてください。")
        if st.button("Google でログイン", type="primary"):
            st.login()
        st.stop()
        return

    user_email = resolve_viewer_email()

    pages = [
        st.Page(
            lambda: render_chat_page(
                settings, tools, agent, usage_logger, user_email
            ),
            title="チャット",
            icon="💬",
            default=True,
        )
    ]
    # 管理者以外にはページ自体を渡さない（ナビゲーションに出さないための措置）。
    # 実際のアクセス制御は render_admin_page の冒頭で毎回行う。
    if settings.is_admin(user_email) or (
        settings.admin_allow_local and not user_email
    ):
        pages.append(
            st.Page(
                lambda: render_admin_page(settings, usage_logger, user_email),
                title="管理",
                icon="🛠️",
                url_path="usage",
            )
        )

    if len(pages) > 1:
        st.navigation(pages).run()
    else:
        # ページが 1 つだけならナビゲーションを出さず、従来どおりの見た目にする
        render_chat_page(settings, tools, agent, usage_logger, user_email)


if __name__ == "__main__":
    main()
