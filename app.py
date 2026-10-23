"""ShiruGPT — BigQuery を自然言語で分析する Streamlit アプリ（プロトタイプ）。

起動:
    streamlit run app.py
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import uuid
from zoneinfo import ZoneInfo
from typing import Any

import streamlit as st
from google.genai import types

from src.admin_page import render_admin_page
from src.agent import (
    AnswerResult,
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
from src.charts import KIND_LABELS, ChartSpec, suggest_chart
from src.presentation import choose_presentation
from src.config import (
    DEFAULT_MONTHLY_LIMIT_USD,
    ConfigError,
    Settings,
    bq_cost_usd,
    credentials_leak_warning,
    gemini_cost_usd,
    human_bytes,
    load_settings,
)
from src.usage_log import UsageLogger, jst_month_bounds_utc
from src.chat_history import HistoryStore, conversation_title, pack, unpack
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
    "GEMINI_THINKING_BUDGET",
    "DRY_RUN_CONFIRM_BYTES",
    "MAX_BYTES_BILLED",
    "DEFAULT_ROW_LIMIT",
    "MAX_RESULT_ROWS",
    "SAMPLE_ROWS",
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
    return st.session_state.ctx


def reset_conversation(
    settings: Settings,
    tools: BigQueryTools,
    usage_logger: UsageLogger,
    user_email: str,
) -> None:
    if st.session_state.get("conversation_owner") != user_email:
        st.session_state.pop("history_entries", None)
        st.session_state.history_list_error = ""
    # 会話を切り替えたとき、以前の図表ウィジェットの選択を持ち越さない。
    for key in list(st.session_state):
        if key.startswith(("chart_", "dl_")):
            del st.session_state[key]
    st.session_state.messages = []
    st.session_state.contents = []
    st.session_state.pending = None
    st.session_state.ctx = _new_context(settings, tools, usage_logger, user_email)
    st.session_state.conversation_owner = user_email
    st.session_state.history_id = uuid.uuid4().hex
    st.session_state.history_generation = 0
    st.session_state.history_dirty = False
    st.session_state.history_error = ""
    st.session_state.history_status = "ready"


@st.cache_resource(show_spinner=False)
def history_store(bucket: str, project: str) -> HistoryStore:
    return HistoryStore(bucket, project)


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
    if st.session_state.history_error:
        st.warning(st.session_state.history_error)
        if st.button("再保存"):
            save_conversation(ctx, st.session_state.history_status)
            st.rerun()
        if st.button("別の会話として保存"):
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
    by_id = {e["id"]: e for e in entries}
    dirty = st.session_state.history_dirty
    for entry in entries:
        selected = entry["id"]
        title = conversation_title([{"role": "user", "text": entry["title"]}])
        clicked = st.button(
            title, key=f"open_history_{selected}", width="stretch", disabled=dirty,
            type="primary" if selected == st.session_state.history_id else "secondary",
            help=entry["updated"].astimezone(ZoneInfo("Asia/Tokyo")).strftime("%Y/%m/%d %H:%M"),
        )
        if clicked:
            open_conversation(settings, tools, usage_logger, user_email, store, selected)
    selected = st.session_state.history_id
    if selected not in by_id:
        return
    confirm_delete = st.checkbox("この会話を削除する", key=f"delete_history_{selected}")
    if st.button("履歴から削除", disabled=not confirm_delete or dirty):
        try:
            store.delete(user_email, selected, by_id[selected]["generation"])
            st.session_state.history_entries = [e for e in entries if e["id"] != selected]
            reset_conversation(settings, tools, usage_logger, user_email)
            st.rerun()
        except Exception:
            logger.warning("履歴の削除に失敗しました。", exc_info=True)
            st.error("削除できませんでした。アプリを開き直して再試行してください。")


def open_conversation(settings, tools, usage_logger, user_email, store, selected) -> None:
    try:
        data, generation = store.load(user_email, selected)
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

    delta = gemini_cost_usd(
        ctx.session_prompt_tokens - ctx.limit_baseline_session_prompt_tokens,
        ctx.session_output_tokens - ctx.limit_baseline_session_output_tokens,
    ) + bq_cost_usd(ctx.session_billed_bytes - ctx.limit_baseline_session_billed_bytes)
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
    kinds = list(KIND_LABELS)
    col1, col2, col3 = st.columns([2, 3, 4])
    kind = col1.selectbox(
        "種類",
        options=kinds,
        index=kinds.index(spec.kind),
        format_func=lambda k: KIND_LABELS[k],
        key=f"{key_prefix}_kind",
    )
    x = col2.selectbox(
        "X 軸",
        options=spec.candidates_x,
        index=spec.candidates_x.index(spec.x) if spec.x in spec.candidates_x else 0,
        key=f"{key_prefix}_x",
    )
    y_options = [c for c in spec.candidates_y if c != x]
    if not y_options:
        st.info("Y 軸に使える数値列がありません。X 軸を変更してください。")
        return
    y = col3.multiselect(
        "Y 軸（数値列）",
        options=y_options,
        default=[c for c in spec.y if c in y_options] or y_options[:1],
        key=f"{key_prefix}_y",
    )
    if not y:
        st.info("Y 軸の列を 1 つ以上選んでください。")
        return

    kwargs = {"x": x, "y": y}
    if kind == "bar":
        # SQL 側の ORDER BY を尊重する（既定では X 軸で並べ替えられてしまう）
        kwargs["sort"] = False
    try:
        CHART_FUNCS[kind](df, **kwargs)
    except Exception:  # noqa: BLE001 - 描画不能な組み合わせは表にフォールバック
        st.warning(
            "この列の組み合わせではグラフを描画できませんでした。"
            "軸の指定を変えるか「データを見る」を開いてください。"
        )
        return
    if spec.reason:
        st.caption(spec.reason)


def render_results(message: dict[str, Any], msg_idx: int, shown: set[str]) -> None:
    runs = message.get("runs") or []
    if not runs:
        return
    plan = choose_presentation(message.get("question", ""), runs, shown)
    if message.get("is_error"):
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
    hidden = [(i, run) for i, run in enumerate(runs)
              if not (plan.mode == "table" and i == plan.index)]
    if hidden:
        with st.expander("データを見る"):
            for idx, run in hidden:
                st.caption(run.purpose or f"結果 {idx + 1}")
                if run.dataframe.empty:
                    st.info("このクエリの結果は 0 件でした。")
                else:
                    render_table(run, idx, msg_idx)
    with st.expander("実行詳細"):
        for idx, run in enumerate(runs):
            st.caption(run.purpose or f"クエリ {idx + 1}")
            st.code(run.sql, language="sql")
            st.caption(f"実行クエリ量: {human_bytes(run.billed_bytes)} · 結果件数: {len(run.dataframe):,} 件")
            for note in run.notes:
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
        if user_email:
            st.write(f"ログイン中: {user_email}")
            if st.button("ログアウト", width="stretch"):
                st.logout()

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
        st.subheader("セッション使用量")
        st.markdown(
            f"""
- モデル: `{settings.gemini_model}`
- 入力トークン合計: **{ctx.session_prompt_tokens:,}**
- 出力トークン合計: **{ctx.session_output_tokens:,}**
- 実行クエリ量合計: **{human_bytes(ctx.session_billed_bytes)}**
"""
        )
        st.caption("この会話の累計です。履歴から再開した場合は以前の利用量を含みます。")
        
        st.divider()
        if st.button("新しいチャット", width="stretch", disabled=st.session_state.history_dirty):
            reset_conversation(settings, tools, usage_logger, user_email)
            st.rerun()

        render_history(settings, tools, usage_logger, user_email, ctx)

        st.link_button(
            "📝 質問・報告フォーム", FEEDBACK_FORM_URL, width="stretch"
        )


# --------------------------------------------------------------------
# ターン処理
# --------------------------------------------------------------------
def process_turn(agent: GeminiAgent, ctx: ToolContext) -> None:
    question = next((m["text"] for m in reversed(st.session_state.messages) if m["role"] == "user"), "")
    try:
        result = agent.run(st.session_state.contents, ctx)
        if isinstance(result, ConfirmationPending):
            st.session_state.pending = result
        elif isinstance(result, AnswerResult):
            st.session_state.pending = None
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "text": result.text,
                    "question": question,
                    "runs": list(ctx.turn_runs),
                    "errors": list(ctx.turn_errors),
                }
            )
        else:
            # 通常発生しない想定外の戻り値（実行環境側の一時的な異常等）。
            # 他の例外と同じ経路で扱い、アプリ全体を落とさない。
            raise AssertionError(f"想定外の戻り値: {result!r}")
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
                "runs": list(ctx.turn_runs),
                "errors": list(ctx.turn_errors),
                "is_error": True,
            }
        )
    save_conversation(ctx)
    st.rerun()


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
        with st.spinner("処理中…"):
            process_turn(agent, ctx)


# --------------------------------------------------------------------
# ページ
# --------------------------------------------------------------------
def render_chat_page(
    settings: Settings,
    tools: BigQueryTools,
    agent: GeminiAgent,
    usage_logger: UsageLogger,
    user_email: str,
) -> None:
    st.title("📊 ShiruGPT")
    st.caption("SHIRUCAFEデータの可視化・分析AI")

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

    prompt = st.chat_input(
        "質問してください（例: 〇〇店の参加率の推移をグラフで見せて）",
        disabled=pending is not None or blocked,
    )
    if not prompt:
        if not st.session_state.messages and pending is None and not blocked:
            st.info(
                "プロトタイプ版です。AIは必ずしも正しい回答を返すとは限りません。"
            )
        return

    st.session_state.messages.append({"role": "user", "text": prompt})
    st.session_state.contents.append(
        types.Content(role="user", parts=[types.Part.from_text(text=prompt)])
    )
    ctx.start_turn()
    usage_logger.log_prompt(
        user_email=user_email,
        session_id=ctx.session_id,
        turn_id=ctx.turn_id,
        prompt_text=prompt,
    )
    save_conversation(ctx, "running")

    with st.chat_message("user"):
        st.markdown(prompt)
    with st.spinner("データベースを調べています…"):
        process_turn(agent, ctx)


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
