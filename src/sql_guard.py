"""Gemini が生成した SQL を実行前に検査する安全弁。

方針:
  1. SELECT / WITH で始まる単一ステートメント以外は実行しない
  2. 参照先データセットが allowlist 内かを検証する
  3. 外側に LIMIT が無ければ自動付与し、その事実を呼び出し元に返す

日本語のデータセット名・テーブル名・列名（バッククォート囲み）を壊さないよう、
キーワード検査用のスキャナと、BigQuery 方言の構文木で検査する。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import sqlglot
from sqlglot import exp
from sqlglot.errors import SqlglotError

# 単独ステートメントとして現れたら危険なキーワード。
# 文字列リテラルとバッククォート識別子はマスク済みの文字列に対して探す。
_FORBIDDEN = (
    "INSERT",
    "UPDATE",
    "DELETE",
    "MERGE",
    "DROP",
    "TRUNCATE",
    "ALTER",
    "CREATE",
    "GRANT",
    "REVOKE",
    "UNDROP",
)

_ALLOWED_STARTS = ("SELECT", "WITH")

_AGGREGATE_RE = re.compile(
    r"(?is)\b(?:GROUP\s+BY|COUNT\s*\(|SUM\s*\(|AVG\s*\(|MIN\s*\(|MAX\s*\("
    r"|ARRAY_AGG\s*\(|STRING_AGG\s*\(|APPROX_COUNT_DISTINCT\s*\()"
)


class SQLGuardError(ValueError):
    """SQL が安全要件を満たさないときに送出する。"""


@dataclass
class GuardResult:
    sql: str
    original_sql: str
    limit_injected: bool = False
    looks_aggregate: bool = False
    referenced_tables: tuple[str, ...] = ()
    notes: list[str] = field(default_factory=list)


def _scan(sql: str, *, mask_backticks: bool) -> str:
    """コメントを除去し、文字列リテラルの中身を空白に潰した文字列を返す。

    長さと位置を保つため、除去した文字は同じ長さの空白に置き換える。
    mask_backticks=True のときはバッククォート識別子の中身も潰す
    （キーワード検査時に識別子名を誤検出しないため）。
    """
    out: list[str] = []
    i = 0
    n = len(sql)
    while i < n:
        ch = sql[i]

        # 行コメント
        if ch == "#" or (ch == "-" and sql.startswith("--", i)):
            while i < n and sql[i] != "\n":
                out.append(" ")
                i += 1
            continue

        # ブロックコメント
        if sql.startswith("/*", i):
            end = sql.find("*/", i + 2)
            end = n if end == -1 else end + 2
            out.append(" " * (end - i))
            i = end
            continue

        # バッククォート識別子
        if ch == "`":
            end = sql.find("`", i + 1)
            end = n - 1 if end == -1 else end
            if mask_backticks:
                out.append(" " * (end - i + 1))
            else:
                out.append(sql[i : end + 1])
            i = end + 1
            continue

        # 文字列リテラル（三重引用符を先に判定）
        if ch in "'\"":
            quote = sql[i : i + 3] if sql.startswith(ch * 3, i) else ch
            j = i + len(quote)
            while j < n:
                if sql[j] == "\\":
                    j += 2
                    continue
                if sql.startswith(quote, j):
                    j += len(quote)
                    break
                j += 1
            else:
                j = n
            out.append(" " * (j - i))
            i = j
            continue

        out.append(ch)
        i += 1

    return "".join(out)


def _strip_trailing(sql: str) -> str:
    return sql.strip().rstrip(";").rstrip()


def _split_path(raw: str) -> list[str]:
    """`proj.dataset.table` / proj.`dataset`.table などを部品に分解する。"""
    cleaned = raw.replace("`", "")
    cleaned = re.sub(r"\s*\.\s*", ".", cleaned).strip()
    return [part for part in cleaned.split(".") if part]


def check_sql(
    sql: str,
    *,
    project: str,
    allowed_datasets: tuple[str, ...] | list[str],
    default_row_limit: int,
) -> GuardResult:
    """SQL を検査し、実行してよい形に整えて返す。問題があれば SQLGuardError。"""
    if not sql or not sql.strip():
        raise SQLGuardError("SQL が空です。")

    original = sql.strip()
    body = _strip_trailing(original)

    # キーワード判定用（文字列・識別子ともにマスク）
    kw_masked = _scan(body, mask_backticks=True)

    if not kw_masked.strip():
        raise SQLGuardError("SQL にコメント以外の内容がありません。")

    # --- 複数ステートメントの拒否 ---------------------------------
    segments = [seg for seg in kw_masked.split(";") if seg.strip()]
    if len(segments) > 1:
        raise SQLGuardError(
            "複数のSQLステートメントは実行できません。SELECT 文を1つだけ指定してください。"
        )

    # --- 先頭キーワードの検査 -------------------------------------
    lead = kw_masked.lstrip().lstrip("(").lstrip()
    first_token = re.match(r"[A-Za-z_]+", lead)
    if not first_token or first_token.group(0).upper() not in _ALLOWED_STARTS:
        found = first_token.group(0).upper() if first_token else lead[:20]
        raise SQLGuardError(
            f"読み取り専用の SELECT クエリのみ実行できます（検出: {found}）。"
        )

    # --- 危険キーワードの検査（多重防御） -------------------------
    for keyword in _FORBIDDEN:
        if re.search(rf"\b{keyword}\b", kw_masked, re.IGNORECASE):
            raise SQLGuardError(
                f"データを変更する可能性のあるキーワード「{keyword}」が含まれているため実行を中止しました。"
            )

    # --- 参照テーブルの allowlist 検証 ----------------------------
    allowed = tuple(allowed_datasets)
    try:
        statements = [node for node in sqlglot.parse(body, read="bigquery") if node is not None]
    except SqlglotError as exc:
        raise SQLGuardError(f"SQL を解析できません。構文を確認してください: {exc}") from exc
    if len(statements) != 1:
        raise SQLGuardError("SELECT 文を1つだけ指定してください。")
    tree = statements[0]
    if not isinstance(tree, (exp.Select, exp.SetOperation, exp.Subquery)) or tree.find(exp.Into):
        raise SQLGuardError("読み取り専用の SELECT クエリのみ実行できます。")
    referenced: list[str] = []

    def validate(parts: list[str], raw: str) -> None:
        if len(parts) >= 3:
            ref_project, dataset, table = parts[0], parts[1], ".".join(parts[2:])
        elif len(parts) == 2:
            ref_project, dataset, table = project, parts[0], parts[1]
        else:
            return  # 単一トークンは CTE / エイリアス等。BigQuery 側で解決される

        if ref_project != project:
            raise SQLGuardError(
                f"許可されていないプロジェクト「{ref_project}」を参照しています"
                f"（許可: {project}）。"
            )
        if dataset not in allowed:
            raise SQLGuardError(
                f"許可されていないデータセット「{dataset}」を参照しています"
                f"（許可: {', '.join(allowed)}）。"
            )
        full = f"{ref_project}.{dataset}.{table}"
        if full not in referenced:
            referenced.append(full)

    # 列・別名・コメント・文字列は Table ノードに含まれない。
    # CTE の定義内、カンマ区切りの結合、UNION 内もすべて検査する。
    for table in tree.find_all(exp.Table):
        if not isinstance(table.this, exp.Identifier):
            raise SQLGuardError("テーブル関数の参照先を検証できないため実行できません。")
        parts = [part.name for part in table.parts]
        validate(parts, table.sql(dialect="bigquery"))

    # 修飾された UDF も allowlist の外へ出さない。SAFE.* は組み込み修飾子。
    for function in tree.find_all(exp.Anonymous):
        parts = _split_path(function.name)
        parent = function.parent
        if isinstance(parent, exp.Dot) and parent.expression is function:
            parts = _split_path(parent.this.sql(dialect="bigquery")) + parts
        if parts and parts[0].upper() == "SAFE":
            parts = parts[1:]
        if len(parts) >= 2:
            validate(parts, function.sql(dialect="bigquery"))

    # --- LIMIT の自動付与 -----------------------------------------
    notes: list[str] = []
    looks_aggregate = bool(_AGGREGATE_RE.search(kw_masked))
    limit_injected = False
    final_sql = body

    if tree.args.get("limit") is None:
        final_sql = f"{body}\nLIMIT {default_row_limit}"
        limit_injected = True
        if looks_aggregate:
            notes.append(
                f"外側に LIMIT が無かったため LIMIT {default_row_limit} を自動付与しました。"
                "集計クエリのため、グループ数がこれを超える場合は結果が切り捨てられます。"
                "全件が必要なら、より絞り込んだ集計にするか LIMIT を明示してください。"
            )
        else:
            notes.append(
                f"外側に LIMIT が無かったため LIMIT {default_row_limit} を自動付与しました。"
            )

    return GuardResult(
        sql=final_sql,
        original_sql=original,
        limit_injected=limit_injected,
        looks_aggregate=looks_aggregate,
        referenced_tables=tuple(referenced),
        notes=notes,
    )
