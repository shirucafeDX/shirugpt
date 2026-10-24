"""チャットコマンドの解析とローカル表示。モデルへの問い合わせは行わない。"""
from __future__ import annotations

from dataclasses import dataclass

from .config import bq_cost_usd, gemini_cost_usd, human_bytes


HELP = """\
| コマンド | 内容 |
| --- | --- |
| `/help` | コマンドの一覧 |
| `/model` | 使用中のモデルを表示 |
| `/new` | 新しいチャットを開始 |
| `/usage` | 月次利用額・残額と、この会話のトークン数・クエリ量 |
| `/thinking` | 思考設定を表示 |
| `/thinking high` | highを明示指定（解除するまで継続） |
| `/thinking medium` | 通常設定へ戻す（既定） |

コマンドはアプリ内で処理し、Geminiへ送信しません。
"""


@dataclass(frozen=True)
class Command:
    name: str
    argument: str = ""


def parse_command(text: str) -> Command | None:
    text = text.strip()
    if not text.startswith("/"):
        return None
    parts = text.split(maxsplit=1)
    name, argument = parts[0], parts[1] if len(parts) > 1 else ""
    return Command(name.lower(), argument.strip())


def usage_text(ctx, monthly_used: float | None, monthly_limit: float | None, reset_at) -> str:
    if monthly_used is None or monthly_limit is None:
        monthly = "月次使用量：取得できません（未ログイン・利用ログ未設定の場合も含みます）。"
    else:
        ratio = monthly_used / monthly_limit * 100 if monthly_limit > 0 else 100.0
        monthly = (
            f"今月の推定利用額：**${monthly_used:,.4f} / ${monthly_limit:,.2f}**（{ratio:.1f}%）  \n"
            f"残額：**${max(0, monthly_limit - monthly_used):,.4f}**  \n"
            f"次回リセット：{reset_at:%Y年%m月%d日 %H:%M}（日本時間）"
        )
    thinking = getattr(ctx, "session_thinking_tokens", None)
    # 古い履歴には思考トークンの累計がないため、最後のターンだけで代用しない。
    thinking_label = f"{thinking:,}" if thinking is not None else "不明（古い履歴）"
    cost = (ctx.gemini_cost_usd() if hasattr(ctx, "gemini_cost_usd") else
            gemini_cost_usd(ctx.session_prompt_tokens, ctx.session_output_tokens)) + bq_cost_usd(ctx.session_billed_bytes)
    return (
        f"{monthly}\n\nこの会話の累計\n\n"
        f"| 項目 | 値 |\n| --- | ---: |\n"
        f"| 入力トークン | {ctx.session_prompt_tokens:,} |\n"
        f"| 出力トークン（思考を含む） | {ctx.session_output_tokens:,} |\n"
        f"| 思考トークン（出力の内数） | {thinking_label} |\n"
        f"| BigQuery課金対象量 | {human_bytes(ctx.session_billed_bytes)} |\n"
        f"| 推定利用額 | ${cost:,.4f} |\n\n"
        "金額は概算です。月次利用額は他の会話も含みます。同時利用分の反映には遅れが生じる場合があります。"
    )
