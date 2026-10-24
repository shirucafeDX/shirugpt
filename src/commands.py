"""チャットコマンドの解析とローカル表示。モデルへの問い合わせは行わない。"""
from __future__ import annotations

from dataclasses import dataclass

from .config import BQ_PRICE_PER_TIB_USD, GEMINI_PRICE_AS_OF, bq_cost_usd, gemini_cost_usd, human_bytes

COMMAND_SUGGESTIONS = [
    {"command": "/help", "description": "コマンドの一覧を表示"},
    {"command": "/model", "description": "使用中のモデルを確認"},
    {"command": "/new", "description": "新しいチャットを開始"},
    {"command": "/usage", "description": "当月のトークン・クエリ量・費用を確認"},
    {"command": "/thinking", "description": "詳細分析モードを切り替え（有効時High）"},
    {"command": "/effort", "description": "Low / Medium / Highを選択"},
]


HELP = """\
| コマンド | 内容 |
| --- | --- |
| `/help` | コマンドの一覧 |
| `/model` | 使用中のモデルを表示 |
| `/new` | 新しいチャットを開始 |
| `/usage` | 当月のInput・Output（思考を含む）、クエリ量、費用と残額 |
| `/thinking` | 詳細分析モードを切り替え。有効時High、解除時は元のeffortへ戻す |
| `/effort` | Low / Medium / Highを選択（1・2・3、上下キーにも対応） |

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
            f"今月の推定利用額：**USD {monthly_used:,.4f} / USD {monthly_limit:,.2f}**（{ratio:.1f}%）  \n"
            f"残額：**USD {max(0, monthly_limit - monthly_used):,.4f}**  \n"
            f"次回リセット：{reset_at:%Y年%m月%d日 %H:%M}（日本時間）"
        )
    cost = (ctx.gemini_cost_usd() if hasattr(ctx, "gemini_cost_usd") else
            gemini_cost_usd(ctx.session_prompt_tokens, ctx.session_output_tokens)) + bq_cost_usd(ctx.session_billed_bytes)
    settings = getattr(ctx, "settings", None)
    model = getattr(settings, "gemini_model", "gemini-3.8-flash")
    location = getattr(settings, "vertex_location", "global")
    input_rate = gemini_cost_usd(1_000_000, 0, model, location)
    output_rate = gemini_cost_usd(0, 1_000_000, model, location)
    bq_cost = bq_cost_usd(ctx.session_billed_bytes)
    ai_cost = cost - bq_cost
    return (
        f"{monthly}\n\nこの会話の累計\n\n"
        f"| 項目 | 値 |\n| --- | ---: |\n"
        f"| 入力トークン | {ctx.session_prompt_tokens:,} |\n"
        f"| 出力トークン（思考を含む） | {ctx.session_output_tokens:,} |\n"
        f"| BigQuery課金対象量 | {human_bytes(ctx.session_billed_bytes)} |\n"
        f"| Gemini推定費用 | USD {ai_cost:,.4f} |\n"
        f"| BigQuery推定費用 | USD {bq_cost:,.4f} |\n"
        f"| 推定利用額 | USD {cost:,.4f} |\n\n"
        f"適用レート（{model} / {location}、{GEMINI_PRICE_AS_OF}時点）：入力 USD {input_rate:.2f} / 100万トークン、"
        f"出力（思考を含む）USD {output_rate:.2f} / 100万トークン、BigQuery USD {BQ_PRICE_PER_TIB_USD:.2f} / TiB。\n\n"
        "金額は期間限定還元・キャッシュ割引・無料枠の適用前の概算です。月次利用額は他の会話も含みます。同時利用分の反映には遅れが生じる場合があります。"
    )


def current_month_usage_text(frame) -> str:
    if frame.empty:
        return "\n\n当月の利用ログはまだありません。"
    row = next(frame.itertuples(index=False))
    bq_usd = bq_cost_usd(row.billed_bytes)
    lines = [
        f"\n\n当月の詳細（{str(row.month)[:7]}・日本時間、金額はUSD）",
        "",
        "| 項目 | 使用量 |",
        "| --- | ---: |",
        f"| Inputトークン | {int(row.input_tokens):,} |",
        f"| Outputトークン（思考を含む） | {int(row.output_tokens):,} |",
        f"| AI呼び出し回数 | {int(row.gemini_calls):,} |",
        f"| クエリ回数 | {int(row.bq_queries):,} |",
        f"| クエリ課金対象量 | {human_bytes(row.billed_bytes)} |",
        f"| Gemini費用 | USD {row.gemini_usd:,.4f} |",
        f"| BigQuery費用 | USD {bq_usd:,.4f} |",
        f"| 合計費用 | USD {row.gemini_usd + bq_usd:,.4f} |",
        "\n出力トークンには思考分を含みます。詳細は取得時点のログに基づくため、上部の今月の利用額と差が出る場合があります。",
    ]
    return "\n".join(lines)
