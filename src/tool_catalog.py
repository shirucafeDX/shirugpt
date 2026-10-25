"""短い目録からツールを選び、選択した詳細定義だけをAPIへ渡す。"""
from __future__ import annotations

from google.genai import types

SELECT_TOOL = "select_tools"
MAX_ACTIVE_TOOLS = 3
TOOL_SUMMARIES = {
    "list_datasets": "利用できるデータセットを確認する。",
    "list_tables": "データセット内のテーブル名を確認する。",
    "get_table_schema": "1テーブルの列・型・説明を確認する。",
    "run_query": "読み取り専用SQLで必要範囲を取得・集計。",
    "ask_clarification": "ユーザーに選択式の確認質問をする。",
    "recall_context": "この会話の保存済み結果・過去の詳細を参照する。",
    "finish_answer": "最終回答と要点・累積要約を返して終了する。",
}

CATALOG_INSTRUCTION = """

## ツールの選択と詳細定義
最初は以下の短い目録とselect_toolsだけを渡す。実行したいツールが現在の定義にない場合、
select_toolsで名前を選ぶ。次の生成で、選んだツールの詳しい説明・引数定義を利用できる。
一度に最大3つ選べる。関連するツールをまとめて選び、同じツールを繰り返す間は再選択しない。
選択を変えると、以前の選択は置き換わる。使用中のツール定義を見て必要なときだけ切り替える。
select_toolsは単独で呼ぶ。定義の準備だけを行い、データの取得やSQL実行はしない。
select_toolsには要点メモを付けず、実際のツール呼び出しにcontext_noteを添える。
""" + "\n".join(f"- {name}: {description}" for name, description in TOOL_SUMMARIES.items())


def validate_selection(value: object) -> list[str]:
    if not isinstance(value, list) or not 1 <= len(value) <= MAX_ACTIVE_TOOLS:
        raise ValueError("namesには使用するツール名を1〜3個の配列で指定してください。")
    if any(not isinstance(name, str) or name not in TOOL_SUMMARIES for name in value):
        raise ValueError("namesにはツール目録にある名前だけを指定してください。")
    if len(set(value)) != len(value):
        raise ValueError("namesのツール名は重複させないでください。")
    return list(value)


def restored_selection(value: object) -> list[str]:
    """旧履歴・不正な保存状態は、詳細定義なしから安全に再選択できる。"""
    try:
        return validate_selection(value)
    except ValueError:
        return []


def selector_declaration() -> types.FunctionDeclaration:
    return types.FunctionDeclaration(
        name=SELECT_TOOL,
        description="使うツールを選び、次の生成でその詳細な引数定義を利用可能にする。単独で呼ぶ。",
        parameters=types.Schema(type=types.Type.OBJECT, properties={
            "names": types.Schema(type=types.Type.ARRAY,
                                  items=types.Schema(type=types.Type.STRING, enum=list(TOOL_SUMMARIES)),
                                  min_items=1, max_items=MAX_ACTIVE_TOOLS),
        }, required=["names"]),
    )


def active_declarations(declarations: list[types.FunctionDeclaration], names: object = None) -> list[types.FunctionDeclaration]:
    selected = restored_selection(names)
    by_name = {declaration.name: declaration for declaration in declarations}
    # 詳細はtoolsに一度だけ載せる。選択ツールの応答に同じスキーマを再掲しない。
    return [selector_declaration(), *(by_name[name] for name in selected)]
