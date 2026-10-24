"""標準のチャット入力を保ち、ブラウザ内だけでコマンド候補を表示する。"""
from pathlib import Path

import streamlit as st
import streamlit.components.v2 as components

from .commands import COMMAND_SUGGESTIONS


_JS = Path(__file__).with_name("command_menu.js").read_text()


def _keyboard_event() -> None:
    payload = st.session_state.get("slash_command_menu", {}).get("action")
    if not isinstance(payload, dict):
        return
    if payload.get("kind") == "command" and payload.get("command") in {item["command"] for item in COMMAND_SUGGESTIONS}:
        st.session_state["keyboard_command"] = payload["command"]
    elif payload.get("kind") == "effort" and payload.get("mode") in ("low", "medium", "high"):
        st.session_state["keyboard_effort"] = payload["mode"]
    elif payload.get("kind") == "close_effort":
        st.session_state.pop("effort_panel", None)
    elif payload.get("kind") == "clarification":
        st.session_state["clarification_keyboard"] = payload


def render_command_menu(*, effort_open=False, effort="medium", running=False) -> None:
    # アプリの実行環境ごとに登録する（AppTestも独立した環境を作る）。
    menu = components.component("slash_command_menu", js=_JS, isolate_styles=False)
    menu(key="slash_command_menu", data={"commands": COMMAND_SUGGESTIONS, "effort_open": effort_open, "effort": effort, "running": running}, height=0, on_action_change=_keyboard_event)
