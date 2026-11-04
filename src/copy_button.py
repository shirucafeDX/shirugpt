"""回答をブラウザーのクリップボードへコピーするアイコン。"""
from pathlib import Path

import streamlit.components.v2 as components


_JS = Path(__file__).with_name('copy_button.js').read_text()


def render_copy_button(text: str, *, key: str) -> None:
    component = components.component('answer_copy_button', js=_JS)
    component(data={'text': text}, key=key, height=40)
