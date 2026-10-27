"""ツール結果はオブジェクトで送信し、空コレクションだけ明示的な印にする。"""
MARKER = '__shirugpt_value__'


def encode(value):
    if value is None:
        return {MARKER: 'null'}
    if isinstance(value, dict):
        if set(value) == {MARKER}:
            raise ValueError('ツール結果が予約済みの空コレクション表現と衝突しています。')
        return {key: encode(item) for key, item in value.items()} if value else {MARKER: 'empty_object'}
    if isinstance(value, list):
        return [encode(item) for item in value] if value else {MARKER: 'empty_list'}
    return value


def decode(value):
    if isinstance(value, dict):
        if value == {MARKER: 'null'}:
            return None
        if value == {MARKER: 'empty_list'}:
            return []
        if value == {MARKER: 'empty_object'}:
            return {}
        return {key: decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode(item) for item in value]
    return value
