"""利用者のセッション内に履歴ZIPを保持する。復元時は毎回別の会話状態を作る。"""
from concurrent.futures import ThreadPoolExecutor


_LOAD_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="history-prefetch")


class HistoryCache:
    def __init__(self, owner):
        self.owner = owner
        self.payloads = {}
        self.pending = {}

    def prefetch(self, store, email, entries):
        for entry in entries:
            key = (entry["id"], entry["generation"])
            if key not in self.payloads and key not in self.pending:
                self.pending[key] = _LOAD_POOL.submit(store.load, email, entry["id"])

    def remember(self, selected, data, generation):
        self.discard(selected)
        self.payloads[(selected, generation)] = (data, generation)

    def load(self, store, email, selected, generation=None):
        key = (selected, generation)
        if key in self.payloads:
            return self.payloads[key]
        # 先読みの失敗を保持せず、次のクリックで再試行できるように取り除く。
        future = self.pending.pop(key, None)
        data, actual = future.result() if future is not None else store.load(email, selected)
        self.remember(selected, data, actual)
        return data, actual

    def discard(self, selected):
        for key in list(self.payloads):
            if key[0] == selected:
                del self.payloads[key]
        for key in list(self.pending):
            if key[0] == selected:
                self.pending.pop(key).cancel()
