"""履歴の先読み、保存・削除・利用者の切り替えを確認する。"""
import unittest
from unittest.mock import Mock

from src.history_cache import HistoryCache


class HistoryCacheTests(unittest.TestCase):
    def test_prefetched_and_opened_items_are_reused(self):
        store = Mock()
        store.load.side_effect = lambda email, selected: (selected.encode(), 1)
        cache = HistoryCache(('test', 'bucket', 'me@example.com'))
        entries = [{'id': str(i), 'generation': 1} for i in range(10)]
        cache.prefetch(store, 'me@example.com', entries[:5])
        for entry in entries[:5]:
            self.assertEqual(cache.load(store, 'me@example.com', entry['id'], 1), (entry['id'].encode(), 1))
        self.assertEqual(store.load.call_count, 5)
        cache.prefetch(store, 'me@example.com', entries[:5])
        cache.load(store, 'me@example.com', '9', 1)
        cache.load(store, 'me@example.com', '9', 1)
        self.assertEqual(store.load.call_count, 6)

    def test_save_updates_payload_and_new_generations_are_reloaded(self):
        store = Mock()
        store.load.return_value = (b'newer', 3)
        cache = HistoryCache(('test', 'bucket', 'me'))
        cache.remember('one', b'first', 1)
        cache.remember('one', b'saved', 2)
        self.assertEqual(cache.load(store, 'me', 'one', 2), (b'saved', 2))
        store.load.assert_not_called()
        self.assertEqual(cache.load(store, 'me', 'one', 3), (b'newer', 3))
        cache.discard('one')
        self.assertFalse(cache.payloads)

    def test_failed_prefetch_can_retry_on_next_open(self):
        store = Mock()
        store.load.side_effect = [RuntimeError('offline'), (b'retry', 1)]
        cache = HistoryCache(('test', 'bucket', 'me'))
        cache.prefetch(store, 'me', [{'id': 'one', 'generation': 1}])
        with self.assertRaises(RuntimeError):
            cache.load(store, 'me', 'one', 1)
        self.assertEqual(cache.load(store, 'me', 'one', 1), (b'retry', 1))
