"""UI を応答可能に保ったままターンを実行し、停止要求を受け付ける。"""
from __future__ import annotations

import logging
from threading import Event, Lock, Thread
from typing import Any

logger = logging.getLogger(__name__)


class ExecutionStopped(Exception):
    """ユーザーによる停止。通常のエラーとは分ける。"""


class ExecutionControl:
    def __init__(self):
        self.stopped = Event()
        self._lock = Lock()
        self._query_job = None
        self._response_stream = None

    def check(self):
        if self.stopped.is_set():
            raise ExecutionStopped()

    def request_stop(self):
        self.stopped.set()
        with self._lock:
            job = self._query_job
            stream = self._response_stream
        if job is not None:
            Thread(target=self._cancel_job, args=(job,), daemon=True).start()
        if stream is not None:
            Thread(target=self._close_stream, args=(stream,), daemon=True).start()

    @staticmethod
    def _cancel_job(job):
        try:
            job.cancel(timeout=10)
        except Exception:
            logger.warning("BigQuery の停止要求に失敗しました。", exc_info=True)

    @staticmethod
    def _close_stream(stream):
        try:
            stream.close()
        except Exception:
            logger.warning("AI応答の受信停止に失敗しました。", exc_info=True)

    def set_response_stream(self, stream):
        with self._lock:
            self._response_stream = stream
        if self.stopped.is_set():
            self.request_stop()

    def clear_response_stream(self, stream):
        with self._lock:
            if self._response_stream is stream:
                self._response_stream = None

    def set_query_job(self, job):
        with self._lock:
            self._query_job = job
        if self.stopped.is_set():
            self.request_stop()

    def clear_query_job(self):
        with self._lock:
            self._query_job = None


class BackgroundTurn:
    def __init__(self, agent, contents, ctx):
        self.control = ExecutionControl()
        self.done = Event()
        self.message = "質問に合う調べ方を考えています…"
        self.result: Any = None
        self.error: Exception | None = None
        self._thread = Thread(target=self._run, args=(agent, contents, ctx), daemon=True)

    def start(self):
        self._thread.start()

    def progress(self, message):
        self.control.check()
        self.message = message

    def _run(self, agent, contents, ctx):
        original_tools = getattr(ctx, "tools", None)
        try:
            from .bq_tools import BigQueryTools

            ctx.execution_control = self.control
            # キャッシュ済みの共用クライアントに停止状態を持たせず、この実行に限定する。
            if isinstance(original_tools, BigQueryTools):
                ctx.tools = BigQueryTools(
                    ctx.settings, client=original_tools.client, execution_control=self.control,
                )
            self.result = agent.run(contents, ctx, on_progress=self.progress)
        except Exception as exc:
            self.error = exc
        finally:
            ctx.tools = original_tools
            ctx.execution_control = None
            self.done.set()
