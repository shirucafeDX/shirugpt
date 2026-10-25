"""本人の会話を GCS に保存する。Gemini のツールからはアクセスさせない。"""
from __future__ import annotations

import base64
import hashlib
import io
import json
import re
from dataclasses import asdict, fields
from datetime import date, datetime, time
from decimal import Decimal
from zipfile import ZipFile, ZIP_DEFLATED

import pandas as pd
from google.genai import types

from .agent import ConfirmationPending, ToolContext
from .bq_tools import QueryRun
from .tool_catalog import restored_selection

# 実行時オブジェクト、認証情報、月次利用量のベースラインは保存しない。
CONTEXT_FIELDS = (
    'active_tool_names', 'analysis_cache', 'context_state', 'schema_cache', 'listing_cache', 'query_cache', 'decisions', 'resume_calls',
    'turn_runs', 'turn_analyses', 'turn_errors', 'turn_prompt_tokens', 'turn_output_tokens',
    'turn_thinking_tokens', 'turn_total_tokens', 'session_prompt_tokens',
    'session_output_tokens', 'session_total_tokens', 'session_billed_bytes',
    'session_thinking_tokens',
    'thinking_mode', 'active_thinking_level', 'session_gemini_cost_usd',
    'thinking_enabled', 'thinking_previous_effort',
    'session_id', 'turn_id',
)


def conversation_title(messages):
    """最初の質問から、一覧に収まる一行のタイトルを作る。追加の AI 呼出は不要。"""
    first = next((m['text'] for m in messages if m['role'] == 'user'), '')
    title = ' '.join(first.split())
    return title[:39] + '…' if len(title) > 40 else title or '新しいチャット'


def policy(settings):
    return [settings.gcp_project, sorted(settings.allowed_datasets), settings.gemini_model]


def pack(messages, contents, ctx, pending, status='ready') -> bytes:
    """JSON + Parquet。pickle は使わず、型と Gemini の署名バイトを保つ。"""
    output = io.BytesIO()
    with ZipFile(output, 'w', ZIP_DEFLATED) as archive:
        frames = {}

        def encode(value):
            if isinstance(value, pd.DataFrame):
                key = id(value)
                if key not in frames:
                    name = f'frames/{len(frames)}.parquet'
                    frames[key] = name
                    archive.writestr(name, value.to_parquet(index=True))
                return {'$type': 'frame', 'value': frames[key]}
            if isinstance(value, QueryRun):
                return {'$type': 'run', 'value': {f.name: encode(getattr(value, f.name)) for f in fields(value)}}
            if isinstance(value, (types.Content, types.FunctionCall)):
                return {'$type': type(value).__name__, 'value': encode(value.model_dump(mode='python'))}
            if isinstance(value, bytes):
                return {'$type': 'bytes', 'value': base64.b64encode(value).decode('ascii')}
            if isinstance(value, (datetime, date, time, Decimal)):
                return {'$type': type(value).__name__, 'value': str(value)}
            if isinstance(value, dict):
                return {k: encode(v) for k, v in value.items()}
            if isinstance(value, (list, tuple)):
                return [encode(v) for v in value]
            if hasattr(value, 'item'):
                return encode(value.item())
            return value

        payload = {
            'version': 1, 'policy': policy(ctx.settings), 'status': status,
            'messages': encode(messages), 'contents': encode(contents),
            'context': encode({key: getattr(ctx, key) for key in CONTEXT_FIELDS}),
            'pending': asdict(pending) if pending else None,
        }
        archive.writestr('conversation.json', json.dumps(payload, ensure_ascii=False))
    return output.getvalue()


def unpack(data, ctx):
    with ZipFile(io.BytesIO(data)) as archive:
        payload = json.loads(archive.read('conversation.json'))
        if payload['version'] != 1:
            raise ValueError('この履歴の保存形式には対応していません。')
        if payload['policy'][:2] != policy(ctx.settings)[:2]:
            raise ValueError('データのアクセス設定が変更されています。この履歴は再開できません。')
        frames = {}

        def decode(value):
            if isinstance(value, list):
                return [decode(v) for v in value]
            if not isinstance(value, dict):
                return value
            kind = value.get('$type')
            raw = value.get('value')
            if kind == 'frame':
                if raw not in frames:
                    frames[raw] = pd.read_parquet(io.BytesIO(archive.read(raw)))
                return frames[raw]
            if kind == 'run':
                return QueryRun(**decode(raw))
            if kind in ('Content', 'FunctionCall'):
                return getattr(types, kind).model_validate(decode(raw))
            if kind == 'bytes':
                return base64.b64decode(raw)
            if kind in ('datetime', 'date', 'time'):
                return {'datetime': datetime, 'date': date, 'time': time}[kind].fromisoformat(raw)
            if kind == 'Decimal':
                return Decimal(raw)
            return {k: decode(v) for k, v in value.items()}

        context = decode(payload['context'])
        for key in CONTEXT_FIELDS:
            if key in context:
                setattr(ctx, key, context[key])
        ctx.context_state = context.get('context_state', {})
        ctx.active_tool_names = restored_selection(context.get('active_tool_names'))
        ctx.thinking_enabled = context.get('thinking_enabled') is True
        previous = context.get('thinking_previous_effort', 'medium')
        ctx.thinking_previous_effort = previous if previous in ('low', 'medium', 'high') else 'medium'
        if 'session_thinking_tokens' not in context:
            ctx.session_thinking_tokens = None
        messages, contents = decode(payload['messages']), decode(payload['contents'])
        pending = ConfirmationPending(**payload['pending']) if payload['pending'] else None
        if payload['policy'][2] != ctx.settings.gemini_model:
            # 別モデルの署名や未完のFunction Callingを送らず、質問と回答を引き継ぐ。
            contents = [types.Content(role='user' if m['role'] == 'user' else 'model',
                                      parts=[types.Part.from_text(text=m['text'])])
                        for m in messages if m.get('text')]
            if pending is not None:
                messages.append({'role': 'assistant', 'text': 'モデルが変更されたため、以前の実行確認は解除しました。必要に応じて、もう一度質問してください。'})
            pending = None
            ctx.resume_calls = None
            ctx.context_state = {}
            ctx.active_tool_names = []
        if ctx.thinking_mode not in ('low', 'medium', 'high'):
            ctx.thinking_mode = 'medium'
        if ctx.active_thinking_level not in ('low', 'medium', 'high'):
            ctx.active_thinking_level = 'medium'
        if payload['status'] == 'running':
            text = '前回の処理は中断されました。必要に応じて、もう一度質問してください。'
            messages.append({'role': 'assistant', 'text': text, 'is_error': True})
            contents.append(types.Content(role='model', parts=[types.Part.from_text(text=text)]))
        return messages, contents, pending


class HistoryStore:
    def __init__(self, bucket_name, project=None, client=None):
        self.bucket_name, self.project, self._client = bucket_name, project, client

    @property
    def client(self):
        if self._client is None:
            from google.cloud import storage
            self._client = storage.Client(project=self.project)
        return self._client

    def _prefix(self, email):
        if not email or not email.strip():
            raise ValueError('履歴の利用にはログインが必要です。')
        owner = hashlib.sha256(email.strip().lower().encode()).hexdigest()
        return f'conversations/{owner}/'

    def _blob(self, email, conversation_id):
        if not re.fullmatch(r'[0-9a-f]{32}', conversation_id):
            raise ValueError('会話IDが不正です。')
        return self.client.bucket(self.bucket_name).blob(self._prefix(email) + conversation_id + '.zip')

    def list(self, email):
        result = []
        for blob in self.client.list_blobs(self.bucket_name, prefix=self._prefix(email), timeout=15):
            if not blob.name.endswith('.zip'):
                continue
            result.append({'id': blob.name.rsplit('/', 1)[-1][:-4],
                           'title': (blob.metadata or {}).get('title', '会話'),
                           'updated': blob.updated, 'generation': int(blob.generation)})
        return sorted(result, key=lambda x: x['updated'], reverse=True)

    def save(self, email, conversation_id, data, title, generation=0):
        blob = self._blob(email, conversation_id)
        blob.metadata = {'title': title[:80]}
        blob.upload_from_string(data, content_type='application/zip',
                                if_generation_match=generation, timeout=20)
        return int(blob.generation)

    def load(self, email, conversation_id):
        blob = self._blob(email, conversation_id)
        blob.reload(timeout=15)
        generation = int(blob.generation)
        return blob.download_as_bytes(if_generation_match=generation, timeout=20), generation

    def delete(self, email, conversation_id, generation):
        self._blob(email, conversation_id).delete(if_generation_match=generation, timeout=15)
