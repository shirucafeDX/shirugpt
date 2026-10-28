"""Managed data analysis: one stateful chat call, with all returned events retained."""
from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd
from google.api_core.exceptions import AlreadyExists, NotFound
from google.cloud import geminidataanalytics as ca
from google.genai import types
from google.protobuf.field_mask_pb2 import FieldMask

from .agent import AnswerResult, ClarificationResult
from .bq_tools import BQToolError, QueryRun, technical_detail
from . import analytics_costs


INSTRUCTION = """language: Japanese
purpose: 質問に応じて簡潔な回答や、データに基づく課題分析・施策検討を行う。
rules:
  - 最初に依頼の目的、対象、必要な情報と分析の順序を整理する。
  - 数値や事実を尋ねる明確で単純な質問は、必要な確認だけ行い簡潔に直接答える。
  - 課題発見や施策検討など複雑な質問では、目的に応じた複数の視点で比較・深掘りする。重要な事実と根拠、考えられる原因や別の解釈、優先順位、具体的な検証方法と限界を詳しく示す。一般論は仮説として扱い、因果関係を断定しない。
  - 分析対象・目的・期間などの不足や候補の曖昧さが結論を左右する場合、勝手に決めず、必要な確認質問を利用者にする。分かっている条件は再質問しない。
  - 確認質問が必要な場合は回答全体を次のJSON形式だけで返す: {"type":"clarification","question":"質問文","options":["選択肢1","選択肢2"],"selection_mode":"single"}。選択肢は2〜5個、複数選択が必要ならselection_modeをmultiにする。自由記述欄は画面が用意するため選択肢に含めない。通常の回答はJSONにしない。
  - 列の業務定義を使う。未確認の名称は実在値を確認する。
  - 店舗などの略称は実在する正式名称と照合する。部分一致で得た候補をそのまま一つの店舗として集計せず、意図する対象を特定できなければ質問する。
  - 対象店舗・比較期間・指標を明示し、未完了期間を区別する。
  - 比率や丸めはSQLまたはPythonで計算する。元の数値を推測で補わない。
  - 保存結果を再利用する。根拠の値が見えなければ取得済み結果を確認する。
  - 質問の文字数・SQL禁止・表示指定を守り、表を文章に再掲しない。
  - 登録されたデータソースの読み取り分析のみ行う。
"""

FIELD_CONTEXT = {
    'connected_id': '来店者の識別子。来店者数は指定期間全体のCOUNT(DISTINCT connected_id)。月別人数の合計ではない。',
    'DU_id': '来店の識別子。来店数はCOUNT(DISTINCT DU_id)。同じ来店のドリンク行を重複集計しない。',
    'store': '店舗の正式名称。「神大店」は「神戸大学前店」を指す。',
    'ordered_at': '日本時間の来店日時。BigQueryのDATETIME型。',
}


def as_dict(message):
    return ca.Message.to_dict(message, use_integers_for_enums=False,
                              always_print_fields_with_no_presence=False)


def clarification_answer(answer):
    try:
        value = json.loads(answer)
    except (ValueError, TypeError):
        return None
    if not isinstance(value, dict) or value.get('type') != 'clarification':
        return None
    question, options, mode = value.get('question'), value.get('options'), value.get('selection_mode')
    if (not isinstance(question, str) or not question.strip() or not isinstance(options, list)
            or not 2 <= len(options) <= 5 or not all(isinstance(x, str) and x.strip() for x in options)
            or mode not in ('single', 'multi')):
        return None
    options = [x.strip() for x in options]
    if len(set(options)) != len(options):
        return None
    return ClarificationResult(question.strip(), options, mode)


def build_agent(settings):
    if getattr(settings, 'analysis_backend', 'gemini') == 'conversational_analytics':
        return ConversationalAnalyticsAgent(settings)
    from .agent import GeminiAgent
    return GeminiAgent(settings)


class ConversationalAnalyticsAgent:
    def __init__(self, settings, *, agent_client=None, chat_client=None):
        self.settings = settings
        location = settings.analytics_location
        endpoint = ('geminidataanalytics.googleapis.com' if location == 'global' else
                    f'geminidataanalytics-{location}.googleapis.com' if '-' in location else
                    f'geminidataanalytics.{location}.rep.googleapis.com')
        options = {'api_endpoint': endpoint}
        self.agents = agent_client or ca.DataAgentServiceClient(client_options=options)
        self.chat = chat_client or ca.DataChatServiceClient(client_options=options)
        self.parent = f'projects/{settings.gcp_project}/locations/{location}'

    def _context(self, ctx):
        dataset = self.settings.default_dataset
        if not self.settings.is_dataset_allowed(dataset):
            raise BQToolError('分析対象データセットが許可されていません。')
        listing = ctx.listing_cache.get(dataset)
        if listing is None:
            listing = ctx.tools.list_tables(dataset)
            ctx.listing_cache[dataset] = listing
        available = {item['table'] for item in listing['tables']}
        names = self.settings.analytics_tables
        unknown = set(names) - available
        if unknown:
            raise BQToolError('分析対象に存在しないテーブルがあります: ' + ', '.join(sorted(unknown)))
        context = ca.Context(system_instruction=INSTRUCTION)
        refs = []
        for name in names:
            key = f'{dataset}.{name}'
            if key not in ctx.schema_cache:
                ctx.schema_cache[key] = ctx.tools.get_table_schema(dataset, name)
            schema = ctx.schema_cache[key]
            ref = ca.BigQueryTableReference(project_id=self.settings.gcp_project,
                                            dataset_id=dataset, table_id=name)
            ref.schema.description = schema.get('description', '')
            ref.schema.fields = [ca.Field(name=c['name'], type_=c['type'],
                                          description=' '.join(filter(None, [c.get('description', ''), FIELD_CONTEXT.get(c['name'], '')])))
                                 for c in schema['columns']]
            refs.append(ref)
        context.datasource_references.bq.table_references = refs
        context.options.analysis.python.enabled = True
        context.options.datasource.big_query_max_billed_bytes = self.settings.max_bytes_billed
        return context

    def _prepare(self, ctx):
        context = self._context(ctx)
        digest = hashlib.sha256(ca.Context.to_json(context, sort_keys=True).encode()).hexdigest()
        digest = hashlib.sha256((self.parent + digest).encode()).hexdigest()
        state = ctx.analytics_state
        if state.get('context_sha256') == digest and state.get('conversation'):
            state.setdefault('authored_context', ca.Context.to_dict(context, use_integers_for_enums=False,
                                                                  always_print_fields_with_no_presence=False))
            self._ensure_logging(state)
            return state
        agent_id = 'shirugpt-' + digest[:24]
        name = f'{self.parent}/dataAgents/{agent_id}'
        try:
            self.agents.get_data_agent(name=name, retry=None, timeout=30)
        except NotFound:
            try:
                self.agents.create_data_agent_sync(
                    parent=self.parent, data_agent_id=agent_id,
                    data_agent=ca.DataAgent(display_name='ShiruGPT',
                        data_analytics_agent=ca.DataAnalyticsAgent(published_context=context)),
                    retry=None, timeout=120)
            except AlreadyExists:
                pass
        conversation = self.chat.create_conversation(
            parent=self.parent, conversation_id='shiru-' + uuid.uuid4().hex,
            conversation=ca.Conversation(agents=[name]), retry=None, timeout=30)
        ctx.analytics_state = {'agent': name, 'conversation': conversation.name,
            'context_sha256': digest, 'instruction': INSTRUCTION,
            'authored_context': ca.Context.to_dict(context, use_integers_for_enums=False,
                                                  always_print_fields_with_no_presence=False),
            'results': {}, 'charts': [], 'job_ids': [], 'turns': 0}
        self._ensure_logging(ctx.analytics_state)
        return ctx.analytics_state

    def _ensure_logging(self, state):
        target = (self.settings.analytics_trace_dataset, self.settings.analytics_trace_table)
        if state.get('trace_target') == list(target):
            return
        state['trace_target'] = list(target)
        try:
            result = self.agents.update_data_agent_sync(data_agent=ca.DataAgent(name=state['agent'],
                bigquery_agent_analytics_enabled=True,
                bigquery_agent_analytics_table=ca.BigQueryTableReference(project_id=self.settings.gcp_project,
                    dataset_id=target[0], table_id=target[1])),
                update_mask=FieldMask(paths=['bigquery_agent_analytics_enabled', 'bigquery_agent_analytics_table']),
                retry=None, timeout=15)
            state['native_logging_status'] = 'enabled' if result.bigquery_agent_analytics_enabled is True else 'not_confirmed'
        except Exception as exc:
            state['native_logging_status'] = 'not_confirmed'
            state['native_logging_error'] = technical_detail(exc)

    def run(self, contents, ctx, on_progress=None):
        if ctx.execution_control:
            ctx.execution_control.check()
        started = time.monotonic()
        event = {'event': 'analytics_call', 'started_at': datetime.now().astimezone().isoformat(),
                 'api': 'conversational_analytics', 'input_tokens': None, 'output_tokens': None,
                 'thinking_tokens': None, 'cached_input_tokens': None, 'error': True}
        ctx.turn_events.append(event)
        stream = None
        try:
            state = self._prepare(ctx)
            event['prepare_seconds'] = round(time.monotonic() - started, 6)
            question = '\n'.join(p.text for p in contents[-1].parts or [] if p.text)
            text = question + '\n基準日時: ' + datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(timespec='microseconds')
            event.update(conversation=state['conversation'], agent=state['agent'],
                         context_sha256=state['context_sha256'], continued=state['turns'] > 0,
                         request_messages=1, request_chars=len(text), request_sha256=hashlib.sha256(text.encode()).hexdigest())
            analytics_costs.start_cost(ctx, event, text)
            request = ca.ChatRequest(parent=self.parent,
                conversation_reference=ca.ConversationReference(conversation=state['conversation'],
                    data_agent_context=ca.DataAgentContext(data_agent=state['agent'])),
                messages=[ca.Message(user_message=ca.UserMessage(text=text), message_id=ctx.turn_id)])
            stream = self.chat.chat(request=request, retry=None, timeout=self.settings.analytics_timeout_seconds)
            # gRPC streams expose cancel(), whereas the shared execution control expects close().
            if ctx.execution_control:
                ctx.execution_control.set_response_stream(_StreamStop(stream))
            final = []
            state['retrieval_groups'] = {}
            state['current_charts'] = []
            count = 0
            for message in stream:
                if ctx.execution_control:
                    ctx.execution_control.check()
                count += 1
                elapsed = round(time.monotonic() - started, 6)
                event.setdefault('first_event_seconds', elapsed)
                raw = as_dict(message)
                ctx.turn_events.append({'event': 'analytics_message', 'received_seconds': elapsed,
                                        'started_at': datetime.now().astimezone().isoformat(), 'message': raw})
                system = raw.get('system_message', {})
                text_message = system.get('text', {})
                if text_message.get('text_type') == 'FINAL_RESPONSE':
                    final.extend(text_message.get('parts', []))
                if on_progress and text_message.get('text_type') in ('PROGRESS', 'THOUGHT'):
                    on_progress('分析APIがデータを確認しています…')
                self._result(system, ctx, state)
            state['turns'] += 1
            event.update(stream_completed=True, stream_events=count)
            event['chat_seconds'] = round(time.monotonic() - started - event['prepare_seconds'], 6)
            answer = '\n'.join(final).strip()
            if not answer:
                raise BQToolError('分析APIが最終回答を返しませんでした。データ・実行詳細を確認してください。')
            lookup_started = time.monotonic()
            self._job_details(ctx, state)
            event['job_lookup_seconds'] = round(time.monotonic() - lookup_started, 6)
            event.update(error=False, status='completed')
            contents.append(types.Content(role='model', parts=[types.Part.from_text(text=answer)]))
            return clarification_answer(answer) or AnswerResult(answer)
        except Exception as exc:
            event['exception'] = technical_detail(exc, limit=10000)
            raise
        finally:
            event['seconds'] = round(time.monotonic() - started, 6)
            analytics_costs.finish_cost(ctx, event)
            if ctx.execution_control:
                ctx.execution_control.set_response_stream(None)
            if stream is not None and hasattr(stream, 'cancel'):
                stream.cancel()
            if ctx.usage_logger:
                ctx.usage_logger.log_analytics_call(user_email=ctx.user_email, session_id=ctx.session_id,
                    turn_id=ctx.turn_id, details={'call': event,
                    'events': [e for e in ctx.turn_events if e is not event]})

    @staticmethod
    def _result(system, ctx, state):
        data = system.get('data', {})
        group = state['retrieval_groups'].setdefault(str(system.get('group_id', 0)), {})
        if data.get('generated_sql'):
            group['sql'] = data['generated_sql']
        if data.get('big_query_job'):
            state['job_ids'].append(data['big_query_job'])
            group['job'] = data['big_query_job']
        if data.get('result'):
            result = data['result']
            state['results'][result['name']] = result
            frame = pd.DataFrame(result.get('data', []))
            for field in result.get('schema', {}).get('fields', []):
                if field.get('name') in frame and field.get('type_') in ('INTEGER', 'INT64'):
                    frame[field['name']] = pd.to_numeric(frame[field['name']], errors='raise').astype('Int64')
            ctx.turn_runs.append(QueryRun(sql=group.get('sql', ''), original_sql=group.get('sql', ''),
                estimated_bytes=0, billed_bytes=0, dataframe=frame,
                summary={'source': 'conversational_analytics', 'result_name': result['name'],
                         'job': dict(group.get('job', {}))},
                notes=['APIから返された結果。処理量はBigQueryジョブ情報で確認します。'],
                purpose=result['name']))
        if system.get('chart'):
            state['current_charts'].append(system['chart'])
        if system.get('error'):
            ctx.turn_errors.append(system['error'].get('text', str(system['error'])))

    @staticmethod
    def _job_details(ctx, state):
        accounted = state.setdefault('accounted_jobs', [])
        ledger = state.setdefault('job_costs', {})
        # group_id is optional; several sequential queries can share its default value.
        jobs = [e['message'].get('system_message', {}).get('data', {}).get('big_query_job', {})
                for e in ctx.turn_events if e.get('event') == 'analytics_message']
        for ref in jobs:
            identity = '/'.join(str(ref.get(k, '')) for k in ('project_id', 'location', 'job_id'))
            if not ref.get('job_id'):
                continue
            if identity in accounted:
                if identity in ledger and not any(e.get('event') == 'analytics_query_job'
                        and e.get('job') == ref for e in ctx.turn_events):
                    ctx.turn_events.append({**ledger[identity], 'billed_bytes': 0, 'reused_job': True})
                continue
            try:
                if ref['project_id'] != ctx.settings.gcp_project:
                    raise ValueError('別プロジェクトのジョブ情報は取得しません。')
                job = ctx.tools.client.get_job(ref['job_id'], project=ref['project_id'],
                                               location=ref.get('location') or None, timeout=10)
                details = {'event': 'analytics_query_job', 'job': ref,
                    'sql': job.query, 'billed_bytes': job.total_bytes_billed,
                    'capacity_pricing': bool(job.reservation_id or job.reservation_usage),
                    'processed_bytes': job.total_bytes_processed, 'cache_hit': job.cache_hit,
                    'started_at': job.started.isoformat() if job.started else None,
                    'ended_at': job.ended.isoformat() if job.ended else None,
                    'seconds': (job.ended - job.started).total_seconds() if job.started and job.ended else None}
                ctx.turn_events.append(details)
                ledger[identity] = details
                for run in ctx.turn_runs:
                    if run.summary.get('job', {}).get('job_id') == ref['job_id']:
                        run.billed_bytes = job.total_bytes_billed or 0
                        run.execution_seconds = details['seconds'] or 0
                if job.total_bytes_billed is not None:
                    ctx.session_billed_bytes += job.total_bytes_billed
                    if ctx.usage_logger:
                        ctx.usage_logger.log_bq_query(user_email=ctx.user_email, session_id=ctx.session_id,
                            turn_id=ctx.turn_id, billed_bytes=job.total_bytes_billed)
                    accounted.append(identity)
            except Exception as exc:
                ctx.turn_events.append({'event': 'analytics_job_lookup', 'job': ref, 'error': str(exc)})


class _StreamStop:
    def __init__(self, stream):
        self.stream = stream

    def close(self):
        self.stream.cancel()
