"""Near-real-time estimates from native usage logs and actual BigQuery jobs.

Neither the visible question nor a CountTokens call measures the managed model's input.
Amounts are public-rate estimates, not Cloud Billing invoices.
"""
from __future__ import annotations

import json
import time
from datetime import date, datetime, timedelta, timezone

from google.cloud import bigquery
from google.api_core.exceptions import NotFound
import sqlglot

from .config import bq_cost_usd

PRICE_AS_OF = '2026-10-28'
PRICE_SOURCE = 'https://cloud.google.com/products/data-agents/pricing?hl=en'
PROMOTION_END = date(2026, 12, 31)
INPUT_USD_PER_MILLION = 3.0
OUTPUT_USD_PER_MILLION = 20.0


def normalize_sql(sql):
    try:
        return sqlglot.parse_one(sql, read='bigquery').sql(dialect='bigquery', normalize=True)
    except (sqlglot.errors.SqlglotError, TypeError):
        return None


def disclosed_queries(events):
    queries = {}
    for event in events:
        system = event.get('message', {}).get('system_message', {})
        candidates = [system.get('data', {}).get('generated_sql', '')]
        if system.get('text', {}).get('text_type') in ('THOUGHT', 'PROGRESS'):
            candidates += system['text'].get('parts', [])
        for sql in candidates:
            if sql.strip().upper().startswith(('SELECT', 'WITH')):
                key = normalize_sql(sql)
                if key:
                    queries[key] = sql
    return queries


def _json(value):
    return json.loads(value) if isinstance(value, str) else value or {}


def aggregate_usage(rows):
    """Deduplicate retries; a completed turn without usage is still unknown."""
    seen, calls = set(), []
    completed = False
    fast_path = False
    for row in rows:
        identity = row.get('event_id') or (str(row.get('timestamp')), row.get('span_id'), row.get('event_type'))
        if identity in seen:
            continue
        seen.add(identity)
        content, attributes = _json(row.get('content')), _json(row.get('attributes'))
        completed |= row.get('event_type') == 'INVOCATION_COMPLETED'
        fast_path |= attributes.get('fast_path') is True
        if row.get('event_type') != 'LLM_RESPONSE':
            continue
        usage = content.get('usage', {})
        metadata = attributes.get('usage_metadata', {})
        prompt, total = usage.get('prompt'), usage.get('total')
        # total-prompt includes reasoning without adding it a second time to completion.
        output = int(total) - int(prompt) if total is not None and prompt is not None else None
        if output is None and metadata.get('candidates_token_count') is not None and metadata.get('thoughts_token_count') is not None:
            output = int(metadata['candidates_token_count']) + int(metadata['thoughts_token_count'])
        calls.append({'prompt': int(prompt) if prompt is not None else None,
                      'output': output if output is not None and output >= 0 else None,
                      'thoughts': metadata.get('thoughts_token_count'),
                      'model': attributes.get('model_version')})
    if not calls:
        return {'status': 'reported' if completed and fast_path else 'pending',
                'prompt_tokens': 0 if completed and fast_path else None,
                'output_tokens': 0 if completed and fast_path else None,
                'thinking_tokens': None, 'model_calls': 0 if completed and fast_path else None}
    def total_field(key):
        return sum(int(c[key]) for c in calls) if all(c[key] is not None for c in calls) else None
    prompt, output = total_field('prompt'), total_field('output')
    return {'status': 'reported' if completed and prompt is not None and output is not None else 'partial',
            'prompt_tokens': prompt, 'output_tokens': output, 'thinking_tokens': total_field('thoughts'),
            'model_calls': len(calls), 'models': sorted({c['model'] for c in calls if c['model']})}


def estimate(entry):
    usage = entry.get('usage', {})
    prompt, output = usage.get('prompt_tokens'), usage.get('output_tokens')
    list_ai = ((prompt * INPUT_USD_PER_MILLION + output * OUTPUT_USD_PER_MILLION) / 1_000_000
               if usage.get('status') == 'reported' and prompt is not None and output is not None else None)
    promotional = datetime.fromisoformat(entry['started_at']).astimezone(timezone.utc).date() <= PROMOTION_END
    ai = 0.0 if promotional else list_ai
    jobs = list(entry.get('jobs', {}).values())
    capacity = any(j.get('capacity_pricing') for j in jobs)
    billed = sum(j.get('billed_bytes') or 0 for j in jobs)
    known_bq = bq_cost_usd(billed) if not capacity else None
    complete_bq = entry.get('bq_complete', False) and not capacity
    monitoring = bq_cost_usd(entry.get('monitoring_billed_bytes', 0))
    return {'ai_usd': ai, 'ai_list_usd': list_ai, 'promotion_applies': promotional,
            'bq_usd': known_bq if complete_bq else None, 'bq_known_usd': known_bq,
            'monitoring_usd': monitoring,
            'total_usd': ai + known_bq + monitoring if ai is not None and complete_bq else None,
            'price_as_of': PRICE_AS_OF, 'price_source': PRICE_SOURCE,
            'scope': '公開単価での分析API＋BigQueryオンデマンド＋使用量確認クエリ概算。保存・配信・契約割引・BigQuery無料枠は含まない。'}


def start_cost(ctx, call, request_text):
    entry = {'turn_id': ctx.turn_id, 'started_at': call['started_at'],
             'conversation': call['conversation'], 'agent': call['agent'],
             'request_text': request_text, 'usage': {'status': 'pending'}, 'jobs': {},
             'expected_sql': [], 'bq_complete': False, 'monitoring_billed_bytes': 0}
    ctx.analytics_state.setdefault('costs', {})[ctx.turn_id] = entry
    return entry


def finish_cost(ctx, call):
    entry = ctx.analytics_state.get('costs', {}).get(ctx.turn_id)
    if entry is None:
        return
    entry['finished_at'] = datetime.now(timezone.utc).isoformat()
    entry['expected_sql'] = list(disclosed_queries(ctx.turn_events))
    entry['stream_completed'] = bool(call.get('stream_completed'))
    for event in ctx.turn_events:
        if event.get('event') == 'analytics_query_job':
            ref = event['job']
            key = '/'.join(ref.get(k, '') for k in ('project_id', 'location', 'job_id'))
            if key not in entry['jobs']:
                entry['jobs'][key] = {**event, 'accounted': True}
    _complete_bq(entry)
    entry['estimate'] = estimate(entry)


def _complete_bq(entry):
    queries = {normalize_sql(j.get('sql')) for j in entry['jobs'].values()}
    entry['bq_complete'] = (entry.get('stream_completed', False)
        and set(entry['expected_sql']) <= queries
        and all(j.get('billed_bytes') is not None for j in entry['jobs'].values()))


def _log_rows(ctx, entry):
    settings = ctx.settings
    table = f'{settings.gcp_project}.{settings.analytics_trace_dataset}.{settings.analytics_trace_table}'
    # Match the exact sent question within its conversation; never use the most recent turn.
    sql = f'''WITH scoped AS (
      SELECT timestamp,event_id,event_type,session_id,span_id,content,attributes
      FROM `{table}`
      WHERE timestamp >= @start AND timestamp < @end
      AND JSON_VALUE(attributes, '$.session_metadata.state."conversation-id"') IN UNNEST(@conversations)
    ), turns AS (
      SELECT DISTINCT session_id FROM scoped WHERE event_type='USER_MESSAGE_RECEIVED'
      AND JSON_VALUE(content,'$.text_summary')=@question
    ) SELECT * FROM scoped WHERE session_id IN (SELECT session_id FROM turns)'''
    started = datetime.fromisoformat(entry['started_at'])
    conversation = entry['conversation']
    job = ctx.tools.client.query(sql, job_config=bigquery.QueryJobConfig(
        maximum_bytes_billed=64 * 1024**2, query_parameters=[
            bigquery.ScalarQueryParameter('start', 'TIMESTAMP', started - timedelta(seconds=3)),
            bigquery.ScalarQueryParameter('end', 'TIMESTAMP', started + timedelta(minutes=15)),
            bigquery.ArrayQueryParameter('conversations', 'STRING', [conversation, conversation.rsplit('/', 1)[-1]]),
            bigquery.ScalarQueryParameter('question', 'STRING', entry['request_text'])]), timeout=10)
    try:
        rows = [dict(row) for row in job.result(timeout=10)]
    finally:
        if job.total_bytes_billed is not None:
            entry['monitoring_billed_bytes'] += job.total_bytes_billed
            ctx.session_billed_bytes += job.total_bytes_billed
            if ctx.usage_logger:
                ctx.usage_logger.log_bq_query(user_email=ctx.user_email, session_id=ctx.session_id,
                    turn_id=entry['turn_id'], billed_bytes=job.total_bytes_billed)
    sessions = {r['session_id'] for r in rows}
    if len(sessions) > 1:
        raise ValueError('同じ質問に複数の計測ターンが一致しました。費用を誤配分しないため未確定にします。')
    if sessions:
        entry['native_session_id'] = next(iter(sessions))
    return rows


def _recover_jobs(ctx, entry):
    if entry.get('bq_complete') or not entry.get('stream_completed'):
        return
    wanted = set(entry['expected_sql']) - {normalize_sql(j.get('sql')) for j in entry['jobs'].values()}
    if not wanted:
        _complete_bq(entry)
        return
    started = datetime.fromisoformat(entry['started_at'])
    ended = datetime.fromisoformat(entry['finished_at'])
    candidates = list(ctx.tools.client.list_jobs(min_creation_time=started - timedelta(seconds=3),
        max_creation_time=ended + timedelta(seconds=3), max_results=101))
    if len(candidates) > 100:
        raise ValueError('照合候補が100ジョブを超えました。BigQuery費用は未確定です。')
    matches = {}
    for candidate in candidates:
        key = normalize_sql(candidate.query)
        if key in wanted:
            matches.setdefault(key, []).append(candidate)
    if any(len(jobs) > 1 for jobs in matches.values()):
        raise ValueError('同じSQLのジョブ候補が複数あります。費用を誤配分しないため未確定にします。')
    for candidate in candidates:
        if normalize_sql(candidate.query) not in wanted:
            continue
        job = ctx.tools.client.get_job(candidate.job_id, location=candidate.location, timeout=10)
        key = '/'.join((job.project, job.location or '', job.job_id))
        if key in entry['jobs']:
            continue
        accounted = ctx.analytics_state.setdefault('accounted_jobs', [])
        reused = key in accounted
        details = {'event': 'analytics_query_job',
                   'job': {'project_id': job.project, 'location': job.location or '', 'job_id': job.job_id},
                   'sql': job.query, 'billed_bytes': 0 if reused else job.total_bytes_billed,
                   'capacity_pricing': bool(job.reservation_id or job.reservation_usage), 'accounted': True}
        entry['jobs'][key] = details
        if reused:
            continue
        if job.total_bytes_billed is not None:
            accounted.append(key)
            ctx.analytics_state.setdefault('job_costs', {})[key] = details
            ctx.session_billed_bytes += job.total_bytes_billed
            if ctx.usage_logger:
                ctx.usage_logger.log_bq_query(user_email=ctx.user_email, session_id=ctx.session_id,
                    turn_id=entry['turn_id'], billed_bytes=job.total_bytes_billed)
    _complete_bq(entry)


def refresh_cost(ctx, turn_id, *, force=False):
    entry = ctx.analytics_state.get('costs', {}).get(turn_id)
    if not entry:
        return None
    now = time.time()
    if not force and now - entry.get('checked_at', 0) < 5:
        return entry
    entry['checked_at'] = now
    try:
        _recover_jobs(ctx, entry)
        previous = entry.get('usage')
        entry['usage'] = aggregate_usage(_log_rows(ctx, entry))
        entry['stable_checks'] = entry.get('stable_checks', 0) + 1 if entry['usage'] == previous and entry['usage']['status'] == 'reported' else 0
        entry.pop('error', None)
    except NotFound:
        entry['error'] = '使用量ログがまだ届いていません。'
    except Exception as exc:
        entry['error'] = str(exc)
    entry['estimate'] = estimate(entry)
    if ctx.usage_logger:
        ctx.usage_logger.log_analytics_cost(user_email=ctx.user_email, session_id=ctx.session_id,
            turn_id=turn_id, details=entry)
    return entry
