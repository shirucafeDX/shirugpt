"""Match SQL disclosed by an analytics turn against the caller's BigQuery jobs."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

from google.cloud import bigquery
import sqlglot

from src.config import load_settings


def normalize(sql):
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
                key = normalize(sql)
                if key:
                    queries[key] = sql
    return queries


def audit(record, client, *, fetch_results=False):
    call = next(e for e in record['events'] if e['event'] == 'analytics_call')
    started = datetime.fromisoformat(call['started_at'])
    expected = disclosed_queries(record['events'])
    matched, seen = [], set()
    if expected:
        jobs = list(client.list_jobs(min_creation_time=started - timedelta(seconds=3),
                    max_creation_time=started + timedelta(seconds=call['seconds'] + 3), max_results=101))
        if len(jobs) > 100:
            raise ValueError('候補ジョブが100件を超えました。確認範囲を絞ってください。')
        for listed in jobs:
            # No all_users option: inspect only the authenticated caller's jobs.
            job = listed if listed.query else client.get_job(listed.job_id, location=listed.location, timeout=10)
            key = normalize(job.query)
            if key not in expected:
                continue
            job = client.get_job(job.job_id, location=job.location, timeout=10)
            item = {'job_id': job.job_id, 'location': job.location, 'query': job.query,
                    'billed_bytes': job.total_bytes_billed, 'processed_bytes': job.total_bytes_processed,
                    'started_at': job.started.isoformat() if job.started else None,
                    'ended_at': job.ended.isoformat() if job.ended else None}
            if fetch_results:
                item['rows'] = [dict(row) for row in job.result(max_results=1000, timeout=30)]
                item['result_limit'] = 1000
            matched.append(item)
            seen.add(key)
    return {'turn_id': record['turn_id'], 'disclosed_sql_count': len(expected),
            'matched_jobs': matched, 'unmatched_sql': [expected[k] for k in expected.keys() - seen],
            'scope': '受信したSQLと自分の実行ジョブの照合。非公開の内部処理の完全性は保証しない。'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('turn', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--results', action='store_true', help='実行済み結果を最大1000行取得。SQL再実行はしない')
    args = parser.parse_args()
    settings = load_settings()
    result = audit(json.loads(args.turn.read_text()), bigquery.Client(project=settings.gcp_project),
                   fetch_results=args.results)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    print(json.dumps({'disclosed': result['disclosed_sql_count'],
                      'matched_jobs': len(result['matched_jobs']), 'unmatched': len(result['unmatched_sql'])}))


if __name__ == '__main__':
    main()
