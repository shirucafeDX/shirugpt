import unittest
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.analytics_costs import aggregate_usage, estimate, finish_cost, refresh_cost, _recover_jobs


def entry():
    return dict(started_at='2026-10-28T00:00:00+00:00', finished_at='2026-10-28T00:00:10+00:00',
                jobs={}, expected_sql=[], stream_completed=True, bq_complete=True,
                monitoring_billed_bytes=0, usage={'status': 'pending'})


class CostTests(unittest.TestCase):
    def test_usage_dedup_and_reasoning_counted_once(self):
        row = {'event_id': '1', 'event_type': 'LLM_RESPONSE',
               'content': {'usage': {'prompt': 1000, 'completion': 20, 'total': 1120}},
               'attributes': {'usage_metadata': {'thoughts_token_count': 100}, 'model_version': 'm'}}
        result = aggregate_usage([row, row, {'event_id': '2', 'event_type': 'INVOCATION_COMPLETED'}])
        self.assertEqual((result['prompt_tokens'], result['output_tokens'], result['thinking_tokens']), (1000, 120, 100))
        self.assertEqual(result['model_calls'], 1)
        self.assertEqual(result['status'], 'reported')

    def test_no_usage_is_unknown_even_when_completed(self):
        result = aggregate_usage([{'event_type': 'INVOCATION_COMPLETED'}])
        self.assertIsNone(result['prompt_tokens'])
        self.assertEqual(result['status'], 'pending')

    def test_promotion_zero_does_not_fabricate_list_cost(self):
        costs = estimate(entry())
        self.assertEqual(costs['ai_usd'], 0)
        self.assertIsNone(costs['ai_list_usd'])
        self.assertEqual(costs['total_usd'], 0)

    def test_after_promotion_unknown_and_partial_not_priced_as_complete(self):
        value = entry()
        value['started_at'] = '2027-01-01T00:00:00+00:00'
        value['usage'] = dict(status='partial', prompt_tokens=1000, output_tokens=100)
        self.assertIsNone(estimate(value)['total_usd'])
        value['usage']['status'] = 'reported'
        self.assertAlmostEqual(estimate(value)['ai_usd'], .005)

    def test_incomplete_and_capacity_jobs_never_report_total(self):
        value = entry()
        value['bq_complete'] = False
        self.assertIsNone(estimate(value)['total_usd'])
        value['bq_complete'] = True
        value['jobs']['j'] = dict(capacity_pricing=True, billed_bytes=10)
        self.assertIsNone(estimate(value)['bq_usd'])

    def test_duplicate_job_event_keeps_first_charge(self):
        value = entry()
        ctx = SimpleNamespace(turn_id='t', analytics_state={'costs': {'t': value}}, turn_events=[
            dict(event='analytics_query_job', job={'job_id': 'j'}, sql='SELECT 1', billed_bytes=100),
            dict(event='analytics_query_job', job={'job_id': 'j'}, sql='SELECT 1', billed_bytes=0, reused_job=True)])
        finish_cost(ctx, {'stream_completed': True})
        self.assertEqual(sum(j['billed_bytes'] for j in value['jobs'].values()), 100)

    @patch('src.analytics_costs._log_rows', return_value=[])
    def test_refresh_throttles_and_unknown_remains_unknown(self, read):
        ctx = SimpleNamespace(analytics_state={'costs': {'t': entry()}}, usage_logger=None)
        refresh_cost(ctx, 't')
        refresh_cost(ctx, 't')
        read.assert_called_once()
        self.assertIsNone(ctx.analytics_state['costs']['t']['usage']['prompt_tokens'])

    def test_recovered_job_is_globally_accounted_once(self):
        from src.analytics_costs import normalize_sql
        value = entry()
        value.update(bq_complete=False, expected_sql=[normalize_sql('SELECT 1')])
        job = SimpleNamespace(query='SELECT 1', project='p', location='US', job_id='j',
                              total_bytes_billed=100, reservation_id=None, reservation_usage=[])
        client = Mock()
        client.list_jobs.return_value = [job]
        client.get_job.return_value = job
        ctx = SimpleNamespace(analytics_state={}, tools=SimpleNamespace(client=client),
                              session_billed_bytes=0, usage_logger=None)
        _recover_jobs(ctx, value)
        self.assertTrue(value['bq_complete'])
        self.assertEqual(ctx.session_billed_bytes, 100)
        second = entry()
        second.update(bq_complete=False, expected_sql=[normalize_sql('SELECT 1')])
        _recover_jobs(ctx, second)
        self.assertEqual(ctx.session_billed_bytes, 100)
        self.assertEqual(next(iter(second['jobs'].values()))['billed_bytes'], 0)

    def test_cost_fragment_displays_unknown_without_network(self):
        from streamlit.testing.v1 import AppTest
        source = '''
import streamlit as st
from types import SimpleNamespace
from src.analytics_costs import estimate
from app import render_analytics_cost
entry = {'started_at':'2026-01-01T00:00:00+00:00','usage':{'status':'pending'},
         'jobs':{},'bq_complete':True,'monitoring_billed_bytes':0}
entry['estimate'] = estimate(entry)
st.session_state.ctx = SimpleNamespace(analytics_state={'costs':{'t':entry}, 'native_logging_status':'not_confirmed'})
render_analytics_cost('t')
'''
        app = AppTest.from_string(source).run(timeout=20)
        self.assertEqual(len(app.exception), 0)
        self.assertEqual(len(app.metric), 4)
        self.assertIn('取得待ち', '\n'.join(c.value for c in app.caption))


if __name__ == '__main__':
    unittest.main()
