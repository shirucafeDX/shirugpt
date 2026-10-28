"""ツール結果のHTTP送信照合。回答の数値検証とは独立。"""
import json
import unittest
import httpx
from src import request_audit, tool_wire


class WireAuditTests(unittest.TestCase):
    def test_object_wire_preserves_empty_null_zero_and_false_without_mutating_archive(self):
        from src.interactions import _wire_input
        data={'result_id':'real','result':{'csv':'n\n987654321\n'},'empty':[],'object':{},'null':None,'zero':0,'false':False}
        step={'type':'function_result','call_id':'q','name':'run_query','result':data}
        output=_wire_input([step])[0]
        self.assertIsInstance(output['result'],dict)
        self.assertEqual(tool_wire.decode(output['result']),data)
        self.assertEqual(data['empty'],[])
        self.assertEqual(request_audit.results([output])[0]['payload_sha256'],request_audit.digest(data))

    def test_actual_wire_matches_and_changed_csv_is_detected_without_logging_contents(self):
        payload={'result':{'csv':'n\n987654321\n'},'null':None}
        steps=[{'type':'function_result','call_id':'q','name':'run_query','result':tool_wire.encode(payload)}]
        metrics={'sent_tool_results':request_audit.results(steps)}
        token=request_audit.active_request.set(metrics)
        try:
            request_audit.capture_http_request(httpx.Request('POST','https://example.invalid/interactions',json={'input':steps},headers={'Authorization':'secret'}))
            self.assertTrue(metrics['wire_results_verified'])
            self.assertNotIn('secret',json.dumps(metrics))
            self.assertNotIn('csv',metrics['wire_tool_results'][0])
            steps[0]['result']['result']['csv']='n\n42\n'
            with self.assertRaises(ValueError):
                request_audit.capture_http_request(httpx.Request('POST','https://example.invalid/interactions',json={'input':steps}))
            self.assertFalse(metrics['wire_results_verified'])
        finally:
            request_audit.active_request.reset(token)
