"""数値のセル参照・計算・自動回答と実送信の照合。"""
from decimal import Decimal
import json
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pandas as pd
from google.genai import types
from streamlit.testing.v1 import AppTest

from src import context_memory as memory, numeric_grounding as ng, analysis_safety as safety, request_audit, tool_wire
from src.agent import GeminiAgent, ToolContext, _sql_key
from src.bq_tools import QueryRun
from src.chat_history import pack, unpack


def reply(name, args):
    return types.GenerateContentResponse(candidates=[types.Candidate(finish_reason='STOP',
        content=types.Content(role='model', parts=[types.Part(function_call=types.FunctionCall(name=name,args=args))]))])


class NumericGroundingTests(unittest.TestCase):
    def setUp(self):
        self.settings = SimpleNamespace(gcp_project='test', allowed_datasets=('data',), gemini_model='test', max_tool_iterations=6)
        self.ctx = ToolContext(settings=self.settings,tools=Mock())
        self.ctx.start_turn()
        self.agent = GeminiAgent(self.settings,client=Mock())
        self.frame = pd.DataFrame({'学年':['学部1年','学部3年'], '予約数':[100,200], '参加数':[60,140]})
        self.rid = self.save(self.frame)
        self.ref = next(iter(ng.data_refs(self.ctx)))

    def save(self, frame, *, sql='SELECT stored', truncated=False):
        self.ctx.query_cache[_sql_key(sql)] = QueryRun(sql,sql,1,1,frame,{},truncated=truncated)
        return memory.register_result(self.ctx,'run_query',{'sql':sql},{'result':{'row_count':len(frame)}})['result_id']

    def source(self, column='参加数', row=0, **kw):
        return {'result_ref':self.ref,'column':column,**({'row_index':row} if row is not None else {}),**kw}

    def claim(self, key='n', operation='cell', sources=None, **kw):
        return {'key':key,'operation':operation,'sources':sources or [self.source()],**kw}

    def render(self, text='{{n}}', claims=None):
        return ng.render(self.ctx,text,claims or [self.claim()],require_grounding=True)

    def test_cells_and_category_labels_come_from_actual_rows(self):
        answer,bindings=self.render('{{grade}}の参加数は{{n}}人。',[
            self.claim('grade',sources=[self.source('学年',1)]),self.claim(sources=[self.source(row=1)])])
        self.assertEqual(answer,'学部3年の参加数は140人。')
        self.assertEqual(bindings[1]['sources'][0]['row_index'],1)
        self.assertEqual(bindings[1]['sources'][0]['result_id'],self.rid)

    def test_aggregates_ratio_difference_and_change_are_computed(self):
        source=self.source(row=None,aggregate='sum');denom=self.source('予約数',None,aggregate='sum')
        answer,_=self.render('{{n}}人、{{rate}}%。',[
            self.claim(sources=[source]),self.claim('rate','ratio',[source,denom],as_percent=True)])
        self.assertEqual(answer,'200人、66.67%。')
        self.assertEqual(self.render(claims=[self.claim(operation='difference',sources=[self.source(row=0),self.source(row=1)])])[0],'80')
        self.assertEqual(self.render(claims=[self.claim(operation='percent_change',sources=[self.source(row=0),self.source(row=1)])])[0],'133.33')
        self.assertEqual(self.render(claims=[self.claim(operation='sum',sources=[self.source(row=0),self.source(row=1)])])[0],'200')
        self.assertEqual(self.render(claims=[self.claim(sources=[self.source(row=None,aggregate='count')])])[0],'2')

    def test_direct_numbers_fullwidth_and_kanji_cannot_bypass_grounding(self):
        for text in ('42件','４２件','五万件','二割です','学部3年です'):
            with self.subTest(text=text),self.assertRaises(ValueError):
                self.render(text)
        self.assertEqual(self.render('1. 確認した人数は{{n}}人。')[0],'1. 確認した人数は60人。')

    def test_invalid_and_ambiguous_sources_are_rejected(self):
        for source in (self.source(row=-1),self.source(row=True),self.source(row=2),
                       {**self.source(),'result_ref':'R999'},self.source('ない列'),
                       self.source(row=None),self.source(row=None,row_filters=[]),
                       self.source(row=0,aggregate='sum')):
            with self.subTest(source=source),self.assertRaises(ValueError):
                self.render(claims=[self.claim(sources=[source])])
        self.save(pd.DataFrame({'key':['x','x'],'n':[1,2]}),sql='SELECT duplicates')
        ref=list(ng.data_refs(self.ctx))[-1]
        with self.assertRaises(ValueError):
            self.render(claims=[self.claim(sources=[{'result_ref':ref,'column':'n','row_filters':[{'column':'key','value':'x'}]}])])

    def test_row_filters_use_original_table_not_paged_display_position(self):
        source=self.source(row=None,row_filters=[{'column':'学年','value':'学部3年'}])
        answer,bindings=self.render(claims=[self.claim(sources=[source])])
        self.assertEqual(answer,'140')
        self.assertEqual(bindings[0]['sources'][0]['row_index'],1)

    def test_missing_values_zero_denominator_and_partial_totals_fail_closed(self):
        for value in (None,float('nan'),float('inf'),True,pd.NaT):
            with self.subTest(value=value):
                rid=self.save(pd.DataFrame({'n':[value]}),sql=f'SELECT missing {repr(value)}')
                with self.assertRaises(ValueError):
                    self.render(claims=[self.claim(sources=[{'result_ref':rid,'column':'n','row_index':0}])])
        zero=self.save(pd.DataFrame({'n':[0]}),sql='SELECT zero')
        source={'result_ref':zero,'column':'n','row_index':0}
        with self.assertRaises(ValueError):self.render(claims=[self.claim(operation='ratio',sources=[self.source(),source])])
        truncated=self.save(self.frame,truncated=True,sql='SELECT limited')
        with self.assertRaises(ValueError):self.render(claims=[self.claim(sources=[{'result_ref':truncated,'column':'参加数','aggregate':'sum'}])])
        self.assertEqual(self.render(claims=[self.claim(sources=[source])])[0],'0')

    def test_mixed_float_and_integer_columns_do_not_lose_integer_precision(self):
        rid = self.save(pd.DataFrame({'n': [9007199254740993], 'rate': [0.1]}), sql='SELECT mixed')
        result, bindings = self.render(claims=[self.claim(sources=[{'result_ref': rid, 'column': 'n', 'row_index': 0}])])
        self.assertEqual(result, '9007199254740993')
        self.assertEqual(bindings[0]['sources'][0]['value'], '9007199254740993')

    def test_decimal_precision_and_declared_rounding(self):
        rid=self.save(pd.DataFrame({'n':[Decimal('12345678901234567890.123456789')]}),sql='SELECT precise')
        source={'result_ref':rid,'column':'n','row_index':0}
        self.assertEqual(self.render(claims=[self.claim(sources=[source])])[0],'12345678901234567890.123456789')
        self.assertEqual(self.render(claims=[self.claim(sources=[source],decimals=2,format='comma')])[0],'12,345,678,901,234,567,890.12')
        with self.assertRaises(ValueError):self.render(claims=[self.claim(decimals=20)])

    def test_large_decimal_column_sums_are_not_silently_rounded(self):
        value = Decimal('9' * 76)
        rid = self.save(pd.DataFrame({'n': [value] * 5000}), sql='SELECT bignumeric')
        source = {'result_ref': rid, 'column': 'n', 'aggregate': 'sum'}
        result = self.render(claims=[self.claim(operation='sum', sources=[source] * 8)])[0]
        self.assertEqual(result, str(int('9' * 76) * 5000 * 8))

    def test_keys_must_match_and_duplicates_do_not_get_silently_overwritten(self):
        for text,claims in [('{{missing}}',[self.claim()]),('{{n}}',[self.claim(),self.claim()]),('取得完了',[self.claim()]),('{{broken}',[self.claim()])]:
            with self.subTest(text=text),self.assertRaises(ValueError):self.render(text,claims)

    def test_aliases_and_bindings_survive_history_roundtrip(self):
        self.render();original=dict(ng.data_refs(self.ctx))
        raw=pack([],[],self.ctx,None);restored=ToolContext(settings=self.settings,tools=Mock())
        unpack(raw,restored)
        self.assertEqual(ng.data_refs(restored),original)
        self.assertEqual(ng.render(restored,'{{n}}',[self.claim()],require_grounding=True)[0],'60')
        self.assertEqual(ng.normalize_note(restored,{'summary':'事実','references':[self.ref]})['references'],[self.rid])

    def test_scalar_answer_is_returned_without_another_model_call(self):
        sql='SELECT COUNT(DISTINCT DU_id) AS n FROM `test.data.来店`'
        self.ctx.tools.list_tables.return_value={'tables':[{'table':'来店'}]}
        self.ctx.tools.get_table_schema.return_value={'columns':[{'name':'DU_id','type':'STRING'}]}
        self.ctx.tools.run_query.return_value=QueryRun(sql,sql,1,1,pd.DataFrame({'n':[987654321]}),{})
        self.agent._generate=Mock(side_effect=[reply('run_query',{'sql':sql})])
        contents=[types.Content(role='user',parts=[types.Part.from_text(text='来店数を数値だけで答えてください。')])]
        result=self.agent.run(contents,self.ctx)
        self.assertEqual(result.text,'987654321')
        self.agent._generate.assert_called_once()
        self.assertEqual(safety.snapshot(self.ctx)['numeric_grounding']['mode'],'scalar')
        self.assertEqual(self.ctx.context_state['current_note']['references'],[memory.result_id_for('run_query',{'sql':sql})])

    def test_supporting_or_ambiguous_result_is_not_chosen_for_scalar_answer(self):
        safety.state(self.ctx)['question']='数値だけで答えて'
        self.assertIsNone(ng.scalar_answer(self.ctx,[('run_query',{'result_id':self.rid})]))
        rid=self.save(pd.DataFrame({'n':[10]}),sql='SELECT scalar')
        self.ctx.query_cache[_sql_key('SELECT scalar')].result_role='supporting'
        self.assertIsNone(ng.scalar_answer(self.ctx,[('run_query',{'result_id':rid})]))

    def test_final_template_overwrites_unverified_summary_numbers(self):
        safety.state(self.ctx)['data_requested']=True
        args={'answer':'参加数は{{n}}人です。','numeric_claims':[self.claim()],
              'context_note':{'summary':'参加数は42人だった','references':[self.ref]},
              'conversation_summary':'参加数42人'}
        payload=self.agent._dispatch(types.FunctionCall(name='finish_answer',args=args),self.ctx).function_response.response
        self.assertEqual(payload['answer'],'参加数は60人です。')
        self.assertNotIn('42',payload['summary']+payload['overview']+self.ctx.context_state['current_note']['summary'])
        self.assertEqual(self.ctx.context_state['current_note']['references'],[self.rid])
        self.assertEqual(safety.snapshot(self.ctx)['numeric_grounding']['status'],'verified')

    def test_numeric_evidence_is_visible_in_details(self):
        app=AppTest.from_string('''
from app import render_results
render_results({'turn_id':'numeric-turn','execution_summary':{'numeric_grounding':{
 'status':'verified','mode':'template','bindings':[{'key':'n','operation':'cell','rendered':'987654321',
 'sources':[{'result_id':'result_real','column':'来店数','row_index':0,'value':'987654321'}]}]}}},0,set())
''').run(timeout=20)
        self.assertFalse(app.exception)
        self.assertIn('数値の根拠',[e.label for e in app.expander])
        self.assertIn('result_real',app.dataframe[0].value['result_id'].tolist())


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
