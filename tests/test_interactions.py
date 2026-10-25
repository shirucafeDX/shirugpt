"""状態ID・差分・復元・課金内訳を実SDKのHTTP送信先を差し替えて検証する。"""
import copy
import json
import time
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import httpx
import pandas as pd
from google import genai
from google.genai import types
from google.genai.errors import ClientError
from google.oauth2.credentials import Credentials
from google.cloud import bigquery

from src import interactions, context_memory as memory
from src.agent import GeminiAgent, ToolContext, ConfirmationPending
from src.bq_tools import ConfirmationRequired, QueryRun, error_code, friendly_error
from src.chat_history import pack, unpack
from src.usage_log import UsageLogger, _SCHEMA
from src.execution import ExecutionControl, ExecutionStopped


def settings(**overrides):
    return SimpleNamespace(**{**dict(gemini_api='interactions', gemini_model='gemini-3.8-flash',
        gcp_project='test-project', allowed_datasets=('data',), vertex_location='global',
        max_tool_iterations=12, thinking_budget=None, interaction_max_input_tokens=32000,
        max_history_tokens=16000), **overrides})


def user(value):
    return types.Content(role='user', parts=[types.Part.from_text(text=value)])


def call(name, args=None, call_id=None):
    return {'type':'function_call', 'name':name, 'arguments':args or {}, 'id':call_id or name}


def output(text):
    return {'type':'model_output','content':[{'type':'text','text':text}]}


def interaction(identifier, steps, *, input_tokens=100, output_tokens=10, thought_tokens=20,
                cached_tokens=30, status=None):
    return {'id':identifier, 'status':status or ('requires_action' if any(s['type']=='function_call' for s in steps) else 'completed'),
            'steps':steps, 'usage':{'total_input_tokens':input_tokens, 'total_output_tokens':output_tokens,
                'total_thought_tokens':thought_tokens, 'total_cached_tokens':cached_tokens,
                'total_tokens':input_tokens+output_tokens+thought_tokens}}


class InteractionTests(unittest.TestCase):
    def setUp(self):
        self.settings = settings()
        self.ctx = ToolContext(tools=Mock(), settings=self.settings, usage_logger=Mock())
        self.ctx.tools.list_datasets.return_value = {'datasets': [{'dataset':'data'}]}
        self.contents = [user('売上を調べて')]

    def sdk(self, replies):
        self.requests = []
        queue = iter(replies)
        def handler(request):
            self.assertIn('/v1beta1/projects/test-project/locations/global/interactions',str(request.url))
            self.requests.append(json.loads(request.content) if request.content else {})
            value=next(queue)
            return value if isinstance(value,httpx.Response) else httpx.Response(200,json=value)
        return genai.Client(enterprise=True, project='test-project', location='global',
            credentials=Credentials(token='local-test'), http_options=types.HttpOptions(
                api_version='v1beta1', retry_options=types.HttpRetryOptions(attempts=0),
                client_args={'transport':httpx.MockTransport(handler)}))

    def test_real_sdk_sends_only_new_results_and_selected_tools(self):
        replies = [interaction('state-1',[call('select_tools',{'names':['list_datasets']})]),
                   interaction('state-2',[call('list_datasets')]), interaction('state-3',[output('調べました')]),
                   interaction('state-4',[output('続きです')])]
        with self.sdk(replies) as client:
            agent = GeminiAgent(self.settings, client=client)
            result = agent.run(self.contents, self.ctx)
            self.assertEqual(result.text,'調べました')
            self.contents.append(user('続けて'))
            agent.run(self.contents, self.ctx)
        first, second, third, fourth = self.requests
        self.assertNotIn('previous_interaction_id',first)
        self.assertEqual(first['input'][0]['content'][0]['text'],'売上を調べて')
        self.assertEqual([t['name'] for t in first['tools']],['select_tools'])
        self.assertEqual(second['previous_interaction_id'],'state-1')
        self.assertEqual([t['name'] for t in second['tools']],['select_tools','list_datasets'])
        self.assertEqual(second['input'][0]['call_id'],'select_tools')
        self.assertEqual(second['input'][0]['result']['available_tools'],['list_datasets'])
        self.assertEqual(third['previous_interaction_id'],'state-2')
        self.assertEqual(len(third['input']),1)
        self.assertIn('datasets',third['input'][0]['result'])
        self.assertEqual(fourth['previous_interaction_id'],'state-3')
        self.assertEqual(fourth['input'],[{'type':'user_input','content':[{'type':'text','text':'続けて'}]}])
        self.assertTrue(all(r['store'] for r in self.requests))
        self.assertTrue(all(r['generation_config']['thinking_level']=='low' for r in self.requests))
        self.assertEqual(self.ctx.tools.list_datasets.call_count,1)
        self.assertEqual(self.ctx.session_prompt_tokens,400)
        self.assertEqual(self.ctx.session_output_tokens,120)
        logged = self.ctx.usage_logger.log_gemini_call.call_args.kwargs
        self.assertEqual(logged['output_tokens'],30)
        self.assertEqual(logged['cached_input_tokens'],30)
        self.assertEqual(logged['api'],'interactions')

    def test_state_and_cursor_roundtrip_resume_only_new_question(self):
        with self.sdk([interaction('state-1',[output('回答')]),interaction('state-2',[output('次の回答')])]) as client:
            agent = GeminiAgent(self.settings, client=client)
            agent.run(self.contents,self.ctx)
            raw = pack([],self.contents,self.ctx,None)
            restored = ToolContext(tools=Mock(),settings=self.settings)
            _, contents, _ = unpack(raw,restored)
            self.assertEqual(restored.interaction_state,self.ctx.interaction_state)
            contents.append(user('追加質問'))
            agent.run(contents,restored)
        self.assertEqual(self.requests[1]['previous_interaction_id'],'state-1')
        self.assertEqual(len(self.requests[1]['input']),1)

    def test_expired_state_bootstraps_memo_and_ids_without_historical_csv_or_sql(self):
        client=Mock()
        client.interactions.create.side_effect=[interaction('state-1',[output('売上は100円です')]),
                                                interaction('state-2',[output('続き')])]
        agent=GeminiAgent(self.settings,client=client)
        agent.run(self.contents,self.ctx)
        rid=memory.register_result(self.ctx,'run_query',{'sql':'SELECT secret'},
            {'result':{'csv':'店舗,売上\nA,100\n','row_count':1}})['result_id']
        memory.accept_note(self.ctx,{'summary':'店舗Aの売上は100円。','references':[rid]})
        memory.finish_turn(self.contents,self.ctx,'店舗Aの売上は100円。','対象店舗はA。')
        self.ctx.interaction_state['chain_started_at']=time.time()-interactions.RETENTION_SECONDS
        self.contents.append(user('前回の数字を比較して'))
        agent.run(self.contents,self.ctx)
        request=client.interactions.create.call_args.kwargs
        self.assertNotIn('previous_interaction_id',request)
        snapshot=json.loads(request['input'][0]['content'][0]['text'])
        self.assertEqual(snapshot['current_question'],'前回の数字を比較して')
        wire=json.dumps(snapshot,ensure_ascii=False)
        self.assertIn('100円',wire)
        self.assertIn(rid,wire)
        self.assertNotIn('SELECT secret',wire)
        self.assertNotIn('店舗,売上',wire)
        self.assertEqual(self.ctx.interaction_state['last_reset_reason'],'expired')

    def test_input_limit_rotates_during_tool_loop_without_orphan_results(self):
        client=Mock()
        client.interactions.create.side_effect=[interaction('state-1',[call('list_datasets',{
            'context_note':{'summary':'対象期間は2026年10月。','references':[]}})],input_tokens=26000),
                                                interaction('state-2',[output('完了')])]
        agent=GeminiAgent(self.settings,client=client)
        agent.run(self.contents,self.ctx)
        request=client.interactions.create.call_args.kwargs
        self.assertNotIn('previous_interaction_id',request)
        self.assertEqual(request['input'][0]['type'],'user_input')
        snapshot=json.loads(request['input'][0]['content'][0]['text'])
        self.assertEqual(snapshot['latest_tool_results'][0]['tool'],'list_datasets')
        self.assertEqual(snapshot['current_working_note']['summary'],'対象期間は2026年10月。')
        self.assertEqual(self.ctx.interaction_state['last_reset_reason'],'input_budget')
        self.ctx.tools.list_datasets.assert_called_once()

    def test_server_expiry_rebuilds_once_but_other_404_does_not(self):
        client=Mock()
        missing=ClientError(404,{'error':{'message':'not found'}})
        client.interactions.create.side_effect=[interaction('state-1',[output('回答')]),missing,
                                                interaction('state-2',[output('復元')])]
        client.interactions.get.side_effect=missing
        agent=GeminiAgent(self.settings,client=client)
        agent.run(self.contents,self.ctx)
        self.contents.append(user('次'))
        self.assertEqual(agent.run(self.contents,self.ctx).text,'復元')
        self.assertNotIn('previous_interaction_id',client.interactions.create.call_args.kwargs)
        client.interactions.get.assert_called_once_with(id='state-1')
        self.contents.append(user('さらに'))
        client.interactions.create.side_effect=missing
        client.interactions.get.side_effect=None
        with self.assertRaises(ClientError): agent.run(self.contents,self.ctx)

    def test_network_failure_does_not_advance_cursor_or_execute_tools(self):
        client=Mock()
        client.interactions.create.side_effect=[interaction('state-1',[output('回答')]),TimeoutError('timeout')]
        agent=GeminiAgent(self.settings,client=client)
        agent.run(self.contents,self.ctx)
        before=copy.deepcopy(self.ctx.interaction_state)
        self.contents.append(user('次'))
        with self.assertRaises(TimeoutError): agent.run(self.contents,self.ctx)
        self.assertEqual(self.ctx.interaction_state,before)
        self.assertEqual(self.ctx.tools.mock_calls,[])
        self.assertEqual(client.interactions.create.call_count,2)

    def test_confirmation_resume_does_not_execute_prior_batch_call_again(self):
        client=Mock()
        client.interactions.create.side_effect=[interaction('state-1',[call('run_query',{'sql':'SELECT 1'},'q1'),
                                                                     call('run_query',{'sql':'SELECT 2'},'q2')]),
                                                interaction('state-2',[output('完了')])]
        frame=pd.DataFrame({'value':[1]})
        run=QueryRun('SELECT 1','SELECT 1',1,1,frame,{'row_count':1})
        self.ctx.tools.run_query.side_effect=[run,ConfirmationRequired('SELECT 2',20,10,[]),run]
        agent=GeminiAgent(self.settings,client=client)
        pending=agent.run(self.contents,self.ctx)
        self.assertIsInstance(pending,ConfirmationPending)
        self.assertEqual(len(self.ctx.tool_receipts),1)
        self.ctx.decisions[pending.key]='approved'
        self.assertEqual(agent.run(self.contents,self.ctx).text,'完了')
        self.assertEqual(self.ctx.tools.run_query.call_count,3)
        sent=client.interactions.create.call_args.kwargs
        self.assertEqual(sent['previous_interaction_id'],'state-1')
        self.assertEqual([item['call_id'] for item in sent['input']],['q1','q2'])

    def test_incomplete_function_call_is_not_executed(self):
        client=Mock()
        client.interactions.create.return_value=interaction('state-1',[call('run_query',{'sql':'SELECT 1'})],status='incomplete')
        result=GeminiAgent(self.settings,client=client).run(self.contents,self.ctx)
        self.assertIn('出力トークン上限',result.text)
        self.assertEqual(self.ctx.tools.mock_calls,[])

    def test_final_answer_tool_is_sent_once_on_followup(self):
        client=Mock()
        client.interactions.create.side_effect=[interaction('state-1',[call('finish_answer',{
            'answer':'最終回答です','conversation_summary':'店舗Aが対象。',
            'context_note':{'summary':'店舗Aが対象。','references':[],'open_questions':[],'resolved_questions':[]}})]),
            interaction('state-2',[output('追加回答')])]
        agent=GeminiAgent(self.settings,client=client)
        self.assertEqual(agent.run(self.contents,self.ctx).text,'最終回答です')
        self.contents.append(user('続けて'))
        agent.run(self.contents,self.ctx)
        sent=client.interactions.create.call_args.kwargs
        self.assertEqual(sent['previous_interaction_id'],'state-1')
        self.assertEqual([v['type'] for v in sent['input']],['function_result','user_input'])
        self.assertEqual(sent['input'][0]['result']['answer'],'最終回答です')

    def test_restore_pending_batch_keeps_receipts_and_original_call_ids(self):
        client=Mock()
        client.interactions.create.side_effect=[interaction('state-1',[call('list_datasets',call_id='first'),
            call('run_query',{'sql':'SELECT 1'},'expensive')]),interaction('state-2',[output('完了')])]
        self.ctx.tools.run_query.side_effect=ConfirmationRequired('SELECT 1',20,10,[])
        agent=GeminiAgent(self.settings,client=client)
        pending=agent.run(self.contents,self.ctx)
        raw=pack([],self.contents,self.ctx,pending)
        tools=Mock()
        frame=pd.DataFrame({'value':[1]})
        tools.run_query.return_value=QueryRun('SELECT 1','SELECT 1',1,1,frame,{'row_count':1})
        restored=ToolContext(tools=tools,settings=self.settings)
        _, contents, confirmation=unpack(raw,restored)
        restored.decisions[confirmation.key]='approved'
        self.assertEqual(agent.run(contents,restored).text,'完了')
        tools.list_datasets.assert_not_called()
        tools.run_query.assert_called_once()
        self.assertEqual(client.interactions.create.call_args.kwargs['previous_interaction_id'],'state-1')
        self.assertEqual([v['call_id'] for v in client.interactions.create.call_args.kwargs['input']],['first','expensive'])

    def test_changed_history_or_project_starts_new_state(self):
        client=Mock()
        client.interactions.create.side_effect=[interaction('state-1',[output('回答')]),interaction('state-2',[output('新規')])]
        agent=GeminiAgent(self.settings,client=client)
        agent.run(self.contents,self.ctx)
        self.contents[0]=user('編集した質問')
        self.contents.append(user('続けて'))
        agent.run(self.contents,self.ctx)
        self.assertNotIn('previous_interaction_id',client.interactions.create.call_args.kwargs)
        self.assertEqual(self.ctx.interaction_state['last_reset_reason'],'history_changed')

    def test_sdk_schema_preserves_nested_argument_names_and_limits(self):
        with self.sdk([interaction('state-1',[output('回答')])]) as client:
            self.ctx.active_tool_names=['run_query']
            GeminiAgent(self.settings,client=client).run(self.contents,self.ctx)
        schema=self.requests[0]['tools'][1]['parameters']
        self.assertEqual(schema['type'],'object')
        self.assertIn('sql',schema['properties'])
        note=schema['properties']['context_note']
        self.assertIn('resolved_questions',note['properties'])
        self.assertEqual(note['properties']['resolved_questions']['maxItems'],12)
        self.assertEqual(note['properties']['resolved_questions']['items']['properties']['conclusion']['type'],'string')

    def test_sdk_does_not_retry_uncertain_post_on_server_error_or_timeout(self):
        for timeout in (False,True):
            sent=[]
            def handler(request):
                sent.append(request)
                if timeout:
                    raise httpx.ReadTimeout('uncertain response',request=request)
                return httpx.Response(500,json={'error':{'code':500,'message':'internal','status':'INTERNAL'}})
            with self.subTest(timeout=timeout), genai.Client(enterprise=True,project='test-project',location='global',
                    credentials=Credentials(token='local-test'),http_options=types.HttpOptions(
                        api_version='v1beta1',retry_options=types.HttpRetryOptions(attempts=0),
                        client_args={'transport':httpx.MockTransport(handler)})) as client:
                ctx=ToolContext(tools=Mock(),settings=self.settings)
                with self.assertRaises(Exception):
                    GeminiAgent(self.settings,client=client).run([user('質問')],ctx)
                self.assertEqual(len(sent),1)
                self.assertEqual(ctx.interaction_state,{})
                self.assertEqual(ctx.tools.mock_calls,[])

    def test_missing_usage_uses_estimate_for_rotation_but_does_not_fabricate_billing(self):
        client=Mock()
        first=interaction('state-1',[output('回答')])
        first.pop('usage')
        client.interactions.create.return_value=first
        GeminiAgent(self.settings,client=client).run(self.contents,self.ctx)
        self.assertTrue(self.ctx.interaction_state['input_tokens_estimated'])
        self.assertGreater(self.ctx.interaction_state['input_tokens'],0)
        self.ctx.usage_logger.log_gemini_call.assert_not_called()

    def test_legacy_history_unknown_empty_part_fields_and_data_nulls_restore(self):
        import io
        from zipfile import ZipFile,ZIP_DEFLATED
        content=types.Content(role='user',parts=[types.Part.from_function_response(
            name='list_tables',response={'value':None})])
        raw=pack([], [content], self.ctx, None)
        target=io.BytesIO()
        with ZipFile(io.BytesIO(raw)) as source,ZipFile(target,'w',ZIP_DEFLATED) as result:
            for filename in source.namelist():
                data=source.read(filename)
                if filename=='conversation.json':
                    value=json.loads(data)
                    value['contents'][0]['value']['parts'][0]['future_optional_field']=None
                    data=json.dumps(value).encode()
                result.writestr(filename,data)
        restored=ToolContext(tools=Mock(),settings=self.settings)
        _,contents,_=unpack(target.getvalue(),restored)
        self.assertIsNone(contents[0].parts[0].function_response.response['value'])

    def test_roll_back_api_discards_pending_unsigned_calls(self):
        self.ctx.interaction_state={'id':'old-state'}
        self.ctx.resume_calls=[types.FunctionCall(name='run_query',args={'sql':'SELECT 1'},id='pending')]
        pending=ConfirmationPending('key','SELECT 1',20,10,[])
        raw=pack([{'role':'user','text':'質問'}],[user('質問')],self.ctx,pending)
        restored=ToolContext(tools=Mock(),settings=settings(gemini_api='generate_content'))
        _,_,confirmation=unpack(raw,restored)
        self.assertIsNone(confirmation)
        self.assertIsNone(restored.resume_calls)
        self.assertEqual(restored.interaction_state,{})

    def test_current_turn_result_can_be_found_after_catalogue_is_shortened(self):
        memory.sync_turns(self.contents,self.ctx)
        for i in range(15):
            memory.register_result(self.ctx,'run_query',{'sql':f'SELECT {i}','purpose':'その他'},
                {'result':{'row_count':1}})
        rid=memory.register_result(self.ctx,'run_query',{'sql':'SELECT 99','purpose':'店舗別の累計売上'},
            {'result':{'row_count':1,'csv':'売上\n100\n'}})['result_id']
        found=memory.recall(self.ctx,{'search':'累計売上'})['matches']
        self.assertEqual(found[0]['turn_id'],self.ctx.context_state['turns'][0]['id'])
        self.assertIn(rid,[r['result_id'] for r in found[0]['results']])

    def test_csv_is_sent_once_with_original_values_then_followup_is_only_new_text(self):
        frame=pd.DataFrame({'value':[1,2]})
        self.ctx.tools.run_query.return_value=QueryRun('SELECT 1','SELECT 1',1,1,frame,{'row_count':2})
        self.ctx.active_tool_names=['run_query']
        replies=[interaction('state-1',[call('run_query',{'sql':'SELECT 1'})]),
                 interaction('state-2',[output('合計は3')]),interaction('state-3',[output('説明')])]
        with self.sdk(replies) as client:
            agent=GeminiAgent(self.settings,client=client)
            agent.run(self.contents,self.ctx)
            self.contents.append(user('説明して'))
            agent.run(self.contents,self.ctx)
        data=self.requests[1]['input'][0]['result']['result']
        self.assertEqual(data['csv'],'value\n1\n2\n')
        self.assertEqual(data['sent_row_count'],2)
        self.assertEqual(len(self.requests[2]['input']),1)
        self.assertEqual(self.requests[2]['input'][0]['content'][0]['text'],'説明して')
        self.ctx.tools.run_query.assert_called_once()

    def test_clarification_answer_continues_same_state_without_duplicate_question_text(self):
        from src.agent import ClarificationResult
        client=Mock()
        client.interactions.create.side_effect=[interaction('state-1',[call('ask_clarification',{
            'scope':'general','question':'何を相談しますか？','options':['集客','採用'],'selection_mode':'single'})]),
            interaction('state-2',[output('集客について回答')])]
        agent=GeminiAgent(self.settings,client=client)
        self.assertIsInstance(agent.run(self.contents,self.ctx),ClarificationResult)
        self.contents.append(user('集客'))
        agent.run(self.contents,self.ctx)
        request=client.interactions.create.call_args.kwargs
        self.assertEqual(request['previous_interaction_id'],'state-1')
        self.assertEqual([v['type'] for v in request['input']],['function_result','user_input'])
        self.assertEqual(request['input'][1]['content'][0]['text'],'集客')

    def test_stop_after_api_response_never_executes_returned_sql(self):
        client=Mock()
        control=ExecutionControl()
        self.ctx.execution_control=control
        def stop_response(**_):
            control.request_stop()
            return interaction('state-1',[call('run_query',{'sql':'SELECT 1'})])
        client.interactions.create.side_effect=stop_response
        agent=GeminiAgent(self.settings,client=client)
        with self.assertRaises(ExecutionStopped): agent.run(self.contents,self.ctx)
        self.ctx.tools.run_query.assert_not_called()
        self.assertEqual(len(self.contents),1)
        self.ctx.interaction_state['reset_required']=True
        self.ctx.execution_control=None
        self.contents.extend([types.Content(role='model',parts=[types.Part.from_text(text='処理を停止しました。')]),user('別の質問')])
        client.interactions.create.side_effect=None
        client.interactions.create.return_value=interaction('state-2',[output('別の回答')])
        agent.run(self.contents,self.ctx)
        self.assertNotIn('previous_interaction_id',client.interactions.create.call_args.kwargs)
        self.ctx.tools.run_query.assert_not_called()

    def test_sdk_429_retries_identical_input_and_counts_only_successful_generation(self):
        rejected=httpx.Response(429,json={'error':{'message':'Resource exhausted. Please try again later.',
                                                  'code':'too_many_requests'}})
        progress=[]
        with self.sdk([rejected,interaction('state-1',[output('回答')])]) as client, \
             patch('src.interactions._wait_for_retry') as wait, \
             patch('src.interactions.random.uniform',return_value=0):
            result=GeminiAgent(self.settings,client=client).run(self.contents,self.ctx,on_progress=progress.append)
        self.assertEqual(result.text,'回答')
        self.assertEqual(len(self.requests),2)
        self.assertEqual(self.requests[0],self.requests[1])
        wait.assert_called_once_with(self.ctx,2)
        self.assertTrue(any('混み合' in text for text in progress))
        self.assertEqual(self.ctx.session_prompt_tokens,100)
        self.ctx.usage_logger.log_gemini_call.assert_called_once()

    def test_sdk_429_exhaustion_has_friendly_message_and_http_code(self):
        replies=[httpx.Response(429,json={'error':{'message':'Resource exhausted.','code':'too_many_requests'}})
                 for _ in range(4)]
        with self.sdk(replies) as client,patch('src.interactions._wait_for_retry') as wait, \
             patch('src.interactions.random.uniform',return_value=0):
            with self.assertRaises(Exception) as caught:
                GeminiAgent(self.settings,client=client).run(self.contents,self.ctx)
        self.assertEqual(len(self.requests),4)
        self.assertEqual([args.args[1] for args in wait.call_args_list],[2,4,8])
        self.assertEqual(error_code(caught.exception),429)
        self.assertIn('同じ質問',friendly_error(caught.exception))
        self.assertNotIn('内容を変えて',friendly_error(caught.exception))
        self.assertEqual(self.ctx.interaction_state,{})
        self.assertEqual(self.ctx.tools.mock_calls,[])
        self.ctx.usage_logger.log_gemini_call.assert_not_called()
        self.assertEqual(self.ctx.usage_logger.log_error.call_args.kwargs['error_code'],429)

    def test_429_after_query_does_not_rerun_sql(self):
        frame=pd.DataFrame({'value':[1]})
        self.ctx.tools.run_query.return_value=QueryRun('SELECT 1','SELECT 1',1,1,frame,{'row_count':1})
        replies=[interaction('state-1',[call('run_query',{'sql':'SELECT 1'})]),
                 httpx.Response(429,json={'error':{'code':'too_many_requests','message':'Resource exhausted.'}}),
                 interaction('state-2',[output('回答')])]
        with self.sdk(replies) as client,patch('src.interactions._wait_for_retry'):
            GeminiAgent(self.settings,client=client).run(self.contents,self.ctx)
        self.ctx.tools.run_query.assert_called_once()
        self.assertEqual(self.requests[1],self.requests[2])
        self.assertEqual(self.requests[1]['previous_interaction_id'],'state-1')
        self.assertEqual(self.ctx.session_prompt_tokens,200)

    def test_rate_limit_wait_can_be_stopped_without_resending(self):
        self.ctx.execution_control=ExecutionControl()
        def stop(seconds):
            self.ctx.execution_control.request_stop()
            return True
        self.ctx.execution_control.stopped.wait=Mock(side_effect=stop)
        with self.sdk([httpx.Response(429,json={'error':{'code':'too_many_requests','message':'busy'}})]) as client:
            with self.assertRaises(ExecutionStopped):
                GeminiAgent(self.settings,client=client).run(self.contents,self.ctx)
        self.assertEqual(len(self.requests),1)
        self.assertEqual(self.ctx.interaction_state,{})

    def test_retry_after_longer_than_wait_budget_is_not_ignored(self):
        reply=httpx.Response(429,headers={'Retry-After':'60'},json={'error':{'code':'too_many_requests','message':'busy'}})
        with self.sdk([reply]) as client,patch('src.interactions._wait_for_retry') as wait:
            with self.assertRaises(Exception) as caught:
                GeminiAgent(self.settings,client=client).run(self.contents,self.ctx)
        self.assertEqual(error_code(caught.exception),429)
        self.assertEqual(len(self.requests),1)
        wait.assert_not_called()

    def test_retry_after_seconds_and_http_date_are_honoured(self):
        from email.utils import formatdate
        now=1700000000
        for header,expected in [('5',5),(formatdate(now+7,usegmt=True),7)]:
            replies=[httpx.Response(429,headers={'Retry-After':header},
                json={'error':{'code':'too_many_requests','message':'busy'}}),interaction('state-1',[output('回答')])]
            ctx=ToolContext(tools=Mock(),settings=self.settings)
            with self.subTest(header=header),self.sdk(replies) as client, \
                 patch('src.interactions.time.time',return_value=now), \
                 patch('src.interactions.random.uniform',return_value=0), \
                 patch('src.interactions._wait_for_retry') as wait:
                GeminiAgent(self.settings,client=client).run([user('質問')],ctx)
                wait.assert_called_once_with(ctx,expected)


class InteractionConfigTests(unittest.TestCase):
    def test_default_api_and_validation(self):
        import os
        from src.config import load_settings,ConfigError
        env={'GCP_PROJECT_ID':'test','BQ_DEFAULT_DATASET':'data','BQ_ALLOWED_DATASETS':'data',
             'BQ_LOG_DATASET':'','GOOGLE_APPLICATION_CREDENTIALS':'','GEMINI_API':'interactions',
             'GEMINI_STATE_MAX_INPUT_TOKENS':'32000','GEMINI_REQUEST_TIMEOUT_SECONDS':'120'}
        with patch.dict(os.environ,env):
            config=load_settings()
            self.assertEqual(config.gemini_api,'interactions')
            self.assertEqual(config.interaction_max_input_tokens,32000)
            self.assertEqual(config.gemini_request_timeout_seconds,120)
            for key,value in [('GEMINI_API','invalid'),('GEMINI_STATE_MAX_INPUT_TOKENS','10'),('GEMINI_REQUEST_TIMEOUT_SECONDS','0')]:
                with self.subTest(key=key),patch.dict(os.environ,{key:value}),self.assertRaises(ConfigError):
                    load_settings()


class InteractionUsageTests(unittest.TestCase):
    def test_usage_schema_migrates_existing_table_and_records_cache_separately(self):
        s=SimpleNamespace(logging_enabled=True,gcp_project='p',log_dataset='logs',log_table='events',
            limit_table='limits',prompt_log_table='prompts',bq_location=None,gemini_model='gemini-3.8-flash')
        client=Mock()
        existing=bigquery.Table('p.logs.events',schema=[f for f in _SCHEMA if f.name not in ('api','cached_input_tokens')])
        client.create_table.side_effect=lambda table, **_: existing if table.table_id=='events' else table
        client.insert_rows_json.return_value=[]
        logger=UsageLogger(s,client=client)
        logger.log_gemini_call(user_email='user',session_id='session',turn_id='turn',prompt_tokens=100,
            output_tokens=30,total_tokens=130,thinking_tokens=20,cached_input_tokens=60,api='interactions')
        client.update_table.assert_called_once_with(existing,['schema'])
        row=client.insert_rows_json.call_args.args[1][0]
        self.assertEqual(row['prompt_tokens'],100)
        self.assertEqual(row['cached_input_tokens'],60)
        self.assertEqual(row['output_tokens'],30)
        self.assertEqual(row['api'],'interactions')
        self.assertTrue(logger._ready)


if __name__=='__main__': unittest.main()
