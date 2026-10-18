#!/usr/bin/env python3
"""Offline read-only telemetry tests; run with SSD TMPDIR and python -B."""
import base64
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
import unittest

RESOURCES = Path(__file__).resolve().parents[1] / 'Resources'
def module(name):
    spec = importlib.util.spec_from_file_location(name, RESOURCES / (name + '.py'))
    result = importlib.util.module_from_spec(spec); spec.loader.exec_module(result); return result
telemetry = module('MethodTelemetry')
bridge = module('DaytonaWorker')

class TelemetryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.base = Path(self.temp.name)
        self.checkout = self.base / 'checkout'; self.operation = self.base / 'private/operations/op'
        self.control = self.checkout / 'SHARED_CONTEXT/methods-fixture'
        self.root = self.checkout / 'compact/.auto-research-methods/runs'
        self.operation.mkdir(parents=True); self.control.mkdir(parents=True)
        self.config = {'provider': 'codex', 'policy': 'fixture'}; self.plan = {'workspace': 'compact'}
        self.put(self.operation/'request.json', {'research_methods': True, 'arguments': [str(self.control)]})
        self.put(self.operation/'status.json', {'phase': 'running', 'started_unix': time.time()-1})
        self.put(self.control/'config.json', self.config); self.put(self.control/'plan.json', self.plan)
        self.run = self.create_run('current')
    def tearDown(self): self.temp.cleanup()
    def put(self, p, value):
        p.parent.mkdir(parents=True, exist_ok=True); p.write_text(json.dumps(value))
    def create_run(self, name, config=None, plan=None):
        run = self.root/name
        self.put(run/'CONFIG.json', self.config if config is None else config)
        self.put(run/'PLAN.json', self.plan if plan is None else plan)
        return run
    def log(self, name, events, run=None):
        path = (run or self.run)/name/'provider.jsonl';path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b''.join(json.dumps(e).encode()+b'\n' for e in events));return path
    def event(self, text='hello'):
        return {'type': 'item.completed', 'item': {'type': 'agent_message', 'id': 'message', 'text': text}}
    def collect(self, cursor=None, limit=100000):
        return telemetry.collect(self.operation, self.checkout, cursor or {}, limit)
    def test_completed_lines_partial_tail_and_exact_retry_keys(self):
        path=self.log('branch-1',[self.event()]);path.write_bytes(path.read_bytes()+b'{"type":')
        first=self.collect();retry=self.collect()
        self.assertEqual(first,retry);self.assertEqual(len(first['method_events']),1)
        self.assertTrue(first['method_partial']);self.assertTrue(first['method_eof'])
        second=self.collect(first['method_cursor']);self.assertEqual(second['method_events'],[])
        with path.open('ab') as out:out.write(b'"turn.completed"}\n')
        third=self.collect(second['method_cursor']);self.assertEqual(third['method_events'][0]['event'],{'type':'turn.completed'})
    def test_filters_reasoning_private_fields_and_credentials(self):
        self.log('branch-1',[{'type':'item.completed','item':{'type':'reasoning','text':'secret chain'}},
            {'type':'thread.started','thread_id':'private-session'},self.event('sk-'+ 'x'*24),
            {'type':'item.completed','item':{'type':'command_execution','command':'cat /research-private/auth/auth.json','aggregated_output':'private token content'}},
            {'type':'turn.failed','error':{'message':'problem','reasoning':'hidden','access_token':'private'}}])
        result=json.dumps(self.collect()['method_events'])
        for bad in ('secret chain','private-session','private token content','hidden','x'*24):self.assertNotIn(bad,result)
        self.assertIn('problem',result);self.assertIn('command_execution',result)
    def test_multiple_branches_and_exact_run_config_scope(self):
        self.log('branch-1',[self.event('one')]);self.log('branch-2',[self.event('two')])
        self.log('branch-3',[self.event('wrong')],self.create_run('other',config={'provider':'codex','policy':'other'}))
        old=self.create_run('old');os.utime(old/'CONFIG.json',(1,1));self.log('branch-4',[self.event('old')],old)
        future=self.create_run('future');os.utime(future/'CONFIG.json',(time.time()+100,time.time()+100))
        self.log('branch-5',[self.event('future')],future)
        self.put(self.operation/'status.json',{'phase':'ended','started_unix':time.time()-1,'ended_unix':time.time()})
        result=self.collect();self.assertEqual({e['branch'] for e in result['method_events']},{'branch-1','branch-2'})
    def test_scope_rejects_traversal_and_symlink_without_reading_target(self):
        self.put(self.control/'plan.json',{'workspace':'../private'})
        with self.assertRaises(ValueError):self.collect()
        self.put(self.control/'plan.json',self.plan)
        target=self.log('real',[self.event('visible')]);(self.run/'symlink').mkdir()
        (self.run/'symlink/provider.jsonl').symlink_to(target)
        self.assertEqual(len(self.collect()['method_events']),1)
    def test_budget_round_robin_and_oversized_record(self):
        self.log('branch-1',[self.event('z'*1000),self.event('after')])
        self.log('branch-2',[{'type':'turn.started'}])
        cursor={};events=[];omitted=0
        for _ in range(40):
            result=self.collect(cursor,128);cursor=result['method_cursor'];events+=result['method_events'];omitted+=result['method_omitted']
            self.assertLessEqual(result['method_bytes_read'],128)
            if result['method_eof']:break
        self.assertEqual(omitted,1);self.assertEqual({e['branch'] for e in events},{'branch-1','branch-2'})
        self.assertEqual(len({e['key'] for e in events}),len(events))
    def test_claude_thinking_and_opaque_tool_payload_are_hidden(self):
        self.config['provider']='claude';self.put(self.control/'config.json',self.config);self.put(self.run/'CONFIG.json',self.config)
        self.log('branch',[{'type':'assistant','session_id':'hidden','message':{'content':[
            {'type':'thinking','thinking':'private reasoning'}, {'type':'text','text':'public finding'},
            {'type':'tool_use','id':'tool','name':'Bash','input':{'command':'secret'}}]}},
            {'type':'user','message':{'content':[{'type':'tool_result','tool_use_id':'tool','content':'secret result'}]}},
            {'type':'result','subtype':'success','result':'finished','session_id':'hidden'}])
        result=json.dumps(self.collect()['method_events']);self.assertIn('public finding',result);self.assertIn('Bash',result)
        for bad in ('private reasoning','secret result','hidden','"command"'):self.assertNotIn(bad,result)
    def test_observation_does_not_write_any_fixture(self):
        self.log('branch',[self.event()]);before={str(p): (p.stat().st_mtime_ns,p.read_bytes()) for p in self.base.rglob('*') if p.is_file()}
        self.collect();after={str(p): (p.stat().st_mtime_ns,p.read_bytes()) for p in self.base.rglob('*') if p.is_file()}
        self.assertEqual(before,after)
    def test_invalid_json_is_skipped_and_truncation_is_explicit(self):
        path=self.log('branch',[self.event()]);path.write_bytes(b'broken\n'+path.read_bytes())
        result=self.collect();self.assertEqual(result['method_omitted'],1)
        path.write_bytes(b'')
        with self.assertRaises(ValueError):self.collect(result['method_cursor'])
    def test_bridge_preserves_raw_offsets_and_summary_without_opt_in(self):
        bridge.R=self.base/'private';bridge.CO=self.checkout
        summary=b'{"summary":"exact method summary"}\n';(self.operation/'stdout').write_bytes(summary)
        (self.operation/'stderr').write_bytes(b'diagnostic\n');self.log('branch',[self.event()])
        request={'id':'op','limit':100000,'stdout_offset':2,'stderr_offset':3}
        old=bridge.poll(request);new=bridge.poll(dict(request,method_cursor={}))
        for key,value in old.items():self.assertEqual(new[key],value)
        self.assertEqual(base64.b64decode(new['stdout']),summary[2:]);self.assertNotIn('method_events',old)
        self.assertEqual(len(new['method_events']),1)
        self.assertFalse((RESOURCES/'__pycache__/MethodTelemetry.cpython-314.pyc').exists())
    def test_control_traversal_and_malformed_metadata_fail_closed(self):
        self.put(self.operation/'request.json',{'research_methods':True,'arguments':[str(self.control/'..'/'methods-fixture')]})
        with self.assertRaises(ValueError):self.collect()
        self.put(self.operation/'request.json',{'research_methods':True,'arguments':[str(self.control)]})
        self.put(self.control/'config.json',[])
        with self.assertRaises(ValueError):self.collect()
    def test_busy_first_branch_does_not_starve_second(self):
        self.log('branch-1',[self.event('x'*2000)])
        self.log('branch-2',[{'type':'turn.started'}])
        first=self.collect(limit=128);second=self.collect(first['method_cursor'],128)
        self.assertEqual(second['method_events'][0]['branch'],'branch-2')
    def test_zero_events_when_ordinary_operation_or_run_not_started(self):
        self.log('branch',[self.event()])
        self.put(self.operation/'status.json',{'phase':'requested'})
        self.assertEqual(self.collect()['method_events'],[])
        self.put(self.operation/'request.json',{'research_methods':False})
        self.assertEqual(self.collect()['method_events'],[])
    def test_bridge_isolates_observer_failure(self):
        bridge.R=self.base/'private';bridge.CO=self.checkout
        result=bridge.poll({'id':'op','limit':100,'method_cursor':{'offsets':[]}})
        self.assertEqual(result['method_events'],[]);self.assertIn('method_error',result)

if __name__=='__main__':unittest.main()
