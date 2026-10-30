"""Regression for a real CLI transport warning incorrectly classified as a native tool."""
from pathlib import Path
import importlib.util,json,os,tempfile,unittest
from unittest.mock import patch
s=importlib.util.spec_from_file_location('bench',Path(__file__).resolve().parents[1]/'bench.py');b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
class NativeGuardTests(unittest.TestCase):
 def test_cli_transport_warning_is_diagnostic(self):
  events=[{'type':'item.completed','item':{'id':'item_0','type':'error','message':'Falling back from WebSockets to HTTPS transport. unexpected status 403 Forbidden'}},{'type':'item.completed','item':{'type':'agent_message','text':'{}'}},{'type':'turn.completed','usage':{'input_tokens':20}}]
  usage,native=b.parse_events('\n'.join(json.dumps(e) for e in events));self.assertEqual(native,[]);self.assertEqual(usage['input_tokens'],20)
 def test_native_execution_and_unknown_events_fail_closed(self):
  for kind in ['command_execution','web_search','mcp_tool_call','unrecognized_tool_kind']:
   _,native=b.parse_events(json.dumps({'type':'item.started','item':{'type':kind}}));self.assertEqual(len(native),1)
class CapacityRetryTests(unittest.TestCase):
 def test_retry_retains_failed_attempt_and_reuses_same_prompt(self):
  parent=Path(os.environ.get('AUTO_RESEARCH_WORKSPACE',b.ROOT));parent.mkdir(parents=True,exist_ok=True)
  with tempfile.TemporaryDirectory(dir=parent) as temporary:
   run=Path(temporary);(run/'model-0000.events.jsonl').write_text('capacity event')
   action={'command':'echo local','done':True,'note':'ready'}
   with patch.object(b,'_sample_once',side_effect=[RuntimeError('Inference failed: Selected model is at capacity'),(action,{})]) as call,patch.object(b.time,'sleep'):
    result=b.sample(run,'same prompt','chosen-model','high',0,20)
   self.assertEqual(result[0],action);self.assertEqual(call.call_count,2);self.assertEqual(call.call_args_list[0].args[1],call.call_args_list[1].args[1]);self.assertTrue((run/'transport-failures/0-1/model-0000.events.jsonl').exists())
 def test_real_native_tool_rejection_is_not_retried(self):
  with patch.object(b,'_sample_once',side_effect=RuntimeError('INVALID_ISOLATION: native command execution')) as call:
   with self.assertRaisesRegex(RuntimeError,'INVALID_ISOLATION'):b.sample(Path('.'),'prompt','chosen-model','high',0,20)
   self.assertEqual(call.call_count,1)
if __name__=='__main__':unittest.main()
