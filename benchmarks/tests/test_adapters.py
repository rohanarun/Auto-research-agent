"""Transport contract tests with local synthetic responses; no paid inference."""
from pathlib import Path
import http.server,importlib.util,json,os,tempfile,threading,unittest
from unittest.mock import patch
CODE=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('adapters',CODE/'adapters.py');a=importlib.util.module_from_spec(s);s.loader.exec_module(a)
SCHEMA={'type':'object'}
class Handler(http.server.BaseHTTPRequestHandler):
 response={};observed=None
 def do_POST(self):
  Handler.observed=json.loads(self.rfile.read(int(self.headers['Content-Length'])))
  self.send_response(200);self.end_headers();self.wfile.write(json.dumps(Handler.response).encode())
 def log_message(self,*args):pass
class TransportTests(unittest.TestCase):
 def setUp(self):
  parent=Path(os.environ.get('AUTO_RESEARCH_WORKSPACE',CODE/'workspace'));parent.mkdir(parents=True,exist_ok=True)
  self.temp=tempfile.TemporaryDirectory(dir=parent);self.run=Path(self.temp.name);self.server=http.server.ThreadingHTTPServer(('127.0.0.1',0),Handler)
  self.thread=threading.Thread(target=self.server.serve_forever,daemon=True);self.thread.start()
  self.config=self.run/'planner.json';self.config.write_text(json.dumps({'kind':'responses','url':'http://127.0.0.1:'+str(self.server.server_port),'key_env':'TEST_MODEL_KEY','allow_http_for_tests':True}))
  self.env=patch.dict(os.environ,{'AUTO_RESEARCH_PLANNER_CONFIG':str(self.config),'TEST_MODEL_KEY':'synthetic-secret'});self.env.start()
 def tearDown(self):self.env.stop();self.server.shutdown();self.server.server_close();self.thread.join();self.temp.cleanup()
 def test_responses_are_tool_free(self):
  Handler.response={'output':[{'type':'message','content':[{'type':'output_text','text':json.dumps({'command':'echo local','done':False,'note':'inspect'})}]}],'usage':{'input_tokens':2,'input_tokens_details':{'cached_tokens':1},'output_tokens_details':{'reasoning_tokens':3}}}
  action,usage=a.sample(self.run,'prompt','user-model','high',0,5,SCHEMA)
  self.assertEqual(action['command'],'echo local');self.assertEqual(usage['input_tokens'],2);self.assertEqual(usage['cached_input_tokens'],1);self.assertEqual(usage['reasoning_output_tokens'],3);self.assertEqual(Handler.observed['tools'],[]);self.assertFalse(Handler.observed['store'])
 def test_rejects_provider_tool_call(self):
  Handler.response={'output':[{'type':'function_call','name':'search'}]}
  with self.assertRaisesRegex(RuntimeError,'INVALID_ISOLATION'):a.sample(self.run,'prompt','user-model','high',0,5,SCHEMA)
 def test_chat_transport(self):
  c=json.loads(self.config.read_text());c['kind']='chat-completions';self.config.write_text(json.dumps(c));Handler.response={'choices':[{'message':{'content':json.dumps({'command':'','done':True,'note':'finished'})}}]}
  action,_=a.sample(self.run,'prompt','user-model','high',0,5,SCHEMA);self.assertTrue(action['done']);self.assertEqual(Handler.observed['tools'],[]);self.assertEqual(Handler.observed['messages'][0]['role'],'system');self.assertIn('command (string)',Handler.observed['messages'][0]['content'])
 def test_cannot_override_contract(self):
  c=json.loads(self.config.read_text());c['parameters']={'tools':[{'type':'web_search'}]};self.config.write_text(json.dumps(c))
  with self.assertRaises(ValueError):a.sample(self.run,'prompt','user-model','high',0,5,SCHEMA)
 def test_action_types(self):
  with self.assertRaises(ValueError):a.validate_action({'command':'x','done':'false','note':'x'})
if __name__=='__main__':unittest.main()
