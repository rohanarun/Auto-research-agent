"""Host-boundary and lifecycle regressions; no provider calls or GPU allocations."""
from pathlib import Path
import hashlib,importlib.util,json,os,sys,tempfile,threading,unittest,uuid
from unittest.mock import patch
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE));sys.path.insert(0,str(CODE/'gpu'))
s=importlib.util.spec_from_file_location('gpu_dispatcher',CODE/'gpu/dispatcher.py');d=importlib.util.module_from_spec(s);s.loader.exec_module(d)
class DispatcherTests(unittest.TestCase):
 def setUp(self):
  root=Path(os.environ.get('AUTO_RESEARCH_WORKSPACE',CODE/'workspace'));root.mkdir(parents=True,exist_ok=True);self.temp=tempfile.TemporaryDirectory(dir=root);self.root=Path(self.temp.name)
 def tearDown(self):self.temp.cleanup()
 def test_safe_file_read_and_symlink_rejection(self):
  q=self.root/'q';q.mkdir();(q/'good').write_bytes(b'candidate');self.assertEqual(d.read_request_file(q,'good',100),b'candidate')
  secret=self.root/'secret';secret.write_bytes(b'private');(q/'bad').symlink_to(secret)
  with self.assertRaises(OSError):d.read_request_file(q,'bad',100)
 def test_hardlink_read_rejected_and_feedback_does_not_modify_target(self):
  q=self.root/'q';q.mkdir();secret=self.root/'secret';secret.write_bytes(b'private');os.link(secret,q/'bad')
  with self.assertRaises(AssertionError):d.read_request_file(q,'bad',100)
  d.feedback(q,'bad',{'safe':True});self.assertEqual(secret.read_bytes(),b'private');self.assertTrue(json.loads((q/'bad').read_text())['safe'])
 def test_feedback_refuses_symlink_directory(self):
  q=self.root/'q';q.mkdir();link=self.root/'link';link.symlink_to(q)
  with self.assertRaises(OSError):d.feedback(link,'result',{})
 def test_no_gpu_when_no_pending_work(self):
  stop=threading.Event();stop.set()
  with patch.object(d,'Lease') as lease:d.serve([],stop);lease.return_value.start.assert_not_called()
 def test_gpu_error_exports_feedback_and_closes_lease(self):
  run=self.root/'runs/trial/task';q=run/'work/.gpu';q.mkdir(parents=True);(run/'result.json').write_text(json.dumps({'status':'running'}));job=uuid.uuid4().hex;source=b'public candidate'
  (q/(job+'.candidate.py')).write_bytes(source);(q/(job+'.request.json')).write_text(json.dumps({'sha256':hashlib.sha256(source).hexdigest(),'cases':[0],'created_at_unix':0}))
  specs=self.root/'gpu/specs';specs.mkdir(parents=True);(specs/'public_problem.json').write_text(json.dumps({'benchmarks':[{}]}));stop=threading.Event()
  class FakeLease:
   sid=None;closed=False
   def start(self):self.sid='fake-owned-lease'
   def close(self):self.closed=True;self.sid=None
  lease=FakeLease()
  def fail(*args,**kwargs):stop.set();raise RuntimeError('synthetic provider failure')
  # The spec is code-root data in the portable runner. Intercept the one public spec read.
  original=Path.read_text
  def read(path,*args,**kwargs):
   if path==CODE/'gpu/specs/public_problem.json':return json.dumps({'benchmarks':[{}]})
   return original(path,*args,**kwargs)
  with patch.object(d,'ROOT',self.root),patch.object(d,'Lease',return_value=lease),patch.object(d.gpu,'run_job',side_effect=fail),patch.object(Path,'read_text',read):d.serve([(run,{'problem':'public_problem'})],stop)
  self.assertTrue(lease.closed);result=json.loads((q/(job+'.result.json')).read_text());self.assertEqual(result['status'],'infrastructure_error')
if __name__=='__main__':unittest.main()
