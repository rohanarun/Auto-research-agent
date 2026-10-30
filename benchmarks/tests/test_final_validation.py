"""Final admission cannot bypass a source audit or strand a lease on evaluator failure."""
from pathlib import Path
import hashlib,importlib.util,json,os,sys,tempfile,unittest
from unittest.mock import patch
CODE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE));sys.path.insert(0,str(CODE/'gpu'))
s=importlib.util.spec_from_file_location('final_validation',CODE/'gpu/validate.py');v=importlib.util.module_from_spec(s);s.loader.exec_module(v)
class FinalValidationTests(unittest.TestCase):
 def setUp(self):
  self.temp=tempfile.TemporaryDirectory(dir=Path(os.environ['AUTO_RESEARCH_WORKSPACE']));self.root=Path(self.temp.name);self.run=self.root/'runs/example/kernel-example';cp=self.run/'checkpoints/0001';cp.mkdir(parents=True);(self.run/'work').mkdir();(self.root/'gpu').mkdir();task=self.root/'tasks/kernel-example';task.mkdir(parents=True)
  self.source=b'candidate';sha=hashlib.sha256(self.source).hexdigest();(cp/'submission.py').write_bytes(self.source);(cp/'submission.json').write_text('{}');(cp/'checkpoint.json').write_text(json.dumps({'at_seconds':12}))
  (self.run/'result.json').write_text(json.dumps({'model':'example-model','status':'submitted','budget_seconds':1800,'task_id':'kernel-example'}));(task/'snapshot-hashes.json').write_text('[]');(self.root/'suite.json').write_text(json.dumps({'tasks':[{'id':'kernel-example','kind':'gpu','problem':'example','target':'2'}]}))
  self.approvals=self.root/'approvals.json';self.approvals.write_text(json.dumps({'example-model/kernel-example':{'sha256':sha,'approved':True,'source_rules_passed':True}}))
 def tearDown(self):self.temp.cleanup()
 def invoke(self):
  with patch.object(v,'ROOT',self.root),patch.object(sys,'argv',['validate.py','--cohort','example','--approvals',str(self.approvals)]),patch('bench.require_ssd'):v.main()
 def test_missing_exact_source_audit_never_allocates(self):
  self.approvals.write_text('{}')
  with patch.object(v.q,'Lease') as lease:
   with self.assertRaisesRegex(RuntimeError,'exact-source audit'):self.invoke()
   lease.assert_not_called()
 def test_worker_failure_always_closes_gpu(self):
  class FakeLease:
   sid='fake';closed=False
   def start(self):pass
   def close(self):self.closed=True
  lease=FakeLease()
  with patch.object(v.q,'Lease',return_value=lease),patch.object(v.g,'run_job',side_effect=[{'result':{'status':'pass','execution_uid':29990}},RuntimeError('synthetic evaluator failure')]):
   with self.assertRaisesRegex(RuntimeError,'synthetic evaluator failure'):self.invoke()
  self.assertTrue(lease.closed);self.assertFalse((self.run/'adjudicated-result.json').exists())
if __name__=='__main__':unittest.main()
