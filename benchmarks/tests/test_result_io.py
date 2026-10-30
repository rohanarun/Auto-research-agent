"""Coordinator never follows worker-controlled links while reading as root."""
from pathlib import Path
import importlib.util,os,tempfile,unittest
s=importlib.util.spec_from_file_location('result_io',Path(__file__).resolve().parents[1]/'gpu/result_io.py');r=importlib.util.module_from_spec(s);s.loader.exec_module(r)
class ResultIOTests(unittest.TestCase):
 def setUp(self):
  parent=Path(os.environ['AUTO_RESEARCH_WORKSPACE']);self.temp=tempfile.TemporaryDirectory(dir=parent);self.root=Path(self.temp.name)
 def tearDown(self):self.temp.cleanup()
 def test_regular_owned_file_and_bounded_tail(self):
  (self.root/'result').write_bytes(b'abcdef');self.assertEqual(r.read_job_file(self.root,'result',[os.getuid()],20,tail=3),b'def')
 def test_cross_job_symlink_and_hardlink_are_rejected(self):
  secret=self.root/'secret';secret.write_bytes(b'private');(self.root/'link').symlink_to(secret)
  with self.assertRaises(OSError):r.read_job_file(self.root,'link',[os.getuid()],20)
  os.link(secret,self.root/'hard')
  with self.assertRaises(PermissionError):r.read_job_file(self.root,'hard',[os.getuid()],20)
 def test_wrong_owner_or_oversized_file_is_rejected(self):
  (self.root/'result').write_bytes(b'abcdef')
  with self.assertRaises(PermissionError):r.read_job_file(self.root,'result',[],20)
  with self.assertRaises(PermissionError):r.read_job_file(self.root,'result',[os.getuid()],2)
if __name__=='__main__':unittest.main()
