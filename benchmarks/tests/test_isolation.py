"""Security controls and provenance tests; no research solutions are embedded here."""
from pathlib import Path
import importlib.util, json, shlex, shutil, tempfile, unittest
CODE_ROOT=Path(__file__).resolve().parents[1]
import os
ROOT=Path(os.environ.get('AUTO_RESEARCH_WORKSPACE',str(CODE_ROOT/'workspace'))).resolve()
s=importlib.util.spec_from_file_location('bench',CODE_ROOT/'bench.py');b=importlib.util.module_from_spec(s);s.loader.exec_module(b)
p=importlib.util.spec_from_file_location('prepare',CODE_ROOT/'setup.py');prep=importlib.util.module_from_spec(p);p.loader.exec_module(prep)

class BoundaryTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  cls.temp=tempfile.TemporaryDirectory(prefix='isolation-',dir=ROOT/'runs')
  cls.original_auth=b.AUTH_SOURCE;b.AUTH_SOURCE=Path(cls.temp.name)/'synthetic-auth.json';b.AUTH_SOURCE.write_text(json.dumps({'synthetic':True}))
  cls.trial=Path(cls.temp.name)/'trial';b.init_run(cls.trial,ROOT/'tests/empty')
 @classmethod
 def tearDownClass(cls):b.AUTH_SOURCE=cls.original_auth;cls.temp.cleanup()
 def test_symlink_cannot_read_curator(self):
  hidden=ROOT/'curator/isolation-canary.txt';link=self.trial/'work/link'
  link.symlink_to(hidden)
  r=b.shell(self.trial,"python -c \"from pathlib import Path; Path('link').read_text()\"",10)
  self.assertNotEqual(r['exit_code'],0);self.assertIn('PermissionError',r['output'])
 def test_no_other_trial_or_auth_reads(self):
  for forbidden in [ROOT/'suite.json',ROOT/'curator/isolation-canary.txt',b.AUTH_SOURCE]:
   command='python -c '+shlex.quote('from pathlib import Path; Path('+repr(str(forbidden))+').read_bytes()')
   r=b.shell(self.trial,command,10);self.assertNotEqual(r['exit_code'],0);self.assertIn('PermissionError',r['output'])
 def test_process_listing_denied(self):
  r=b.shell(self.trial,'ps -axo command',10);self.assertNotEqual(r['exit_code'],0)
 def test_local_compilation_works(self):
  r=b.shell(self.trial,"printf '#include <stdio.h>\\nint main(){puts(\"local\");}\\n' > .tmp/check.c; cc .tmp/check.c -o .tmp/check && .tmp/check",30)
  self.assertEqual(r['exit_code'],0,r['output']);self.assertEqual(r['output'].strip(),'local')
 def test_offline_positive_and_negative_controls(self):
  code="""import socket
print(17*19)
for host in ['1.1.1.1','127.0.0.1']:
 try:
  s=socket.socket();s.settimeout(1);s.connect((host,443))
 except PermissionError:print('DENIED')
 else:raise AssertionError('network escape')
"""
  r=b.shell(self.trial,'python -c '+shlex.quote(code),10)
  self.assertEqual(r['exit_code'],0,r['output']);self.assertEqual(r['output'].splitlines(),['323','DENIED','DENIED'])
 def test_no_git_objects_or_client_configuration(self):
  for task in json.loads((ROOT/'suite.json').read_text())['tasks']:
   snapshot=ROOT/'tasks'/task['id']/'snapshot'
   self.assertFalse((snapshot/'.git').exists());self.assertFalse((snapshot/'.codex').exists())
   h,_=prep.treehash(snapshot);self.assertEqual(h,task['snapshot_sha256'])

if __name__=='__main__':unittest.main()
