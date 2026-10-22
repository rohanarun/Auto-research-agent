import io,json,os,sys,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'Resources/LeanSearch'))
from remote import execute

class RemoteTests(unittest.TestCase):
    def check(self, provider, response):
        config={'provider':provider,'base_url':'https://provider.example','instance_id':'owned-id','key_env':'TEST_LEAN_KEY','transport_timeout_seconds':30,'control_timeout_seconds':10}
        with patch.dict(os.environ,{'TEST_LEAN_KEY':'not-a-real-key'}), patch('urllib.request.build_opener') as opener:
            opener.return_value.open.return_value=io.BytesIO(json.dumps(response).encode())
            result=execute(config,'printf test')
            req=opener.return_value.open.call_args.args[0]
            self.assertNotIn('not-a-real-key',req.full_url)
            self.assertNotIn('not-a-real-key',req.data.decode())
            return req,result
    def test_agent37_nonzero_is_preserved(self):
        req,result=self.check('agent37',{'exit_code':7,'stdout':'test','stderr':'failed','truncated':False})
        self.assertEqual(result['exit_code'],7)
        self.assertEqual(req.get_header('X-agent37-key'),'not-a-real-key')
        self.assertTrue(req.full_url.endswith('/v1/instances/owned-id/exec'))
    def test_daytona_contract(self):
        req,result=self.check('daytona',{'exitCode':0,'result':'test'})
        self.assertEqual(result['stdout'],'test')
        self.assertEqual(req.get_header('Authorization'),'Bearer not-a-real-key')
        self.assertTrue(req.full_url.endswith('/toolbox/owned-id/process/execute'))
    def test_truncated_output_is_not_accepted(self):
        with self.assertRaises(ValueError):self.check('agent37',{'truncated':True})

if __name__=='__main__':unittest.main()
