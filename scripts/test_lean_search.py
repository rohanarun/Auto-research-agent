"""Real-kernel and HTTP regression checks; all artifacts remain in the workspace."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
import urllib.error
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Resources/LeanSearch'))
from cli import request
from core import audit_axioms, digest
from server import initialize


class LeanServiceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        lean = os.environ.get('LEAN_TEST_BINARY')
        if not lean:
            raise unittest.SkipTest('Set LEAN_TEST_BINARY to the actual installed Lean binary')
        cls.root = Path(tempfile.mkdtemp(prefix='lean-tests-', dir=ROOT / 'tmp'))
        cls.config = initialize(cls.root, lean)
        cls.start()

    @classmethod
    def start(cls):
        cls.log = (cls.root / ('server-' + uuid.uuid4().hex + '.log')).open('w')
        cls.process = subprocess.Popen([sys.executable, '-B', str(ROOT / 'Resources/LeanSearch/cli.py'), 'serve', '--config', str(cls.config)], stdout=cls.log, stderr=cls.log)
        for _ in range(100):
            p = cls.root / 'server.json'
            if p.exists():
                data = json.loads(p.read_text())
                if data['pid'] == cls.process.pid:
                    cls.url = data['url'];cls.token = (cls.root / 'server-token').read_text().strip();return
            if cls.process.poll() is not None: raise RuntimeError('Server failed: ' + cls.log.name)
            time.sleep(.05)
        raise RuntimeError('Test server startup timed out')

    @classmethod
    def stop(cls):
        cls.process.terminate();cls.process.wait(timeout=15);cls.log.close()

    @classmethod
    def tearDownClass(cls):
        cls.stop()
        print('Lean service test evidence:', cls.root)

    def submit(self, source, identifier=None):
        payload = {'id': identifier or str(uuid.uuid4()), 'profile': 'kernel-check', 'files': {'Candidate.lean': source}}
        job = request(self.url + '/v1/jobs', self.token, payload)
        for _ in range(200):
            job = request(self.url + '/v1/jobs/' + job['id'], self.token)
            if job['status'] not in ('QUEUED', 'RUNNING'):return job
            time.sleep(.05)
        self.fail('Test candidate did not finish')

    def test_valid_kernel_and_source_restore(self):
        baseline = (self.root / 'profiles/kernel-check/Candidate.lean').read_bytes()
        job = self.submit('theorem candidate : 1 + 1 = 2 := by rfl\n')
        self.assertEqual(job['status'], 'LEAN_CHECKED', job)
        self.assertEqual((self.root / 'profiles/kernel-check/Candidate.lean').read_bytes(), baseline)
        self.assertTrue(job['receipt']['source_sha256'])

    def test_sorry_and_axiom_are_rejected(self):
        for source in ['theorem candidate : 1 + 1 = 2 := by sorry\n', 'axiom candidate : 1 + 1 = 2\n']:
            self.assertEqual(self.submit(source)['status'], 'REJECTED')

    def test_weakened_statement_is_rejected(self):
        self.assertEqual(self.submit('theorem candidate : True := by trivial\n')['status'], 'REJECTED')

    def test_unauthorized_read_is_rejected(self):
        with self.assertRaises(urllib.error.HTTPError) as e:
            request(self.url + '/v1/health', 'wrong-token')
        self.assertEqual(e.exception.code, 401)

    def test_candidate_cannot_change_challenge_or_escape(self):
        for name in ['Audit.lean', '../server-token']:
            with self.assertRaises(urllib.error.HTTPError):
                request(self.url + '/v1/jobs', self.token, {'id': str(uuid.uuid4()), 'profile':'kernel-check','files':{name:'bad'}})

    def test_idempotent_submit_and_conflict(self):
        identifier = str(uuid.uuid4());source = 'theorem candidate : 1 + 1 = 2 := by decide\n'
        first = self.submit(source, identifier)
        self.assertEqual(self.submit(source, identifier), first)
        with self.assertRaises(urllib.error.HTTPError):self.submit(source+'\n', identifier)

    def test_sandbox_cannot_read_token(self):
        secret_path = json.dumps(str(self.root / 'server-token'))
        source = 'theorem candidate : 1 + 1 = 2 := by decide\n#eval IO.FS.readFile ' + secret_path + '\n'
        job = self.submit(source)
        self.assertEqual(job['status'], 'REJECTED')
        log = (self.root/'jobs'/job['id']/'verification/0-build.log').read_text()
        self.assertNotIn(self.token, log)
        self.assertTrue('denied' in log.lower() or 'not permitted' in log.lower(), log)

    def test_duplicate_audit_is_not_accepted(self):
        with self.assertRaises(ValueError):
            audit_axioms("'x' does not depend on any axioms\n'x' depends on axioms: [sorryAx]", ['x'], [])

    def test_receipt_survives_restart(self):
        job = self.submit('theorem candidate : 1 + 1 = 2 := by decide\n')
        self.stop();self.start()
        self.assertEqual(request(self.url+'/v1/jobs/'+job['id'],self.token),job)


if __name__ == '__main__':unittest.main()
