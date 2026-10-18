"""Check benchmark scoring and process isolation without a model account."""
import concurrent.futures
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

import benchmark_methods as b
from check_workspace import validate


class BenchmarkTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp(prefix='benchmark-test-', dir=validate(b.ROOT/'tmp')))

    def tearDown(self):
        shutil.rmtree(self.root)

    def test_oracle_matches_all_balanced_binary_vectors_with_negative_weights(self):
        task={'n':4,'edges':[[0,1,-100],[0,2,3],[1,3,4]],'baseline':7}
        reference=b.oracle(task)
        full=max(sum(w for a,c,w in task['edges'] if bits[a]!=bits[c])
                 for bits in itertools.product((0,1),repeat=4) if sum(bits)==2)
        self.assertEqual(reference['optimum'],7)
        self.assertEqual(reference['optimum'],full)

    def test_task_mutation_and_invalid_high_scoring_claim_rejected(self):
        task={'n':4,'edges':[[0,1,100]],'baseline':100}
        raw=json.dumps(task).encode();(self.root/'task.json').write_bytes(raw)
        (self.root/'candidate.json').write_text('[1,0,1,0]')
        evaluator=self.root/'checker.py'
        evaluator.write_text(b.EVALUATOR.replace('EXPECTED_HASH',repr(hashlib.sha256(raw).hexdigest())))
        def check():return json.loads(subprocess.check_output([sys.executable,'-B',str(evaluator),str(self.root)]))
        self.assertEqual(check()['score'],100)
        (self.root/'candidate.json').write_text('[999,0,0,0]');self.assertFalse(check()['valid'])
        (self.root/'candidate.json').write_text('[1,0,1,0]')
        task['edges'][0][2]=999999;(self.root/'task.json').write_text(json.dumps(task))
        self.assertFalse(check()['valid'])

    def test_two_arms_have_isolated_stdout_and_complete_reports(self):
        cli=self.root/'fixture-cli'
        cli.write_text("#!/usr/bin/python3\nimport sys,json,uuid\nfrom pathlib import Path\np=sys.stdin.read()\nprint(json.dumps({'type':'thread.started','thread_id':str(uuid.uuid4())}))\nprint(json.dumps({'type':'turn.completed'}))\n")
        cli.chmod(0o700)
        settings={'codexPath':str(cli),'codexModel':'fixture'}
        with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
            jobs=[pool.submit(b.trial,self.root,settings,[],seed,'fork') for seed in (17,41)]
            reports=[j.result() for j in jobs]
        for report in reports:
            self.assertTrue(report['valid'],report['failure'])
            self.assertIsNone(report['failure'])
            p=self.root/('instance-%s-fork'%report['seed'])
            rows=[json.loads(s) for s in (p/'runner.log').read_text().splitlines()]
            self.assertEqual(len(rows),4)
            self.assertEqual(json.loads((p/'REPORT.json').read_text())['seed'],report['seed'])


if __name__=='__main__':unittest.main()
