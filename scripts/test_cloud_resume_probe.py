#!/usr/bin/env python3
"""Offline resume preflight tests. Fixtures never contain actual credentials."""
import importlib.util, json, os, tempfile, unittest
from pathlib import Path
spec = importlib.util.spec_from_file_location('probe', Path(__file__).resolve().parents[1] / 'Resources/CloudResumeProbe.py')
probe = importlib.util.module_from_spec(spec); spec.loader.exec_module(probe)

class ProbeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.base = Path(self.temp.name)
        self.root = self.base/'research'; self.checkout = self.base/'checkout'; self.proc = self.base/'proc'
        self.state = self.root/'providers/state/state'; (self.state/'sessions').mkdir(parents=True)
        (self.state/'auth.json').write_text('fixture only; never read')
        (self.state/'sessions/rollout-provider.jsonl').write_text('fixture only; never read')
        (self.root/'operations').mkdir(); self.checkout.mkdir(); self.proc.mkdir()
    def tearDown(self): self.temp.cleanup()
    def check(self): return probe.inspect('state','provider',self.root,self.checkout,self.proc)['status']
    def process(self, pid, cwd=None, parent=0):
        p=self.proc/str(pid);p.mkdir(exist_ok=True);(p/'stat').write_text(f'{pid} (fixture) S {parent} 0')
        if cwd is not None:(p/'cwd').symlink_to(cwd)
    def test_restores_existing_without_reading_auth_or_rollout(self): self.assertEqual(self.check(),'READY_TO_RESUME_EXISTING_STATE')
    def test_missing_provider_state(self):
        (self.state/'sessions/rollout-provider.jsonl').unlink();self.assertEqual(self.check(),'BLOCKED_MISSING_PROVIDER_SESSION')
    def test_active_recorded_operation(self):
        folder=self.root/'operations/op';folder.mkdir();(folder/'cli.json').write_text(json.dumps({'pid':98765}))
        self.process(98765);self.assertEqual(self.check(),'BLOCKED_ACTIVE_OPERATION')
    def test_unrecorded_checkout_child(self):
        self.process(98765,self.checkout);self.assertEqual(self.check(),'BLOCKED_CHECKOUT_PROCESS')
    def test_own_control_ancestors_allowed_other_shell_rejected(self):
        self.process(os.getpid(),self.checkout,parent=98764);self.process(98764,self.checkout)
        self.assertEqual(self.check(),'READY_TO_RESUME_EXISTING_STATE')
        self.process(98765,self.checkout);self.assertEqual(self.check(),'BLOCKED_CHECKOUT_PROCESS')
    def test_stopped_receipt_with_dead_pid_can_resume(self):
        folder=self.root/'operations/op';folder.mkdir();(folder/'status.json').write_text(json.dumps({'phase':'running'}));(folder/'cli.json').write_text(json.dumps({'pid':98765}))
        self.assertEqual(self.check(),'READY_TO_RESUME_EXISTING_STATE')
    def test_undrained_descendant_blocks(self):
        folder=self.root/'operations/op';folder.mkdir();(folder/'status.json').write_text(json.dumps({'phase':'ended','cleanup':{'remaining_pids':[98765]}}))
        self.process(98765);self.assertEqual(self.check(),'BLOCKED_UNDRAINED_OPERATION')
    def test_malformed_receipt_fails_closed(self):
        folder=self.root/'operations/op';folder.mkdir();(folder/'cli.json').write_text('invalid')
        with self.assertRaises(ValueError):self.check()

if __name__=='__main__': unittest.main()
