from pathlib import Path
import importlib.util,json,unittest
CODE=Path(__file__).resolve().parents[1]
s=importlib.util.spec_from_file_location('scoring',CODE/'gpu/scoring.py');sc=importlib.util.module_from_spec(s);s.loader.exec_module(sc)
class ScoringTests(unittest.TestCase):
 def setUp(self):self.policy=json.loads((CODE/'gpu/scoring-policy.json').read_text())
 def test_strict_point_cutoff(self):
  t=self.policy['speedup_threshold']
  self.assertFalse(sc.assess(t,t,self.policy,2)['measured_speedup_exceeds_threshold'])
  self.assertTrue(sc.assess(t+1e-8,t-1e-8,self.policy,2)['measured_speedup_exceeds_threshold'])
 def test_uncertainty_does_not_silently_change_requested_cutoff(self):
  r=sc.assess(self.policy['speedup_threshold']+0.001,self.policy['speedup_threshold']-0.02,self.policy,2);self.assertTrue(r['measured_speedup_exceeds_threshold']);self.assertFalse(r['timing_supported_above_threshold']);self.assertFalse(r['historical_reference_recovered'])
if __name__=='__main__':unittest.main()
