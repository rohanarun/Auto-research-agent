from pathlib import Path
import importlib.util,unittest
s=importlib.util.spec_from_file_location('check_results',Path(__file__).resolve().parents[1]/'tools/check_results.py');c=importlib.util.module_from_spec(s);s.loader.exec_module(c)
class ScoreContractTests(unittest.TestCase):
 def base(self):return {'budget_seconds':1800,'elapsed_seconds':100,'task_id':'kernel-example','benchmark_outcome':'pass','time_to_validated_breakthrough_seconds':90,'candidate_correct':True,'source_rules_passed':True,'one_sided_95_lower':1.97,'reference_speedup':2.0}
 def test_fully_bound_pass(self):c.check_record(self.base())
 def test_rejects_late_or_missing_witness(self):
  for time in [None,1800.1]:
   record=self.base();record['time_to_validated_breakthrough_seconds']=time
   with self.assertRaises(AssertionError):c.check_record(record)
 def test_rejects_self_reported_or_below_threshold_pass(self):
  for field,value in [('candidate_correct',False),('source_rules_passed',None),('one_sided_95_lower',1.95)]:
   record=self.base();record[field]=value
   with self.assertRaises(AssertionError):c.check_record(record)
if __name__=='__main__':unittest.main()
