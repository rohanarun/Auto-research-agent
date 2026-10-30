"""Check published score consistency and source/timing bindings without renting hardware."""
from pathlib import Path
import hashlib,json,math,statistics
CODE=Path(__file__).resolve().parents[1]
def check_record(r):
 assert r['budget_seconds']>0 and r['elapsed_seconds']>=0
 if r.get('benchmark_outcome')=='pass':
  t=r.get('time_to_validated_breakthrough_seconds');assert t is not None and 0<=t<=r['budget_seconds'],'A pass needs an in-budget admitted checkpoint'
  if r['task_id'].startswith('kernel-'):
   assert r.get('candidate_correct') is True and r.get('source_rules_passed') is True,'A kernel pass needs correctness and source-rule admission'
   assert r['speedup_vs_baseline']>r['speedup_threshold'],'A kernel pass must exceed the baseline speedup cutoff'
   assert r['time_to_validated_breakthrough_seconds']==r['final_source_sealed_at_seconds'],'Time must bind the exact evaluated source'
 if r.get('benchmark_outcome')=='right_censored_no_submission':assert r.get('time_to_validated_breakthrough_seconds') is None

def main():
 policy=json.loads((CODE/'gpu/scoring-policy.json').read_text());dataset=json.loads((CODE/'data/results.json').read_text());cohort=json.loads((CODE/'data/cohort.json').read_text());rows=dataset['results'];index={}
 for r in rows:
  key=(r['model'],r['task_id']);assert key not in index,'Duplicate valid trial cell';index[key]=r;check_record(r)
  if not r['task_id'].startswith('kernel-'):
   for artifact,sha in [('proof_artifact','proof_sha256'),('witness_artifact','witness_sha256')]:
    if r.get(artifact):
     path=(CODE/r[artifact]).resolve();assert path.is_relative_to(CODE) and not path.is_symlink();assert hashlib.sha256(path.read_bytes()).hexdigest()==r[sha]
   if r.get('proof_review'):assert r['proof_review']['proof_sha256']==r['proof_sha256']
   if r.get('benchmark_outcome')=='pass':assert r['submission_sealed_at_seconds']==r['time_to_validated_breakthrough_seconds']
   continue
  if r.get('speedup_vs_baseline') is not None:
   assert r['speedup_threshold']==policy['speedup_threshold'] and r['scoring_policy_version']==policy['version']
   if r.get('candidate_correct') and r.get('source_rules_passed'):assert (r['benchmark_outcome']=='pass')==(r['speedup_vs_baseline']>policy['speedup_threshold'])
  if r.get('time_to_measured_speedup_seconds') is not None:assert 0<=r['time_to_measured_speedup_seconds']<=r['budget_seconds'] and r['time_to_measured_speedup_seconds']==r['final_source_sealed_at_seconds']
  if r.get('candidate_artifact'):
   source=(CODE/r['candidate_artifact']).resolve();assert source.is_relative_to(CODE) and not source.is_symlink();assert hashlib.sha256(source.read_bytes()).hexdigest()==r['candidate_sha256']
   evaluation=json.loads((CODE/r['evaluation_artifact']).read_text());assert evaluation['source_sha256']==r['candidate_sha256']
   banks=[b for b in evaluation['banks'] if b]
   for bank in banks:
    assert bank['source_sha256']==r['candidate_sha256'];assert bank['role']=='contestant' and 10000<=bank['execution_uid']<60000
   if r.get('candidate_correct'):
    assert len(banks)==2 and all(b['status']=='pass' for b in banks);assert len({b['heldout_seed'] for b in banks})==2
    spec=json.loads((CODE/'gpu/specs'/(r['task_id'][len('kernel-'):]+'.json')).read_text())
    for bank in banks:
     assert len(bank['cases'])==len(spec['benchmarks']);assert sorted(c['index'] for c in bank['cases'])==list(range(len(spec['benchmarks'])))
     total_b=total_c=0.
     for case in bank['cases']:
      assert case['correct'] and case['fresh_graph_replay_checked'];assert case['config']==spec['benchmarks'][case['index']];assert len(case['paired_rounds'])==5
      median_b=statistics.median(p['baseline'] for p in case['paired_rounds']);median_c=statistics.median(p['candidate'] for p in case['paired_rounds']);assert math.isclose(median_b,case['baseline_us'],rel_tol=1e-10);assert math.isclose(median_c,case['candidate_us'],rel_tol=1e-10)
      w=case['config'].get('weight',1);total_b+=w*median_b;total_c+=w*median_c
     assert math.isclose(total_b/total_c,bank['speedup_vs_baseline'],rel_tol=1e-10)
    point=math.exp(statistics.mean(math.log(b['speedup_vs_baseline']) for b in banks));assert math.isclose(point,r['speedup_vs_baseline'],rel_tol=1e-10)
 for m in cohort['models']:
  for task in cohort['core_tasks']:assert (m,task) in index,'Missing core comparison cell'
 assert sum(r['scope']=='core' for r in rows)==len(cohort['models'])*len(cohort['core_tasks'])
 print('RESULT_SOURCE_AND_TIMING_BINDINGS_PASS',len(rows),'records')
if __name__=='__main__':main()
