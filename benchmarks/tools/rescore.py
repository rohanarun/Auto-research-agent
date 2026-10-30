"""Reclassify committed measurements without model calls, source repairs or GPU allocations."""
from pathlib import Path
import argparse,ast,copy,datetime,json,sys
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE/'gpu'))
from scoring import assess

def main():
 p=argparse.ArgumentParser();p.add_argument('--input',type=Path,default=CODE/'data/results-historical-target-v1.json');p.add_argument('--output',type=Path,default=CODE/'data/results.json');a=p.parse_args()
 data=json.loads(a.input.read_text());policy=json.loads((CODE/'gpu/scoring-policy.json').read_text())
 data=copy.deepcopy(data);data['scoring_policy_version']=policy['version'];data['rescore_provenance']={'applied_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'method':'Same sealed sources, measurements and timestamps; primary kernel cutoff changed retrospectively at user request. Initial actors pursued historical targets. No new GPU work.','original_results':'data/results-historical-target-v1.json','original_policy':'gpu/scoring-policy-historical-v1.json','time_scope':'Earliest in-budget checkpoint containing byte-identical final evaluated source. Earlier different source versions were not exhaustively evaluated.'}
 for r in data['results']:
  if not r['task_id'].startswith('kernel-'):continue
  r['historical_benchmark_outcome']=r['benchmark_outcome'];r['historical_reference_target']=r['target'];r['target']=str(policy['speedup_threshold']);r['scoring_policy_version']=policy['version'];r['time_to_validated_breakthrough_seconds']=None
  if r.get('candidate_artifact'):
   candidate=CODE/r['candidate_artifact'];baseline=CODE/'data/kernel-starters'/r['task_id'][len('kernel-'):]/'baseline.py'
   r['source_equals_baseline_ast']=ast.dump(ast.parse(candidate.read_text()))==ast.dump(ast.parse(baseline.read_text()))
  point=r.get('speedup_vs_baseline')
  if point is None:continue
  r.update(assess(point,r['one_sided_95_lower'],policy,r['reference_speedup']))
  r['time_to_measured_speedup_seconds']=r['final_source_sealed_at_seconds'];r['time_scope']=data['rescore_provenance']['time_scope']
  passed=r.get('candidate_correct') is True and r.get('source_rules_passed') is True and r['measured_speedup_exceeds_threshold']
  r['benchmark_outcome']='pass' if passed else 'below_threshold'
  if passed:r['time_to_validated_breakthrough_seconds']=r['final_source_sealed_at_seconds']
  if r['source_equals_baseline_ast']:r['source_progress_note']='AST-identical to the frozen baseline; measured timing deviation is not an algorithmic improvement.'
 a.output.write_text(json.dumps(data,indent=2)+'\n');print('RESCORED_WITHOUT_GPU',sum(r.get('benchmark_outcome')=='pass' and r['task_id'].startswith('kernel-') for r in data['results']))
if __name__=='__main__':main()
