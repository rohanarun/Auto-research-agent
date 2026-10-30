"""Evaluate model-designated final sealed sources; no repairs or source changes after clock."""
from pathlib import Path
import argparse,datetime,hashlib,importlib.util,json,secrets,sys
sys.path.insert(0,str(Path(__file__).resolve().parent));import session as g
from timing_stats import bootstrap
s=importlib.util.spec_from_file_location('gpu_queue',Path(__file__).with_name('dispatcher.py'));q=importlib.util.module_from_spec(s);s.loader.exec_module(q)
ROOT=g.ROOT
def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
def main():
 p=argparse.ArgumentParser();p.add_argument('--approvals',type=Path,required=True);p.add_argument('--cohort',action='append',required=True,help='Exact completed cohort directory name under the configured workspace/runs');a=p.parse_args();approvals=json.loads(a.approvals.read_text());suite=json.loads((ROOT/'suite.json').read_text());tasks={t['id']:t for t in suite['tasks'] if t['kind']=='gpu'};work=[]
 policy=json.loads((Path(__file__).parent/'scoring-policy.json').read_text())
 __import__('bench').require_ssd()
 for cohort in a.cohort:
  directory=(ROOT/'runs'/cohort).resolve();assert directory.is_relative_to((ROOT/'runs').resolve()) and directory.is_dir(),'Invalid cohort'
  for run in sorted(directory.iterdir()):
   if not run.is_dir() or run.name not in tasks:continue
   task=tasks[run.name];meta=json.loads((run/'result.json').read_text());model=meta['model'];assert meta['status'] in ['submitted','budget_exhausted','stopped_without_submission'],'Replace infrastructure-invalid attempts separately before scoring'
   assert not (run/'adjudicated-result.json').exists(),'Do not silently repeat completed final measurements'
   result={**meta,'actor_host_cuda_available':False,'gpu_judge':'NVIDIA H200','gpu_available':True,'reference_speedup':float(task['target']),'time_to_validated_breakthrough_seconds':None,'evidence_class':'none','benchmark_outcome':'right_censored_no_submission','candidate_correct':None,'source_rules_passed':None,'submission_selection':'Final sealed source only; time is earliest sealed checkpoint containing that exact source. Other versions are not exhaustively judged.'}
   checkpoints=[]
   for cp in sorted((run/'checkpoints').glob('*')):
    src=cp/'submission.py'
    if src.is_file() and not src.is_symlink() and (cp/'submission.json').is_file() and json.loads((cp/'checkpoint.json').read_text())['at_seconds']<=meta['budget_seconds']:checkpoints.append(cp)
   if not checkpoints:save(run/'adjudicated-result.json',result);continue
   final=checkpoints[-1];data=(final/'submission.py').read_bytes();sha=hashlib.sha256(data).hexdigest();first=next(cp for cp in checkpoints if hashlib.sha256((cp/'submission.py').read_bytes()).hexdigest()==sha);at=json.loads((first/'checkpoint.json').read_text())['at_seconds']
   decision=approvals.get(model+'/'+task['id'])
   if not decision or decision.get('sha256')!=sha:raise RuntimeError('Missing exact-source audit for '+model+'/'+task['id'])
   result.update(candidate_sha256=sha,selected_checkpoint=final.name,first_identical_checkpoint=first.name,final_source_sealed_at_seconds=at,source_audit=decision,source_rules_passed=decision.get('source_rules_passed',decision['approved']))
   if not decision['approved']:result.update(benchmark_outcome='rule_violation',evidence_class='source_audit');save(run/'adjudicated-result.json',result);continue
   frozen_hashes=json.loads((ROOT/'tasks'/task['id']/'snapshot-hashes.json').read_text())
   if any(not (run/'work'/rel).is_file() or (run/'work'/rel).is_symlink() or hashlib.sha256((run/'work'/rel).read_bytes()).hexdigest()!=digest for rel,digest in frozen_hashes):result.update(benchmark_outcome='rule_violation',evidence_class='frozen_baseline_modified');save(run/'adjudicated-result.json',result);continue
   staged=ROOT/'gpu/staged'/('sealed-'+sha+'.py');staged.parent.mkdir(exist_ok=True);staged.write_bytes(data);work.append((run,task,model,staged,result))
 lease=q.Lease();stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+secrets.token_hex(3)
 try:
  if work:
   lease.start()
   smoke_task=work[0][1]
   probe_id='final-runner-preflight-'+stamp
   probe=g.run_job(lease.sid,smoke_task['problem'],ROOT/'tasks'/smoke_task['id']/'snapshot/baseline.py',probe_id,29990,seconds=300,cases=[0],rounds=2)
   assert probe.get('result',{}).get('status')=='pass' and probe['result']['execution_uid']==29990,'Shared final worker preflight failed; do not score any candidate'
   save(ROOT/'gpu/final-runner-preflight.json',{'passed':True,'job_id':probe_id,'execution_uid':29990,'scope':'Actual CUDA worker under unprivileged supervisor; safe no-follow output reads and protected coordinator lifecycle.'})
  for n,(run,task,model,src,result) in enumerate(work):
   banks=[];jobs=[]
   for bank in range(2):
    jobid='final-'+str(n)+'-'+task['problem']+'-'+str(bank)+'-'+stamp;state=g.run_job(lease.sid,task['problem'],src,jobid,30000+n*2+bank,seconds=900);r=state.get('result');jobs.append(jobid)
    if not r:raise RuntimeError('Missing held-out judge result '+jobid+' '+state.get('log','')[-1500:])
    if r['status']!='pass':
     if str(r.get('error','')).startswith('baseline'):raise RuntimeError('Shared baseline infrastructure failure '+str(r.get('error')))
     result.update(benchmark_outcome='incorrect',candidate_correct=False,evidence_class='heldout_correctness_failure',failure=r.get('error'),evaluation_jobs=jobs);save(run/'adjudicated-result.json',result);break
    banks.append(r)
   if len(banks)==2:
    stats=bootstrap(banks,samples=policy['bootstrap_samples'],seed=policy['bootstrap_seed']);meets_target=stats['one_sided_95_lower']>=(1-policy['equivalence_tolerance'])*float(task['target']);passed=meets_target and result['source_rules_passed'] is True;result.update(candidate_correct=True,benchmark_outcome=('below_target' if not meets_target else ('pending_rule_review' if result['source_rules_passed'] is None else ('pass' if passed else 'rule_violation'))),evidence_class='two_bank_correctness_source_audit_paired_timing',speedup_vs_baseline=stats['point'],timing_interval_95=stats['central_95'],one_sided_95_lower=stats['one_sided_95_lower'],timing_statistics=stats,time_to_validated_breakthrough_seconds=result['final_source_sealed_at_seconds'] if passed else None,evaluation_jobs=jobs,evaluation_finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat());save(run/'adjudicated-result.json',result)
   print('KERNEL_ADJUDICATED',model,task['id'],result['benchmark_outcome'],result.get('speedup_vs_baseline'),flush=True)
 finally:lease.close()
if __name__=='__main__':main()
