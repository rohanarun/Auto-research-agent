"""Same 30-minute actor protocol across all 15 cells; serialized offline GPU diagnostics."""
from pathlib import Path
from types import SimpleNamespace
import argparse,concurrent.futures,hashlib,importlib.util,json,os,shutil,sys,threading,time
CODE_ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE_ROOT));sys.path.insert(0,str(CODE_ROOT/'gpu'))
from urllib.parse import quote
import bench as b
ROOT=b.ROOT
# Avoid Python stdlib queue collision.
s=importlib.util.spec_from_file_location('gpu_queue',CODE_ROOT/'gpu/dispatcher.py');q=importlib.util.module_from_spec(s);s.loader.exec_module(q)
MODELS=[];COHORT=''
GPU_PROMPT='''\nGPU BENCHMARK PROTOCOL: Your Mac shell has no CUDA. A file-only bridge submits a source file to an isolated H200 with Torch 2.14.1+cu130 / Triton 3.8.0 and SGLang. Run `python gpu_test.py solution/submission.py` to queue diagnostic public case 0, or add `--cases 0,1` for at most two public case indices. Poll with the printed command. Do not invoke okc, remote services, package installation, Internet or account tools. GPU tests are serialized fairly across all contestants; queue time counts in the 1,800-second wall-clock cap and is reported separately. The GPU is deleted whenever its queue empties. You may continue CPU analysis while waiting. Budget at most three diagnostic requests, preferably focused on correctness, then the strongest revision. Selected-case speed is not the full target. Preserve all output dtypes and shapes, dynamic fresh graph replay, original precision and TASK rules. Optimizing by input/output caching, skipped work, checker/timer manipulation, monkeypatching evaluator globals or reading private files invalidates a submission. A later full independent held-out check determines the score. Save the final self-contained kernel to solution/submission.py AND create solution/submission.json with claim and assumptions, even if diagnostics are pending. Set done=true only after the final files are saved. Be candid about unverified changes.'''
GPU_PROMPT=GPU_PROMPT.replace('at most three diagnostic requests','at most '+str(q.POLICY['max_diagnostic_requests'])+' diagnostic requests').replace('at most two public case indices','at most '+str(q.POLICY['max_cases_per_request'])+' public case indices')
original_init=b.init_run
original_sample=b.sample
def init(run,snapshot):
 original_init(run,snapshot);shutil.copyfile(CODE_ROOT/'gpu/gpu_test.py',run/'work/gpu_test.py');(run/'work/.gpu').mkdir()
def sample(run,prompt,*args,**kw):return original_sample(run,prompt+GPU_PROMPT,*args,**kw)
b.init_run=init;b.sample=sample

def main():
 global MODELS,COHORT
 p=argparse.ArgumentParser();p.add_argument('--model',action='append',required=True);p.add_argument('--cohort',required=True);p.add_argument('--task',action='append');p.add_argument('--seconds',type=int,default=1800);p.add_argument('--workers',type=int,default=3);a=p.parse_args();MODELS=a.model;COHORT=a.cohort
 b.require_ssd();assert json.loads((ROOT/'preflight.json').read_text())['passed'];assert json.loads((ROOT/'gpu/runtime-prepared.json').read_text())['passed'],'Run gpu/prepare.py before the research clock';tasks=[t for t in json.loads((ROOT/'suite.json').read_text())['tasks'] if t['kind']=='gpu']
 if a.task:tasks=[t for t in tasks if t['id'] in a.task]
 assert tasks,'No selected GPU tasks'
 for t in tasks:assert t['target_status']=='calibrated_private_h200_v1',t['id']
 runs=[(ROOT/'runs'/(COHORT+'-'+quote(m,safe=''))/t['id'],t) for t in tasks for m in MODELS]
 for run,t in runs:assert not run.exists(),'No silent trial replays'
 stop=threading.Event();thread=threading.Thread(target=q.serve,args=(runs,stop),daemon=True);thread.start()
 b.write(ROOT/'kernel-matrix.json',{'status':'running','cohort':COHORT,'models':MODELS,'tasks':[t['id'] for t in tasks],'budget_seconds':a.seconds,'harness':'codex-cli-broker-v1-gpu-queue','started_at':b.utc(),'prompt_sha256':hashlib.sha256((b.SYSTEM_PROMPT(a.seconds)+GPU_PROMPT).encode()).hexdigest(),'concurrency':a.workers,'gpu_diagnostic_policy':q.POLICY,'shared_gpu':'one at a time; wall-clock includes queue wait'})
 try:
  with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
   fs={pool.submit(b.run_task,t,SimpleNamespace(model=m,effort='high',seconds=a.seconds),COHORT+'-'+quote(m,safe='')):(t,m) for t in tasks for m in MODELS}
   for f in concurrent.futures.as_completed(fs):
    t,m=fs[f]
    try:r=f.result();print('KERNEL_ACTOR_COMPLETE',m,t['id'],r['status'],flush=True)
    except Exception as e:print('KERNEL_ACTOR_ERROR',m,t['id'],repr(e),flush=True)
 finally:
  stop.set();thread.join(timeout=1200)
  if thread.is_alive():raise RuntimeError('GPU queue did not drain; cleanup must be checked')
 d=json.loads((ROOT/'kernel-matrix.json').read_text());d.update(status='actors_complete_needs_final_validation',finished_at=b.utc());b.write(ROOT/'kernel-matrix.json',d)
if __name__=='__main__':main()
