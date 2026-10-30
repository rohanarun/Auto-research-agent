"""Trusted host queue; only ready jobs justify an H200 lease. No agent credentials leave host."""
from pathlib import Path
import datetime,hashlib,json,os,secrets,stat,sys,time,traceback
sys.path.insert(0,str(Path(__file__).resolve().parent))
import session as gpu
ROOT=gpu.ROOT
POLICY_PATH=Path(os.environ.get('AUTO_RESEARCH_GPU_POLICY',str(Path(__file__).with_name('research-policy.json'))))
POLICY=json.loads(POLICY_PATH.read_text())
assert all(type(POLICY[k]) is int and POLICY[k]>0 for k in ['max_diagnostic_requests','max_cases_per_request','diagnostic_deadline_seconds','diagnostic_rounds'])

def save(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
def read_request_file(directory,name,limit):
 fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  file=os.open(name,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK,dir_fd=fd)
  try:
   info=os.fstat(file);assert stat.S_ISREG(info.st_mode) and info.st_nlink==1 and info.st_size<=limit,'Invalid request file type/size'
   with os.fdopen(file,'rb',closefd=False) as stream:return stream.read(limit+1)
  finally:os.close(file)
 finally:os.close(fd)
def feedback(directory,name,value):
 fd=os.open(directory,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
 try:
  temporary='.feedback-'+secrets.token_hex(16)
  file=os.open(temporary,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600,dir_fd=fd)
  with os.fdopen(file,'w') as stream:json.dump(value,stream,indent=2);stream.write('\n')
  os.replace(temporary,name,src_dir_fd=fd,dst_dir_fd=fd)
 finally:os.close(fd)

class Lease:
 def __init__(self):self.sid=None;self.directory=None
 def start(self):
  stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+secrets.token_hex(3)
  self.directory=ROOT/'gpu'/('lease-'+stamp);self.directory.mkdir(parents=True,exist_ok=False)
  image_data=json.loads((Path(__file__).resolve().parent/'runtime-image.json').read_text());image='lmsysorg/sglang@'+next(i['digest'] for i in image_data['images'] if i['architecture']=='amd64')
  created=gpu.dc.create('ar-bench-'+stamp,image);self.sid=created['id'];save(self.directory/'created.json',created);print('GPU_QUEUE_CREATED',self.sid,flush=True)
  began=time.monotonic()
  while True:
   r=gpu.dc.request('control','/sandbox/'+self.sid);state=r.get('data',{});save(self.directory/'status.json',r)
   if state.get('state')=='started':break
   if state.get('state') in ['error','destroyed','stopped'] or time.monotonic()-began>600:raise RuntimeError('GPU readiness failure '+str(state.get('state')))
   time.sleep(3)
  assert state.get('gpuType')=='H200' and state.get('networkBlockAll') is True
  smoke=gpu.call(self.sid,'/sbin/ldconfig && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader',30);save(self.directory/'smoke.json',smoke)
  if smoke.get('exitCode',0):raise RuntimeError('GPU runtime infrastructure failed')
  cache=ROOT/'gpu/public-sglang-cache.tar.gz'
  if cache.exists():
   r=gpu.dc.upload(self.sid,'/tmp/ar-public-cache.tar.gz',cache.read_bytes())
   if not r.get('ok'):raise RuntimeError('Public dependency cache upload failed')
   save(self.directory/'public-cache.json',{'sha256':hashlib.sha256(cache.read_bytes()).hexdigest(),'bytes':cache.stat().st_size,'source':'Public SGLang dependency JIT cache; private optimized kernels and other contestant caches excluded.'})
 def close(self):
  if not self.sid:return
  save(self.directory/'deleted.json',gpu.dc.delete(self.sid))
  for _ in range(20):
   r=gpu.dc.request('control','/sandbox/'+self.sid)
   if r.get('http_status')==404 or r.get('data',{}).get('state')=='destroyed':break
   time.sleep(2)
  save(self.directory/'deletion-confirmed.json',r);print('GPU_QUEUE_DELETED',self.sid,r.get('http_status',r.get('data',{}).get('state')),flush=True)
  if r.get('http_status')!=404 and r.get('data',{}).get('state')!='destroyed':raise RuntimeError('GPU deletion not confirmed')
  self.sid=None

def requests(runs):
 found=[]
 for run,task in runs:
  work=run/'work';queue=work/'.gpu'
  metadata=run/'result.json'
  if metadata.exists():
   try:
    if json.loads(metadata.read_text()).get('status')!='running':continue
   except json.JSONDecodeError:continue
  if not queue.exists() or queue.is_symlink():continue
  for f in queue.glob('*.request.json'):
   job=f.name.split('.')[0]
   if (queue/(job+'.result.json')).exists() or (ROOT/'gpu/queue-claims'/(job+'.json')).exists():continue
   try:found.append((f.stat().st_mtime,run,task,f,job))
   except FileNotFoundError:continue
 return sorted(found,key=lambda x:x[0])

def serve(runs,stop):
 lease=Lease();counter=0
 try:
  while not stop.is_set():
   pending=requests(runs)
   if not pending:
    lease.close();stop.wait(1);continue
   _,run,task,f,job=pending[0];dest=f.parent/(job+'.result.json');claim=ROOT/'gpu/queue-claims'/(job+'.json')
   try:
    assert len(job)==32 and all(c in '0123456789abcdef' for c in job)
    assert not f.parent.is_symlink() and f.parent.resolve().is_relative_to(run/'work')
    existing=list((ROOT/'gpu/queue-claims').glob('*.json')) if (ROOT/'gpu/queue-claims').exists() else []
    own=sum(json.loads(p.read_text()).get('run')==str(run) for p in existing);assert own<POLICY['max_diagnostic_requests'],'Diagnostic request budget exhausted'
    req=json.loads(read_request_file(f.parent,f.name,5000));data=read_request_file(f.parent,job+'.candidate.py',2_000_000);assert hashlib.sha256(data).hexdigest()==req['sha256']
    cases=req.get('cases',[0]);spec=json.loads((Path(__file__).resolve().parent/'specs'/(task['problem']+'.json')).read_text())
    assert isinstance(cases,list) and 1<=len(cases)<=POLICY['max_cases_per_request'] and all(type(i) is int and 0<=i<len(spec['benchmarks']) for i in cases),'Diagnostic case-count budget exhausted'
    staged=ROOT/'gpu/staged'/(job+'.py');staged.parent.mkdir(exist_ok=True);staged.write_bytes(data)
    save(claim,{'job_id':job,'run':str(run),'problem':task['problem'],'request':req,'claimed_at':time.time()})
    if not lease.sid:lease.start()
    counter+=1;jobid='diagnostic-'+task['problem']+'-'+job
    gpu_started=time.time();state=gpu.run_job(lease.sid,task['problem'],staged,jobid,10000+counter,seconds=POLICY['diagnostic_deadline_seconds'],cases=cases,rounds=POLICY['diagnostic_rounds'],cancel_event=stop)
    value=state.get('result')
    if value is None:raise RuntimeError('No GPU evaluator result: '+state.get('log','')[-1500:])
    value.update(job_id=job,scope='diagnostic_only',queue_wait_seconds=round(gpu_started-req['created_at_unix'],3),gpu_job_elapsed_seconds=round(time.time()-gpu_started,3),queue_wait_basis='Same-host bridge timestamp; queue wait is separate from execution and does not determine admission',note='Public selected-case feedback; a full independent hidden-bank final check is required for a benchmark pass.')
    feedback(f.parent,dest.name,value)
   except Exception as e:
    save(claim,{'job_id':job,'error':str(e),'traceback':traceback.format_exc()});feedback(f.parent,dest.name,{'job_id':job,'status':'cancelled_after_research_clock' if isinstance(e,InterruptedError) else 'infrastructure_error','error':str(e),'scope':'diagnostic_only'})
    lease.close()
 finally:lease.close()
