"""One H200 lease for a nonempty batch; export every result and delete in finally."""
from pathlib import Path
import argparse,base64,datetime,hashlib,io,json,os,secrets,shlex,subprocess,sys,tarfile,time
sys.path.insert(0,str(Path(__file__).resolve().parent))
import daytona_control as dc
CODE_ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(CODE_ROOT))
from bench import ROOT
RESULT_IO_SOURCE=Path(__file__).with_name('result_io.py').read_text()

def call(sid,cmd,timeout=45):
 r=dc.execute(sid,cmd,timeout)
 if not r.get('ok'):raise RuntimeError(str(r))
 return r['data']
def quote(s):return shlex.quote(str(s))
def run_job(sid,problem,candidate,jobid,uid,seconds=600,cases=None,rounds=5,cancel_event=None):
 probe=call(sid,'python3 -c '+quote("import pathlib,sys,os,json; print(json.dumps({'home':str(pathlib.Path.home()),'python':sys.executable,'uid':os.geteuid(),'env':{k:os.environ[k] for k in ['PATH','LD_LIBRARY_PATH','CUDA_HOME','CUDA_PATH'] if k in os.environ}}))"))
 runtime=json.loads(probe['result']);assert runtime['uid']==0,'Trusted GPU coordinator must be root';assert 10000<=uid<60000;work='/home/auto-research-benchmark/jobs/'+jobid
 files={'candidate.py':Path(candidate).read_bytes(),'judge.py':(CODE_ROOT/'gpu/judge.py').read_bytes(),'offline_guard.py':(CODE_ROOT/'gpu/offline_guard.py').read_bytes(),'landlock_guard.py':(CODE_ROOT/'gpu/landlock_guard.py').read_bytes()}
 spec=json.loads((Path(__file__).resolve().parent/'specs'/(problem+'.json')).read_text())
 for name,source in spec['files'].items():
  if name!='template.py':files[name]=source.encode()
 files['problem.json']=json.dumps({'benchmarks':spec['benchmarks']}).encode()
 buf=io.BytesIO()
 with tarfile.open(fileobj=buf,mode='w:gz') as tar:
  for name,data in files.items():
   info=tarfile.TarInfo(name);info.size=len(data);info.mode=0o600;tar.addfile(info,io.BytesIO(data))
 mkdir=call(sid,'python3 -c '+quote('from pathlib import Path;p=Path('+repr(work)+');p.mkdir(parents=True,exist_ok=False);p.chmod(0o700)'))
 if mkdir.get('exitCode',0):raise RuntimeError('GPU workspace creation failed '+str(mkdir)[:500])
 upload=dc.upload(sid,work+'/bundle.tar.gz',buf.getvalue())
 if not upload.get('ok'):raise RuntimeError('GPU file upload failed '+str(upload)[:1000])
 script='import tarfile,pathlib,os; p=pathlib.Path('+repr(work)+');t=tarfile.open(p/"bundle.tar.gz");t.extractall(p,filter="data");(p/"bundle.tar.gz").unlink();(p/"home").mkdir(mode=0o700);cache=pathlib.Path("/tmp/ar-public-cache.tar.gz");tarfile.open(cache).extractall(p/"home/.cache/sglang/jit",filter="data") if cache.exists() else None;[(f.chmod(0o700 if f.is_dir() else 0o600),os.chown(f,'+str(uid)+','+str(uid)+')) for f in p.rglob("*")];os.chown(p,'+str(uid)+','+str(uid)+')'
 extraction=call(sid,'python3 -c '+quote(script))
 if extraction.get('exitCode',0):raise RuntimeError('GPU bundle extraction failed '+str(extraction)[:1000])
 seed=secrets.randbelow(2**30)
 environment={**runtime['env'],'HOME':work+'/home','TMPDIR':work+'/home','XDG_CACHE_HOME':work+'/home/.cache','TRITON_CACHE_DIR':work+'/home/.triton','CUDA_VISIBLE_DEVICES':'0','PYTHONDONTWRITEBYTECODE':'1','USER':'benchmark','LOGNAME':'benchmark'}
 arguments=[runtime['python'],'judge.py','--problem',problem,'--seed',str(seed),'--rounds',str(rounds)]
 if cases is not None:arguments.extend(['--cases',','.join(str(i) for i in cases)])
 if jobid.startswith('reference-'):arguments.append('--trusted-reference')
 supervisor='import os,subprocess,ctypes;assert ctypes.CDLL(None).prctl(4,0,0,0,0)==0;result=subprocess.run('+repr(arguments)+');fd=os.open("exitcode",os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600);os.write(fd,str(result.returncode).encode());os.close(fd)'
 bootstrap='import os,ctypes;os.chdir('+repr(work)+');os.setgroups([]);os.setgid('+str(uid)+');os.setuid('+str(uid)+');assert ctypes.CDLL(None).prctl(38,1,0,0,0)==0;os.execve('+repr(runtime['python'])+','+repr([runtime['python'],'-c',supervisor])+','+repr(environment)+')'
 command=quote(runtime['python'])+' -c '+quote(bootstrap)
 launch='nohup '+command+' > '+quote(work+'/run.log')+' 2>&1 </dev/null & echo $!'
 r=call(sid,launch);started=time.monotonic();out=ROOT/'gpu/results'/jobid;out.mkdir(parents=True,exist_ok=False)
 (out/'intent.json').write_text(json.dumps({'sandbox_id':sid,'problem':problem,'candidate_sha256':hashlib.sha256(Path(candidate).read_bytes()).hexdigest(),'uid':uid,'seed':seed,'remote_work':work,'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'launch':r},indent=2)+'\n')
 while time.monotonic()-started<seconds:
  if cancel_event is not None and cancel_event.is_set():
   (out/'cancelled.json').write_text(json.dumps({'status':'cancelled_after_research_clock','reason':'No actors can consume additional diagnostic feedback.'})+'\n');raise InterruptedError('Research clocks ended; cancel unused diagnostic GPU work')
  script=RESULT_IO_SOURCE+'\nimport json\np='+repr(work)+'\nuid='+str(uid)+'''
try:exit_code=int(read_job_file(p,'exitcode',[uid],128))
except FileNotFoundError:r={'done':False}
else:
 try:result=json.loads(read_job_file(p,'result.json',[uid],8*1024*1024))
 except FileNotFoundError:result=None
 log=read_job_file(p,'run.log',[0,uid],16*1024*1024,tail=4000).decode(errors='replace')
 r={'done':True,'exit_code':exit_code,'log':log,'result':result}
print(json.dumps(r))
'''
  r=call(sid,'python3 -c '+quote(script));text=r.get('result','')
  try:state=json.loads(text)
  except Exception:raise RuntimeError('Malformed GPU job observation '+str(r)[:800])
  if state['done']:
   (out/'result.json').write_text(json.dumps(state,indent=2)+'\n');print('GPU_JOB_DONE',problem,jobid,state.get('exit_code'),state.get('result',{}).get('status') if state.get('result') else None,flush=True);return state
  time.sleep(3)
 raise TimeoutError('GPU job deadline exceeded; entire benchmark lease will be deleted')
