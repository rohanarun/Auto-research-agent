"""Owned cloud process and checkpoint bridge. No Daytona API key enters this file."""
from pathlib import Path
import os,sys,json,time,uuid,subprocess,signal,base64,hashlib,tarfile,stat,re,urllib.request
R=Path('/research-private');CO=Path('/workspace/checkout');CLI='/opt/autoresearch/bin/codex'
def read(p,default=None):
 try:return json.loads(p.read_text())
 except FileNotFoundError:return default
def write(p,x):
 p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.incoming');q.write_text(json.dumps(x));q.chmod(0o600);q.replace(p)
def safe_id(x):
 assert re.fullmatch(r'[A-Za-z0-9_-]+',x);return x
def safe_env(state):
 home=state.parent/'home';home.mkdir(parents=True,exist_ok=True);(CO/'tmp').mkdir(parents=True,exist_ok=True)
 return {'PATH':'/opt/autoresearch/bin:/usr/local/bin:/usr/bin:/bin','HOME':str(home),'CODEX_HOME':str(state),'TMPDIR':str(CO/'tmp'),'PYTHONDONTWRITEBYTECODE':'1','GIT_CONFIG_NOSYSTEM':'1','GIT_CONFIG_GLOBAL':'/dev/null','LANG':'C.UTF-8','NO_COLOR':'1'}
def alive(pid):
 try:return Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()[0]!='Z'
 except (OSError,IndexError):return False
def terminate(proc):
 rows=subprocess.check_output(['ps','-eo','pid=,ppid=,sid='],text=True);pairs=[tuple(map(int,l.split())) for l in rows.splitlines()];children=[]
 def walk(pid):
  for child,parent,sid in pairs:
   if parent==pid:walk(child);children.append(child)
 walk(proc.pid)
 children=list(set(children+[pid for pid,parent,sid in pairs if sid==proc.pid and pid!=proc.pid]))
 for sig in [signal.SIGTERM,signal.SIGKILL]:
  for pid in children+[proc.pid]:
   if alive(pid):
    try:os.kill(pid,sig)
    except ProcessLookupError:pass
  end=time.monotonic()+2
  while any(alive(pid) for pid in children+[proc.pid]) and time.monotonic()<end:time.sleep(.05)
 return {'remaining_pids':[pid for pid in children+[proc.pid] if alive(pid)]}
def spawn(action,identifier):
 folder=R/'operations'/safe_id(identifier);folder.mkdir(parents=True,exist_ok=True)
 with (folder/'launcher.log').open('ab') as log:
  proc=subprocess.Popen([sys.executable,__file__,action,identifier],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 write(folder/'launch.json',{'pid':proc.pid,'started_unix':time.time()});return {'pid':proc.pid,'id':identifier}
def setup(request):
 identifier=safe_id(request['id']);folder=R/'operations'/identifier;old=read(folder/'request.json')
 if old is not None:
  assert old==request;return read(folder/'status.json',{'phase':'requested'})
 write(folder/'request.json',request);return spawn('setup-run',identifier)
def setup_run(identifier):
 folder=R/'operations'/identifier;request=read(folder/'request.json');write(folder/'status.json',{'phase':'preparing'})
 try:
  package=request['codex_package'];assert re.fullmatch(r'@openai/codex(?:@[A-Za-z0-9.+_-]+)?',package)
  runtime=read(R/'runtime.json')
  if not Path(CLI).exists():
   import platform,shutil
   runtime=request.get('runtime')
   if not runtime:
    version=package.split('@')[-1] if package!='@openai/codex' else 'latest'
    endpoint='https://api.github.com/repos/openai/codex/releases/'+('latest' if version=='latest' else 'tags/rust-v'+version)
    with urllib.request.urlopen(urllib.request.Request(endpoint,headers={'User-Agent':'AutoResearch'}),timeout=90) as f:release=json.load(f)
    arch={'x86_64':'x86_64','aarch64':'aarch64'}[platform.machine()];name='codex-package-'+arch+'-unknown-linux-musl.tar.gz'
    asset=next(x for x in release['assets'] if x['name']==name)
    runtime={'url':asset['browser_download_url'],'digest':asset['digest'],'release':release['tag_name']}
   assert runtime['url'].startswith('https://github.com/openai/codex/releases/download/') and runtime['digest'].startswith('sha256:')
   bundle=R/'codex-package.tar.gz';h=hashlib.sha256()
   with urllib.request.urlopen(runtime['url'],timeout=90) as f,bundle.open('wb') as out:
    for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk);out.write(chunk)
   assert h.hexdigest()==runtime['digest'].split(':',1)[1],'Official CLI asset hash mismatch'
   install=Path('/opt/autoresearch');install.mkdir(parents=True,exist_ok=True)
   with tarfile.open(bundle,'r:gz') as tf:
    for m in tf:
     target=(install/m.name).resolve();assert target.is_relative_to(install.resolve()) and (m.isfile() or m.isdir())
     tf.extract(m,install,filter='data')
   found=list(install.rglob('codex'));assert len(found)==1
   if found[0]!=Path(CLI):shutil.copy2(found[0],CLI)
   Path(CLI).chmod(0o755);write(R/'runtime.json',runtime)
  if not (CO/'.git').exists():
   if request.get('repository'):
    repository=request['repository'];assert re.fullmatch(r'[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+',repository) and all(p not in ['.','..'] for p in repository.split('/'))
    CO.parent.mkdir(parents=True,exist_ok=True);subprocess.run(['git','clone','--','https://github.com/'+repository+'.git',str(CO)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
   else:
    CO.mkdir(parents=True,exist_ok=True)
    if request.get('source_archive'):
     with tarfile.open(R/'source.tar.gz','r:gz') as tf:
      for m in tf:
       target=(CO/m.name).resolve();assert target.is_relative_to(CO.resolve())
       if m.issym():assert (target.parent/m.linkname).resolve().is_relative_to(CO.resolve())
       assert not m.islnk() and (m.isfile() or m.isdir() or m.issym())
       tf.extract(m,CO,filter='data')
    subprocess.run(['git','init','-q',str(CO)],check=True)
    subprocess.run(['git','-C',str(CO),'add','.'],check=True)
    subprocess.run(['git','-C',str(CO),'-c','user.name=Auto Research','-c','user.email=autoresearch@localhost','-c','commit.gpgsign=false','-c','core.hooksPath=/dev/null','commit','--allow-empty','-qm','Private cloud research snapshot'],check=True)
  (R/'auth').mkdir(parents=True,exist_ok=True);(R/'connection').mkdir(parents=True,exist_ok=True)
  version=subprocess.check_output([CLI,'--version'],text=True).strip()
  write(folder/'status.json',{'phase':'ready','cli_version':version,'runtime':runtime,'checkout':str(CO),'completed_unix':time.time()})
 except BaseException as exc:write(folder/'status.json',{'phase':'failed','error_class':type(exc).__name__,'message':'Cloud setup failed; installation diagnostics and checkout retained.'});raise
def launch(request):
 identifier=safe_id(request['id']);folder=R/'operations'/identifier;old=read(folder/'request.json')
 if old is not None:
  assert old==request;return read(folder/'status.json',{'phase':'requested'})
 write(folder/'request.json',request);return spawn('run',identifier)
def run(identifier):
 folder=R/'operations'/identifier;request=read(folder/'request.json');state=R/'connection' if request.get('login') else R/'providers'/safe_id(request['state_id'])/'state';state.mkdir(parents=True,exist_ok=True)
 if not request.get('login') and not (state/'auth.json').exists():
  import shutil
  previous=request.get('previous_state_id');source=R/'providers'/safe_id(previous)/'state/auth.json' if previous else R/'auth/auth.json'
  assert source.is_file(),'Cloud Codex sign-in is required';shutil.copy2(source,state/'auth.json');(state/'auth.json').chmod(0o600)
 args=[CLI,'--no-daemon']+request['arguments'];env=safe_env(state);started=time.time();write(folder/'status.json',{'phase':'running','started_unix':started,'state_id':request.get('state_id')})
 try:
  with (folder/'stdout').open('wb') as out,(folder/'stderr').open('wb') as err:
   proc=subprocess.Popen(args,cwd=CO,env=env,stdin=subprocess.PIPE,stdout=out,stderr=err,start_new_session=True)
   proc.stdin.write(request.get('input','').encode());proc.stdin.close();write(folder/'cli.json',{'pid':proc.pid,'started_unix':time.time()});cleanup=None
   while proc.poll() is None:
    if (folder/'stop.json').exists():cleanup=terminate(proc);break
    time.sleep(.1)
   code=proc.wait()
   if cleanup is None:cleanup=terminate(proc)
  write(folder/'status.json',{'phase':'ended','exit_code':code,'started_unix':started,'ended_unix':time.time(),'cleanup':cleanup,'state_id':request.get('state_id')})
 except BaseException as exc:
  if 'proc' in locals() and proc.poll() is None:terminate(proc)
  write(folder/'status.json',{'phase':'failed','exit_code':1,'error_class':type(exc).__name__,'message':'Owned CLI invocation failed; all run files retained.'});raise
def poll(request):
 folder=R/'operations'/safe_id(request['id']);out={};limit=int(request['limit']);assert limit>0
 for name in ['stdout','stderr']:
  path=folder/name;offset=int(request.get(name+'_offset',0));raw=b''
  if path.exists():
   with path.open('rb') as f:f.seek(offset);raw=f.read(limit)
  out[name]=base64.b64encode(raw).decode();out[name+'_offset']=offset+len(raw);out[name+'_eof']=not path.exists() or offset+len(raw)==path.stat().st_size
 out['status']=read(folder/'status.json',{'phase':'requested'});return out
def stop(request):
 folder=R/'operations'/safe_id(request['id']);write(folder/'stop.json',{'unix':time.time()});pid=read(folder/'launch.json',{}).get('pid')
 while alive(pid):time.sleep(.1)
 return read(folder/'status.json',{'phase':'not_started'})
def auth_status():
 state=R/'connection';state.mkdir(exist_ok=True,parents=True);p=subprocess.run([CLI,'--no-daemon','login','status'],env=safe_env(state),cwd=CO,capture_output=True,text=True)
 return {'authenticated':p.returncode==0,'description':(p.stdout+p.stderr).strip()}
def fork_auth(request):
 url=request['url'];assert url.startswith('https://')
 auth=(R/'connection/auth.json').read_bytes();encoded=base64.b64encode(auth).decode()
 # Credential data stays cloud-to-cloud; only a non-secret receipt returns to Mac.
 code='import pathlib,base64,os;p=pathlib.Path("/research-private/auth/auth.json");p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode('+repr(encoded)+'));p.chmod(0o600)'
 import shlex
 body=json.dumps({'command':'sudo -n python3 -c '+shlex.quote(code),'timeout':60}).encode()
 req=urllib.request.Request(url.rstrip('/')+'/process/execute',data=body,headers={'Content-Type':'application/json','X-Daytona-Preview-Token':request['token']})
 class NoRedirect(urllib.request.HTTPRedirectHandler):
  def redirect_request(self,*args,**kwargs):raise RuntimeError('Private cloud credential redirects are forbidden')
 with urllib.request.build_opener(NoRedirect()).open(req,timeout=75) as f:response=json.load(f)
 assert response['exitCode']==0;return {'copied':True,'host_auth_read':False,'provider_history_copied':False}
def checkpoint():
 def contents(p):
  if not p.exists():return None
  assert p.is_file() and p.resolve().is_relative_to(CO.resolve())
  return base64.b64encode(p.read_bytes()).decode()
 return {'checkpoint':contents(CO/'RESEARCH_CHECKPOINT.md'),'notes':contents(CO/'RESEARCH_HANDOFF.md'),'topics':{str(p.relative_to(CO/'RESEARCH_CONTEXT')):contents(p) for p in sorted((CO/'RESEARCH_CONTEXT').rglob('*.md'))}}
def put_context(request):
 root=CO/'SHARED_CONTEXT'/safe_id(request['round']);root.mkdir(parents=True,exist_ok=True)
 for name,value in request['files'].items():
  p=(root/name).resolve();assert p.is_relative_to(root.resolve());p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(base64.b64decode(value))
 return {'written':len(request['files'])}
def manifest():
 result={}
 for root in [CO,R/'providers',R/'operations']:
  if not root.exists():continue
  for p in [root,*sorted(root.rglob('*'))]:
   if p.name=='auth.json' and p.parent.name=='state':continue
   st=p.lstat();name=str(p.relative_to('/'))
   if p.is_symlink():v={'kind':'symlink','target':os.readlink(p)}
   elif p.is_dir():v={'kind':'dir'}
   elif p.is_file():
    h=hashlib.sha256()
    with p.open('rb') as f:
     for b in iter(lambda:f.read(1048576),b''):h.update(b)
    v={'kind':'file','size':st.st_size,'sha256':h.hexdigest()}
   else:raise RuntimeError('Unsupported recovery entry')
   result[name]=v
 return result
def pack():
 export=R/'export';export.mkdir(exist_ok=True);entries=manifest();write(export/'manifest.json',{'entries':entries,'all_provider_credentials_excluded':True})
 with tarfile.open(export/'checkout-state.tar.gz','w:gz') as tf:
  for name in entries:tf.add('/'+name,arcname=name,recursive=False)
 p=export/'checkout-state.tar.gz';h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 receipt={'sha256':h.hexdigest(),'bytes':p.stat().st_size,'entries':len(entries),'manifest_bytes':(export/'manifest.json').stat().st_size};write(export/'receipt.json',receipt);return receipt
def export_chunk(request):
 assert request['file'] in ['checkout-state.tar.gz','manifest.json'];p=R/'export'/request['file']
 with p.open('rb') as f:f.seek(int(request['offset']));raw=f.read(int(request['limit']))
 return {'data':base64.b64encode(raw).decode(),'offset':int(request['offset'])+len(raw),'eof':int(request['offset'])+len(raw)==p.stat().st_size}
if __name__=='__main__':
 action=sys.argv[1]
 if action in ['setup-run','run']:
  try:(setup_run if action=='setup-run' else run)(sys.argv[2])
  except BaseException as exc:
   folder=R/'operations'/safe_id(sys.argv[2]);old=read(folder/'status.json',{})
   if old.get('phase') not in ['ended','failed']:write(folder/'status.json',{'phase':'failed','exit_code':1,'error_class':type(exc).__name__,'message':'Owned operation failed; inspect retained diagnostics.'})
   raise
 else:
  request=json.load(sys.stdin)
  if action=='setup':result=setup(request)
  elif action=='source-chunk':
   p=R/'source.tar.gz';p.parent.mkdir(parents=True,exist_ok=True);offset=int(request['offset']);data=base64.b64decode(request['data'])
   if p.exists() and p.stat().st_size>offset:
    with p.open('rb') as f:f.seek(offset);assert f.read(len(data))==data
   else:
    assert offset==(p.stat().st_size if p.exists() else 0)
    with p.open('ab') as f:f.write(data)
    p.chmod(0o600)
   result={'offset':offset+len(data)}
  elif action=='setup-status':result=read(R/'operations'/safe_id(request['id'])/'status.json',{'phase':'requested'})
  elif action=='launch':result=launch(request)
  elif action=='poll':result=poll(request)
  elif action=='stop':result=stop(request)
  elif action=='auth-status':result=auth_status()
  elif action=='fork-auth':result=fork_auth(request)
  elif action=='checkpoint':result=checkpoint()
  elif action=='put-context':result=put_context(request)
  elif action=='pack':result=pack()
  elif action=='export-chunk':result=export_chunk(request)
  else:raise RuntimeError('Unknown owned action')
  print(json.dumps(result))
