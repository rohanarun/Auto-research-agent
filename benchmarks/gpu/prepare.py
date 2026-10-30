"""Curator-only public dependency warmup and actual confined-worker preflight."""
from pathlib import Path
import datetime,hashlib,importlib.util,json,sys
sys.path.insert(0,str(Path(__file__).resolve().parent));import session as g
s=importlib.util.spec_from_file_location('gpu_dispatcher',Path(__file__).with_name('dispatcher.py'));d=importlib.util.module_from_spec(s);s.loader.exec_module(d)
CODE=Path(__file__).resolve().parents[1];ROOT=g.ROOT

def main():
 tasks=[t for t in json.loads((ROOT/'suite.json').read_text())['tasks'] if t['kind']=='gpu'];assert tasks,'Prepare at least one kernel scenario first'
 lease=d.Lease();jobs=[];stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
 try:
  lease.start()
  for i,task in enumerate(tasks):
   job='runtime-warmup-'+task['problem']+'-'+stamp;jobs.append(job)
   candidate=ROOT/'tasks'/task['id']/'snapshot/baseline.py';r=g.run_job(lease.sid,task['problem'],candidate,job,40000+i,seconds=900,cases=[0],rounds=2)
   if r.get('result',{}).get('status')!='pass':raise RuntimeError('Public baseline/boundary preflight failed: '+str(r)[-1800:])
  script='''from pathlib import Path
import json,tarfile
jobs=JOBS
files={}
for name in jobs:
 cache=Path('/home/auto-research-benchmark/jobs')/name/'home/.cache/sglang/jit'
 for lib in cache.rglob('*.so'):
  if not lib.parent.name.startswith('deps-') or not lib.parent.parent.parent.name.startswith('sgl_kernel_jit_'):continue
  manifest=lib.parent/'sgl_deps.json'
  if not manifest.exists():continue
  if not all(row[0] in ['kernels','sglang','sys','toolkit','tvm_ffi'] for row in json.loads(manifest.read_text())):raise RuntimeError('Non-public dependency namespace')
  for f in lib.parent.iterdir():
   if f.is_file() and not f.is_symlink():files[str(f.relative_to(cache))]=f
with tarfile.open('/tmp/ar-public-cache-export.tar.gz','w:gz') as archive:
 for name,path in files.items():archive.add(path,arcname=name)
print(json.dumps({'files':len(files)}))
'''.replace('JOBS',repr(jobs))
  r=g.call(lease.sid,'python3 -c '+g.quote(script),45)
  if r.get('exitCode',0):raise RuntimeError('Public cache export failed')
  data=g.dc.download(lease.sid,'/tmp/ar-public-cache-export.tar.gz');(ROOT/'gpu/public-sglang-cache.tar.gz').write_bytes(data)
  report={'passed':True,'completed_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'jobs':jobs,'cache_sha256':hashlib.sha256(data).hexdigest(),'cache_bytes':len(data),'scope':'Public frozen baseline case0 only, actual unprivileged GPU and network guard; no private reference or actor caches.'};(ROOT/'gpu/runtime-prepared.json').write_text(json.dumps(report,indent=2)+'\n');print('GPU_RUNTIME_PREPARED',len(jobs),'public baseline checks')
 finally:lease.close()
if __name__=='__main__':main()
