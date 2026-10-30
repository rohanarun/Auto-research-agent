#!/usr/bin/env python3
"""Curator-only preparation. Downloads happen before any offline trial."""
from pathlib import Path
import argparse,concurrent.futures,hashlib,io,json,os,shutil,subprocess,sys,tarfile,urllib.request,venv
CODE=Path(__file__).resolve().parent

def fetch(url):
 req=urllib.request.Request(url,headers={'User-Agent':'auto-research-benchmark-setup','Accept':'application/vnd.github+json'})
 with urllib.request.urlopen(req,timeout=120) as r:return r.read()
def export(repo,sha,dest):
 archive=fetch('https://api.github.com/repos/'+repo+'/tarball/'+sha)
 with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
  first=tar.getmembers()[0].name.split('/')[0]
  for member in tar:
   parts=Path(member.name).parts
   if not parts or parts[0]!=first:raise ValueError('Unexpected archive prefix')
   relative=Path(*parts[1:])
   if not relative.parts:continue
   if relative.parts[0] in ['.git','.codex']:continue
   if not (member.isdir() or member.isfile()) or not (dest/relative).resolve().is_relative_to(dest.resolve()):raise ValueError('Unsafe archive member')
   member.name=str(relative);tar.extract(member,dest,filter='data')
def prefetch(snapshot):
 manifest=snapshot/'source-manifest.json'
 if not manifest.exists():return
 m=json.loads(manifest.read_text())
 for f in m['files']:
  relative=Path(f['path'])
  if relative.is_absolute() or '..' in relative.parts:raise ValueError('Invalid source manifest')
  p=snapshot/'upstream'/relative;p.parent.mkdir(parents=True,exist_ok=True)
  data=fetch('https://raw.githubusercontent.com/openai/math/'+m['commit']+'/'+f['path'])
  if hashlib.sha256(data).hexdigest()!=f['sha256']:raise ValueError('Upstream source hash mismatch')
  p.write_bytes(data)
def treehash(root):
 rows=[[str(p.relative_to(root)),hashlib.sha256(p.read_bytes()).hexdigest()] for p in sorted(root.rglob('*')) if p.is_file()]
 return hashlib.sha256(json.dumps(rows,separators=(',',':')).encode()).hexdigest(),rows

def main():
 p=argparse.ArgumentParser();p.add_argument('--workspace',type=Path,default=Path(os.environ.get('AUTO_RESEARCH_WORKSPACE',CODE/'workspace')));p.add_argument('--task',action='append');p.add_argument('--no-dependencies',action='store_true');a=p.parse_args();root=a.workspace.expanduser().resolve()
 if any(s in root.parts for s in ['Documents','Desktop','Mobile Documents']):raise SystemExit('Use a non-synced workspace outside Documents/Desktop/iCloud')
 mount=os.environ.get('AUTO_RESEARCH_REQUIRE_MOUNT')
 if mount and not Path(mount).is_mount():raise SystemExit('Required volume is not mounted')
 root.mkdir(parents=True,exist_ok=True)
 suite=json.loads((CODE/'scenarios.json').read_text());tasks=suite['tasks']
 if a.task:tasks=[t for t in tasks if t['id'] in a.task]
 for d in ['curator','tasks','runs','tests/empty']:(root/d).mkdir(parents=True,exist_ok=True)
 for task in tasks:
  dest=root/'tasks'/task['id'];snapshot=dest/'snapshot';dest.mkdir(exist_ok=True)
  if not snapshot.exists():
   snapshot.mkdir()
   if task['kind']=='gpu':shutil.copytree(CODE/'data/kernel-starters'/task['problem'],snapshot,dirs_exist_ok=True)
   else:export(task['repo'],task['base_sha'],snapshot);prefetch(snapshot)
  actual,rows=treehash(snapshot)
  if actual!=task['snapshot_sha256']:raise ValueError('Frozen snapshot hash mismatch: '+task['id'])
  (dest/'task.json').write_text(json.dumps(task,indent=2)+'\n');(dest/'snapshot-hashes.json').write_text(json.dumps(rows,indent=2)+'\n')
  if task['kind']!='gpu':
   ref=root/'curator'/('reference-'+task['id'])
   if not ref.exists():ref.mkdir();export(task['repo'],task['breakthrough_sha'],ref)
  print('Prepared',task['id'],actual,flush=True)
 (root/'suite.json').write_text(json.dumps({**suite,'tasks':tasks},indent=2)+'\n')
 (root/'curator/isolation-canary.txt').write_text('This curator file must never be visible to a contestant.')
 if not a.no_dependencies:
  environment=root/'.venv'
  if not environment.exists():venv.EnvBuilder(with_pip=True).create(environment)
  subprocess.run([str(environment/'bin/python'),'-m','pip','install','--cache-dir',str(root/'.cache/pip'),'sympy==1.14.0','numpy==2.0.2','scipy==1.13.1'],check=True)
 if sys.platform=='darwin' and not a.no_dependencies:
  developer=subprocess.check_output(['xcode-select','-p'],text=True).strip()
  choices=[Path(developer)/'Toolchains/XcodeDefault.xctoolchain/usr/bin',Path(developer)/'usr/bin']
  binaries=next((p for p in choices if (p/'clang').is_file()),None)
  if binaries is None:raise RuntimeError('Developer compiler directory not found; set up Xcode/Command Line Tools')
  for name in ['cc','c++','clang','clang++','ar','ranlib']:
   target=binaries/name;link=root/'.venv/bin'/name
   if target.exists() and not link.exists():link.symlink_to(target)
 print('Set AUTO_RESEARCH_WORKSPACE='+str(root)+' before running the benchmark.')
if __name__=='__main__':main()
