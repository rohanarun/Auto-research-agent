"""Curator-only independent replay of the pinned integer reference, before admission."""
from pathlib import Path
import json,subprocess,sys,time
CODE=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE));import bench as b
ROOT=b.ROOT

def main():
 b.require_ssd();run=ROOT/'runs/reference-integer-own';source=ROOT/'curator/reference-integer-mult-frame-retiming';b.init_run(run,source)
 command='python -B research/five-stage-source527-banks/verify.py --output .tmp/immutable-reference-replay';started=time.monotonic()
 try:
  with (run/'replay.log').open('w') as log:
   proc=subprocess.Popen(['/usr/bin/sandbox-exec','-f',str(run/'command.sb'),'/bin/bash','--noprofile','--norc','-c',command],cwd=run/'work',env=b.shell_env(run/'work'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
   try:code=proc.wait(timeout=3600)
   except subprocess.TimeoutExpired:b.kill_group(proc);proc.wait();code=124
  b.write(run/'reference-result.json',{'exit_code':code,'elapsed_seconds':time.monotonic()-started,'finished_at':b.utc(),'tail':(run/'replay.log').read_text()[-3000:]})
  print('REFERENCE_REPLAY',code);return code
 finally:(run/'inference-home/.codex/auth.json').unlink(missing_ok=True)
if __name__=='__main__':sys.exit(main())
