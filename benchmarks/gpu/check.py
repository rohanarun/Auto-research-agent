"""Standalone full held-out evaluation of a sealed kernel on a fresh short-lived H200."""
from pathlib import Path
import argparse,datetime,importlib.util,json,secrets,sys
sys.path.insert(0,str(Path(__file__).resolve().parent));import session as g
s=importlib.util.spec_from_file_location('gpu_queue',Path(__file__).with_name('dispatcher.py'));q=importlib.util.module_from_spec(s);s.loader.exec_module(q)
def main():
 p=argparse.ArgumentParser();p.add_argument('--problem',required=True);p.add_argument('--candidate',required=True,type=Path);p.add_argument('--banks',type=int,default=2);p.add_argument('--output',required=True,type=Path);a=p.parse_args()
 assert a.banks>=2,'Two independent banks required for final validation';assert a.candidate.is_file() and not a.candidate.is_symlink()
 lease=q.Lease();results=[];stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+secrets.token_hex(3)
 try:
  lease.start()
  for bank in range(a.banks):
   state=g.run_job(lease.sid,a.problem,a.candidate,'heldout-'+a.problem+'-'+str(bank)+'-'+stamp,10000+bank,seconds=900)
   results.append(state)
   if not state.get('result') or state['result'].get('status')!='pass':break
 finally:
  lease.close();a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps({'problem':a.problem,'banks':results,'source_audit_required':True,'note':'Correctness/timing checks alone do not establish compliance with the task rules.'},indent=2)+'\n')
if __name__=='__main__':main()
