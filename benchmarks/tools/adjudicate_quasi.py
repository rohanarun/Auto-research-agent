"""Exact certificate admission using only the unchanged frozen predecessor verifier."""
from pathlib import Path
from fractions import Fraction as Q
import ast,concurrent.futures,hashlib,json,shlex,sys
CODE_ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(CODE_ROOT));import bench as b
ROOT=b.ROOT
TASK='quasi-rh-nested-levels'
def literal(n):
 if isinstance(n,ast.Constant) and type(n.value) is int:return n.value
 if isinstance(n,ast.Tuple):return [literal(x) for x in n.elts]
 if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='Q':return str(Q(*(literal(x) for x in n.args)))
 raise ValueError('Unsupported parameter literal')
def params(cp):
 p=cp/'parameters.json'
 if p.exists():return json.loads(p.read_text())
 tree=ast.parse((cp/'certify.py').read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main');branch=next(n for n in main.body if isinstance(n,ast.If))
 while len(branch.orelse)==1 and isinstance(branch.orelse[0],ast.If):branch=branch.orelse[0]
 values={n.targets[0].id:literal(n.value) for n in branch.orelse if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['q','s']}
 return {**values,'theta':'874809963/1000000000'}
SCRIPT='''import sys,json
from fractions import Fraction as Q
sys.path.insert(0,'depth3')
import certify_depth3 as c,model as m
p=json.load(open('parameters.json'));sub=json.load(open('claim.json'));theta=Q(p['theta'])
chain,interval,reserves=c.certify([Q(x) for x in p['q']],Q(p['s']),theta,'independent frozen-verifier replay')
c.width_floor_room(chain)
ell,h=Q(sub['ell']),Q(sub['h'])
assert ell==m.ell_at(theta) and h>0
A,B,C=m.capped_quad(chain.lam,chain.B,theta,h,chain.s)
points=[Q(0),Q(3,4)]
if A>0 and 0<=-B/(2*A)<=Q(3,4):points.append(-B/(2*A))
assert min(A*x*x+B*x+C for x in points)>=0,'rational h fails complete capped quadratic'
assert all(ok for _,ok,_ in c.RESULTS)
json.dump({'passed':True,'checks':len(c.RESULTS),'metric':str(theta),'ell':str(ell),'h':str(h),'exact_capped_quad_minimum':str(min(A*x*x+B*x+C for x in points)),'scope':'Conditional H1-H5 arithmetic only, no complete Lean or analytic endpoint.'},open('verified.json','w'),indent=2)
'''
def judge(run):
 meta=json.loads((run/'result.json').read_text());hashes=json.loads((ROOT/'tasks'/TASK/'snapshot-hashes.json').read_text());assert all((run/'work'/p).is_file() and hashlib.sha256((run/'work'/p).read_bytes()).hexdigest()==h for p,h in hashes)
 audits=[];accepted=None
 for cp in sorted((run/'checkpoints').glob('*')):
  try:
   p=params(cp);sub=json.loads((cp/'submission.json').read_text());at=json.loads((cp/'checkpoint.json').read_text())['at_seconds'];assert at<=1800 and Q(sub['metric'])<=Q('874809963/1000000000');assert (cp/'proof.md').stat().st_size>500
   audit=ROOT/'runs'/('judge-quasi-'+run.parent.name+'-'+cp.name);b.init_run(audit,ROOT/'tasks'/TASK/'snapshot');b.write(audit/'work/parameters.json',p);b.write(audit/'work/claim.json',sub);(audit/'work/independent.py').write_text(SCRIPT)
   r=b.shell(audit,'python -B independent.py',120);(audit/'inference-home/.codex/auth.json').unlink(missing_ok=True);b.write(audit/'result.json',r)
   verified=audit/'work/verified.json';ok=r['exit_code']==0 and verified.exists();audits.append({'checkpoint':cp.name,'at_seconds':at,'passed':ok,'judge':str(audit)})
   if ok:accepted=(at,json.loads(verified.read_text()));break
  except Exception as e:audits.append({'checkpoint':cp.name,'error':str(e)})
 result={**meta,'benchmark_outcome':'pass' if accepted else 'pending_review','evidence_class':'independent_exact_frozen_verifier' if accepted else 'unadmitted_certificate','time_to_validated_breakthrough_seconds':accepted[0] if accepted else None,'adjudication':'Conditional H1-H5 arithmetic certificate; no end-to-end Lean nonvanishing theorem or independent analytic proof.','audits':audits}
 b.write(run/'adjudicated-result.json',result);print(meta['model'],result['benchmark_outcome'],result['time_to_validated_breakthrough_seconds'],flush=True)
def main():
 runs=list((ROOT/'runs').glob('*/'+TASK))
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(judge,runs))
if __name__=='__main__':main()
