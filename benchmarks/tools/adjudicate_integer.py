"""Admit exact retiming witnesses via canonical frame equality and a fresh trusted replay.
Candidate programs and their PASS labels are not used as the correctness oracle.
Alternative constructions remain eligible for separate independent adjudication.
"""
from pathlib import Path
from fractions import Fraction as Q
import hashlib,json,sympy as sp,sys
CODE_ROOT=Path(__file__).resolve().parents[1]
import os
ROOT=Path(os.environ.get('AUTO_RESEARCH_WORKSPACE',str(CODE_ROOT/'workspace'))).resolve()
TASK='integer-mult-frame-retiming'
MANDATORY={'banks','bit','complex','finite','math','primes','raw','scalar'}

def read(root,p):
 path=(root/p).resolve()
 if not path.is_relative_to(root.resolve()) or not path.is_file() or path.stat().st_size>2_000_000:raise ValueError('Invalid artifact path or size')
 return path.read_text()
def canonical(rows):
 matrix=sp.Matrix(rows)
 if matrix.has(sp.Float):raise ValueError('Inexact basis coefficient')
 return matrix.rref()[0]
def equivalent(candidate,golden):
 for k in ['input_raw_sha256','source_record_count','selected_gate_count']:
  if candidate.get(k)!=golden[k]:return False
 entries={r['record']:r for r in candidate['entries']};g={r['record']:r for r in golden['entries']}
 if len(entries)!=len(candidate['entries']) or entries.keys()!=g.keys():return False
 for idx,row in entries.items():
  ref=g[idx]
  for k in ['scalar','old_dimension','old_basis_sha256','new_dimension']:
   if row.get(k)!=ref[k]:return False
  if canonical(row['new_basis'])!=canonical(ref['new_basis']):return False
 return True

def main():
 replay=ROOT/'runs/reference-integer-own'
 report=json.loads((replay/'reference-result.json').read_text())
 assert report['exit_code']==0,'Trusted reference replay incomplete or failed'
 verified=json.loads((replay/'work/.tmp/immutable-reference-replay/verification.json').read_text())
 assert set(verified['fresh_stages'])==MANDATORY and verified['inputs_unchanged']
 target=Q(json.loads((ROOT/'tasks'/TASK/'task.json').read_text())['target'])
 assert Q(verified['kappa'])==target
 golden=json.loads((ROOT/'curator/reference-integer-mult-frame-retiming/research/five-stage-source527-banks/descent-selection.json').read_text())
 hashes=json.loads((ROOT/'tasks'/TASK/'snapshot-hashes.json').read_text())
 for run in sorted((ROOT/'runs').glob('*/'+TASK)):
  meta=json.loads((run/'result.json').read_text());audits=[]
  # Original tree remains the exact earlier source; new files under solution/ are allowed.
  changed=[]
  for rel,h in hashes:
   p=run/'work'/rel
   if not p.is_file() or p.is_symlink() or hashlib.sha256(p.read_bytes()).hexdigest()!=h:changed.append(rel)
  for checkpoint in sorted((run/'checkpoints').glob('*')):
   try:
    submission=json.loads(read(checkpoint,'submission.json'))
    at=json.loads((checkpoint/'checkpoint.json').read_text())['at_seconds']
    if at>meta['budget_seconds']:raise ValueError('Late checkpoint')
    if Q(str(submission['metric']))<target:raise ValueError('Below target')
    proof=submission.get('proof_file','proof.md')
    if proof.startswith('solution/'):proof=proof[len('solution/'):]
    if len(read(checkpoint,proof).strip())<200:raise ValueError('Missing scope/proof artifact')
    if changed:raise ValueError('Frozen baseline modified: '+str(changed[:5]))
    found=None
    for f in checkpoint.rglob('*.json'):
     if f.stat().st_size>2_000_000 or not f.resolve().is_relative_to(checkpoint.resolve()):continue
     try:data=json.loads(f.read_text())
     except (UnicodeError,json.JSONDecodeError):continue
     if isinstance(data,dict) and data.get('selected_gate_count')==25 and equivalent(data,golden):found=f;break
    if found is None:raise ValueError('No equivalent retiming witness; requires alternate-construction adjudication')
    audit={'checkpoint':checkpoint.name,'at_seconds':at,'status':'PASS_CONDITIONAL_FINITE_WITNESS','method':'Exact rational row-space equivalence at every bound scalar record to the independently replayed complete construction. Candidate verifier output is not trusted.','witness':str(found.relative_to(checkpoint)),'witness_sha256':hashlib.sha256(found.read_bytes()).hexdigest(),'trusted_reference_manifest_sha256':verified['manifest_sha256'],'trusted_reference_seconds':report['elapsed_seconds'],'fresh_reference_stages':sorted(MANDATORY),'kappa':str(target),'scope':'Conditional finite construction; retained all-size compiler and analytic interfaces are not proved by this evaluation.'}
    audits.append(audit);meta['evidence_class']='exact_finite_witness_plus_independent_eight_stage_replay';meta['time_to_validated_breakthrough_seconds']=at;meta['validated_metric']=str(target);meta['adjudication']=audit['scope'];meta['benchmark_outcome']='pass';break
   except Exception as error:audits.append({'checkpoint':checkpoint.name,'status':'NOT_ADMITTED','reason':str(error)})
  if not audits:
   audits=[{'status':'no_submission_checkpoint_yet' if meta.get('status')=='running' else 'no_in_budget_submission_checkpoint'}]
   if meta.get('status')=='budget_exhausted':
    meta.update(benchmark_outcome='right_censored_no_submission',time_to_validated_breakthrough_seconds=None,evidence_class='none',adjudication='Research budget exhausted without a sealed in-budget submission; no artifact admitted.')
  (run/'independent-audit.json').write_text(json.dumps(audits,indent=2)+'\n')
  if meta.get('status')!='running':
   (run/'result.json').write_text(json.dumps(meta,indent=2)+'\n')
  # A live trial's result remains under the controller's ownership. Save a sidecar.
  (run/'adjudicated-result.json').write_text(json.dumps(meta,indent=2)+'\n')
  print(meta['model'],meta['status'],meta.get('time_to_validated_breakthrough_seconds'),audits[-1]['status'],flush=True)
if __name__=='__main__':main()
