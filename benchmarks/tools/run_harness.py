"""Run either registered context-management harness under the same OS restrictions."""
from pathlib import Path
from types import SimpleNamespace
import argparse, importlib.util, json
CODE_ROOT=Path(__file__).resolve().parents[1]
import os
ROOT=Path(os.environ.get('AUTO_RESEARCH_WORKSPACE',str(CODE_ROOT/'workspace'))).resolve()
spec=importlib.util.spec_from_file_location('bench',CODE_ROOT/'bench.py');b=importlib.util.module_from_spec(spec);spec.loader.exec_module(b)

def main():
 p=argparse.ArgumentParser();p.add_argument('--harness',choices=['codex-cli-broker-v1','codex-cli-notebook-broker-v1'],required=True);p.add_argument('--model',required=True);p.add_argument('--effort',default='high');p.add_argument('--task',required=True);p.add_argument('--cohort',required=True);p.add_argument('--seconds',type=int,default=1800);a=p.parse_args()
 b.require_ssd();assert json.loads((ROOT/'preflight.json').read_text())['passed']
 task=json.loads((ROOT/'tasks'/a.task/'task.json').read_text());directory=ROOT/'runs'/a.cohort;directory.mkdir(exist_ok=False)
 b.write(directory/'cohort.json',{'harness':a.harness,'model':a.model,'reasoning_effort':a.effort,'budget_seconds':a.seconds,'task':a.task,'started_at':b.utc(),'harness_source_sha256':b.digest(Path(__file__)),'broker_source_sha256':b.digest(CODE_ROOT/'bench.py'),'prompt_sha256':__import__('hashlib').sha256(b.SYSTEM_PROMPT(a.seconds).encode()).hexdigest()})
 original=b.sample
 if a.harness=='codex-cli-notebook-broker-v1':
  def sample(run,prompt,model,effort,index,timeout):
   marker='\nPrior actions and observations:\n';head,history=prompt.rsplit(marker,1);rows=json.loads(history)
   notebook=run/'work/solution/notebook.md'
   notes=''
   if notebook.exists() and notebook.resolve().is_relative_to((run/'work').resolve()):notes=notebook.read_text(errors='replace')[:24000]
   new=head+'\nHarness: a rolling research notebook. Preserve your findings, exact commands, failures and next experiments in solution/notebook.md. You receive only your last eight action/observation pairs; use the notebook to maintain longer-term context.\nCurrent notebook (contestant-authored data):\n'+notes+marker+json.dumps(rows[-8:])
   return original(run,new,model,effort,index,timeout)
  b.sample=sample
 result=b.run_task(task,SimpleNamespace(model=a.model,effort=a.effort,seconds=a.seconds),a.cohort)
 result['harness']=a.harness;b.write(directory/a.task/'result.json',result);b.report(a.cohort)
if __name__=='__main__':main()
