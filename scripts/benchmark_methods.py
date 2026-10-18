"""Bounded live, paired comparison of research modes; no benchmark answers in prompts.

Uses the production runner, the configured model and independent finite scoring.
Small pilots can refute usefulness on these instances, not establish broad superiority.
"""
import argparse
import concurrent.futures
import contextlib
import hashlib
import importlib.util
import itertools
import json
import os
from pathlib import Path
import random
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location('methods', ROOT / 'Resources/ResearchMethods.py')
methods = importlib.util.module_from_spec(spec)
spec.loader.exec_module(methods)

EVALUATOR = '''import hashlib,json,sys
from pathlib import Path
TASK_HASH=EXPECTED_HASH
def value(task, bits):
 if not isinstance(bits,list) or len(bits)!=task['n'] or any(type(x)!=int or x not in (0,1) for x in bits) or sum(bits)!=task['n']//2:raise ValueError('balanced binary assignment required')
 return sum(w for a,b,w in task['edges'] if bits[a]!=bits[b])
def evaluate(root):
 raw=(root/'task.json').read_bytes()
 if hashlib.sha256(raw).hexdigest()!=TASK_HASH:raise ValueError('immutable task changed')
 task=json.loads(raw);bits=json.loads((root/'candidate.json').read_text());current=value(task,bits)
 probes=json.loads((root/'PROBES.json').read_text()) if (root/'PROBES.json').exists() else []
 if not isinstance(probes,list) or len(probes)>1000:raise ValueError('bounded probe schema')
 unique={tuple(p):value(task,p) for p in probes}
 return {'valid':True,'score':current,'metrics':{'cut':current,'gain':current-task['baseline'],'best_probe':max([current]+list(unique.values())),'unique_probes':len(unique),'distinct_probe_scores':len(set(unique.values()))}}
if __name__=='__main__':
 root=Path(sys.argv[-1]).resolve()
 try:
  result=evaluate(root);out=root/'.method-evaluation';out.mkdir(exist_ok=True)
  (out/'receipt.json').write_text(json.dumps(result));result['evidence']=['.method-evaluation/receipt.json']
 except Exception as error:result={'valid':False,'reason':str(error)}
 print(json.dumps(result))
'''

PROMPT = '''Find the best balanced weighted graph cut you can, using task.json.
candidate.json is a binary list of length n with exactly n/2 ones. Maximize the
total weight of edges whose endpoints differ. verify.py scores your workspace.
The current feasible incumbent is supplied. Choose useful algorithms/experiments,
implement and run them, and independently check the candidate before returning.
You may create arbitrary algorithms and intermediate files inside this workspace;
the scientific requirements are only the fixed task and a balanced assignment.
Retain your best candidate separately from research history. Save up to 1000
distinct evaluated assignments in PROBES.json; the frozen evaluator recomputes
their validity and scores. Save factual findings and next directions in NOTES.md.
Do not change task.json or claim a proof of optimality without supplying one.
No network, MCP, publication or nested agents. One CPU thread per computation.
Finish and checkpoint inside the assigned window. A practical solution is useful
even when no gain is found. Return immediately when your bounded experiment is done.
'''


def make_task(seed):
    rng = random.Random(seed)
    n = 20
    edges = [[a, b, rng.randrange(-5, 16)] for a in range(n) for b in range(a+1, n) if rng.random() < .35]
    start = [int(i < n//2) for i in range(n)]
    baseline = sum(w for a,b,w in edges if start[a] != start[b])
    return {'n': n, 'edges': edges, 'baseline': baseline}, start


def oracle(task):
    # Independent representation: balanced vertex subsets and edge crossing sums.
    best, witness = None, None
    for subset in itertools.combinations(range(1, task['n']), task['n']//2-1):
        left = {0, *subset}
        val = sum(w for a,b,w in task['edges'] if (a in left) != (b in left))
        if best is None or val > best:
            best, witness = val, sorted(left)
    return {'optimum': best, 'left_vertices': witness, 'enumerated_up_to_complement': True}


def usage(output):
    totals = {'input_tokens': 0, 'cached_input_tokens': 0, 'output_tokens': 0}
    terminal = 0
    for p in output.glob('*/provider.jsonl'):
        for line in p.open():
            try: event = json.loads(line)
            except ValueError: continue
            if event.get('type') == 'turn.completed':
                terminal += 1
                for name in totals: totals[name] += event.get('usage', {}).get(name, 0)
    return {'reported_completed_turns': terminal, **totals}


def trial(root, settings, disabled, seed, mode):
    path = root / ('instance-%s-%s' % (seed, mode)); path.mkdir()
    workspace = path/'source'; workspace.mkdir()
    task,start = make_task(seed)
    raw = (json.dumps(task, sort_keys=True)+'\n').encode()
    (workspace/'task.json').write_bytes(raw)
    (workspace/'candidate.json').write_text(json.dumps(start))
    (workspace/'PROBES.json').write_text('[]')
    evaluator = path/'evaluator.py';evaluator.write_text(EVALUATOR.replace('EXPECTED_HASH', repr(hashlib.sha256(raw).hexdigest())))
    (workspace/'verify.py').write_bytes(evaluator.read_bytes())
    # Same maximum 360 provider-seconds in each arm when RSI stalls occur.
    # Actual consumption and missed interventions are reported, never concealed.
    schedule = {'ordinary': (2, 1, 3), 'fork': (2, 1, 1),
                'rsi': (3, 1, 2/3), 'combined': (2, 1, 1)}[mode]
    rounds,explore,continuation = schedule
    options = dict(rsiTop=mode in ('rsi','combined'), forkAndFlush=mode in ('fork','combined'),
                   branches=2, rounds=rounds, explorationMinutes=explore, exploitationMinutes=continuation)
    args = ['--no-daemon','-c','model_reasoning_effort="high"','-c','features.memories=false',
            '-c','features.multi_agent=false','exec','--disable','plugins','-c','approval_policy="never"',
            '-c','sandbox_mode="workspace-write"']
    for name in disabled: args += ['-c','mcp_servers.%s.enabled=false' % name]
    args += ['--json','--skip-git-repo-check','--model',settings['codexModel'],'-']
    config = dict(provider='codex',executable=settings['codexPath'],arguments=args,
                  policy='Bounded benchmark authorized by the user. All writes, temporary files and computations stay inside the supplied SSD workspace. No network, nested agents or external messages.',options=options)
    plan = dict(objective='Maximize exact balanced cut weight on the pinned instance',worker_prompt=PROMPT,
                evaluator=str(evaluator),evaluation_seconds=3,direction='maximize',
                metrics=[dict(name=n,definition=d) for n,d in [('cut','Exact feasible incumbent cut'),('gain','Cut minus pinned starting cut'),('best_probe','Best independently rescored feasible probe'),('unique_probes','Distinct independently valid assignments, not quality'),('distinct_probe_scores','Different exact outcomes in retained probes')]],
                selection_metrics=[dict(name='best_probe',direction='maximize')],
                metric_subset=3,portfolio_size=2,stall_seconds=3,meaningful_gain=0)
    methods.write(path/'CONFIG.json',config);methods.write(path/'PLAN.json',plan)
    runner = methods.Runner(config,plan,workspace,path/'run')
    started = time.monotonic();result = None;failure = None
    try:
        with (path/'runner.log').open('w') as log, contextlib.redirect_stdout(log): result = runner.run()
    except Exception as error: failure = type(error).__name__+': '+str(error)
    elapsed = time.monotonic()-started
    # Score from independently computed vertices, never evaluator text alone.
    final = None;valid = False
    if result:
        leader = Path(result['leader']['workspace']);bits = json.loads((leader/'candidate.json').read_text())
        valid = len(bits)==task['n'] and all(type(b)==int and b in (0,1) for b in bits) and sum(bits)==task['n']//2
        if valid:
            left={i for i,b in enumerate(bits) if b};final=sum(w for a,b,w in task['edges'] if (a in left)!=(b in left))
    reference = oracle(task)
    launches = [json.loads(p.read_text()) for p in (path/'run').glob('*/LAUNCH.json')]
    report = dict(seed=seed,mode=mode,valid=valid,score=final,baseline=task['baseline'],oracle=reference,
                  regret=reference['optimum']-final if valid else None,wall_seconds=elapsed,
                  failure=failure,usage=usage(path/'run'),fresh_sessions=sum(x['fresh_context'] for x in launches),
                  exploration_calls=sum(x['phase']=='explore' for x in launches),
                  actual_rsi_intervention=options['rsiTop'] and any(x['phase']=='explore' for x in launches),
                  stop_reason=result.get('stop_reason') if result else None,
                  pending_workspaces=len(result.get('pending_workspaces',[])) if result else 0,
                  production_runner_sha256=hashlib.sha256((ROOT/'Resources/ResearchMethods.py').read_bytes()).hexdigest())
    methods.write(path/'REPORT.json',report)
    print(json.dumps({k:report[k] for k in ['seed','mode','valid','score','regret','failure','wall_seconds']}),flush=True)
    return report


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--settings',type=Path,required=True);args=parser.parse_args()
    sys.path.insert(0,str(ROOT/'scripts'));from check_workspace import validate
    root=validate(args.output);root.mkdir(parents=True,exist_ok=False)
    settings=json.loads(args.settings.read_text())
    servers=subprocess.check_output([settings['codexPath'],'--disable','plugins','mcp','list','--json'],text=True)
    disabled=[x['name'] for x in json.loads(servers)]
    pairs=[(seed,mode) for seed in (17,41) for mode in ('ordinary','fork','rsi','combined')]
    random.Random(20261018).shuffle(pairs)
    protocol=dict(status='PREREGISTERED_BEFORE_LIVE_TRIALS',instances=[17,41],arms=['ordinary','fork','rsi','combined'],
                  order=pairs,maximum_concurrent_arms=2,maximum_provider_seconds_per_arm=360,
                  model=settings['codexModel'],reasoning='high',primary='Exact regret versus independent exhaustive optimum; lower is better',
                  secondary=['validity','reported token usage','wall seconds','actual RSI interventions','fresh sessions'],
                  analysis='Paired per-instance arm-minus-ordinary regret and usage; n=2 exploratory only. Report all failures, no retry or exclusions. No statistical superiority claim.',
                  limitations=['Synthetic finite optimization, not mathematical discovery','User account load and provider latency may confound timing','Actual consumption differs if arms stop early; report it','No equal-token claim; input/output/cache usage is measured','RSI stall threshold uses elapsed time; verify intervention actually occurred'])
    methods.write(root/'PROTOCOL.json',protocol)
    # Runner progress is process stdout. Separate processes prevent one arm's
    # log redirection from closing or capturing another arm's output.
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        futures=[pool.submit(trial,root,settings,disabled,seed,mode) for seed,mode in pairs]
        reports=[future.result() for future in futures]
    comparisons=[]
    for seed in (17,41):
        rows={r['mode']:r for r in reports if r['seed']==seed};base=rows['ordinary']
        for mode in ('fork','rsi','combined'):
            row=rows[mode];comparisons.append(dict(seed=seed,mode=mode,
                regret_delta=row['regret']-base['regret'] if row['valid'] and base['valid'] else None,
                output_token_delta=row['usage']['output_tokens']-base['usage']['output_tokens'],
                wall_seconds_delta=row['wall_seconds']-base['wall_seconds']))
    methods.write(root/'RESULTS.json',dict(reports=reports,paired=comparisons,
        conclusion='Pilot complete. Interpret paired scores, costs and failures; n=2 cannot establish general superiority.'))

if __name__=='__main__':main()
