"""Private GPU worker. Run only inside the fresh network-blocked H200 sandbox.
Each invocation sees one candidate and public frozen correctness sources; never other solutions.
"""
from pathlib import Path
import argparse,importlib.util,json,random,statistics,time,traceback,os,sys
p=argparse.ArgumentParser();p.add_argument('--problem',required=True);p.add_argument('--candidate',default='candidate.py');p.add_argument('--seed',type=int,required=True);p.add_argument('--rounds',type=int,default=5);p.add_argument('--trusted-reference',action='store_true');p.add_argument('--cases');a=p.parse_args()
os.environ.setdefault('USER','benchmark');os.environ.setdefault('LOGNAME','benchmark')
import torch
torch.cuda.init()
if not a.trusted_reference:
 assert 10000<=os.geteuid()<60000,'Contestants must run as a dedicated unprivileged UID'
 from offline_guard import install as confine_network
 confine_network(sync=True)
import torch

def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m

def graph(call):
 # Correctness is checked outside timing. Clone mutable input tensors for each module.
 stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
 with torch.cuda.stream(stream):
  for _ in range(3):call()
 stream.synchronize()
 g=torch.cuda.CUDAGraph()
 with torch.cuda.graph(g):output=call()
 return g,output

def time_graph(g,replays):
 begin=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
 begin.record()
 for _ in range(replays):g.replay()
 end.record();end.synchronize();return begin.elapsed_time(end)*1000/replays

def clone(value):
 if isinstance(value,torch.Tensor):return value.clone()
 if isinstance(value,dict):return {k:clone(v) for k,v in value.items()}
 if isinstance(value,list):return [clone(v) for v in value]
 if isinstance(value,tuple):return tuple(clone(v) for v in value)
 return value

result={'problem':a.problem,'status':'error','gpu':torch.cuda.get_device_name(0),'torch':torch.__version__,'cuda':torch.version.cuda,'cases':[],'source_sha256':__import__('hashlib').sha256(Path(a.candidate).read_bytes()).hexdigest(),'judge_sha256':__import__('hashlib').sha256(Path(__file__).read_bytes()).hexdigest(),'role':'trusted_reference_calibration' if a.trusted_reference else 'contestant','execution_uid':os.geteuid()}
try:
 ref=load('reference','reference.py');baseline=load('baseline','baseline.py')
 candidate=load('candidate',a.candidate)
 spec=json.loads(Path('problem.json').read_text());cases=spec['benchmarks'];rng=random.Random(a.seed)
 weighted_baseline=0.;weighted_candidate=0.
 # Independent weights and held-out input seeds are hidden from the contestant until execution.
 weights=ref.generate_weights(a.seed) if hasattr(ref,'generate_weights') else None
 bw=weights;cw=clone(weights) if weights is not None else None
 bp=baseline.prepare(bw) if weights is not None and hasattr(baseline,'prepare') else bw
 cp=candidate.prepare(cw) if weights is not None and hasattr(candidate,'prepare') else cw
 for index,case in enumerate(cases):
  if a.cases is not None and index not in [int(i) for i in a.cases.split(',')]:continue
  cfg={**case,'seed':a.seed+1009*(index+1)};data=ref.generate_input(**cfg)
  bd,cd=clone(data),clone(data)
  def call(mod,inp,w):return mod.kernel_fn(inp,w) if weights is not None else mod.kernel_fn(inp)
  def check(inp,out):
   packed=ref.collect(inp,out) if hasattr(ref,'collect') else out
   return ref.check_implementation(inp,packed,weights) if weights is not None else ref.check_implementation(inp,packed)
  def same_contract(left,right):
   if type(left) is torch.Tensor:return type(right) is torch.Tensor and left.dtype==right.dtype and left.shape==right.shape and left.device==right.device
   if isinstance(left,(tuple,list)):return isinstance(right,(tuple,list)) and len(left)==len(right) and all(same_contract(x,y) for x,y in zip(left,right))
   return type(left) is type(right)
  output=call(candidate,cd,cp);torch.cuda.synchronize()
  ok,msg=check(cd,output)
  if not ok:raise AssertionError('case '+str(index)+': '+msg)
  base_output=call(baseline,bd,bp);torch.cuda.synchronize()
  assert same_contract(base_output,output),'Candidate output contract differs from baseline'
  bok,bmsg=check(bd,base_output)
  if not bok:raise RuntimeError('baseline correctness failure '+bmsg)
  bg,bout=graph(lambda:call(baseline,bd,bp));cg,cout=graph(lambda:call(candidate,cd,cp))
  fresh=ref.generate_input(**{**cfg,'seed':cfg['seed']+1000003});changed=[]
  for key in ['x','hidden_states','qkv','stream','attn_input','topk_ids','topk_weights']:
   if key in data and isinstance(data[key],torch.Tensor) and fresh[key].shape==data[key].shape:
    bd[key].copy_(fresh[key]);cd[key].copy_(fresh[key]);changed.append(key)
  if not changed:raise RuntimeError('No dynamic input field validated for graph replay')
  bg.replay();cg.replay();torch.cuda.synchronize()
  ok,msg=check(cd,cout)
  if not ok:raise AssertionError('fresh graph replay case '+str(index)+': '+msg)
  assert same_contract(bout,cout),'Graph output contract differs from baseline'
  bok,bmsg=check(bd,bout)
  if not bok:raise RuntimeError('baseline fresh graph failure '+bmsg)
  routing_changed=[k for k in ['topk_ids','topk_weights'] if k in changed]
  # Test genuinely fresh routing, then retain the original predeclared timing
  # workload: fresh activations with this bank's original routing metadata.
  for key in routing_changed:
   bd[key].copy_(data[key]);cd[key].copy_(data[key])
  if routing_changed:
   bg.replay();cg.replay();torch.cuda.synchronize()
   ok,msg=check(cd,cout)
   if not ok:raise AssertionError('restored-routing graph replay case '+str(index)+': '+msg)
   bok,bmsg=check(bd,bout)
   if not bok:raise RuntimeError('baseline restored-routing graph failure '+bmsg)
  pairs=[];replays=max(1,min(20,int(case.get('repeats',10))))
  # Cold-L2 paired order. Cache clearing runs outside the measured graphs.
  flush=torch.empty(64*1024*1024,device='cuda',dtype=torch.uint8)
  for _ in range(a.rounds):
   order=[('baseline',bg),('candidate',cg)];rng.shuffle(order);times={}
   for name,g in order:
    flush.zero_();torch.cuda.synchronize();times[name]=time_graph(g,replays)
   pairs.append(times)
  b=statistics.median(x['baseline'] for x in pairs);c=statistics.median(x['candidate'] for x in pairs)
  weight=case.get('weight',1);weighted_baseline+=b*weight;weighted_candidate+=c*weight
  row={'index':index,'config':case,'correct':True,'fresh_graph_replay_checked':True,'changed_input_fields':changed,'fresh_routing_replay_checked':bool(routing_changed),'timing_uses_original_routing':bool(routing_changed),'correctness':msg,'baseline_us':b,'candidate_us':c,'paired_rounds':pairs}
  result['cases'].append(row);print(json.dumps({'progress':index+1,'total':len(cases),'case':row}),flush=True)
  del bg,cg,bd,cd,data,fresh,bout,cout,output,base_output,flush;torch.cuda.empty_cache()
 result.update(status='pass',correct=True,weighted_baseline_us=weighted_baseline,weighted_candidate_us=weighted_candidate,speedup_vs_baseline=weighted_baseline/weighted_candidate,heldout_seed=a.seed)
except BaseException as error:result.update(status='failed',correct=False,error=str(error),traceback=traceback.format_exc())
Path('result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
