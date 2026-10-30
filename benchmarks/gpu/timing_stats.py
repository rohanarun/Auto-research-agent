"""Paired timing bootstrap. This uncertainty describes timings, not model variation."""
import math,random,statistics

def speedup(banks):return math.exp(statistics.mean(math.log(b['speedup_vs_baseline']) for b in banks))
def bootstrap(banks,samples=10000,seed=20261030):
 rng=random.Random(seed);draws=[]
 for _ in range(samples):
  ratios=[]
  for bank in banks:
   base=candidate=0.
   for case in bank['cases']:
    pairs=case['paired_rounds'];resampled=rng.choices(pairs,k=len(pairs));w=case['config'].get('weight',1)
    base+=w*statistics.median(x['baseline'] for x in resampled);candidate+=w*statistics.median(x['candidate'] for x in resampled)
   ratios.append(base/candidate)
  draws.append(math.exp(statistics.mean(math.log(x) for x in ratios)))
 draws.sort();return {'point':speedup(banks),'one_sided_95_lower':draws[int(.05*samples)],'central_95':[draws[int(.025*samples)],draws[int(.975*samples)]],'samples':samples,'seed':seed,'scope':'Paired timing variability conditional on these two banks; not research-trial/model uncertainty'}
