"""Rebuild publication figures from committed adjudicated result data."""
from pathlib import Path
import csv,json,math,os
CODE=Path(__file__).resolve().parents[1]
os.environ.setdefault('MPLCONFIGDIR',str(Path(os.environ.get('AUTO_RESEARCH_WORKSPACE',CODE/'workspace'))/'.cache/matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Patch

def main():
 data=json.loads((CODE/'data/results.json').read_text());rows=data['results'];cohort=json.loads((CODE/'data/cohort.json').read_text())
 models=cohort['models'];tasks=cohort['core_tasks'];labels=cohort['task_labels'];model_labels=cohort['model_labels'];out=CODE/'charts';out.mkdir(exist_ok=True)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','savefig.facecolor':'white'})
 index={(r['model'],r['task_id']):r for r in rows};palette=['#2563eb','#8b5cf6','#ea580c']
 fig,ax=plt.subplots(figsize=(11,5.8));ax.set_xlim(0,len(models));ax.set_ylim(0,len(tasks));ax.invert_yaxis()
 for y,task in enumerate(tasks):
  for x,model in enumerate(models):
   r=index.get((model,task),{});status=r.get('benchmark_outcome','not_run');color='#eef2f7';message='No scored result'
   if status=='pass':
    color='#d6f2e2';t=r.get('time_to_validated_breakthrough_seconds');message='Recovered'+ ('\n'+f'{int(t)//60}m {int(t)%60:02d}s' if t is not None else '')
   elif status.startswith('right_censored'):color='#e8edf4';message='No admitted submission\n30-minute cap'
   elif status=='incorrect':color='#fce1df';message='Incorrect candidate'
   elif status=='below_target':color='#e8edf4';message='Correct on tested shapes\nBelow target'
   elif status=='rule_violation':color='#fce1df';message='Source rules failed'
   elif status=='infrastructure_error':color='#fff0cf';message='Infrastructure failure'
   elif status=='pending_review':color='#e0ebff';message='Review pending'
   ax.add_patch(Rectangle((x+.025,y+.025),.95,.95,facecolor=color,edgecolor='white',linewidth=2));ax.text(x+.5,y+.5,message,ha='center',va='center',fontsize=10)
 ax.set_xticks([i+.5 for i in range(len(models))],[model_labels[m] for m in models]);ax.xaxis.tick_top();ax.set_yticks([i+.5 for i in range(len(tasks))],[labels[t] for t in tasks]);ax.tick_params(length=0,pad=12)
 for s in ax.spines.values():s.set_visible(False)
 fig.suptitle('Recovering our next research improvement',fontsize=19,fontweight='bold',y=.99);fig.text(.01,.015,'One valid trial per cell • High reasoning • 30-minute cap • Shared GPU queue; see cohort and retry details',fontsize=10,color='#475569')
 fig.tight_layout(rect=[0,.05,1,.93]);fig.savefig(out/'recovery-matrix.png',dpi=180);fig.savefig(out/'recovery-matrix.svg');plt.close(fig)
 integer=[index[(m,'integer-mult-frame-retiming')] for m in models]
 fig,ax=plt.subplots(figsize=(10,3.9))
 for i,(model,r) in enumerate(zip(models,integer)):
  t=r.get('time_to_validated_breakthrough_seconds');budget=r['budget_seconds']/60
  if t is not None:
   ax.barh(i,t/60,height=.5,color=palette[i%len(palette)]);ax.text(t/60+.45,i,f'{t/60:.2f} min',va='center',fontweight='bold')
  else:
   ax.barh(i,budget,height=.5,color='#eef2f7',edgecolor='#64748b',hatch='///');ax.plot([budget],[i],marker='>',color='#475569',markersize=9);ax.text(budget-.4,i,'No admitted submission by 30 min',ha='right',va='center',fontsize=10)
 ax.set_yticks(range(len(models)),[model_labels[m] for m in models]);ax.invert_yaxis();ax.set_xlim(0,34);ax.set_xlabel('Research minutes to an independently admitted finite witness');ax.set_title('Integer multiplication: exact conditional milestone',loc='left',fontsize=16,fontweight='bold',pad=18);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
 fig.text(.01,.01,'The 8-stage reference replay and exact frame equivalence do not prove inherited all-size analytic/compiler assumptions.',fontsize=8.5,color='#475569');fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(out/'integer-recovery.png',dpi=180);fig.savefig(out/'integer-recovery.svg');plt.close(fig)
 kernel_tasks=[t for t in tasks if t.startswith('kernel-')];fig,ax=plt.subplots(figsize=(10,6))
 for y,task in enumerate(kernel_tasks):
  for j,model in enumerate(models):
   r=index.get((model,task),{});ref=r.get('reference_speedup');value=r.get('speedup_vs_baseline');ci=r.get('timing_interval_95');offset=(j-(len(models)-1)/2)*.22
   if value is not None and ref:
    point=value/ref;error=[[max(0,point-ci[0]/ref)],[max(0,ci[1]/ref-point)]] if ci else None
    ax.errorbar(point,y+offset,xerr=error,fmt='o',color=palette[j%len(palette)],capsize=3,markersize=6)
   else:
    status=r.get('benchmark_outcome','not_run');ax.text(.02,y+offset,model_labels[model]+': '+status.replace('_',' '),fontsize=8,color=palette[j%len(palette)],va='center',transform=ax.get_yaxis_transform())
 ax.axvline(1,color='#334155',linestyle='--',linewidth=1,label='Our reference target');ax.axvline(.98,color='#94a3b8',linestyle=':',linewidth=1);ax.set_yticks(range(len(kernel_tasks)),[labels[t] for t in kernel_tasks]);ax.invert_yaxis();ax.set_xlabel('Weighted aggregate speedup / our calibrated reference target');ax.set_title('Kernel performance after correctness checks',loc='left',fontsize=16,fontweight='bold',pad=18);ax.grid(axis='x',alpha=.15)
 handles=[plt.Line2D([],[],marker='o',linestyle='',color=palette[i%len(palette)],label=model_labels[m]) for i,m in enumerate(models)];ax.legend(handles=handles,loc='upper left',bbox_to_anchor=(0,1.015),ncol=3,frameon=False,fontsize=9);ax.text(1,.97,'Reference',rotation=90,ha='right',va='top',fontsize=8,color='#334155',transform=ax.get_xaxis_transform())
 fig.text(.01,.01,'Two hidden banks; bars show paired-timing uncertainty only. Missing measurements are labeled, never assigned zero.',fontsize=9,color='#475569');fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(out/'kernel-performance.png',dpi=180);fig.savefig(out/'kernel-performance.svg');plt.close(fig)
 fields=['model','task_id','benchmark_outcome','elapsed_seconds','time_to_validated_breakthrough_seconds','speedup_vs_baseline','reference_speedup','budget_seconds']
 with (CODE/'data/results.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(rows)
 for path in out.glob('*.svg'):path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
if __name__=='__main__':main()
