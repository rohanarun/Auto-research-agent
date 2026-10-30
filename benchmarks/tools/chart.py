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
 threshold=json.loads((CODE/'gpu/scoring-policy.json').read_text())['speedup_threshold'];models=cohort['models'];tasks=cohort['core_tasks'];labels=cohort['task_labels'];model_labels=cohort['model_labels'];out=CODE/'charts';out.mkdir(exist_ok=True)
 plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','savefig.facecolor':'white'})
 index={(r['model'],r['task_id']):r for r in rows};palette=['#2563eb','#8b5cf6','#ea580c','#10b981','#b45309','#0f766e','#be185d']
 fig,ax=plt.subplots(figsize=(4+2.3*len(models),5.8));ax.set_xlim(0,len(models));ax.set_ylim(0,len(tasks));ax.invert_yaxis()
 for y,task in enumerate(tasks):
  for x,model in enumerate(models):
   r=index.get((model,task),{});status=r.get('benchmark_outcome','not_run');color='#eef2f7';message='No scored result'
   if status=='pass':
    color='#d6f2e2';t=r.get('time_to_validated_breakthrough_seconds');message=('Observed '+f"{r['speedup_vs_baseline']:.4f}x" if task.startswith('kernel-') else 'Recovered')+ ('\n'+f'{int(t)//60}m {int(t)%60:02d}s' if t is not None else '')
    if task.startswith('kernel-') and not r.get('timing_supported_above_threshold'):message+=' *'
    if r.get('source_equals_baseline_ast'):message+='\nUnchanged baseline'
   elif status.startswith('right_censored'):color='#e8edf4';message='No admitted submission\n30-minute cap'
   elif status=='stopped_without_submission':color='#e8edf4';message='Stopped early\nno submission'
   elif status=='incorrect':color='#fce1df';message='Incorrect candidate'
   elif status=='below_threshold':color='#e8edf4';message=f'Correct on tested shapes\nBelow {threshold}x cutoff'
   elif status=='rule_violation':color='#fce1df';message='Source rules failed'
   elif status=='infrastructure_error':color='#fff0cf';message='Infrastructure failure'
   elif status=='pending_review':color='#e0ebff';message='Review pending'
   ax.add_patch(Rectangle((x+.025,y+.025),.95,.95,facecolor=color,edgecolor='white',linewidth=2));ax.text(x+.5,y+.5,message,ha='center',va='center',fontsize=10)
 ax.set_xticks([i+.5 for i in range(len(models))],[model_labels[m].replace(' ','\n',1) if len(models)>3 else model_labels[m] for m in models]);ax.xaxis.tick_top();ax.set_yticks([i+.5 for i in range(len(tasks))],[labels[t] for t in tasks]);ax.tick_params(length=0,pad=12)
 for s in ax.spines.values():s.set_visible(False)
 fig.suptitle('Measured research progress within 30 minutes',fontsize=19,fontweight='bold',y=.99);fig.text(.01,.015,f'Kernel cutoff >{threshold}x • Time = exact evaluated source first sealed',fontsize=10,color='#475569')
 fig.tight_layout(rect=[0,.05,1,.93]);fig.savefig(out/'recovery-matrix.png',dpi=180);fig.savefig(out/'recovery-matrix.svg');plt.close(fig)
 integer=[index[(m,'integer-mult-frame-retiming')] for m in models]
 fig,ax=plt.subplots(figsize=(10,1.6+0.75*len(models)))
 for i,(model,r) in enumerate(zip(models,integer)):
  t=r.get('time_to_validated_breakthrough_seconds');budget=(r['elapsed_seconds'] if r.get('benchmark_outcome')=='stopped_without_submission' else r['budget_seconds'])/60
  if t is not None:
   ax.barh(i,t/60,height=.5,color=palette[i%len(palette)]);ax.text(t/60+.45,i,f'{t/60:.2f} min',va='center',fontweight='bold')
  else:
   ax.barh(i,budget,height=.5,color='#eef2f7',edgecolor='#64748b',hatch='///');ax.plot([budget],[i],marker='>',color='#475569',markersize=9);ax.text(budget+.45 if r.get('benchmark_outcome')=='stopped_without_submission' else budget-.4,i,(f'Stopped at {budget:.1f} min, no submission' if r.get('benchmark_outcome')=='stopped_without_submission' else 'No admitted submission by 30 min'),ha='left' if r.get('benchmark_outcome')=='stopped_without_submission' else 'right',va='center',fontsize=10)
 ax.set_yticks(range(len(models)),[model_labels[m] for m in models]);ax.invert_yaxis();ax.set_xlim(0,34);ax.set_xlabel('Research minutes to an independently admitted finite witness');ax.set_title('Integer multiplication: exact conditional milestone',loc='left',fontsize=16,fontweight='bold',pad=18);ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
 fig.text(.01,.01,'The 8-stage reference replay and exact frame equivalence do not prove inherited all-size analytic/compiler assumptions.',fontsize=8.5,color='#475569');fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(out/'integer-recovery.png',dpi=180);fig.savefig(out/'integer-recovery.svg');plt.close(fig)
 kernel_tasks=[t for t in tasks if t.startswith('kernel-')];fig,ax=plt.subplots(figsize=(11,8))
 for y,task in enumerate(kernel_tasks):
  for j,model in enumerate(models):
   r=index.get((model,task),{});value=r.get('speedup_vs_baseline');ci=r.get('timing_interval_95');offset=(j-(len(models)-1)/2)*(.7/len(models))
   if value is not None:
    point=value;error=[[max(0,point-ci[0])],[max(0,ci[1]-point)]] if ci else None
    ax.errorbar(point,y+offset,xerr=error,fmt='o',color=palette[j%len(palette)],capsize=3,markersize=6)
   else:
    status=r.get('benchmark_outcome','not_run');ax.text(.02,y+offset,model_labels[model]+': '+status.replace('_',' '),fontsize=8,color=palette[j%len(palette)],va='center',transform=ax.get_yaxis_transform())
 threshold=json.loads((CODE/'gpu/scoring-policy.json').read_text())['speedup_threshold']
 values=[r.get('speedup_vs_baseline') for r in rows if r['task_id'].startswith('kernel-') and r.get('speedup_vs_baseline')]
 if values and min(values)<0.5:ax.set_xscale('log')
 ax.axvline(threshold,color='#334155',linestyle='--',linewidth=1);ax.set_yticks(range(len(kernel_tasks)),[labels[t] for t in kernel_tasks]);ax.invert_yaxis();ax.set_xlabel('Measured weighted aggregate speedup versus baseline');ax.set_title('Kernel performance after correctness checks',loc='left',fontsize=16,fontweight='bold',pad=44);ax.grid(axis='x',alpha=.15)
 handles=[plt.Line2D([],[],marker='o',linestyle='',color=palette[i%len(palette)],label=model_labels[m]) for i,m in enumerate(models)];ax.legend(handles=handles,loc='upper left',bbox_to_anchor=(0,1.07),ncol=4,frameon=False,fontsize=9);ax.text(threshold,.97,f'{threshold}x cutoff',rotation=90,ha='right',va='top',fontsize=8,color='#334155',transform=ax.get_xaxis_transform())
 fig.text(.01,.01,'Two fresh banks; bars show timing uncertainty only. Raw cutoff crossings are distinct from confident improvements.',fontsize=9,color='#475569');fig.tight_layout(rect=[0,.04,1,1]);fig.savefig(out/'kernel-performance.png',dpi=180);fig.savefig(out/'kernel-performance.svg');plt.close(fig)
 fields=['model','task_id','benchmark_outcome','elapsed_seconds','time_to_validated_breakthrough_seconds','speedup_vs_baseline','reference_speedup','budget_seconds','time_to_measured_speedup_seconds','timing_supported_above_threshold','source_equals_baseline_ast','historical_benchmark_outcome','scoring_policy_version']
 with (CODE/'data/results.csv').open('w',newline='') as f:
  writer=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(rows)

 # --- Figure 4: Cost/Speed Pareto Frontier ---
 pr_pricing = {
  'claude-haiku-5-5': {'in': 0.10, 'out': 0.50, 'cch': 0.01, 'color': '#be185d'},
  'claude-sonnet-5-5': {'in': 2.0, 'out': 10.0, 'cch': 0.10, 'color': '#0f766e'},
  'gemini-3.8-flash-opencode': {'in': 0.75, 'out': 3.75, 'cch': 0.075, 'color': '#10b981'},
  'claude-opus-5-5': {'in': 4.0, 'out': 20.0, 'cch': 0.20, 'color': '#b45309'},
  'gpt-5.6-sol': {'in': 1.5, 'out': 7.5, 'cch': 0.075, 'color': '#ea580c'},
  'gpt-6.1-sol': {'in': 2.0, 'out': 10.0, 'cch': 0.10, 'color': '#2563eb'},
  'gpt-6-astra': {'in': 10.0, 'out': 50.0, 'cch': 1.0, 'color': '#8b5cf6'}
 }
 m_data = []
 for m in models:
  p = pr_pricing.get(m, {'in': 2.0, 'out': 10.0, 'cch': 0.1, 'color': '#334155'})
  m_runs = [r for r in rows if r['model'] == m and r['task_id'] in tasks]
  tot_c = sum(((max(0, r.get('usage', {}).get('input_tokens', 0) - r.get('usage', {}).get('cached_input_tokens', 0)) * p['in'] + r.get('usage', {}).get('cached_input_tokens', 0) * p['cch'] + r.get('usage', {}).get('output_tokens', 0) * p['out']) / 1e6) for r in m_runs)
  k_sp = [r.get('speedup_vs_baseline') or 0.5 for r in m_runs if r['task_id'].startswith('kernel-')]
  g_sp = math.exp(sum(math.log(max(0.01, s)) for s in k_sp) / len(k_sp)) if k_sp else 1.0
  ir = next((r for r in m_runs if r['task_id'] == 'integer-mult-frame-retiming'), {})
  it = (ir.get('time_to_validated_breakthrough_seconds') or ir.get('elapsed_seconds')) / 60.0 if ir.get('benchmark_outcome') == 'pass' else None
  m_data.append({'id': m, 'name': model_labels[m], 'cost': tot_c, 'spd': g_sp, 'int_t': it, 'color': p['color']})
 fig, (a1, a2) = plt.subplots(1, 2, figsize=(16, 6.8))
 s_c = sorted(m_data, key=lambda x: x['cost'])
 pf_s, max_s = [], -1
 for m in s_c:
  if m['spd'] > max_s: pf_s.append(m); max_s = m['spd']
 a1.plot([m['cost'] for m in pf_s], [m['spd'] for m in pf_s], '--', color='#059669', linewidth=2.2, alpha=0.9, zorder=3, label='Pareto Frontier (Optimal Tradeoff)')
 a1.fill_between([m['cost'] for m in pf_s], [m['spd'] for m in pf_s], 0.05, color='#10b981', alpha=0.08, zorder=1)
 ann1 = {'claude-haiku-5-5': (0.33, 0.28, 'left'), 'claude-sonnet-5-5': (2.3, 0.96, 'right'), 'gemini-3.8-flash-opencode': (8.5, 2.18, 'left'), 'claude-opus-5-5': (16.0, 1.15, 'left'), 'gpt-5.6-sol': (15.5, 0.72, 'left'), 'gpt-6.1-sol': (31.0, 0.98, 'left'), 'gpt-6-astra': (118.0, 1.08, 'left')}
 for m in m_data:
  a1.scatter(m['cost'], m['spd'], color=m['color'], s=140, zorder=5, edgecolors='black', linewidth=1.2)
  tx, ty, al = ann1.get(m['id'], (m['cost'] * 1.15, m['spd'], 'left'))
  a1.text(tx, ty, f"{m['name']}\n({m['spd']:.2f}x, ${m['cost']:.2f})", fontsize=9.5, fontweight='bold' if m in pf_s else 'normal', color=m['color'], ha=al, va='center')
 a1.set_xscale('log'); a1.set_xlim(0.15, 220); a1.set_ylim(0.05, 2.45); a1.set_xlabel('Total Cohort Inference Cost (USD, Log Scale)', fontsize=12, fontweight='bold', labelpad=10)
 a1.set_ylabel('GPU Kernel GeoMean Speedup vs Baseline (x)', fontsize=12, fontweight='bold', labelpad=10); a1.set_title('Kernel Performance vs Inference Cost\n(Geometric Mean across 5 GPU Kernels)', fontsize=14, fontweight='bold', pad=15)
 a1.grid(True, linestyle=':', alpha=0.4, which='both'); a1.axhline(1.01, color='#64748b', linestyle=':', linewidth=1.2, label='1.01x Speedup Cutoff'); a1.legend(loc='lower right', frameon=True, fontsize=9.5)
 p_int = [m for m in m_data if m['int_t'] is not None]; f_int = [m for m in m_data if m['int_t'] is None]
 pf_t, min_t = [], 999
 for m in sorted(p_int, key=lambda x: x['cost']):
  if m['int_t'] < min_t: pf_t.append(m); min_t = m['int_t']
 a2.plot([m['cost'] for m in pf_t], [m['int_t'] for m in pf_t], '--', color='#2563eb', linewidth=2.2, alpha=0.9, zorder=3, label='Pareto Frontier (Fastest Witness)')
 for m in p_int:
  a2.scatter(m['cost'], m['int_t'], color=m['color'], s=140, zorder=5, edgecolors='black', linewidth=1.2)
  ha = 'right' if m['id'] == 'gpt-6-astra' else 'left'
  ox = 0.85 if m['id'] == 'gpt-6-astra' else 1.15
  oy = 1.2 if m['id'] == 'gpt-6-astra' else -0.2
  a2.annotate(f"{m['name']}\n({m['int_t']:.1f}m, ${m['cost']:.2f})", (m['cost'], m['int_t']), xytext=(m['cost'] * ox, m['int_t'] + oy), fontsize=9.5, fontweight='bold', color=m['color'], ha=ha)
 stag_y = [29.2, 31.0, 27.5, 32.5]
 for i, m in enumerate(sorted(f_int, key=lambda x: x['cost'])):
  y_p = stag_y[i % len(stag_y)]
  a2.scatter(m['cost'], 30.0, color='#94a3b8', s=100, marker='X', zorder=5, linewidth=1.5)
  a2.plot([m['cost'], m['cost']], [30.0, y_p], color='#cbd5e1', linestyle='-', linewidth=0.8, zorder=2)
  a2.text(m['cost'] * 1.12, y_p, f"{m['name']}\n(No submission)", fontsize=8.5, color='#475569', va='center')
 a2.set_xscale('log'); a2.set_xlim(0.15, 220); a2.set_ylim(6, 34); a2.set_xlabel('Total Cohort Inference Cost (USD, Log Scale)', fontsize=12, fontweight='bold', labelpad=10)
 a2.set_ylabel('Research Minutes to Validated Witness (Minutes)', fontsize=12, fontweight='bold', labelpad=10); a2.set_title('Research Velocity vs Inference Cost\n(Integer Retiming Exact Milestone)', fontsize=14, fontweight='bold', pad=15)
 a2.grid(True, linestyle=':', alpha=0.4, which='both'); a2.axhline(30.0, color='#ef4444', linestyle=':', linewidth=1.2, label='30-Minute Budget Cap (Timeout)'); a2.legend(loc='lower left', frameon=True, fontsize=9.5)
 fig.suptitle('Cost/Speed Pareto Frontier: Autonomous Research Benchmark', fontsize=17, fontweight='bold', y=0.99)
 fig.text(0.01, 0.01, 'Inference costs computed from token usage logs with provider API pricing. Kernel speedups evaluate 5 Muse kernels on NVIDIA H200.', fontsize=9, color='#64748b')
 fig.tight_layout(rect=[0, 0.03, 1, 0.95]); fig.savefig(out/'pareto-frontier.png', dpi=200); fig.savefig(out/'pareto-frontier.svg'); plt.close(fig)

 for path in out.glob('*.svg'):path.write_text('\n'.join(line.rstrip() for line in path.read_text().splitlines())+'\n')
if __name__=='__main__':main()
