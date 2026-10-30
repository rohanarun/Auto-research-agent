"""Import the Claude CLI (Linux broker) cohort into the published result data. No model calls or GPU work.

Copies each sealed final kernel source and its two held-out judge banks into data/candidates/<model>/<task>/,
adds one row per (model, task) to data/results.json, the models to data/cohort.json, and the infrastructure
exclusions of this cohort. Integer rows come from the integer cohorts; no submission was sealed in any of them.
"""
from pathlib import Path
import argparse, datetime, hashlib, json, shutil

CODE = Path(__file__).resolve().parents[1]
HARNESS = 'claude-cli-broker-linux-v1'
LABELS = {'claude-opus-5-5': 'Claude Opus 5.5', 'claude-sonnet-5-5': 'Claude Sonnet 5.5', 'claude-haiku-5-5': 'Claude Haiku 5.5'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--workspace', type=Path, required=True)
    p.add_argument('--kernel-cohort', required=True, help='prefix, e.g. claude-kernels-r4')
    p.add_argument('--integer-cohort', required=True, help='prefix, e.g. claude-int-r2')
    p.add_argument('--excluded', action='append', default=[], help='workspace run directories excluded as infrastructure-invalid')
    a = p.parse_args()
    ws = a.workspace
    data = json.loads((CODE / 'data/results.json').read_text()); cohort = json.loads((CODE / 'data/cohort.json').read_text())
    policy = json.loads((CODE / 'gpu/scoring-policy.json').read_text())
    rows = [r for r in data['results'] if r['model'] not in LABELS]
    new = []
    for model in LABELS:
        # integer
        run = ws / 'runs' / (a.integer_cohort + '-' + model) / 'integer-mult-frame-retiming'
        m = json.loads((run / 'result.json').read_text())
        new.append({**{k: m.get(k) for k in ['task_id', 'model', 'reasoning_effort', 'budget_seconds', 'started_at', 'finished_at', 'elapsed_seconds',
                                             'steps', 'usage', 'snapshot_sha256', 'time_to_first_submission_seconds']},
                    'harness': HARNESS, 'time_to_validated_breakthrough_seconds': None,
                    'benchmark_outcome': 'stopped_without_submission' if m['status'] == 'stopped_without_submission' else 'right_censored_no_submission',
                    'evidence_class': 'none', 'research_execution_status': m['status'], 'scope': 'core', 'metric': 'kappa', 'score_direction': 'max',
                    'target': '710436534421701/1000000000000000000', 'trial_id': run.parent.name + '/' + run.name, 'cohort': run.parent.name,
                    'adjudication': 'The actor stopped before the cap without sealing solution/submission.json; nothing to adjudicate.'})
        # kernels
        for run in sorted((ws / 'runs' / (a.kernel_cohort + '-' + model)).glob('kernel-*')):
            r = json.loads((run / 'adjudicated-result.json').read_text())
            dest = CODE / 'data/candidates' / model / run.name; dest.mkdir(parents=True, exist_ok=True)
            src = run / 'checkpoints' / r['selected_checkpoint'] / 'submission.py'
            shutil.copyfile(src, dest / 'submission.py')
            banks = []
            for job in r.get('evaluation_jobs', []):
                state = json.loads((ws / 'gpu/results' / job / 'result.json').read_text()); banks.append(state.get('result'))
            (dest / 'evaluation.json').write_text(json.dumps({'source_sha256': r['candidate_sha256'], 'banks': banks, 'source_audit': r['source_audit'],
                                                             'scope': 'Two fresh held-out banks after the research clock; final sealed source only.'}, indent=2) + '\n')
            row = {k: v for k, v in r.items() if k not in ('evaluation_jobs',)}
            row.update(harness=HARNESS + '-gpu-queue', research_execution_status=r['status'], scope='core', score_direction='max', metric='speedup_vs_baseline',
                       speedup_threshold=policy['speedup_threshold'], trial_id=run.parent.name + '/' + run.name, cohort=run.parent.name,
                       candidate_artifact=str((dest / 'submission.py').relative_to(CODE)), evaluation_artifact=str((dest / 'evaluation.json').relative_to(CODE)),
                       time_scope='Earliest in-budget checkpoint containing byte-identical final evaluated source. Earlier different source versions were not exhaustively evaluated.')
            new.append(row)
    data['results'] = rows + new
    excl = [e for e in data.get('infrastructure_exclusions', []) if e.get('harness') != HARNESS]
    for d in a.excluded:
        for rf in sorted((ws / 'runs').glob(d + '*/*/result.json')):
            m = json.loads(rf.read_text())
            excl.append({'harness': HARNESS, 'trial': rf.parent.parent.name + '/' + rf.parent.name, 'model': m.get('model'), 'task_id': m.get('task_id'),
                         'status': m.get('status'), 'reason': m.get('infrastructure_exclusion') or m.get('error', '')[:400]})
    data['infrastructure_exclusions'] = excl
    data['claude_cohort_provenance'] = {
        'harness': HARNESS, 'added_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'inference': 'Claude Code CLI print mode; all built-in tools disabled (--tools ""), no MCP, no settings/memory files, structured {command,done,note} output via --json-schema; any non-structured tool event invalidates the trial.',
        'execution': 'Linux namespace broker: uid 65534, fresh mount/net/pid/ipc/uts namespaces, read-only system dirs and venv, read-write trial workspace only; no network.',
        'prompt_difference': 'Identical prompts except "Your Mac shell" -> "Your Linux shell" in the GPU paragraph.',
        'gpu': 'Same Daytona H200 dispatcher, research policy, judge and scoring policy as the initial cohort; provisioning retried up to 3x per lease (infrastructure).',
        'comparability': 'Different inference client, host OS and sandbox from the codex-cli-broker-v1 cohort; compare across harnesses with care.',
        'source_audits': 'Curator audit by the operating Claude session (diff review + AST scan for cross-call state, caching, file access, timer/checker patching).'}
    (CODE / 'data/results.json').write_text(json.dumps(data, indent=2) + '\n')
    for m, l in LABELS.items():
        if m not in cohort['models']:
            cohort['models'].append(m)
        cohort['model_labels'][m] = l
    cohort['claude_cohort'] = {'harness': HARNESS, 'kernel_cohort': a.kernel_cohort, 'integer_cohort': a.integer_cohort, 'models': list(LABELS),
                               'reasoning': 'high (--effort high)', 'budget_seconds': 1800, 'kernel_concurrency': 5}
    (CODE / 'data/cohort.json').write_text(json.dumps(cohort, indent=2) + '\n')
    print('IMPORTED', len(new), 'rows;', len(excl), 'exclusions')


if __name__ == '__main__':
    main()
