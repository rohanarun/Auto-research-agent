"""Supplementary independent artifact review; never rewrites registered outcomes."""
import argparse
import hashlib
import json
from pathlib import Path

from check_workspace import validate


def review(root):
    recorded = json.loads((root/'RESULTS.json').read_text())
    reports = recorded['reports']
    artifacts = []
    outside_commands = []
    for report in reports:
        arm = root/('instance-%s-%s' % (report['seed'], report['mode']))
        raw = (arm/'source/task.json').read_bytes()
        expected = hashlib.sha256(raw).hexdigest()
        task = json.loads(raw)
        accepted, rejected = [], []
        for candidate in (arm/'run').glob('*/workspace/candidate.json'):
            try:
                if hashlib.sha256((candidate.parent/'task.json').read_bytes()).hexdigest() != expected:
                    raise ValueError('task mutation')
                bits = json.loads(candidate.read_text())
                if not isinstance(bits, list) or len(bits) != task['n'] or any(type(b) != int or b not in (0,1) for b in bits) or sum(bits) != task['n']//2:
                    raise ValueError('invalid balanced assignment')
                left = {i for i,b in enumerate(bits) if b}
                value = sum(w for a,b,w in task['edges'] if (a in left) != (b in left))
                accepted.append({'path': str(candidate), 'score': value,
                                 'sha256': hashlib.sha256(candidate.read_bytes()).hexdigest()})
            except (ValueError, TypeError, OSError) as error:
                rejected.append({'path': str(candidate), 'reason': str(error)})
        for log in (arm/'run').glob('*/provider.jsonl'):
            for line in log.open():
                try: event = json.loads(line)
                except ValueError: continue
                item = event.get('item', {})
                if item.get('type') == 'command_execution':
                    command = item.get('command', '')
                    if '../' in command or '/Volumes/' in command or 'REPORT.json' in command:
                        outside_commands.append({'arm': arm.name, 'log': str(log), 'command': command})
        best = max([a['score'] for a in accepted], default=None)
        artifacts.append({'seed': report['seed'], 'mode': report['mode'],
                          'best_saved_score': best, 'oracle': report['oracle']['optimum'],
                          'saved_optimum': best == report['oracle']['optimum'],
                          'accepted': accepted, 'rejected': rejected})
    modes = {}
    for mode in ['ordinary','fork','rsi','combined']:
        rows = [r for r in reports if r['mode']==mode]
        modes[mode] = {'trials':len(rows), 'valid_selected_incumbents':sum(r['valid'] for r in rows),
                       'selected_optima':sum(r['valid'] and r['regret']==0 for r in rows),
                       'deadline_handoffs':sum(r.get('stop_reason')=='phase_budget_reached' for r in rows),
                       'actual_rsi_interventions':sum(r['actual_rsi_intervention'] for r in rows),
                       'saved_optima_supplementary':sum(r['saved_optimum'] for r in artifacts if r['mode']==mode),
                       'reported_output_tokens':sum(r['usage']['output_tokens'] for r in rows),
                       'reported_completed_turns':sum(r['usage']['reported_completed_turns'] for r in rows)}
    return {'registered_outcomes_unchanged': True, 'modes': modes, 'artifacts': artifacts,
            'commands_needing_scope_review': outside_commands,
            'interpretation': ['Primary results are the preregistered selected-incumbent outcomes.',
                               'Saved-artifact scores are supplementary, after-budget descriptive review; they do not retroactively complete a provider turn or change a registered outcome.',
                               'Two instances of one small finite optimization family cannot establish general research superiority.',
                               'Standalone RSI cannot be judged when its stall intervention did not activate.',
                               'An incomplete turn may omit usage totals; reported token sums are not complete billing or cost measurements.',
                               'Phase allocations differ by mode under the same aggregate ceiling; deadline sensitivity is an implementation/schedule result.',
                               'This benchmark does not score the semantic value of negative Integer Mult research; that needs an explicit maintainer/user rubric.']}


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path);args=parser.parse_args()
    root=validate(args.root);result=review(root)
    (root/'SUPPLEMENTARY-REVIEW.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result['modes'],indent=2))
