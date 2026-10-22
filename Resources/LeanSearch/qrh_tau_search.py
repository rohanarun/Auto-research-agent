#!/usr/bin/env python3
"""Finite PR #9 tau search followed by an exact-target local Lean server job.

No model calls. The operator supplies pinned proof templates and a matching profile.
"""
import argparse
from fractions import Fraction as F
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import zipfile
from cli import request
from core import digest, save_json
from cpu_search import overview
from qrh import search


def pinned(plan, key, directory):
    path = (directory / plan[key]).resolve()
    if digest(path.read_bytes()) != plan[key + '_sha256']:
        raise ValueError('Pinned input changed: ' + key)
    return path


def enumerate_candidates(plan):
    low, high = plan['denominator_min'], plan['denominator_max']
    if type(low) is not int or type(high) is not int or not 0 < low <= high:
        raise ValueError('Require a nonempty positive integer denominator range')
    rows = []
    for n in range(low, high + 1):
        tau = F(1, n)
        rows.append(dict(N=n, tau=str(tau), theta=str(F(7, 8) - tau / 4),
                         central_budget=F(plan['central_cost']) * tau < F(plan['central_margin'])))
    passing = [row for row in rows if row['central_budget']]
    if not passing:
        raise ValueError('No candidate passes the configured central budget')
    return rows, min(passing, key=lambda row: F(row['theta']))


def checked_receipt(status, candidate, verification):
    """Accept only the submitted job under the operator's exact proof plan."""
    if status['status'] not in ('LEAN_CHECKED', 'COMPARATOR_ACCEPTED'):
        return False
    if status.get('id') != candidate['id'] or status.get('profile') != candidate['profile']:
        raise ValueError('Verification response belongs to a different job or profile')
    expected_request = digest(json.dumps(candidate, sort_keys=True).encode())
    if status.get('request_sha256') != expected_request:
        raise ValueError('Verification receipt does not bind the submitted proof source')
    receipt = status.get('receipt', {})
    if receipt.get('status') != status['status']:
        raise ValueError('Missing or inconsistent verification receipt')
    if receipt.get('plan_sha256') != verification['plan_sha256']:
        raise ValueError('Verification used a different proof plan or exact-target audit')
    if set(receipt.get('targets', [])) != set(verification['targets']):
        raise ValueError('Verification did not check all configured terminal targets')
    return True


def run(plan_path, output, overview_path):
    plan_path = Path(plan_path).resolve()
    plan = json.loads(plan_path.read_text())
    output = Path(output).resolve(); output.mkdir(parents=True, exist_ok=False)
    save_json(output / 'input-plan.json', plan)
    source = pinned(plan, 'certifier', plan_path.parent)
    retune = pinned(plan, 'retune_script', plan_path.parent)
    archive = pinned(plan, 'template_archive', plan_path.parent)
    if [[F(x) for x in edge] for edge in plan['domain']] != [[F(0), F(5, 6)], [F(0), F(1, 2)]]:
        raise ValueError('Require the full QRH parameter domain')
    rows, best = enumerate_candidates(plan)
    save_json(output / 'candidates.json', rows)
    baseline = F(7, 8) - F(1, 4 * plan['baseline_denominator'])
    if F(best['theta']) >= baseline:
        raise ValueError('No strict improvement over the configured Lean anchor')
    if F(best['theta']) != F(plan['verification']['theta']):
        raise ValueError('Search selected a different theta; prepare its exact-target proof profile before submitting')
    objective = 'Search τ = 1/N to lower θ = 7/8 − τ/4 within the existing PR #9 proof family.'
    method = (f'Enumerate N from {plan["denominator_min"]} to {plan["denominator_max"]}; '
              'check the central budget and full-domain exact certificate, then submit the retuned proof to Lean. No model calls.')
    overview(overview_path, objective, method, 'Exact search complete; checking the selected candidate and controls.')
    selected = list(dict.fromkeys([*plan.get('controls', []), best['N']]))
    points = [dict(N=n, ell=str(F(1, 6) + F(1, n)), h=str(F(13, 16) + F(3, 2*n))) for n in selected]
    exact = search(dict(scope=plan['scope'], certifier=str(source), certifier_sha256=plan['certifier_sha256'],
                        box=plan['domain'], maxdepth=plan['maxdepth'], candidates=points), output / 'exact',
                   on_candidate=lambda i, row: print(f'Exact check {i+1}/{len(points)}: N={row["candidate"]["N"]} {row["status"]}', flush=True))
    if next(row for row in exact['candidates'] if row['candidate']['N'] == best['N'])['status'] != 'EXACT_CONDITIONAL':
        raise ValueError('Selected candidate failed exact screening; no proof submitted')
    spec = importlib.util.spec_from_file_location('pinned_tau_retune', retune)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    target = output / 'candidate-source'; target.mkdir()
    with zipfile.ZipFile(archive) as z:
        for name in module.FILES:
            path = target / name
            if not path.resolve().is_relative_to(target) or Path(name).is_absolute():
                raise ValueError('Proof template path escapes the candidate directory')
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(z.read('project/' + name))
    with (output / 'retune.log').open('w') as log:
        subprocess.run([sys.executable, '-B', str(retune), '1', str(best['N'])], cwd=target,
                       stdout=log, stderr=subprocess.STDOUT, check=True)
    connection_path = Path(os.environ['LEAN_VERIFIER_CONNECTION']).resolve()
    connection = json.loads(connection_path.read_text())
    token_path = Path(connection['token_file'])
    if not token_path.is_absolute(): token_path = connection_path.parent / token_path
    token = token_path.read_text().strip(); url = connection['url'].rstrip('/')
    profiles = request(url + '/v1/profiles', token)
    profile = plan['verification']['profile']
    if profile not in profiles:
        raise ValueError('Matching Lean profile is unavailable: ' + profile)
    if set(profiles[profile]['targets']) != set(plan['verification']['targets']):
        raise ValueError('Server profile terminal targets do not match the search plan')
    candidate = dict(id=str(uuid.uuid4()), profile=profile, files={name:(target/name).read_text() for name in module.FILES},
                     metadata=dict(theta=best['theta'], tau=best['tau'], plan_sha256=digest(plan_path.read_bytes()),
                                   script_sha256=digest(Path(__file__).read_bytes()), model_calls=0))
    save_json(output / 'candidate.json', candidate)
    save_json(output / 'job-handle.json', dict(id=candidate['id'], url=url))
    status = request(url + '/v1/jobs', token, candidate)
    last = None
    while True:
        save_json(output / 'verification-status.json', status)
        if status['status'] != last:
            last = status['status']; print('Lean verification: ' + last, flush=True)
            overview(overview_path, objective, method, f'Candidate θ = {best["theta"]}. Lean job {candidate["id"]}: {last}.')
        if last not in ('QUEUED', 'RUNNING'): break
        time.sleep(plan['verification']['poll_seconds'])
        status = request(url + '/v1/jobs/' + candidate['id'], token)
    accepted = checked_receipt(status, candidate, plan['verification'])
    summary = (f'Searched {len(rows)} candidates. Selected θ = {best["theta"]} at τ = {best["tau"]}; '
               f'change from PR #9 anchor: {baseline - F(best["theta"])}. Lean server: {last}. ')
    summary += ('Exact terminal statements and axiom audit passed. ' if accepted else 'No verified improvement. ')
    summary += ('Official Comparator not run; ' if last != 'COMPARATOR_ACCEPTED' else '')
    summary += 'no tracker-record claim. No model was used.'
    result = dict(status='COMPLETED' if accepted else 'INCOMPLETE', summary=summary, selected=best,
                  baseline_theta=str(baseline), theta_improvement=str(baseline-F(best['theta'])),
                  verification=last, job_id=candidate['id'], model_calls=0,
                  search_plan_sha256=digest(plan_path.read_bytes()))
    save_json(output / 'cpu-result.json', result)
    overview(overview_path, objective, method, summary)
    print(summary, flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for key in ('plan', 'output', 'overview'): p.add_argument('--' + key, required=True)
    a = p.parse_args()
    try:
        r = run(a.plan, a.output, a.overview)
        sys.exit(0 if r['status'] == 'COMPLETED' else 1)
    except Exception as error:
        print('Search/verification incomplete: ' + str(error), file=sys.stderr, flush=True)
        sys.exit(1)
