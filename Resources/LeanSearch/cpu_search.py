#!/usr/bin/env python3
"""Run a configured, finite QRH rational search. No model or network calls."""
import argparse
from fractions import Fraction
import json
from pathlib import Path
import platform
import sys
import time

from core import digest, save_json
from qrh import search


def overview(path, objective, method, meaning):
    # Atomic replacement lets the UI read while a candidate completes.
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp')
    save_json(temporary, dict(objective=objective, method=method, meaning=meaning))
    temporary.replace(destination)


def run(plan_path, output, overview_path):
    plan_path = Path(plan_path).resolve()
    plan_bytes = plan_path.read_bytes()
    plan = json.loads(plan_bytes)
    if plan['adapter'] != 'qrh-rational-grid':
        raise ValueError('This generated script supports qrh-rational-grid only')
    source = (plan_path.parent / plan['certifier']).resolve()
    if digest(source.read_bytes()) != plan['certifier_sha256']:
        raise ValueError('Certifier changed: review it and update the source pin before searching')
    ns = plan['n_values']
    heights = plan['h_values']
    if not ns or not heights or any(type(n) is not int or n <= 0 for n in ns):
        raise ValueError('n_values must be a nonempty list of positive integers')
    if any(not isinstance(h, str) for h in heights):
        raise ValueError('h_values must be exact rational strings')
    if type(plan['maxdepth']) is not int or plan['maxdepth'] < 0:
        raise ValueError('maxdepth must be a nonnegative integer')
    if [[Fraction(x) for x in edge] for edge in plan['box']] != [[Fraction(0), Fraction(5, 6)], [Fraction(0), Fraction(1, 2)]]:
        raise ValueError('QRH requires the full delta=[0,5/6], y=[0,1/2] domain')
    candidates = list(plan.get('controls', []))
    candidates += [dict(ell=str(Fraction(1, 6) + Fraction(1, n)), h=str(Fraction(h)), N=n)
                   for n in ns for h in heights]
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / 'input-plan.json', plan)
    save_json(output / 'source.json', dict(plan_sha256=digest(plan_bytes),
        script_sha256=digest(Path(__file__).read_bytes()), certifier_sha256=plan['certifier_sha256'],
        qrh_adapter_sha256=digest(Path(__file__).with_name('qrh.py').read_bytes()),
        python=sys.version, platform=platform.platform(), model_calls=0))
    objective = 'Search rational ℓ and h to lower the QRH exponent bound σ₀ = 11/12 − ℓ/4 while preserving all source inequalities.'
    method = (f'Python script, no model. Vary ℓ = 1/6 + 1/N for N = {ns}; '
              f'test h = {heights}. Check F, J and every side inequality over the full configured domain with exact arithmetic at depth {plan["maxdepth"]}.')
    scope = plan['scope']
    total = len(candidates)
    overview(overview_path, objective, method, f'0/{total} candidates checked. {scope}')
    print(f'CPU script started: {total} candidates; no model calls.', flush=True)
    started = time.monotonic()

    def progress(index, row):
        message = f'{index + 1}/{total}: ell={row["candidate"]["ell"]}, h={row["candidate"]["h"]} → {row["status"]}'
        print(message, flush=True)
        overview(overview_path, objective, method, message + '. ' + scope)

    exact_plan = dict(scope=scope, certifier=str(source), certifier_sha256=plan['certifier_sha256'],
                      box=plan['box'], maxdepth=plan['maxdepth'], candidates=candidates)
    receipt = search(exact_plan, output / 'screen', on_candidate=progress)
    if digest(source.read_bytes()) != plan['certifier_sha256']:
        raise ValueError('Certifier changed during execution; results require review')
    rows = receipt['candidates']
    accepted = [row for row in rows if row['status'] == 'EXACT_CONDITIONAL']
    incomplete = sum(row['status'] == 'INCOMPLETE' for row in rows)
    best = min(accepted, key=lambda row: Fraction(row['sigma'])) if accepted else None
    summary = f'Checked {total} candidates: {len(accepted)} exact conditional certificates, {total - len(accepted) - incomplete} not certified, {incomplete} incomplete.'
    if best:
        summary += f' Best σ₀ = {best["sigma"]} at ℓ = {best["candidate"]["ell"]}, h = {best["candidate"]["h"]}.'
    summary += ' No Lean theorem or record claim. No model was used.'
    result = dict(status='INCOMPLETE' if incomplete else 'COMPLETED', summary=summary,
                  total=total, certified=len(accepted), incomplete=incomplete,
                  best=None if best is None else dict(candidate=best['candidate'], sigma=best['sigma']),
                  wall_seconds=time.monotonic() - started, model_calls=0,
                  screen_sha256=digest((output / 'screen/receipt.json').read_bytes()))
    save_json(output / 'cpu-result.json', result)
    overview(overview_path, objective, method, summary)
    print(summary, flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--overview', required=True)
    args = parser.parse_args()
    try:
        result = run(args.plan, args.output, args.overview)
        sys.exit(0 if result['status'] == 'COMPLETED' else 1)
    except Exception as error:
        print(f'CPU search failed: {error}', file=sys.stderr, flush=True)
        sys.exit(1)
