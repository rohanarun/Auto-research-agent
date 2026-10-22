"""Exact CPU screening against a source-pinned QRH certificate implementation."""
import importlib.util
import json
from pathlib import Path
from fractions import Fraction
from core import digest, save_json


def search(plan, output):
    import sympy as sp
    source = Path(plan['certifier']).resolve()
    if digest(source.read_bytes()) != plan['certifier_sha256']:
        raise ValueError('QRH certifier hash mismatch')
    spec = importlib.util.spec_from_file_location('qrh_certifier', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save_json(output / 'plan.json', plan)
    box = tuple(tuple(sp.Rational(x) for x in edge) for edge in plan['box'])
    results = []
    for index, candidate in enumerate(plan['candidates']):
        # Rational strings prevent silent conversion from binary floating point.
        if not all(isinstance(candidate[k], str) for k in ('ell', 'h')):
            raise ValueError('ell and h must be exact rational strings')
        ell, h = Fraction(candidate['ell']), Fraction(candidate['h'])
        row = {'candidate': candidate, 'status': 'INCOMPLETE', 'scope': plan['scope']}
        try:
            f, j = module.adaptive_poly(ell, h)
            fl, jl = [], []
            fp = bool(module.prove_positive(f, box, maxdepth=plan['maxdepth'], log=fl))
            jp = bool(module.prove_positive(j, box, maxdepth=plan['maxdepth'], log=jl))
            side = module.side_constraints(ell, h)
            row.update({'F_positive': fp, 'J_positive': jp,
                        'side_constraints': {k: str(v) for k, v in side.items()},
                        'F_boxes': [{'box': [[str(v) for v in e] for e in b], 'lower_bound': str(v)} for b, v in fl],
                        'J_boxes': [{'box': [[str(v) for v in e] for e in b], 'lower_bound': str(v)} for b, v in jl],
                        'sigma': str(Fraction(11, 12) - ell / 4)})
            row['status'] = 'EXACT_CONDITIONAL' if fp and jp and side and all(v > 0 for v in side.values()) else 'NOT_CERTIFIED'
        except Exception as error:
            row['error'] = str(error)
        save_json(output / ('candidate-' + str(index) + '.json'), row)
        results.append(row)
    result = {'status': 'SCREENED', 'certifier_sha256': plan['certifier_sha256'],
              'sympy_version': sp.__version__, 'candidates': results,
              'claim': 'Exact finite certificate only, conditional on the source theorem. No Lean or global record claim.'}
    save_json(output / 'receipt.json', result)
    return result
