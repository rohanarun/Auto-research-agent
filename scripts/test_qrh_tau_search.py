"""Acceptance boundaries for the model-free tau search client."""
import copy
from fractions import Fraction
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'Resources/LeanSearch'))
from core import digest
from qrh_tau_search import checked_receipt, enumerate_candidates


class TauSearchTests(unittest.TestCase):
    def test_strict_central_budget_boundary(self):
        rows, best = enumerate_candidates(dict(denominator_min=33722, denominator_max=40000,
                                               central_cost='15/8', central_margin='49/881280'))
        self.assertFalse(rows[0]['central_budget'])
        self.assertEqual(best['N'], 33723)
        self.assertEqual(Fraction(best['theta']), Fraction(236059, 269784))
        self.assertLess(Fraction(best['theta']), Fraction(139999, 160000))

    def receipt(self):
        candidate = dict(id='test-job', profile='test-profile', files={'Candidate.lean': 'source'})
        verification = dict(plan_sha256=digest(b'exact-target-plan'), targets=['exact_target'])
        status = dict(id=candidate['id'], profile=candidate['profile'], status='LEAN_CHECKED',
                      request_sha256=digest(json.dumps(candidate, sort_keys=True).encode()),
                      receipt=dict(status='LEAN_CHECKED', **verification))
        return candidate, verification, status

    def test_success_requires_exact_source_and_operator_plan(self):
        candidate, verification, status = self.receipt()
        self.assertTrue(checked_receipt(status, candidate, verification))
        for key, value in [('id', 'another-job'), ('profile', 'another-profile'),
                           ('request_sha256', digest(b'different-source')), ('receipt', {})]:
            changed = copy.deepcopy(status); changed[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                checked_receipt(changed, candidate, verification)
        for key, value in [('plan_sha256', digest(b'weaker-target-plan')),
                           ('targets', ['scalar_only']), ('status', 'REJECTED')]:
            changed = copy.deepcopy(status); changed['receipt'][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                checked_receipt(changed, candidate, verification)

    def test_failed_or_incomplete_proofs_are_never_promoted(self):
        candidate, verification, status = self.receipt()
        for state in ['QUEUED', 'RUNNING', 'REJECTED', 'ERROR', 'INTERRUPTED', 'RESTORE_FAILED']:
            status['status'] = state
            with self.subTest(state=state):
                self.assertFalse(checked_receipt(status, candidate, verification))


if __name__ == '__main__':
    unittest.main()
