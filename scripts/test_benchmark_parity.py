import copy
import unittest
from benchmark_parity import REQUIRED_EQUAL, compare_initial_pair, cli_request, digest


class ParityTests(unittest.TestCase):
    def setUp(self):
        self.request = {'arguments': ['exec', '--model', 'fixture'], 'input': 'Entire instructions and user request'}
        self.a = {k: 'fixture-equal-' + k for k in REQUIRED_EQUAL}
        self.a.update(actual_id='fixture-cli', actual_user_messages=[self.request['input']],
                      actual_arguments=self.request['arguments'], prior_sessions=0,
                      auth_only_state=True, observed_network_denied=True)
        self.b = copy.deepcopy(self.a); self.b['actual_id'] = 'fixture-native'

    def test_equal_complete_fixture_and_every_mismatched_control(self):
        self.assertTrue(compare_initial_pair(self.a, self.b, self.request)['passed'])
        for key in REQUIRED_EQUAL:
            with self.subTest(key=key):
                changed = copy.deepcopy(self.b); changed[key] = 'different'
                self.assertFalse(compare_initial_pair(self.a, changed, self.request)['passed'])

    def test_missing_effort_in_both_is_unknown_not_equal(self):
        self.a['resolved_effort'] = self.b['resolved_effort'] = None
        self.assertFalse(compare_initial_pair(self.a, self.b, self.request)['passed'])

    def test_hidden_prompt_state_and_resume_differences_reject(self):
        for key, value in [('actual_user_messages', ['shorter task']), ('prior_sessions', 1),
                           ('actual_user_messages', [self.request['input'], 'hidden extra instruction']),
                           ('auth_only_state', False), ('actual_arguments', ['exec', 'resume']),
                           ('observed_network_denied', False), ('actual_id', self.a['actual_id'])]:
            with self.subTest(key=key):
                changed = copy.deepcopy(self.b); changed[key] = value
                self.assertFalse(compare_initial_pair(self.a, changed, self.request)['passed'])

    def test_control_uses_full_native_bytes_and_cannot_use_fixture_successor(self):
        row = dict(sharing=False, agents=1, interval_seconds=0, agent_index=1,
                   generation=1, input=self.request['input'], input_sha256=digest(self.request['input']),
                   arguments=self.request['arguments'], fixture_context=False)
        self.assertEqual(cli_request({'requests': [row]}, False, 1, 0, 1), self.request)
        row['input'] += '\nhidden amendment'
        with self.assertRaises(ValueError): cli_request({'requests': [row]}, False, 1, 0, 1)


if __name__ == '__main__':
    unittest.main()
