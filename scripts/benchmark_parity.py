"""Credential-free request controls. Does not launch a CLI or call a provider."""
import hashlib
import json
import re
from pathlib import Path


def digest(value):
    return hashlib.sha256(value.encode()).hexdigest()


def cli_request(contracts, sharing, agents, interval_seconds, agent_index):
    """The CLI control consumes the exact production-native first request bytes.

    This deliberately tests a prompt-matched CLI, not vanilla Codex defaults.
    Actual successor content is generated per trial, never taken from a fixture.
    """
    rows = [r for r in contracts['requests'] if
            (r['sharing'], r['agents'], r['interval_seconds'], r['agent_index'], r['generation'])
            == (sharing, agents, interval_seconds, agent_index, 1)]
    if len(rows) != 1:
        raise ValueError('Exactly one initial production-native request required')
    row = rows[0]
    if digest(row['input']) != row['input_sha256'] or row['fixture_context']:
        raise ValueError('Contract changed or fixture context supplied for research')
    return {'arguments': list(row['arguments']), 'input': row['input']}


def actual_transcript(path):
    """Read actual model-visible context, not just requested launcher settings."""
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    meta = next(r['payload'] for r in rows if r['type'] == 'session_meta')
    turns = [r['payload'] for r in rows if r['type'] == 'turn_context']
    users, developers = [], []
    for row in rows:
        p = row.get('payload', {})
        if row['type'] == 'response_item' and p.get('role') in ('user', 'developer'):
            texts = [v.get('text', '') for v in p.get('content', [])]
            (users if p['role'] == 'user' else developers).extend(texts)
    # This is the sole normalization: a fresh private provider-state UUID.
    # No prose, instructions, workspace paths or checkpoint bytes are changed.
    developers = [re.sub(r'/research-private/providers/[0-9A-Fa-f-]+/state',
                         '/research-private/providers/STATE/state', t) for t in developers]
    turn = turns[0] if turns else {}
    effort = turn.get('reasoning_effort') or turn.get('effort') or turn.get('collaboration_mode', {}).get('settings', {}).get('reasoning_effort')
    return {'actual_id': meta['id'], 'cli_version': meta.get('cli_version'),
            'account_identity_sha256': digest(meta['creator_account_id']) if meta.get('creator_account_id') else None,
            'base_instructions_sha256': digest(meta['base_instructions']['text']) if meta.get('base_instructions', {}).get('text') else None,
            'developer_context_sha256': digest(json.dumps(developers, ensure_ascii=False)) if developers else None,
            'actual_user_messages': users, 'actual_model': turn.get('model'),
            'resolved_effort': effort, 'approval_policy': turn.get('approval_policy'),
            'sandbox_policy': turn.get('sandbox_policy'), 'cwd': meta.get('cwd')}


REQUIRED_EQUAL = ('cli_version', 'cli_binary_sha256', 'container_image_digest',
                  'python_version', 'compiler_version', 'baseline_tree_sha256',
                  'worker_resources', 'checker_resources', 'checker_admission',
                  'account_identity_sha256', 'base_instructions_sha256',
                  'developer_context_sha256', 'actual_model', 'resolved_effort',
                  'approval_policy', 'sandbox_policy', 'cwd', 'clock_boundary',
                  'schedule_block', 'tools_schema_sha256', 'model_catalog_sha256')


def compare_initial_pair(cli, native, request):
    errors = []
    for name in REQUIRED_EQUAL:
        if cli.get(name) is None or native.get(name) is None:
            errors.append(name + ': actual evidence missing')
        elif cli[name] != native[name]:
            errors.append(name + ': mismatch')
    if cli.get('actual_user_messages') != native.get('actual_user_messages'):
        errors.append('Full actual user-message sequence mismatch, including extra instructions')
    for arm, observed in [('cli', cli), ('native', native)]:
        if request['input'] not in observed.get('actual_user_messages', []):
            errors.append(arm + ': full actual initial prompt mismatch')
        if observed.get('actual_arguments') != request['arguments']:
            errors.append(arm + ': actual CLI arguments mismatch')
        if observed.get('prior_sessions') != 0 or observed.get('auth_only_state') is not True:
            errors.append(arm + ': fresh auth-only state unverified')
        if observed.get('observed_network_denied') is not True:
            errors.append(arm + ': actual network restriction unverified')
    if not cli.get('actual_id') or not native.get('actual_id') or cli.get('actual_id') == native.get('actual_id'):
        errors.append('Fresh distinct actual provider sessions required')
    return {'passed': not errors, 'errors': errors,
            'scope': 'Initial request/runtime parity only, not a research result',
            'later_context_content_expected_to_diverge': True}


def main(root):
    root = Path(root)
    contracts = json.loads((root / 'native-request-contracts.json').read_text())
    controls = []
    for row in contracts['requests']:
        if row['generation'] != 1:
            continue
        request = cli_request(contracts, row['sharing'], row['agents'], row['interval_seconds'], row['agent_index'])
        assert request['input'] == row['input'] and request['arguments'] == row['arguments']
        controls.append({k: row[k] for k in ('sharing', 'agents', 'interval_seconds', 'agent_index', 'input_sha256')} | request)
    proof = {'status': 'OFFLINE ONLY; actual provider parity not verified',
             'initial_requests': len(controls), 'requests': controls,
             'byte_equal_to_actual_native_local_capture': True,
             'model_calls': 0, 'launch_allowed': False,
             'pending': ['Actual transcript and runtime pair audit', 'Resolved actual reasoning effort',
                         'Common launch barrier integration into both versioned drivers',
                         'Corrected target endpoint driver integration', 'Provider capacity']}
    (root / 'cli-request-contracts.json').write_text(json.dumps(proof, indent=2) + '\n')
    return {k: v for k, v in proof.items() if k != 'requests'}


if __name__ == '__main__':
    import sys
    print(json.dumps(main(sys.argv[1]), indent=2))
