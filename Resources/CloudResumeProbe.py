"""Read-only preflight: disclose no credentials, prompts, command lines or file contents."""
from pathlib import Path
import json, os, re, sys


def inspect(state_id, provider_id, root=Path('/research-private'), checkout=Path('/workspace/checkout'), proc=Path('/proc')):
    if not all(re.fullmatch(r'[A-Za-z0-9_-]+', value) for value in (state_id, provider_id)):
        return {'status': 'BLOCKED_INVALID_ID'}
    state = root / 'providers' / state_id / 'state'
    if not checkout.is_dir() or not state.is_dir() or not (state / 'auth.json').is_file():
        return {'status': 'BLOCKED_MISSING_STATE'}
    if not any((state / 'sessions').rglob('*' + provider_id + '*.jsonl')):
        return {'status': 'BLOCKED_MISSING_PROVIDER_SESSION'}

    def alive(pid):
        if not isinstance(pid, int) or pid <= 0:
            return False
        try:
            return (proc / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()[0] != 'Z'
        except FileNotFoundError:
            return False
        # Permissions/parse errors deliberately propagate: uncertainty blocks resume.

    for folder in (root / 'operations').iterdir():
        if not folder.is_dir():
            continue
        for filename in ('launch.json', 'cli.json'):
            path = folder / filename
            if path.exists() and alive(json.loads(path.read_text()).get('pid')):
                return {'status': 'BLOCKED_ACTIVE_OPERATION'}
        status = folder / 'status.json'
        if status.exists():
            remaining = json.loads(status.read_text()).get('cleanup') or {}
            if any(alive(pid) for pid in remaining.get('remaining_pids', [])):
                return {'status': 'BLOCKED_UNDRAINED_OPERATION'}
    # Catch unrecorded/orphaned research processes as well as owned operation receipts.
    # Ignore this probe and its control ancestors (sudo/shell/exec server), not
    # other shells or researchers sharing the checkout. Only numeric PIDs leave proc.
    ancestors = set()
    pid = os.getpid()
    while pid > 0 and pid not in ancestors:
        ancestors.add(pid)
        try:
            pid = int((proc / str(pid) / 'stat').read_text().rsplit(')', 1)[1].split()[1])
        except FileNotFoundError:
            break
    for process in proc.iterdir():
        if not process.name.isdigit() or int(process.name) in ancestors:
            continue
        try:
            cwd = (process / 'cwd').resolve(strict=True)
            if cwd == checkout.resolve() or checkout.resolve() in cwd.parents:
                if alive(int(process.name)):
                    return {'status': 'BLOCKED_CHECKOUT_PROCESS'}
        except FileNotFoundError:
            continue
    return {'status': 'READY_TO_RESUME_EXISTING_STATE'}


if __name__ == '__main__':
    try:
        result = inspect(*sys.argv[1:])
    except Exception:
        result = {'status': 'BLOCKED_UNCERTAIN_PROCESS_OR_STATE'}
    print(json.dumps(result))
