#!/usr/bin/env python3
"""Linux + Claude CLI variant of the structured-action broker (harness "claude-cli-broker-linux-v1").

Same contract as codex-cli-broker-v1: a tool-free model returns one {command, done, note} JSON action per step; a
separate broker runs the command and returns the observation. Differences, recorded as a distinct harness version:
  * inference: Claude Code CLI in print mode, every built-in tool disabled (--tools ""), no MCP servers, no settings or
    memory files, structured output enforced with --json-schema, run from an empty private directory. Any tool call
    other than the structured-output channel invalidates the trial (fail closed, as in the Codex path).
  * command broker: Linux namespaces instead of macOS Seatbelt. Each command runs as uid 65534 in fresh mount, network,
    pid, ipc and uts namespaces. The private root holds read-only system directories, the read-only benchmark venv
    and the read-write trial workspace (both at their host paths); nothing else from the host is mounted (no /home
    contents, /root, /mnt, curator files, credentials or other trials). There is no network interface but loopback.
  * prompts: identical text except "Your Mac shell" -> "Your Linux shell" in the GPU protocol paragraph.
Usage:
  python3.12 linux_claude.py preflight
  python3.12 linux_claude.py kernels --model M [--model M2 ...] --cohort C [--workers N]
  python3.12 linux_claude.py task --model M --task ID --cohort C
"""
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote
import argparse, concurrent.futures, hashlib, importlib.util, json, os, shlex, shutil, subprocess, sys, threading, time

CODE = Path(__file__).resolve().parent
sys.path.insert(0, str(CODE)); sys.path.insert(0, str(CODE / 'gpu'))
import bench as b

HARNESS = 'claude-cli-broker-linux-v1'
CLAUDE = os.environ.get('AUTO_RESEARCH_CLAUDE_BINARY') or shutil.which('claude') or 'claude'
JAIL_UID = 65534
SYSTEM_DIRS = ['usr', 'bin', 'lib', 'lib64', 'sbin', 'etc']

JAIL = r'''
set -e
N=$(mktemp -d /run/arjail.XXXXXX); mount -t tmpfs -o size=64m tmpfs "$N"
for d in __SYSDIRS__; do
  [ -e /$d ] || continue
  if [ -L /$d ]; then ln -s "$(readlink /$d)" "$N/$d"; else mkdir -p "$N/$d"; mount --rbind "/$d" "$N/$d"; mount -o remount,ro,bind "$N/$d" 2>/dev/null || true; fi
done
mkdir -p "$N/dev" "$N/proc" "$N/tmp" "$N$AR_VENV" "$N$AR_WORK"
mount --rbind /dev "$N/dev"; mount -t proc proc "$N/proc"; mount -t tmpfs -o size=4g tmpfs "$N/tmp"; chmod 1777 "$N/tmp"
mount --bind "$AR_VENV" "$N$AR_VENV"; mount -o remount,ro,bind "$N$AR_VENV"
mount --bind "$AR_WORK" "$N$AR_WORK"
ip link set lo up 2>/dev/null || true
exec chroot --userspec=__UID__:__UID__ "$N" /usr/bin/env -i PATH="$AR_VENV/bin:/usr/local/bin:/usr/bin:/bin" HOME="$AR_WORK/.home" TMPDIR="$AR_WORK/.tmp" \
  PYTHONDONTWRITEBYTECODE=1 PYTHONPYCACHEPREFIX="$AR_WORK/.cache/pycache" XDG_CACHE_HOME="$AR_WORK/.cache" OPENBLAS_NUM_THREADS=1 \
  OMP_NUM_THREADS=1 GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null bash --noprofile --norc -c "cd \"$AR_WORK\" && $AR_CMD"
'''.replace('__SYSDIRS__', ' '.join(SYSTEM_DIRS)).replace('__UID__', str(JAIL_UID))


def jail_cmd(work, command):
    env = {'AR_WORK': str(work), 'AR_VENV': str(b.ROOT / '.venv'), 'AR_CMD': command, 'PATH': '/usr/sbin:/usr/bin:/sbin:/bin'}
    return ['unshare', '--mount', '--net', '--pid', '--ipc', '--uts', '--fork', '--kill-child', 'bash', '-c', JAIL], env


def shell(run, command, timeout=120):
    work = run / 'work'; stamp = str(time.time_ns()); out = run / ('shell-' + stamp + '.log')
    started = time.monotonic()
    argv, env = jail_cmd(work, command)
    with out.open('w') as f:
        proc = subprocess.Popen(argv, cwd='/', env=env, stdout=f, stderr=subprocess.STDOUT, start_new_session=True)
        with (run / 'command-groups.jsonl').open('a') as groups:
            groups.write(json.dumps({'pgid': proc.pid}) + '\n')
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            b.kill_group(proc); code = 124; proc.wait()
    content = out.read_text(errors='replace'); limit = 24000
    return {'exit_code': code, 'elapsed_seconds': round(time.monotonic() - started, 3),
            'output': content if len(content) <= limit else content[:16000] + '\n[output truncated; full output remains in broker log]\n' + content[-8000:],
            'log': out.name}


def init_run(run, snapshot):
    assert run.resolve().is_relative_to((b.ROOT / 'runs').resolve()), 'Trial destination escapes configured workspace'
    run.mkdir(parents=True, exist_ok=False); shutil.copytree(snapshot, run / 'work')
    work = run / 'work'
    for x in ['.home', '.tmp', '.cache', 'solution']:
        (work / x).mkdir(exist_ok=True)
    (run / 'inference-home').mkdir()
    subprocess.run(['chown', '-R', f'{JAIL_UID}:{JAIL_UID}', str(work)], check=True)
    os.chmod(run, 0o755)


def _sample_once(run, prompt, model, effort, index, timeout):
    home = run / 'inference-home'; home.mkdir(exist_ok=True)
    empty = home / 'cwd'; empty.mkdir(exist_ok=True)
    events = run / f'model-{index:04d}.events.jsonl'; err = run / f'model-{index:04d}.stderr.log'
    (run / f'model-{index:04d}.prompt.txt').write_text(prompt)
    cmd = [CLAUDE, '-p', '--model', model, '--effort', effort, '--tools', '', '--strict-mcp-config', '--setting-sources', '',
           '--disallowedTools', 'mcp__*', '--json-schema', json.dumps(b.ACTION_SCHEMA), '--output-format', 'stream-json',
           '--verbose', '--no-session-persistence']
    env = {k: v for k, v in os.environ.items() if not k.startswith(('DAYTONA', 'SUPER_EXPERIMENTS', 'AUTO_RESEARCH', 'QUEUE_'))}
    with events.open('w') as f, err.open('w') as e:
        proc = subprocess.Popen(cmd, cwd=empty, env=env, stdin=subprocess.PIPE, stdout=f, stderr=e, start_new_session=True)
        try:
            proc.communicate(prompt.encode(), timeout=max(1, timeout))
        except subprocess.TimeoutExpired:
            b.kill_group(proc); proc.wait(); raise TimeoutError('Model reached trial time limit')
    usage, action, native, error = {}, None, [], None
    for line in events.read_text().splitlines():
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = ev.get('type')
        if t == 'system' and ev.get('subtype') == 'init':
            tools = [x for x in ev.get('tools', []) if x != 'StructuredOutput']
            if tools:
                native.append({'init_tools': tools})
        elif t == 'assistant':
            for c in ev.get('message', {}).get('content', []):
                if c.get('type') == 'tool_use' and c.get('name') != 'StructuredOutput':
                    native.append(c)
        elif t == 'result':
            u = ev.get('usage', {})
            usage = {'input_tokens': u.get('input_tokens', 0) + u.get('cache_creation_input_tokens', 0) + u.get('cache_read_input_tokens', 0),
                     'cached_input_tokens': u.get('cache_read_input_tokens', 0), 'output_tokens': u.get('output_tokens', 0)}
            action = ev.get('structured_output')
            if ev.get('is_error') or ev.get('subtype') != 'success':
                error = str(ev.get('result') or ev.get('subtype'))
    if native:
        raise RuntimeError('INVALID_ISOLATION: inference invoked built-in tools: ' + json.dumps(native)[:500])
    if proc.returncode or action is None:
        detail = (error or '') + ' ' + err.read_text()[-1800:] + ' ' + events.read_text()[-1500:]
        # Transient transport failures (provider auth/network/capacity) are retried by bench.sample within the same
        # research clock (wall-clock charged), exactly like the Codex path's capacity retries.
        if any(w in detail.lower() for w in ['authentication', 'network issue', 'overloaded', 'rate limit', 'api_error', 'timed out', 'econnreset']):
            detail = 'transient provider transport (503): ' + detail
        raise RuntimeError('Inference failed: ' + detail)
    if set(action) != set(b.ACTION_SCHEMA['required']) or not isinstance(action['command'], str) or not isinstance(action['done'], bool):
        raise ValueError('Invalid action schema')
    (home / f'model-{index:04d}.json').write_text(json.dumps(action) + '\n')
    return action, usage


def system_prompt(seconds):
    return b.PROMPT.replace('30 minutes', str(seconds) + ' seconds') + '\nTASK:\n'


def install():
    b.shell = shell; b.init_run = init_run; b._sample_once = _sample_once; b.SYSTEM_PROMPT = system_prompt
    original = b.run_task

    def run_task(task, args, cohort):
        meta = original(task, args, cohort)
        meta['harness'] = HARNESS
        meta['machine'] = 'linux-namespaces-broker'
        b.write(b.ROOT / 'runs' / cohort / task['id'] / 'result.json', meta)
        return meta
    b.run_task = run_task
    # The finally-block process sweep uses BSD ps flags; make it portable.
    real_run = subprocess.run

    def run_patch(args, *a, **k):
        if isinstance(args, list) and args[:2] == ['ps', '-axo']:
            args = ['ps', '-eo', 'pid,command']
        return real_run(args, *a, **k)
    b.subprocess.run = run_patch


def preflight():
    b.require_ssd(); run = b.ROOT / 'runs' / ('preflight-linux-' + time.strftime('%Y%m%dT%H%M%S'))
    empty = b.ROOT / 'tests/empty'; empty.mkdir(parents=True, exist_ok=True)
    init_run(run, empty)
    hidden = b.ROOT / 'curator/isolation-canary.txt'
    if not hidden.exists():
        hidden.write_text('hidden-curator-canary')
    probes = {'curator': str(hidden), 'real_repo': str(CODE / 'scenarios.json'), 'other_trial': str(b.ROOT / 'suite.json'),
              'host_home': str(Path.home() / '.bash_history'), 'credential': os.environ.get('AUTO_RESEARCH_CREDENTIAL_PROBE', str(Path.home() / '.netrc'))}
    code = ("import socket, pathlib, json\nr={}\nr['compute']=17*19\np=pathlib.Path('offline-test.txt');p.write_text('local-only');"
            "r['workspace_write']=p.read_text()=='local-only'\n"
            "for name,path in " + repr(sorted(probes.items())) + ":\n try:pathlib.Path(path).read_bytes();r[name]='LEAK'\n"
            " except PermissionError:r[name]='denied'\n except FileNotFoundError:r[name]='denied'\n except Exception as e:r[name]=type(e).__name__\n"
            "for host in ['1.1.1.1','127.0.0.1','github.com']:\n try:\n  s=socket.socket();s.settimeout(2);s.connect((host,443));r[host]='LEAK';s.close()\n"
            " except Exception as e:r[host]='denied' if isinstance(e,(PermissionError,ConnectionRefusedError,OSError)) else type(e).__name__\n"
            "import subprocess\nr['host_processes_visible']=len(subprocess.run(['ps','-e'],capture_output=True,text=True).stdout.splitlines())-1\n"
            "import os\nr['uid']=os.getuid()\nprint(json.dumps(r))\n")
    result = shell(run, shlex.quote(str(b.PYTHON)) + ' -c ' + shlex.quote(code), 60)
    assert result['exit_code'] == 0 and result['output'].strip(), repr(result)
    t = json.loads(result['output'].splitlines()[-1]); result['tests'] = t
    result['passed'] = (t['compute'] == 323 and t['workspace_write'] and all(t[k] == 'denied' for k in probes)
                        and all(t[h] == 'denied' for h in ['1.1.1.1', '127.0.0.1', 'github.com']) and t['uid'] == JAIL_UID
                        and t['host_processes_visible'] <= 5)
    result['harness'] = HARNESS
    b.write(run / 'result.json', result); b.write(b.ROOT / 'preflight.json', {'run': str(run), **result})
    print(json.dumps(result, indent=2)); return result['passed']


def kernels(a):
    install()
    s = importlib.util.spec_from_file_location('run_kernels', CODE / 'tools/run_kernels.py'); rk = importlib.util.module_from_spec(s)
    sys.argv = ['run_kernels.py']
    s.loader.exec_module(rk)   # patches b.init_run / b.sample on top of ours (gpu_test bridge + GPU prompt)
    rk.GPU_PROMPT = rk.GPU_PROMPT.replace('Your Mac shell', 'Your Linux shell')
    base_feedback = rk.q.feedback

    def feedback(directory, name, value):
        # The trusted dispatcher writes results root-owned 0600; hand them to the jailed actor's uid (read-only for
        # it is enough, but the actor owns its workspace, so ownership is the simple fix).
        base_feedback(directory, name, value)
        path = Path(directory) / name
        os.chown(path, JAIL_UID, JAIL_UID); os.chmod(path, 0o644)
    rk.q.feedback = feedback
    base_start = rk.q.Lease.start

    def start(self):
        # Provider provisioning failures are infrastructure failures: retry (wall-clock charged) before giving up.
        for attempt in range(3):
            try:
                return base_start(self)
            except Exception as e:
                print('GPU_LEASE_RETRY', attempt, type(e).__name__, str(e)[:200], flush=True)
                try:
                    self.close()
                except Exception:
                    pass
                if attempt == 2:
                    raise
                time.sleep(10)
    rk.q.Lease.start = start
    base_init = rk.original_init

    def init(run, snapshot):
        base_init(run, snapshot)
        shutil.copyfile(CODE / 'gpu/gpu_test.py', run / 'work/gpu_test.py'); (run / 'work/.gpu').mkdir()
        subprocess.run(['chown', '-R', f'{JAIL_UID}:{JAIL_UID}', str(run / 'work')], check=True)
    b.init_run = init
    sys.argv = ['run_kernels.py'] + sum((['--model', m] for m in a.model), []) + ['--cohort', a.cohort, '--workers', str(a.workers), '--seconds', str(a.seconds)] + sum((['--task', t] for t in (a.task or [])), [])
    rk.main()
    d = json.loads((b.ROOT / 'kernel-matrix.json').read_text()); d['harness'] = HARNESS + '-gpu-queue'
    d['prompt_sha256'] = hashlib.sha256((system_prompt(a.seconds) + rk.GPU_PROMPT).encode()).hexdigest(); b.write(b.ROOT / 'kernel-matrix.json', d)


def task(a):
    install()
    t = json.loads((b.ROOT / 'tasks' / a.task / 'task.json').read_text()); directory = b.ROOT / 'runs' / a.cohort; directory.mkdir(exist_ok=False)
    b.write(directory / 'cohort.json', {'harness': HARNESS, 'model': a.model, 'reasoning_effort': a.effort, 'budget_seconds': a.seconds, 'task': a.task,
                                         'started_at': b.utc(), 'harness_source_sha256': b.digest(Path(__file__)), 'broker_source_sha256': b.digest(CODE / 'bench.py'),
                                         'prompt_sha256': hashlib.sha256(system_prompt(a.seconds).encode()).hexdigest(),
                                         'cli_version': subprocess.run([CLAUDE, '--version'], capture_output=True, text=True).stdout.strip()})
    r = b.run_task(t, SimpleNamespace(model=a.model, effort=a.effort, seconds=a.seconds), a.cohort)
    b.report(a.cohort); return r


def main():
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest='cmd', required=True)
    sub.add_parser('preflight')
    k = sub.add_parser('kernels'); k.add_argument('--model', action='append', required=True); k.add_argument('--cohort', required=True)
    k.add_argument('--workers', type=int, default=3); k.add_argument('--seconds', type=int, default=1800); k.add_argument('--task', action='append')
    t = sub.add_parser('task'); t.add_argument('--model', required=True); t.add_argument('--task', required=True); t.add_argument('--cohort', required=True)
    t.add_argument('--effort', default='high'); t.add_argument('--seconds', type=int, default=1800)
    a = p.parse_args()
    if a.cmd == 'preflight':
        install(); sys.exit(0 if preflight() else 1)
    if a.cmd == 'kernels':
        kernels(a)
    else:
        task(a)


if __name__ == '__main__':
    main()
