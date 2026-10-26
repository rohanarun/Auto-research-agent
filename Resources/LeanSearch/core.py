"""Source-bound Lean checks. Plans are trusted operator configuration, never uploads."""
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.incoming')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def inside(root, name):
    root = Path(root).resolve()
    path = root / name
    if not name or Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('Expected a relative file path')
    if not path.resolve().is_relative_to(root) or path.is_symlink():
        raise ValueError('File escapes the assigned directory or is a symlink')
    return path


def manifest(root):
    """Bind source and Lake configuration; dependency pins live in lake-manifest."""
    root = Path(root)
    result = {}
    for folder, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in ('.git', '.lake', '.lean-search')
                         and not (Path(folder) / d).is_symlink())
        for name in sorted(files):
            p = Path(folder) / name
            if name.endswith('.lean') or name in ('lean-toolchain', 'lake-manifest.json', 'lakefile.toml'):
                if p.is_symlink():
                    raise ValueError('Source symlinks are not supported: ' + str(p))
                result[str(p.relative_to(root))] = digest(p.read_bytes())
    return result


def audit_axioms(output, targets, allowed):
    found = {}
    for name, values in re.findall(r"'([^']+)' depends on axioms:\s*\[([^\]]*)\]", output, re.S):
        if name in found:
            raise ValueError('Duplicate axiom audit for ' + name)
        found[name] = [v.strip() for v in values.split(',') if v.strip()]
    for name in re.findall(r"'([^']+)' does not depend on any axioms", output):
        if name in found:
            raise ValueError('Duplicate axiom audit for ' + name)
        found[name] = []
    if not targets or any(name not in found for name in targets):
        raise ValueError('Missing terminal theorem axiom audit')
    if any(set(found[name]) - set(allowed) for name in targets):
        raise ValueError('Unapproved axioms in terminal theorem')
    return {name: found[name] for name in targets}


def run_command(argv, cwd, log, prefix=(), timeout=None):
    if not isinstance(argv, list) or not argv or not all(isinstance(a, str) for a in argv):
        raise ValueError('Commands must be nonempty argument arrays')
    start = time.time()
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', TMPDIR=str(Path(cwd) / '.lean-search/tmp'), LEAN_PATH=str(Path(cwd) / '.lean-search'))
    # Verification children do not inherit server or cloud credentials.
    env = {k: v for k, v in env.items() if not any(s in k.upper() for s in ('TOKEN', 'SECRET', 'API_KEY', 'PASSWORD'))}
    Path(env['TMPDIR']).mkdir(parents=True, exist_ok=True)
    with Path(log).open('wb') as output:
        process = subprocess.Popen([*prefix, *argv], cwd=cwd, env=env,
                                   stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        process_file = Path(str(log) + '.process.json')
        save_json(process_file, {'pid': process.pid, 'state': 'running', 'argv': [*prefix, *argv]})
        try:
            code = process.wait(timeout=timeout)
            state = 'completed'
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            code, state = process.returncode, 'timeout'
        except BaseException:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait()
            save_json(process_file, {'pid': process.pid, 'state': 'interrupted', 'exit_code': process.returncode})
            raise
        save_json(process_file, {'pid': process.pid, 'state': state, 'exit_code': code})
    return {'argv': argv, 'exit_code': code, 'state': state,
            'seconds': time.time() - start, 'log': Path(log).name,
            'log_sha256': digest(Path(log).read_bytes())}


def verify(plan, project, output, prefix=()):
    project, output = Path(project).resolve(), Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    if not plan.get('scope') or not plan.get('targets'):
        raise ValueError('A verification plan needs an exact scope and terminal targets')
    stages = plan['stages']
    kinds = [s['kind'] for s in stages]
    if kinds.count('build') != 1 or kinds.count('axioms') != 1 or kinds.count('comparator') > 1:
        raise ValueError('Require one build, one axiom audit, and at most one Comparator')
    if kinds != sorted(kinds, key={'build': 0, 'axioms': 1, 'comparator': 2}.__getitem__):
        raise ValueError('Check build, axiom audit, then Comparator')
    before = manifest(project)
    save_json(output / 'plan.json', plan)
    save_json(output / 'sources.json', before)
    result = {'status': 'RUNNING', 'scope': plan['scope'], 'targets': plan['targets'],
              'plan_sha256': digest((output / 'plan.json').read_bytes()),
              'source_sha256': digest(json.dumps(before, sort_keys=True).encode()), 'stages': [],
              'dependency_scope': 'Lake lockfile and operator-supplied dependency pins; not an independent dependency audit'}
    save_json(output / 'receipt.json', result)
    try:
        for path, expected in plan.get('pins', {}).items():
            if digest(inside(project, path).read_bytes()) != expected:
                raise ValueError('Pinned input changed: ' + path)
        for i, stage in enumerate(stages):
            log = output / (str(i) + '-' + stage['kind'] + '.log')
            record = run_command(stage['argv'], project, log, prefix, plan.get('timeout_seconds'))
            record['kind'] = stage['kind']
            result['stages'].append(record)
            if record['state'] != 'completed' or record['exit_code'] != 0:
                result['status'] = 'INCOMPLETE' if record['state'] == 'timeout' else 'REJECTED'
                break
            text = log.read_text(errors='replace')
            if stage['kind'] == 'axioms':
                record['axioms'] = audit_axioms(text, plan['targets'], plan['permitted_axioms'])
            if stage['kind'] == 'comparator':
                if not stage.get('success_markers') or not all(m in text for m in stage['success_markers']):
                    raise ValueError('Comparator did not confirm acceptance')
            save_json(output / 'receipt.json', result)
        else:
            result['status'] = 'COMPARATOR_ACCEPTED' if 'comparator' in kinds else 'LEAN_CHECKED'
        for path, expected in plan.get('pins', {}).items():
            if digest(inside(project, path).read_bytes()) != expected:
                raise ValueError('Pinned input changed during verification: ' + path)
        if before != manifest(project):
            raise ValueError('Source changed during verification; result cannot be reused')
    except Exception as error:
        result['status'], result['error'] = 'REJECTED', str(error)
    save_json(output / 'receipt.json', result)
    return result
