"""Authenticated, persistent verification queue with operator-owned profiles."""
import argparse
import base64
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import queue
import secrets
import shutil
import signal
import sys
import threading
import uuid
import fcntl
from core import digest, inside, read_json, save_json, verify


def mac_sandbox(project, compiler_root, destination, toolchain_roots=()):
    if sys.platform != 'darwin':
        raise ValueError('macOS sandbox requested on another platform')
    def literal(p):
        return json.dumps(str(Path(p).resolve()))
    reads = ['/System', '/usr', '/bin', '/sbin', '/Library', '/dev', compiler_root, project, *toolchain_roots]
    profile = '(version 1)\n(deny default)\n(allow process*)\n(allow sysctl-read)\n'
    profile += '(allow mach-lookup)\n(allow file-read-metadata)\n(allow file-read* (literal "/"))\n'
    profile += '(allow file-read* ' + ' '.join('(subpath ' + literal(p) + ')' for p in reads) + ')\n'
    profile += '(allow file-write* (require-all (subpath ' + literal(Path(project) / '.lake') + ') (require-not (subpath ' + literal(Path(project) / '.lake/packages') + '))) (subpath ' + literal(Path(project) / '.lean-search') + ') (literal "/dev/null"))\n'
    profile += '(deny file-write* (subpath ' + literal(Path(project) / '.lake/packages') + '))\n'
    Path(destination).write_text(profile)
    return ['/usr/bin/sandbox-exec', '-f', str(destination)]


class Service:
    def __init__(self, config_path):
        self.config_path = Path(config_path).resolve()
        self.config = read_json(config_path)
        self.root = Path(self.config['state_directory']).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock_file = (self.root / 'server.lock').open('a')
        fcntl.flock(self.lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        self.jobs = self.root / 'jobs'
        self.jobs.mkdir(exist_ok=True)
        self.key = Path(self.config['token_file']).read_text().strip()
        if len(self.key) < 32:
            raise ValueError('Generate a server token with at least 32 characters')
        self.queue = queue.Queue()
        self.guard = threading.Lock()
        self.closing = False
        self.profile_locks = []
        for profile in self.config['profiles'].values():
            folder = Path(profile['project']).resolve()
            if not folder.is_dir():
                raise ValueError('Prepared profile project is missing: ' + str(folder))
            lock = (folder / '.lean-verifier.lock').open('a')
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.profile_locks.append(lock)
        # Restore uploads from interrupted runs before accepting another candidate.
        for folder in self.jobs.iterdir():
            if not (folder / 'status.json').exists():
                save_json(folder / 'status.json', {'id': folder.name, 'status': 'INTERRUPTED', 'error': 'Admission was interrupted; inspect the retained request before resubmitting'})
                continue
            status = read_json(folder / 'status.json')
            if status['status'] in ('QUEUED', 'RUNNING'):
                for process_file in (folder / 'verification').glob('*.process.json'):
                    process = read_json(process_file)
                    if process['state'] == 'running':
                        try:
                            os.kill(process['pid'], 0)
                        except ProcessLookupError:
                            pass
                        else:
                            raise ValueError('A previous verifier process may still be running; inspect job ' + folder.name)
                self.restore(folder)
                status['status'] = 'INTERRUPTED'
                save_json(folder / 'status.json', status)
        self.worker = threading.Thread(target=self.work)
        self.worker.start()

    def restore(self, folder):
        backup = folder / 'restore.json'
        if not backup.exists():
            return
        data = read_json(backup)
        for name, content in data['files'].items():
            target = inside(data['project'], name)
            if content is None:
                target.unlink(missing_ok=True)
            else:
                target.write_bytes(base64.b64decode(content))
        backup.rename(folder / 'restored.json')

    def submit(self, request):
        identifier = request.get('id')
        if str(uuid.UUID(identifier)) != identifier:
            raise ValueError('Use a canonical UUID as the idempotency key')
        profile = self.config['profiles'][request['profile']]
        files = request['files']
        if not isinstance(files, dict) or not files:
            raise ValueError('Provide candidate source files')
        if set(files) - set(profile['editable_files']):
            raise ValueError('Candidate changes a file outside the approved profile')
        for name, value in files.items():
            inside(profile['project'], name)
            if not isinstance(value, str):
                raise ValueError('File content must be UTF-8 text')
        encoded = json.dumps(request, sort_keys=True).encode()
        request_hash = digest(encoded)
        folder = self.jobs / identifier
        with self.guard:
            if folder.exists():
                status = read_json(folder / 'status.json')
                if status.get('request_sha256') != request_hash:
                    raise ValueError('Idempotency key already belongs to a different candidate')
                return status
            if self.closing:
                raise ValueError('Server is draining; submit after restart')
            folder.mkdir()
            save_json(folder / 'request.json', request)
            save_json(folder / 'profile.json', profile)
            status = {'id': identifier, 'status': 'QUEUED', 'profile': request['profile'], 'request_sha256': request_hash}
            save_json(folder / 'status.json', status)
            self.queue.put(identifier)
            return status

    def work(self):
        while True:
            identifier = self.queue.get()
            if identifier is None:
                self.queue.task_done()
                return
            if self.closing:
                self.queue.task_done()
                continue
            folder = self.jobs / identifier
            status = read_json(folder / 'status.json')
            profile = read_json(folder / 'profile.json')
            request = read_json(folder / 'request.json')
            project = Path(profile['project']).resolve()
            status['status'] = 'RUNNING'
            save_json(folder / 'status.json', status)
            try:
                old = {name: base64.b64encode(inside(project, name).read_bytes()).decode()
                       if inside(project, name).exists() else None for name in request['files']}
                save_json(folder / 'restore.json', {'project': str(project), 'files': old})
                for name, value in request['files'].items():
                    path = inside(project, name)
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(value)
                if profile['isolation']['kind'] == 'macos':
                    prefix = mac_sandbox(project, profile['isolation']['compiler_root'], folder / 'sandbox.sb',
                                         profile['isolation'].get('toolchain_roots', []))
                else:
                    raise ValueError('This host requires an implemented isolation backend')
                result = verify(profile['plan'], project, folder / 'verification', prefix)
                status['status'] = result['status']
                status['receipt'] = result
            except Exception as error:
                status['status'], status['error'] = 'ERROR', str(error)
            finally:
                try:
                    self.restore(folder)
                except Exception as error:
                    status['status'], status['error'] = 'RESTORE_FAILED', str(error)
                    self.closing = True
                save_json(folder / 'status.json', status)
                self.queue.task_done()


def serve(config_path):
    service = Service(config_path)
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def authorized(self):
            value = self.headers.get('Authorization', '')
            if not hmac.compare_digest(value, 'Bearer ' + service.key):
                self.reply(401, {'error': 'Authentication required'})
                return False
            return True

        def do_GET(self):
            if not self.authorized(): return
            if self.path == '/v1/health':
                return self.reply(200, {'status': 'draining' if service.closing else 'ready',
                                        'profiles': list(service.config['profiles'])})
            if self.path == '/v1/profiles':
                return self.reply(200, {name: {'scope': p['plan']['scope'], 'targets': p['plan']['targets'], 'editable_files': p['editable_files']} for name, p in service.config['profiles'].items()})
            parts = self.path.split('/')
            if len(parts) in (4, 6) and parts[:3] == ['', 'v1', 'jobs']:
                try:
                    identifier = str(uuid.UUID(parts[3]))
                    folder = service.jobs / identifier
                    if len(parts) == 4:
                        return self.reply(200, read_json(folder / 'status.json'))
                    if parts[4] == 'logs':
                        name = parts[5]
                        if not name.endswith('.log') or '/' in name or '..' in name:
                            raise ValueError('Invalid log name')
                        path = folder / 'verification' / name
                        # Explicit bounded log reads; full files are retained on the host.
                        count = service.config['log_tail_chars']
                        with path.open('rb') as log:
                            log.seek(max(0, path.stat().st_size - count * 4))
                            text = log.read().decode(errors='replace')[-count:]
                        return self.reply(200, {'text': text})
                except (ValueError, FileNotFoundError):
                    pass
            self.reply(404, {'error': 'Not found'})

        def do_POST(self):
            if not self.authorized(): return
            if self.path != '/v1/jobs': return self.reply(404, {'error': 'Not found'})
            try:
                length = int(self.headers.get('Content-Length', '0'))
                if not 0 < length <= service.config['max_upload_bytes']:
                    return self.reply(413, {'error': 'Candidate exceeds configured upload limit'})
                request = json.loads(self.rfile.read(length))
                self.reply(202, service.submit(request))
            except (ValueError, KeyError, TypeError) as error:
                self.reply(400, {'error': str(error)})

    server = ThreadingHTTPServer(('127.0.0.1', service.config['port']), Handler)
    endpoint = 'http://127.0.0.1:' + str(server.server_port)
    save_json(service.root / 'server.json', {'url': endpoint, 'pid': os.getpid()})
    print(json.dumps({'url': endpoint, 'status': 'ready'}), flush=True)
    def stop(*_):
        service.closing = True
        # Finish the running proof and preserve queued jobs for explicit resubmission.
        service.queue.put(None)
        threading.Thread(target=server.shutdown, daemon=True).start()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        server.serve_forever()
        service.worker.join()
    finally:
        server.server_close()
        (service.root / 'server.json').unlink(missing_ok=True)


def initialize(directory, lean):
    root = Path(directory).resolve()
    root.mkdir(parents=True, exist_ok=True)
    config_path = root / 'server-config.json'
    if config_path.exists(): return config_path
    lean = Path(lean).resolve()
    project = root / 'profiles/kernel-check'
    project.mkdir(parents=True)
    (project / 'Candidate.lean').write_text('theorem candidate : 1 + 1 = 2 := by decide\n')
    (project / 'Audit.lean').write_text('import Candidate\nexample : 1 + 1 = 2 := candidate\n#print axioms candidate\n')
    token = root / 'server-token'
    token.write_text(secrets.token_urlsafe(32))
    token.chmod(0o600)
    # This kernel fixture exercises the server. It is deliberately not QRH certification.
    plan = {'scope': 'Kernel smoke test: candidate : 1 + 1 = 2', 'targets': ['candidate'],
            'permitted_axioms': ['propext', 'Quot.sound', 'Classical.choice'],
            'stages': [{'kind': 'build', 'argv': [str(lean), '-o', '.lean-search/Candidate.olean', 'Candidate.lean']},
                       {'kind': 'axioms', 'argv': [str(lean), 'Audit.lean']}]}
    plan['stages'][1]['argv'] = [str(lean), 'Audit.lean']
    plan['pins'] = {'Audit.lean': digest((project / 'Audit.lean').read_bytes())}
    save_json(config_path, {'state_directory': str(root), 'token_file': str(token), 'port': 0,
                           'max_upload_bytes': 8388608, 'log_tail_chars': 65536,
                           'profiles': {'kernel-check': {'project': str(project), 'editable_files': ['Candidate.lean'],
                            'isolation': {'kind': 'macos', 'compiler_root': str(lean.parent.parent)}, 'plan': plan}}})
    return config_path
