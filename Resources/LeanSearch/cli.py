#!/usr/bin/env python3
"""Portable CPU search, verification, and shared-server client."""
import argparse
import json
import os
from pathlib import Path
import sys
import urllib.request
import uuid
from core import read_json, save_json, verify


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *args, **kwargs):
        raise ValueError('Refusing to forward credentials to a redirect')


def request(url, token, payload=None):
    from urllib.parse import urlparse
    parsed = urlparse(url)
    if parsed.scheme != 'https' and not (parsed.scheme == 'http' and parsed.hostname in ('127.0.0.1', 'localhost')):
        raise ValueError('Remote verification requires HTTPS')
    if parsed.username or parsed.password:
        raise ValueError('Credentials must not appear in the URL')
    req = urllib.request.Request(url, data=None if payload is None else json.dumps(payload).encode(),
                                 headers={'Authorization': 'Bearer ' + token, 'Content-Type': 'application/json'})
    # No mutation retry: poll the same UUID after an uncertain submission.
    with urllib.request.build_opener(NoRedirect()).open(req) as response:
        return json.load(response)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    for name in ('qrh-search', 'verify'):
        p = sub.add_parser(name)
        p.add_argument('--plan', required=True)
        p.add_argument('--output', required=True)
        if name == 'verify': p.add_argument('--project', required=True)
    p = sub.add_parser('remote-exec')
    p.add_argument('--config', required=True)
    p.add_argument('--command-file', required=True)
    p = sub.add_parser('init-server')
    p.add_argument('--directory', required=True)
    p.add_argument('--lean', required=True, help='Actual installed Lean binary, not an elan download shim')
    p = sub.add_parser('serve')
    p.add_argument('--config', required=True)
    for name in ('submit', 'status', 'health', 'profiles'):
        p = sub.add_parser(name)
        p.add_argument('--connection', required=True, help='JSON with url and token_file; token never appears in CLI arguments')
        if name == 'submit':
            p.add_argument('--candidate', required=True, help='JSON containing profile, files, and optional metadata')
            p.add_argument('--handle', required=True, help='New durable job-handle file saved before HTTP submission')
        elif name == 'status': p.add_argument('--handle', required=True)
    args = parser.parse_args()
    if args.action == 'qrh-search':
        from qrh import search
        result = search(read_json(args.plan), args.output)
    elif args.action == 'verify':
        result = verify(read_json(args.plan), args.project, args.output)
    elif args.action == 'remote-exec':
        from remote import execute
        result = execute(read_json(args.config), Path(args.command_file).read_text())
    elif args.action == 'init-server':
        from server import initialize
        result = {'config': str(initialize(args.directory, args.lean))}
    elif args.action == 'serve':
        from server import serve
        serve(args.config)
        return
    else:
        config = read_json(args.connection)
        url = config['url'].rstrip('/')
        token_path = Path(config['token_file'])
        if not token_path.is_absolute(): token_path = Path(args.connection).resolve().parent / token_path
        token = token_path.read_text().strip()
        if args.action in ('health', 'profiles'):
            result = request(url + '/v1/' + args.action, token)
        elif args.action == 'status':
            handle = read_json(args.handle)
            result = request(url + '/v1/jobs/' + str(uuid.UUID(handle['id'])), token)
        else:
            candidate = read_json(args.candidate)
            candidate['id'] = str(uuid.uuid4())
            # Exclusive creation prevents overwriting evidence or accidental resubmission.
            with Path(args.handle).open('x') as output:
                json.dump({'id': candidate['id'], 'url': url}, output)
            result = request(url + '/v1/jobs', token, candidate)
    print(json.dumps(result, indent=2))
    if result.get('status') in ('REJECTED', 'INCOMPLETE', 'ERROR', 'RESTORE_FAILED') or result.get('exit_code', 0) != 0:
        sys.exit(1)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps({'status': 'ERROR', 'error': str(error)}), file=sys.stderr)
        sys.exit(1)
