"""Optional Agent37/Daytona control adapters for already-owned remote search nodes.

Use these only for bounded control commands. Run long searches under the provider's
persistent worker or nohup and retain the operation handle; never retry a timed-out
mutation. Lean compilation belongs on the shared verification server.
"""
import json
import os
from urllib.parse import quote, urlparse
import urllib.request
from cli import NoRedirect


def execute(config, command):
    provider = config['provider']
    key = os.environ[config['key_env']]
    identifier = quote(config['instance_id'], safe='')
    if provider == 'agent37':
        url = config['base_url'].rstrip('/') + '/v1/instances/' + identifier + '/exec'
        headers = {'X-Agent37-Key': key}
        body = {'command': command}
    elif provider == 'daytona':
        url = config['base_url'].rstrip('/') + '/toolbox/' + identifier + '/process/execute'
        headers = {'Authorization': 'Bearer ' + key}
        body = {'command': command, 'timeout': config['control_timeout_seconds']}
    else:
        raise ValueError('Unknown remote backend')
    if urlparse(url).scheme != 'https':
        raise ValueError('Provider credentials require HTTPS')
    headers['Content-Type'] = 'application/json'
    req = urllib.request.Request(url, json.dumps(body).encode(), headers)
    with urllib.request.build_opener(NoRedirect()).open(req, timeout=config['transport_timeout_seconds']) as response:
        value = json.load(response)
    if provider == 'agent37':
        if value.get('truncated'):
            raise ValueError('Provider truncated output; retrieve retained log files')
        return {'exit_code': value['exit_code'], 'stdout': value['stdout'], 'stderr': value['stderr']}
    return {'exit_code': value['exitCode'], 'stdout': value.get('result', ''), 'stderr': ''}
