#!/usr/bin/env python3
"""Generate private installation defaults without changing existing runtime settings."""
import json
import os
import pathlib
import shutil
import urllib.request

from check_workspace import validate


def configure():
    project = validate(pathlib.Path(__file__).resolve().parent.parent)
    resources = project / 'Resources'
    settings = json.loads((resources / 'Defaults.example.json').read_text())
    runtime = validate(os.environ.get('RESEARCH_WORKSPACE', str(project / 'Runtime')))
    runtime.mkdir(parents=True, exist_ok=True)
    settings['workspaceRoot'] = str(runtime)
    for provider, command in (('codex', 'codex'), ('claude', 'claude'),
                              ('opencode', 'opencode'), ('antigravity', 'agy')):
        override = os.environ.get('RESEARCH_' + provider.upper() + '_PATH')
        executable = override or shutil.which(command) or ''
        if executable:
            executable = os.path.abspath(os.path.expanduser(executable))
        settings[provider + 'Path'] = executable
        if not executable:
            print(provider + ' CLI not found; set its executable path in Connections after installing it.')
    defaults = resources / 'Defaults.json'
    defaults.write_text(json.dumps(settings, indent=2) + '\n')
    defaults.chmod(0o600)
    catalog = resources / 'catalog-source.html'
    if not catalog.exists():
        # This fetched website snapshot is a build input, not part of the source distribution.
        with urllib.request.urlopen(settings['catalogURL']) as response:
            catalog.write_bytes(response.read())
    for directory in ('tmp', 'build-cache', 'evidence'):
        (project / directory).mkdir(exist_ok=True)
    print('Configured local workspace:', runtime)


if __name__ == '__main__':
    configure()
