#!/usr/bin/env python3
"""Configure the web edition workspace on Windows, Linux or macOS.

Creates the runtime workspace, discovers the provider CLIs from PATH and
writes a private Defaults.json, mirroring scripts/configure.py without the
macOS-only external-volume rule. Existing settings are preserved.
"""
import json
import os
import pathlib
import shutil
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent


def configure() -> pathlib.Path:
    workspace = pathlib.Path(os.environ.get("RESEARCH_WORKSPACE") or (ROOT / "Runtime"))
    workspace.mkdir(parents=True, exist_ok=True)
    for component in ("projects", "sessions", "cache", "tmp"):
        (workspace / component).mkdir(parents=True, exist_ok=True)

    resources = ROOT / "Resources"
    defaults_file = resources / "Defaults.json"
    settings = {}
    if defaults_file.exists():
        settings = json.loads(defaults_file.read_text(encoding="utf-8"))
    else:
        example = resources / "Defaults.example.json"
        settings = json.loads(example.read_text(encoding="utf-8")) if example.exists() else {}
    settings["workspaceRoot"] = str(workspace)
    for provider, command in (("codex", "codex"), ("claude", "claude"),
                              ("opencode", "opencode"), ("antigravity", "agy")):
        override = os.environ.get("RESEARCH_" + provider.upper() + "_PATH")
        executable = override or shutil.which(command) or ""
        if executable:
            executable = str(pathlib.Path(os.path.abspath(os.path.expanduser(executable))))
        settings[provider + "Path"] = executable
        if not executable:
            print(provider + " CLI not found; set its executable path in Connections "
                  "after installing it.")
    defaults_file.write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    try:
        defaults_file.chmod(0o600)
    except OSError:
        pass
    print("Configured web edition workspace:", workspace)
    print("Start the dashboard: python -m webapp.server --workspace \"" + str(workspace) + "\"")
    return workspace


if __name__ == "__main__":
    try:
        configure()
    except OSError as error:
        print("Setup failed: " + str(error))
        sys.exit(1)
