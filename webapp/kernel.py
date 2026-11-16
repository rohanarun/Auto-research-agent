"""Built-in kernel setup drafts, mirroring ResearchCore's KernelResearchSetup
and OKCLogin.

A draft holds the user-supplied form values for one built-in project. Saving a
draft never starts research; only an explicit Play renders the configured
request into a visible chat. Validation rules, the rendered JSON block and
the account check mirror the native app exactly, so ``kernel-setups.json``
interchanges between editions.
"""
from __future__ import annotations

import json
import os
import pathlib
import re
import subprocess

from .models import ResearchError

ACCOUNT_PATTERN = re.compile(r"\A[a-zA-Z0-9][a-zA-Z0-9_-]*\Z")
ANSI_PATTERN = re.compile(r"\x1B\[[0-?]*[ -/]*[@-~]")
SIGNED_IN_PATTERN = re.compile(r"(?im)^\s*signed in as\s+@?([a-zA-Z0-9][a-zA-Z0-9_-]*)\s*$")
SUPER_EXPERIMENTS_URL = "https://app.getsupers.com/experiments"

DRAFT_FIELDS = {
    "expectedAccount": "",
    "sourcePath": "",
    "checkBudget": "",
    "allowScoredSubmissions": False,
    "scoreBudget": "",
    "scoreCriterion": "",
    "researchPrompt": "",
    "useSuperExperiments": True,
}


def normalize_account(text: str) -> str:
    value = (text or "").strip()
    return (value[1:] if value.startswith("@") else value).lower()


def is_account_name(text: str) -> bool:
    return ACCOUNT_PATTERN.fullmatch(text or "") is not None


def account_from_output(output: str) -> str | None:
    """Interpret the official whoami identity line only. Unknown output is
    never mistaken for a verified account."""
    clean = ANSI_PATTERN.sub("", output or "")
    bare = clean.strip()
    if is_account_name(bare):
        return normalize_account(bare)
    match = SIGNED_IN_PATTERN.search(clean)
    if match:
        return normalize_account(match.group(1))
    return None


def default_draft(suggested_prompts: list[str] | None = None) -> dict:
    draft = dict(DRAFT_FIELDS)
    if suggested_prompts:
        draft["researchPrompt"] = suggested_prompts[0]
    return draft


def validate_draft(fields: dict) -> dict:
    """Mirror KernelResearchSetup's rules; returns the normalized draft."""
    draft = dict(DRAFT_FIELDS)
    draft.update({k: v for k, v in (fields or {}).items() if k in DRAFT_FIELDS})

    expected = normalize_account(str(draft.get("expectedAccount", "")))
    if not is_account_name(expected):
        raise ResearchError("Enter the GitHub/OKC username you intend to use, then verify "
                            "that account before starting research.")
    draft["expectedAccount"] = expected

    source = str(draft.get("sourcePath", "")).strip()
    if "\0" in source:
        raise ResearchError("The kernel path contains an invalid character.")
    draft["sourcePath"] = source

    draft["checkBudget"] = _budget(draft.get("checkBudget"), "check", positive=False)
    scored = bool(draft.get("allowScoredSubmissions"))
    draft["allowScoredSubmissions"] = scored
    draft["scoreBudget"] = _budget(draft.get("scoreBudget"), "scored submission",
                                   positive=True) if scored else ""

    prompt = str(draft.get("researchPrompt", "")).strip()
    if not prompt:
        raise ResearchError("Enter a research prompt before preparing the request.")
    draft["researchPrompt"] = prompt

    criterion = str(draft.get("scoreCriterion", "")).strip()
    draft["scoreCriterion"] = criterion
    if scored and not criterion:
        raise ResearchError("Describe when a scored submission is allowed, including your "
                            "first-baseline or improvement criterion.")

    draft["useSuperExperiments"] = bool(draft.get("useSuperExperiments"))
    return draft


def _budget(value, name: str, positive: bool) -> str:
    text = str(value if value is not None else "").strip()
    if not text or any(not (48 <= ord(ch) <= 57) for ch in text):
        raise ResearchError(f"Enter a {'positive' if positive else 'nonnegative'} whole-number "
                            f"{name} budget.")
    count = int(text)
    if count < (1 if positive else 0):
        raise ResearchError(f"Enter a {'positive' if positive else 'nonnegative'} whole-number "
                            f"{name} budget.")
    return text


def render_prompt(template: str, draft: dict, project_id: str, project_name: str) -> str:
    """Replace {{PROJECT_INPUTS_JSON}} once, with pretty-printed sorted keys."""
    marker = "{{PROJECT_INPUTS_JSON}}"
    sections = template.split(marker)
    if len(sections) != 2:
        raise ResearchError("The kernel setup template must contain its input placeholder "
                            "exactly once.")
    scored = int(draft.get("scoreBudget") or 0)
    inputs = {
        "projectID": project_id,
        "projectName": project_name,
        "expectedAccount": draft.get("expectedAccount", ""),
        "sourcePath": draft.get("sourcePath") or None,
        "maximumOfficialChecks": int(draft.get("checkBudget") or 0),
        "maximumScoredSubmissions": scored,
        "authorizeScoredSubmissions": scored > 0,
        "researchPrompt": draft.get("researchPrompt", ""),
        "scoreCriterion": draft.get("scoreCriterion") or None,
        "useSuperExperiments": bool(draft.get("useSuperExperiments")),
        "superExperimentsURL": SUPER_EXPERIMENTS_URL,
    }
    # Swift's JSONEncoder omits nil optionals; drop them for byte parity.
    inputs = {key: value for key, value in inputs.items() if value is not None}
    block = json.dumps(inputs, indent=2, sort_keys=True, ensure_ascii=False)
    return sections[0] + block + sections[1]


def okc_executable() -> pathlib.Path | None:
    """Find okc on PATH or ~/.okc/bin/okc, like the native OKCLogin launcher."""
    import shutil
    found = shutil.which("okc") or shutil.which("okc.exe")
    if found:
        return pathlib.Path(found)
    home = pathlib.Path(os.path.expanduser("~"))
    for candidate in (home / ".okc" / "bin" / "okc.exe", home / ".okc" / "bin" / "okc"):
        if candidate.is_file():
            return candidate
    return None


def verify_account(workspace_root: str, environment: dict) -> str:
    """Read-only account inspection: only a parsed username leaves this
    function; neither raw output nor credentials are persisted."""
    executable = okc_executable()
    if executable is None:
        raise ResearchError("OKC CLI is not installed. Follow the installation instructions, "
                            "then choose your account.")
    args = [str(executable), "whoami"]
    # Windows .cmd shims need the shell wrapper, like the provider runner.
    if os.name == "nt" and executable.suffix.lower() in (".cmd", ".bat"):
        args = ["cmd", "/c"] + args
    try:
        result = subprocess.run(args, cwd=workspace_root,
                                capture_output=True, timeout=30, env=environment)
    except (OSError, subprocess.SubprocessError):
        raise ResearchError("OKC could not verify a signed-in account. Use Choose account, "
                            "complete login in the terminal, then refresh the account.") from None
    if result.returncode != 0:
        raise ResearchError("OKC could not verify a signed-in account. Use Choose account, "
                            "complete login in the terminal, then refresh the account.")
    output = result.stdout.decode("utf-8", "replace")
    account = account_from_output(output)
    if account is None:
        raise ResearchError("The OKC account could not be identified. Run okc whoami in a "
                            "terminal and verify your account before trying again.")
    return account


def login_script(directory: pathlib.Path, project_directory: str, home: pathlib.Path,
                 choose_account: bool = False) -> pathlib.Path:
    """Write only a terminal launcher. Authentication stays in the official
    CLI and the user's own terminal; no output or secrets are collected."""
    import uuid
    directory.mkdir(parents=True, exist_ok=True)
    name = "okc-login-" + str(uuid.uuid4()).upper()
    if os.name == "nt":
        file = directory / (name + ".cmd")
        body = ("@echo off\r\n"
                f"cd /d {project_directory}\r\n"
                "set OKC=\r\n"
                "where okc >nul 2>nul && set OKC=okc\r\n"
                "if not defined OKC if exist \"%USERPROFILE%\\.okc\\bin\\okc.exe\" "
                "set OKC=\"%USERPROFILE%\\.okc\\bin\\okc.exe\"\r\n"
                "if not defined OKC (\r\n"
                "  echo OKC CLI is not installed. Follow the official instructions at"
                " https://openkernelchallenge.dev/ and then return to Choose account.\r\n"
                "  exit /b 1\r\n"
                ")\r\n"
                "echo Sign in with your own account and personally review any challenge"
                " rules. No checks or scored submissions run here.\r\n"
                "%OKC% login" + (" --force" if choose_account else "") + "\r\n"
                "%OKC% whoami\r\n")
        file.write_text(body, encoding="utf-8")
    else:
        file = directory / (name + ".command")
        fallback = home / ".okc" / "bin" / "okc"
        login_arguments = "login --force" if choose_account else "login"
        body = ("#!/bin/zsh\n"
                f"cd -- '{project_directory}' || exit 1\n"
                "okc_cli=$(command -v okc 2>/dev/null)\n"
                'if [[ -z "$okc_cli" || ! -x "$okc_cli" ]]; then\n'
                f"  okc_cli='{fallback}'\n"
                "fi\n"
                'if [[ ! -x "$okc_cli" ]]; then\n'
                "  printf '%s\\n' 'OKC CLI is not installed. Follow the official instructions"
                " at https://openkernelchallenge.dev/ and then return to Choose account.'\n"
                "  exit 1\n"
                "fi\n"
                "printf '%s\\n' 'Sign in with your own account and personally review any"
                " challenge rules. No checks or scored submissions run here.'\n"
                f'if "$okc_cli" {login_arguments}; then\n'
                '  "$okc_cli" whoami\n'
                "else\n"
                "  exit $?\n"
                "fi\n") + "\n"
        file.write_text(body, encoding="utf-8")
        try:
            file.chmod(0o700)
        except OSError:
            pass
    return file
