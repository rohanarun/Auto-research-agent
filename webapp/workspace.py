"""Workspace validation and preparation, mirroring ResearchCore's Workspace.swift
with cross-platform rules.

The native app requires a mounted external SSD under ``/Volumes``. The web
edition runs on any writable directory on any OS and instead rejects the
obviously synced locations (Documents, Desktop, OneDrive, iCloud), which is
the same hazard that rule guards against. Existing checkouts are preserved.
"""
import os
import pathlib
import subprocess
import urllib.parse

from .models import ResearchError

SYNCED_COMPONENTS = ("Documents", "Desktop", "Mobile Documents", "OneDrive",
                     "iCloud Drive", "iCloud~")


def _reject_synced(path: pathlib.Path):
    for part in path.parts:
        if part in SYNCED_COMPONENTS:
            raise ResearchError(
                "Choose a workspace outside synced folders (Documents, Desktop, OneDrive or iCloud). "
                "Synced locations can corrupt active research checkouts.")
    return path


def validate(path: str) -> pathlib.Path:
    resolved = pathlib.Path(os.path.abspath(path))
    _reject_synced(resolved)
    # Walk to the nearest existing ancestor; it must be writable.
    ancestor = resolved
    while not ancestor.exists():
        parent = ancestor.parent
        if parent == ancestor:
            break
        ancestor = parent
    if not ancestor.exists():
        raise ResearchError("The workspace directory does not exist and cannot be created. Reconnect the drive.")
    if not os.access(ancestor, os.W_OK):
        raise ResearchError("The workspace is not writable.")
    if resolved.exists() and not os.access(resolved, os.W_OK):
        raise ResearchError("The workspace is not writable.")
    return resolved


def directory(root: str, component: str) -> pathlib.Path:
    base = validate(root)
    destination = base / component
    # The destination must never escape the workspace (path containment).
    base_resolved = pathlib.Path(os.path.abspath(base))
    destination_resolved = pathlib.Path(os.path.abspath(destination))
    try:
        destination_resolved.relative_to(base_resolved)
    except ValueError:
        raise ResearchError("The destination escapes the workspace.")
    _reject_synced(destination_resolved)
    destination_resolved.mkdir(parents=True, exist_ok=True)
    return destination_resolved


def reject_symlink(path: pathlib.Path):
    if path.is_symlink():
        raise ResearchError(f"The workspace path {path} is a symbolic link, which is not permitted here.")


def instruction_file(root: pathlib.Path, tracked_files: list[str]) -> pathlib.Path | None:
    """Prefer the literal repository entry point, then case-insensitive
    equivalents ordered by path depth, like Workspace.instructionFile."""
    def depth(value: str) -> int:
        return len(value.split("/"))

    candidates = [f for f in tracked_files
                  if f.rsplit("/", 1)[-1].lower() == "auto-research.md"]
    candidates.sort(key=lambda f: (0 if f == "auto-research.md" else 1, depth(f), f))
    for candidate in candidates:
        file = root / candidate
        try:
            file.relative_to(root.resolve())
        except ValueError:
            continue
        if file.is_file():
            return file
    literal = root / "auto-research.md"
    try:
        literal.relative_to(root.resolve())
    except ValueError:
        return None
    return literal if literal.is_file() else None


def executable_check(path: str) -> bool:
    if not path:
        return False
    return os.path.isfile(path) and os.access(path, os.X_OK)


def resolve_executable(path: str, settings_key: str = "") -> str:
    """Expand PATH references so Windows .cmd shims and bare names work."""
    if not path:
        return ""
    candidate = pathlib.Path(os.path.expandvars(os.path.expanduser(path)))
    if candidate.is_absolute():
        return str(candidate)
    return ""


def git_command(repository: str, destination: pathlib.Path) -> tuple[list[str], str]:
    """`git clone -- https://github.com/<owner>/<repo>.git <destination>`
    like the app: the repository's slash is part of the URL, never encoded."""
    return (["git", "clone", "--", f"https://github.com/{repository}.git", str(destination)],
            f"https://github.com/{repository}.git")


def clone(command: list[str], parent: pathlib.Path, environment: dict,
          on_line=None) -> tuple[int, str, str]:
    """Run the clone with streaming output, returning (code, stdout, stderr)."""
    process = subprocess.Popen(command, cwd=str(parent), stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, text=True,
                               encoding="utf-8", errors="replace",
                               env=environment)
    out: list[str] = []
    try:
        for line in process.stdout or []:
            out.append(line.rstrip("\n"))
            if on_line:
                on_line(line.rstrip("\n"))
    finally:
        if process.stdout is not None:
            process.stdout.close()
    code = process.wait()
    return code, "\n".join(out), ""


def tracked_files(directory_path: pathlib.Path, environment: dict) -> list[str]:
    try:
        result = subprocess.run(["git", "ls-files", "-z"], cwd=str(directory_path),
                                capture_output=True, env=environment, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return []
    if result.returncode != 0:
        return []
    return [part for part in result.stdout.decode("utf-8").split("\0") if part]
