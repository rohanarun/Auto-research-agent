"""Shared Lean verification server for the web edition.

Mirrors the native LeanServerView.swift + LeanServerController.swift: a
dedicated server folder, the installed Lean executable, start/stop of the
LeanSearch server (init-server then serve), an ngrok share, and the access
token. Commands match the native controller; on hosts where the LeanSearch
tools cannot run (their server module is POSIX-only) the real error surfaces
honest instead of pretending success.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path

from . import spawn as spawn_module
from .models import ResearchError


class LeanError(ResearchError):
    """A readable, user-facing Lean server failure."""


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def _write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _default_runner(command: list, cwd: str, timeout: float = 120.0):
    result = subprocess.run(command, cwd=cwd, capture_output=True, timeout=timeout,
                            creationflags=spawn_module.NO_WINDOW)
    return result.returncode, result.stdout.decode("utf-8", "replace"), \
        result.stderr.decode("utf-8", "replace")


def _default_popen(command: list, cwd: str):
    return subprocess.Popen(command, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                            errors="replace", creationflags=spawn_module.NO_WINDOW)


def _default_health(url: str, token: str, timeout: float = 10.0) -> int:
    request = urllib.request.Request(url, headers={"Authorization": "Bearer " + token})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return int(response.status)


def _default_sleep(seconds: float):
    time.sleep(seconds)


def detect_lean() -> str:
    """The first installed elan toolchain binary, like the native controller."""
    home = Path.home() / ".elan" / "toolchains"
    if home.is_dir():
        for toolchain in sorted(home.iterdir()):
            binary = toolchain / "bin" / ("lean.exe" if os.name == "nt" else "lean")
            if binary.is_file():
                return str(binary)
    return "lean.exe" if os.name == "nt" else "lean"


class LeanStore:
    """Persisted Lean server settings plus the live server/ngrok processes."""

    def __init__(self, workspace_root: str, resources_dir: Path,
                 runner=None, popen=None, health=None, sleep=None):
        self._lock = threading.RLock()
        root = Path(workspace_root)
        self._file = root / "lean-settings.json"
        self._resources = Path(resources_dir)
        self._root = root
        self._runner = runner or _default_runner
        self._popen = popen or _default_popen
        self._health = health or _default_health
        self._sleep = sleep or _default_sleep
        data = _read_json(self._file) or {}
        self.storage_path = data.get("storagePath") or ""
        self.lean_path = data.get("leanPath") or detect_lean()
        self.ngrok_path = data.get("ngrokPath") or "ngrok"
        self.status = data.get("status") or "Stopped"
        self.local_url = data.get("localURL") or ""
        self.public_url = data.get("publicURL") or ""
        self.server_process = None
        self.ngrok_process = None
        self.busy = ""

    # ---------- persistence ----------

    def _persist(self):
        with self._lock:
            _write_json(self._file, {
                "storagePath": self.storage_path,
                "leanPath": self.lean_path,
                "ngrokPath": self.ngrok_path,
                "status": self.status,
                "localURL": self.local_url,
                "publicURL": self.public_url,
            })

    # ---------- views ----------

    def view(self) -> dict:
        with self._lock:
            return {
                "storagePath": self.storage_path,
                "leanPath": self.lean_path,
                "ngrokPath": self.ngrok_path,
                "status": self.status,
                "localURL": self.local_url,
                "publicURL": self.public_url,
                "running": self.server_process is not None and self.server_process.poll() is None,
                "sharing": self.ngrok_process is not None and self.ngrok_process.poll() is None,
                "busy": self.busy,
            }

    # ---------- settings ----------

    def save_updates(self, updates: dict) -> dict:
        with self._lock:
            if "storagePath" in updates:
                self.storage_path = str(updates["storagePath"] or "").strip()
            if "leanPath" in updates:
                self.lean_path = str(updates["leanPath"] or "").strip() or detect_lean()
            if "ngrokPath" in updates:
                self.ngrok_path = str(updates["ngrokPath"] or "").strip() or "ngrok"
        self._persist()
        return self.view()

    # ---------- lifecycle ----------

    def _tools_cli(self) -> Path:
        candidate = self._resources / "LeanSearch" / "cli.py"
        if not candidate.is_file():
            raise LeanError("The Lean tools are not installed with this app.")
        return candidate

    def start(self) -> dict:
        """init-server, then serve, then the health check — native parity."""
        if self.busy:
            raise LeanError("The Lean server is already " + self.busy + ".")
        if self.view()["running"]:
            raise LeanError("The server is already running.")
        storage = self.storage_path.strip()
        if not storage:
            raise LeanError("Choose a dedicated Lean server folder first.")
        tools = self._tools_cli()
        python = sys.executable
        lean = Path(os.path.expanduser(self.lean_path))
        if not lean.is_file():
            raise LeanError("Set the installed Lean executable first (found nothing at "
                            + str(lean) + ").")
        storage_path = Path(os.path.expanduser(storage))
        try:
            storage_path.mkdir(parents=True, exist_ok=True)
        except OSError as error:
            raise LeanError("The server folder could not be created: " + str(error)) from None
        self.busy = "starting"
        try:
            code, out, err = self._runner(
                [python, "-B", str(tools), "init-server", "--directory", str(storage_path),
                 "--lean", str(lean)], str(self._root))
            if code != 0:
                raise LeanError("init-server failed with status " + str(code) + ": "
                                + (err.strip() or out.strip() or "no output"))
            server = self._popen(
                [python, "-B", str(tools), "serve", "--config",
                 str(storage_path / "server-config.json")], str(self._root))
            with self._lock:
                self.server_process = server
            config = None
            for _ in range(20):
                if server.poll() is not None:
                    raise LeanError("The server exited with status " + str(server.returncode)
                                    + ". The LeanSearch server needs a POSIX host in this "
                                      "edition; its modules import fcntl.")
                self._sleep(0.5)
                config = _read_json(storage_path / "server.json")
                if config and config.get("url"):
                    break
            if not config or not config.get("url"):
                raise LeanError("The server never reported its address in server.json.")
            url = str(config.get("url"))
            token_path = storage_path / "server-token"
            try:
                token = token_path.read_text(encoding="utf-8").strip()
            except FileNotFoundError:
                token = ""
            try:
                self._health(url + "/v1/health", token)
            except Exception as error:
                raise LeanError("The health check failed: " + str(error)
                                + ". The server may still be warming up; try again.")
            with self._lock:
                self.status = "Ready · localhost only"
                self.local_url = url
            self._persist()
            return self.view()
        finally:
            self.busy = ""

    def stop(self) -> dict:
        with self._lock:
            self.status = "Draining the active verification job"
        self.stop_sharing()
        server = self.server_process
        if server is not None and server.poll() is None:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
        with self._lock:
            self.server_process = None
            self.status = "Stopped"
            self.local_url = ""
        self._persist()
        return self.view()

    # ---------- ngrok sharing ----------

    def share(self) -> dict:
        if self.view()["sharing"]:
            raise LeanError("The server is already shared.")
        if not self.local_url:
            raise LeanError("Start the server before sharing it.")
        ngrok = os.path.expanduser(self.ngrok_path or "ngrok")
        self.busy = "sharing"
        try:
            process = self._popen(
                [ngrok, "http", self.local_url, "--inspect=false", "--log=stdout",
                 "--log-format=json"], str(self._root))
            with self._lock:
                self.ngrok_process = process
            public = ""
            deadline_lines = 200
            while deadline_lines > 0:
                if process.poll() is not None:
                    raise LeanError("ngrok exited with status " + str(process.returncode)
                                    + ". Uses your existing ngrok login; check the executable path.")
                line = process.stdout.readline() if process.stdout else ""
                if not line:
                    self._sleep(0.2)
                    deadline_lines -= 1
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                text = json.dumps(payload)
                start = text.find('"url"')
                if start == -1:
                    continue
                for candidate in payload.get("cfg") or {}, payload:
                    value = candidate.get("url") if isinstance(candidate, dict) else None
                    if isinstance(value, str) and value.startswith("https://"):
                        public = value
                        break
                if public:
                    break
                deadline_lines -= 1
            if not public:
                self.stop_sharing()
                raise LeanError("ngrok never reported a public HTTPS address. "
                                "Stopping sharing; local verification stays available.")
            with self._lock:
                self.public_url = public
                self.status = "Shared through ngrok · bearer token required"
            self._persist()
            return self.view()
        finally:
            self.busy = ""

    def stop_sharing(self) -> dict:
        process = self.ngrok_process
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        with self._lock:
            self.ngrok_process = None
            self.public_url = ""
            if self.status.startswith("Shared through"):
                self.status = "Ready · localhost only" if self.local_url else "Stopped"
        self._persist()
        return self.view()

    def token_text(self) -> str:
        storage = Path(os.path.expanduser(self.storage_path or ""))
        if not self.storage_path or not storage.is_dir():
            raise LeanError("Choose a dedicated Lean server folder first.")
        try:
            return (storage / "server-token").read_text(encoding="utf-8").strip()
        except FileNotFoundError:
            raise LeanError("The server has never started, so no access token exists yet.") from None

    def open_folder(self) -> None:
        storage = Path(os.path.expanduser(self.storage_path or ""))
        if not self.storage_path:
            raise LeanError("Choose a dedicated Lean server folder first.")
        if not storage.is_dir():
            storage.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(str(storage))  # noqa: S606 — opens Explorer on the host
        elif sys.platform == "darwin":
            subprocess.Popen(["open", str(storage)])
        else:
            subprocess.Popen(["xdg-open", str(storage)])

    def shutdown(self):
        try:
            self.stop()
        except Exception:
            pass
