"""Auto Research desktop launcher for Windows.

Wraps the web edition server unchanged: the same harness, the same workspace
layout, the same port and token the manual ``python webapp\\server.py`` process
uses today, so saved tailnet links keep working. A pywebview window (Edge
WebView2) shows the existing dashboard UI at the tokenized dashboard URL.

Behavior on top of the server:

- Single instance: a named Windows mutex plus a port check. A second launch,
  or a launch while the manual server already owns the port, shows an
  "already running" note and opens the dashboard against the existing server.
- First run seeds ``%APPDATA%\\AutoResearch\\config.json`` with this device's
  defaults (the install copy's workspace, port 8765, the device token, LAN
  access). A fresh machine gets a generated token.
- Closing the window stops the server cleanly. If a research session is still
  running, the app asks first, like the native app does when quitting.
- If WebView2 cannot start, the tokenized URL opens in the default browser and
  the server keeps serving; the launcher logs to ``launcher.log`` instead of
  crashing.

Helpers are pure functions so scripts/test_webapp.py can exercise them
without opening a window. ``webview`` and the ``webapp`` package are imported
only inside the window/server entry points.
"""
from __future__ import annotations

import ctypes
import json
import os
import secrets
import socket
import sys
import threading
import time
import urllib.parse
import webbrowser
from pathlib import Path

if os.name == "nt":  # ctypes.windll exists only on Windows
    import ctypes.wintypes

    _kernel32 = ctypes.windll.kernel32
    _kernel32.CreateMutexW.restype = ctypes.wintypes.HANDLE
    _kernel32.CloseHandle.argtypes = (ctypes.wintypes.HANDLE,)
else:
    _kernel32 = None

INSTANCE_MUTEX = "AutoResearch.SingleInstance"
ERROR_ALREADY_EXISTS = 183
# The token this device's manual server runs with (Runtime/server.log and the
# saved tailnet links embed it). A machine without the install copy generates
# its own on first run.
DEVICE_TOKEN = "lTvSAN8IVidADF2yrAXKPKK0"
DEFAULT_INSTALL_NAME = "Auto-research-agent"
CONFIG_FIELDS = ("workspaceRoot", "port", "token", "access")
WINDOW_TITLE = "Auto Research"
RUNNING_PROMPT = "Research is running \u2013 quit anyway?"
ALREADY_RUNNING_MESSAGE = ("Auto Research is already running. "
                           "The dashboard opens in your browser.")


# ---------- config ----------

def appdata_dir() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home() / "AppData" / "Roaming")
    return Path(base) / "AutoResearch"


def config_path() -> Path:
    return appdata_dir() / "config.json"


def log_path() -> Path:
    return appdata_dir() / "launcher.log"


def device_defaults(home: Path | None = None) -> dict:
    """Seed values for this device.

    With the install copy present (its ``Runtime`` directory holds the user's
    projects), the seed keeps that workspace and this device's existing token
    so saved tailnet links keep working. A fresh machine gets a generated
    token in the same format the server prints.
    """
    home = home or Path.home()
    root = home / DEFAULT_INSTALL_NAME
    if (root / "Runtime").is_dir():
        token = DEVICE_TOKEN
    else:
        token = generate_token()
    return {"workspaceRoot": str(root), "port": 8765, "token": token,
            "access": "lan"}


def seed_config(path: Path, defaults: dict, overwrite: bool = False) -> bool:
    """Write the config when the file is absent (or when overwriting a corrupt
    file). True when a file was written."""
    if path.is_file() and not overwrite:
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(defaults, indent=2) + "\n", encoding="utf-8")
    return True


def load_config(path: Path | None = None, defaults: dict | None = None) -> dict:
    """Read the config, seeding it on first run and repairing a corrupt file.

    Unknown values fall back to the defaults field by field, so a hand-edited
    config cannot break the server assembly.
    """
    path = path or config_path()
    defaults = defaults or device_defaults()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        seed_config(path, defaults, overwrite=True)
        data = dict(defaults)
    if not isinstance(data, dict):
        seed_config(path, defaults, overwrite=True)
        data = dict(defaults)
    config = dict(defaults)
    for field in CONFIG_FIELDS:
        value = data.get(field)
        if isinstance(value, (str, int)) and str(value).strip():
            config[field] = value
    try:
        port = int(config["port"])
    except (TypeError, ValueError):
        port = defaults["port"]
    config["port"] = port if 1 <= port <= 65535 else defaults["port"]
    if config["access"] not in ("local", "lan"):
        config["access"] = defaults["access"]
    config["token"] = str(config["token"]).strip() or generate_token()
    return config


def resolve_workspace(root: str) -> str:
    """The server's workspace: the configured root's ``Runtime`` directory
    when it exists (the install checkout), otherwise the root itself."""
    if not root:
        return root
    candidate = Path(root) / "Runtime"
    if candidate.is_dir():
        return str(candidate)
    return root


def generate_token() -> str:
    """The same token format webapp.server.main() generates for lan access."""
    return secrets.token_urlsafe(18)


# ---------- single instance ----------

def acquire_instance_lock(name: str = INSTANCE_MUTEX) -> tuple[int, bool]:
    """Create (or open) the single-instance mutex.

    Returns ``(handle, held)``: held is True when another Auto Research
    process already owns the mutex. The handle stays open for the process
    lifetime; Windows releases it at exit.
    """
    if _kernel32 is None:
        raise RuntimeError("The single-instance mutex requires Windows.")
    handle = _kernel32.CreateMutexW(None, False, name)
    if not handle:
        return 0, True  # cannot create or open: assume another owner
    return int(handle), ctypes.GetLastError() == ERROR_ALREADY_EXISTS


def release_instance_lock(handle: int) -> None:
    if handle and _kernel32 is not None:
        _kernel32.CloseHandle(handle)


def port_busy(port: int, host: str = "127.0.0.1", timeout: float = 0.3) -> bool:
    """True when something already serves on this port (the manual server,
    another Auto Research instance, or any other listener)."""
    try:
        with socket.create_connection((host, int(port)), timeout=timeout):
            return True
    except OSError:
        return False


def wait_for_dashboard(config: dict, timeout: float = 5.0) -> None:
    """Give a starting instance a moment to bind the port before opening the
    dashboard against it."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if port_busy(config["port"]):
            return
        time.sleep(0.25)


# ---------- paths ----------

def repository_root() -> Path:
    """The checkout directory containing ``webapp/`` and ``Resources/``.

    Unfrozen runs resolve from this file (``launcher/launcher.py``). A frozen
    exe resolves to the PyInstaller extraction directory (``sys._MEIPASS``),
    which carries ``webapp`` and ``Resources`` as bundled data — the same
    layout ``webapp/server.py`` computes from ``__file__`` when frozen.
    """
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", "")
        return Path(meipass) if meipass else Path(sys.executable).parent
    return Path(__file__).resolve().parent.parent


def resources_dir() -> Path:
    return repository_root() / "Resources"


def window_icon() -> str | None:
    """Unfrozen runs use the generated launcher icon; a frozen exe's window
    icon comes from the embedded exe resource."""
    if getattr(sys, "frozen", False):
        return None
    icon = Path(__file__).resolve().parent / "app.ico"
    return str(icon) if icon.is_file() else None


def window_url(config: dict) -> str:
    query = urllib.parse.urlencode({"token": config["token"]})
    return f"http://127.0.0.1:{config['port']}/?{query}"


# ---------- diagnostics ----------

def log(message: str) -> None:
    """Append a timestamped line to launcher.log. The windowed exe has no
    console, and http.server writes request logs to stderr, which is None in
    windowed processes — main() redirects both to this file."""
    try:
        path = log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        with path.open("a", encoding="utf-8") as file:
            file.write(f"[{stamp}] {message}\n")
    except OSError:
        pass


def notify(message: str) -> None:
    """A native message box. Windowed exes have no console to print to."""
    try:
        ctypes.windll.user32.MessageBoxW(None, message, WINDOW_TITLE, 0x40)
    except Exception:  # pragma: no cover - a message box is best effort
        log(message)


# ---------- server assembly ----------

def start_server(workspace: str, port: int, access: str, token: str,
                 resources: Path):
    """Build the server exactly like webapp.server.main() does: validate the
    workspace, load the store, bind, refresh the catalog, start auth
    detection. Returns (server, store)."""
    root = str(repository_root())
    if root not in sys.path:
        sys.path.insert(0, root)
    from webapp import workspace as workspace_module
    from webapp.server import Server
    from webapp.store import Store

    workspace_module.validate(workspace)
    store = Store(resources)
    store.load_workspace(workspace)
    store._port = port
    store._access = access
    host = "0.0.0.0" if access == "lan" else "127.0.0.1"
    server = Server(store, token, access, host, port)
    store.refresh_catalog()  # may be offline; the saved catalog remains
    threading.Thread(target=store.refresh_auth, daemon=True).start()
    return server, store


def running_session_ids(store) -> list:
    return [session.id for session in store.state.sessions
            if session.state == "Running"]


def stop_running_sessions(store, timeout: float = 10.0) -> list:
    """Stop active runs before quitting, like the native app's quit prompt."""
    ids = set(running_session_ids(store)) | set(getattr(store, "_runners", {}))
    for session_id in sorted(ids):
        try:
            store.stop(session_id)
        except Exception as error:  # noqa: BLE001 - best effort at quit
            log(f"Stopping session {session_id} failed: {error}")
    deadline = time.time() + timeout
    while time.time() < deadline and getattr(store, "_runners", None):
        time.sleep(0.1)
    return sorted(ids)


# ---------- window and fallback ----------

def fallback_to_browser(config: dict, server, thread: threading.Thread) -> int:
    """WebView2 failed: open the tokenized URL in the default browser and
    keep serving, like the manual server does."""
    log("Opening the dashboard in the default browser; the server keeps running.")
    webbrowser.open(window_url(config))
    notify("Auto Research could not open its desktop window.\n\n"
           "The dashboard opens in your browser instead, and the server keeps "
           "running on port " + str(config["port"]) + ".")
    try:
        while thread.is_alive():
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()
        thread.join(timeout=5)
    return 0


def run_window(config: dict, server, store, thread: threading.Thread) -> int:
    url = window_url(config)

    def on_closing(window):
        # Event handlers return False to abort the close (pywebview's contract).
        try:
            if not running_session_ids(store):
                return True
            answer = window.create_confirmation_dialog(WINDOW_TITLE, RUNNING_PROMPT)
            return bool(answer)
        except Exception as error:  # noqa: BLE001 - never trap the close
            log(f"Close confirmation failed: {error}")
            return True

    try:
        import webview
    except Exception as error:  # noqa: BLE001 - ImportError or loader failure
        log(f"pywebview import failed: {error}")
        return fallback_to_browser(config, server, thread)
    try:
        window = webview.create_window(WINDOW_TITLE, url, width=1280, height=800,
                                       min_size=(980, 640))
        window.events.closing += on_closing
        webview.start(icon=window_icon(), private_mode=False,
                      storage_path=str(appdata_dir() / "webview"))
    except Exception as error:  # noqa: BLE001 - WebView2 init failure
        log(f"The desktop window failed to start: {error}")
        return fallback_to_browser(config, server, thread)
    stop_running_sessions(store)
    server.stop()
    thread.join(timeout=5)
    log("Server stopped; exiting.")
    return 0


def main() -> int:
    config = load_config()
    try:
        handle, held = acquire_instance_lock()
    except RuntimeError as error:
        log(f"Single-instance check failed: {error}")
        handle, held = 0, False
    if held or port_busy(config["port"]):
        wait_for_dashboard(config)
        log("Already running: opening the dashboard against the existing server.")
        webbrowser.open(window_url(config))
        notify(ALREADY_RUNNING_MESSAGE)
        release_instance_lock(handle)
        return 0

    # Windowed exes have no console: http.server logs every request to
    # stderr, which is None there, so both streams go to launcher.log.
    try:
        appdata_dir().mkdir(parents=True, exist_ok=True)
        stream = log_path().open("a", encoding="utf-8")
        if sys.stdout is None or sys.stderr is None:
            sys.stdout = stream
            sys.stderr = stream
    except OSError:
        stream = None

    workspace = resolve_workspace(str(config["workspaceRoot"]))
    try:
        server, store = start_server(workspace, config["port"], config["access"],
                                     config["token"], resources_dir())
    except Exception as error:  # noqa: BLE001 - visible instead of silent
        log(f"Server startup failed: {error}")
        if stream:
            try:
                sys.stderr = sys.__stderr__ or stream  # type: ignore[union-attr]
            except Exception:  # noqa: BLE001
                pass
        notify("Auto Research failed to start:\n\n" + str(error))
        return 1
    thread = threading.Thread(target=server.serve_forever, name="dashboard-server",
                              daemon=True)
    thread.start()
    log(f"Serving {config['access']} on port {config['port']}; "
        f"workspace {workspace}")
    try:
        return run_window(config, server, store, thread)
    finally:
        release_instance_lock(handle)


if __name__ == "__main__":
    sys.exit(main())
