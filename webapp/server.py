"""Auto Research web edition server.

Serves the research harness and the dashboard UI over HTTP. By default the
server binds the loopback address; ``--access lan`` binds all interfaces so
devices on the same network (for example a Tailscale tailnet) can reach the
same dashboard, with a generated token guarding the API.

    python webapp/server.py                      # localhost only
    python webapp/server.py --access lan         # tailnet/LAN + token auth
    python webapp/server.py --port 8765 --token my-secret
"""
from __future__ import annotations

import argparse
import json
import secrets
import shutil
import sys
import threading
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    # Direct execution (`python webapp/server.py`): make the package importable.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from webapp import workspace as workspace_module  # noqa: E402
from webapp.models import ResearchError  # noqa: E402
from webapp.store import Store  # noqa: E402

VERSION = "1.6.0-web"
STATIC_DIR = Path(__file__).resolve().parent / "static"


def detect_cli_paths() -> dict:
    """Discover provider CLIs from PATH, like scripts/configure.py."""
    found = {}
    for provider, command in (("codex", "codex"), ("claude", "claude"),
                              ("opencode", "opencode"), ("antigravity", "agy")):
        path = shutil.which(command) or ""
        found[provider + "Path"] = str(Path(path).resolve()) if path else ""
    return found


class Handler(BaseHTTPRequestHandler):
    server_version = "AutoResearch/" + VERSION
    store: Store
    token: str | None
    access: str
    host: str
    port: int
    quiet: bool = False

    def log_message(self, format, *args):  # noqa: A002 - http.server signature
        if not self.quiet:
            super().log_message(format, *args)

    # ---------- helpers ----------

    def _json(self, status: int, payload) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _error(self, status: int, message: str) -> None:
        self._json(status, {"error": message})

    def _body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        if not raw:
            return {}
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise ResearchError("Request body is not valid JSON.")
        return payload if isinstance(payload, dict) else {}

    def _authorized(self) -> bool:
        if not self.token:
            return True
        supplied = (self.headers.get("X-Auth-Token")
                    or urllib.parse.parse_qs(urllib.parse.urlsplit(self.path).query).get("token", [""])[0])
        return bool(supplied) and secrets.compare_digest(supplied, self.token)

    def _route(self) -> tuple[str, dict]:
        url = urllib.parse.urlsplit(self.path)
        params = urllib.parse.parse_qs(url.query)
        return url.path, {key: values[0] for key, values in params.items()}

    # ---------- requests ----------

    def do_GET(self):
        path, params = self._route()
        if path == "/healthz":
            return self._json(200, {"ok": True, "uptimeSeconds": round(time.time() - self.store.started_at, 1),
                                    "version": VERSION})
        if not path.startswith("/api"):
            return self._static(path)
        if not self._authorized():
            return self._error(401, "This dashboard requires its access token. Open the address shown by "
                                    "the server, including ?token=…")
        try:
            store = self.store
            if path == "/api/state":
                since = int(params.get("since", "0") or 0)
                if since:
                    store.wait_for_version(since, 25.0)
                return self._json(200, store.snapshot())
            if path == "/api/dashboard":
                view = store.dashboard(params.get("projectID") or store.state.selected_project_id or "")
                return self._json(200, view if view is not None else {"empty": True})
            if path == "/api/settings":
                return self._json(200, {**store.settings.to_json(),
                                        "detectedPaths": detect_cli_paths(),
                                        "auth": dict(store.auth)})
            if path == "/api/logs":
                session_id = params.get("sessionID", "")
                return self._json(200, {"sessionID": session_id, "log": store.logs(session_id)})
            if path == "/api/auth/status":
                store.refresh_auth()
                return self._json(200, {"auth": dict(store.auth)})
            if path == "/api/policy":
                return self._json(200, {"text": store.policy_text()})
            if path == "/api/instructions":
                file_path, text = store.instructions(params.get("projectID") or "")
                return self._json(200, {"path": str(file_path), "text": text})
            if path == "/api/setup":
                return self._json(200, store.setup_overview(params.get("projectID") or ""))
            if path == "/api/mcp":
                return self._json(200, store.mcp_servers())
            if path == "/api/cloud/status":
                return self._json(200, store.cloud_status())
            if path == "/api/lean/settings":
                return self._json(200, store.lean_view())
            if path == "/api/lean/token":
                return self._json(200, {"token": store.lean_token()})
            return self._error(404, "No such endpoint: " + path)
        except ResearchError as error:
            return self._error(400, str(error))
        except Exception as error:  # pragma: no cover
            return self._error(500, f"Server error: {error}")

    def do_POST(self):
        path, _params = self._route()
        if path == "/healthz":
            return self._json(200, {"ok": True})
        if not self._authorized():
            return self._error(401, "This dashboard requires its access token. Open the address shown by "
                                    "the server, including ?token=…")
        try:
            body = self._body()
            store = self.store
            if path == "/api/select":
                store.select(project_id=body.get("projectID"), session_id=body.get("sessionID"))
                return self._json(200, {"ok": True})
            if path == "/api/projects/refresh":
                return self._json(200, store.refresh_catalog())
            if path == "/api/projects/add":
                project, seeded = store.add_project(repository=body.get("repository") or "",
                                                    local_path=body.get("localPath") or "",
                                                    instructions=body.get("instructions") or "")
                return self._json(200, {"ok": True, "project": project.to_json(),
                                        "seeded": seeded})
            if path == "/api/projects/edit":
                project = store.edit_project(body.get("projectID") or "", body)
                return self._json(200, {"ok": True, "project": project.to_json()})
            if path == "/api/projects/remove":
                store.remove_project(body.get("projectID") or "")
                return self._json(200, {"ok": True})
            if path == "/api/chats/create":
                session = store.create_chat(body.get("projectID") or "", body.get("provider"))
                return self._json(200, {"sessionID": session.id})
            if path == "/api/play":
                session_id = store.play(body.get("projectID") or "", body.get("prompt"),
                                        body.get("provider"),
                                        keep_researching=bool(body.get("keepResearching")),
                                        parallel=bool(body.get("parallel")),
                                        agent_count=body.get("agentCount"),
                                        share_context=(None if body.get("shareContext") is None
                                                       else bool(body.get("shareContext"))))
                return self._json(200, {"sessionID": session_id})
            if path == "/api/chats/send":
                session_id = store.send(body.get("sessionID") or "", body.get("text") or "",
                                        keep_researching=bool(body.get("keepResearching")))
                return self._json(200, {"sessionID": session_id})
            if path == "/api/chats/stop":
                store.stop(body.get("sessionID") or "")
                return self._json(200, {"ok": True})
            if path == "/api/chats/close":
                store.close(body.get("sessionID") or "")
                return self._json(200, {"ok": True})
            if path == "/api/chats/reopen":
                store.reopen(body.get("sessionID") or "")
                return self._json(200, {"ok": True})
            if path == "/api/settings":
                store.save_settings(body)
                return self._json(200, {"ok": True})
            if path == "/api/policy":
                store.save_policy(body.get("text") or "")
                return self._json(200, {"ok": True})
            if path == "/api/instructions":
                store.save_instructions(body.get("projectID") or "", body.get("text") or "")
                return self._json(200, {"ok": True})
            if path == "/api/setup":
                draft = store.save_setup_draft(body.get("projectID") or "", body.get("draft") or {})
                return self._json(200, {"ok": True, "draft": draft})
            if path == "/api/setup/verify-account":
                return self._json(200, {"account": store.verify_okc_account()})
            if path == "/api/setup/login-script":
                script = store.login_script_file(body.get("projectID") or "",
                                                 choose_account=bool(body.get("chooseAccount")))
                return self._json(200, store.spawn_terminal(script))
            if path == "/api/mcp":
                store.set_mcp_selection(body.get("enabled") or [])
                return self._json(200, {"ok": True})
            if path == "/api/mcp/refresh":
                return self._json(200, store.refresh_mcp_servers())
            if path == "/api/auth/login":
                if body.get("spawn", True):
                    try:
                        script = store.provider_login_script(body.get("provider") or "")
                        return self._json(200, store.spawn_terminal(script))
                    except ResearchError:
                        raise
                return self._json(200, {"instructions": store.login(body.get("provider") or "")})
            if path == "/api/cloud/provider":
                return self._json(200, store.cloud_set_provider(body.get("provider") or ""))
            if path == "/api/cloud/run-target":
                return self._json(200, store.cloud_set_run_target(body.get("target") or ""))
            if path == "/api/cloud/key":
                return self._json(200, store.cloud_save_key(body.get("provider") or "",
                                                            body.get("key") or ""))
            if path == "/api/cloud/remove-key":
                return self._json(200, store.cloud_remove_key(body.get("provider") or ""))
            if path == "/api/cloud/connect":
                return self._json(200, store.cloud_connect(body.get("provider") or ""))
            if path == "/api/cloud/stop":
                return self._json(200, store.cloud_stop(body.get("provider") or ""))
            if path == "/api/cloud/codex-login":
                script = store.codex_device_login_script()
                return self._json(200, store.spawn_terminal(script))
            if path == "/api/lean/settings":
                return self._json(200, store.lean_save(body))
            if path == "/api/lean/start":
                return self._json(200, store.lean_start())
            if path == "/api/lean/stop":
                return self._json(200, store.lean_stop())
            if path == "/api/lean/share":
                return self._json(200, store.lean_share())
            if path == "/api/lean/stop-share":
                return self._json(200, store.lean_stop_sharing())
            if path == "/api/lean/open-logs":
                store.lean_open_logs()
                return self._json(200, {"ok": True})
            if path == "/api/workspace/open":
                store.open_workspace_folder()
                return self._json(200, {"ok": True})
            return self._error(404, "No such endpoint: " + path)
        except ResearchError as error:
            return self._error(400, str(error))
        except Exception as error:  # pragma: no cover
            return self._error(500, f"Server error: {error}")

    # ---------- static files ----------

    _CONTENT_TYPES = {".html": "text/html; charset=utf-8",
                      ".js": "text/javascript; charset=utf-8",
                      ".css": "text/css; charset=utf-8",
                      ".svg": "image/svg+xml", ".png": "image/png",
                      ".ico": "image/x-icon", ".json": "application/json"}

    def _static(self, path: str) -> None:
        if path in ("/", ""):
            path = "/index.html"
        file = (STATIC_DIR / path.lstrip("/")).resolve()
        if not file.is_relative_to(STATIC_DIR.resolve()) or not file.is_file():
            # SPA behaviour: unknown paths load the app shell.
            file = STATIC_DIR / "index.html"
            if not file.is_file():
                return self._error(404, "The dashboard UI is missing from webapp/static.")
        data = file.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", self._CONTENT_TYPES.get(file.suffix.lower(),
                                                                  "application/octet-stream"))
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class Server:
    def __init__(self, store: Store, token: str | None, access: str, host: str, port: int):
        self.store = store
        self.token = token
        self.access = access
        self.host = host
        self.port = port
        handler = type("BoundHandler", (Handler,), {"store": store, "token": token,
                                                    "access": access, "host": host, "port": port})
        self.httpd = ThreadingHTTPServer((host, port), handler)
        self.httpd.daemon_threads = True

    def serve_forever(self):
        self.httpd.serve_forever()

    def stop(self):
        self.httpd.shutdown()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Auto Research web edition")
    parser.add_argument("--workspace", help="Runtime workspace directory "
                        "(default: <repo>/Runtime)")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--access", choices=("local", "lan"), default="local",
                        help="lan binds all interfaces so tailnet or LAN devices "
                             "can reach the dashboard")
    parser.add_argument("--host", default=None, help="Override the bind address")
    parser.add_argument("--token", default=None, help="Access token (generated "
                        "automatically for lan access)")
    parser.add_argument("--quiet", action="store_true", help="Reduce request logging")
    args = parser.parse_args(argv)

    resources = Path(__file__).resolve().parent.parent / "Resources"
    workspace = args.workspace or str(Path(__file__).resolve().parent.parent / "Runtime")
    workspace_module.validate(workspace)
    store = Store(resources)
    store.load_workspace(workspace)
    store._port = args.port
    store._access = args.access

    token = args.token
    if args.access == "lan" and not token:
        token = secrets.token_urlsafe(18)
    host = args.host or ("0.0.0.0" if args.access == "lan" else "127.0.0.1")
    server = Server(store, token, args.access, host, args.port)
    store.refresh_catalog()  # may be offline; the saved catalog remains
    threading.Thread(target=store.refresh_auth, daemon=True).start()

    print("Auto Research web edition " + VERSION)
    local = f"http://127.0.0.1:{args.port}"
    if args.access == "lan":
        print(f"Dashboard (network):  http://<this device's address>:{args.port}")
        print(f"Dashboard (loopback): {local}")
        if token:
            print(f"Access token:         {token}")
            print(f"Open from another device: http://<this device's address>:{args.port}/?token={token}")
        print("On a Tailscale network, <this device's address> is the device's Tailscale name or address.")
    else:
        print(f"Dashboard: {local}")
        print("Start with --access lan to reach the dashboard from other devices on your network.")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
