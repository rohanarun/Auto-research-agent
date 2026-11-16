"""Offline tests for the web edition's cloud and Lean settings.

Follows the repository's harness conventions: synthetic fixtures, injected
HTTP fakes instead of network calls, no account data. Covers the Daytona and
Agent37 management-API clients (Bearer auth, paging, install-ID filtering,
stop calls), key storage that never leaks through the API, the Lean server
lifecycle with injected processes, and the new dashboard routes.
"""
from pathlib import Path
import io
import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
(ROOT / "tmp").mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "scripts"))

from webapp import cloud, lean, server, store, transport  # noqa: E402
from test_webapp import fake_cli_directory, resources_copy  # noqa: E402


class CloudStoreTests(unittest.TestCase):
    def _store(self, tmp, request=None) -> cloud.CloudStore:
        tmp = Path(tmp)
        (tmp / "workspace").mkdir(parents=True, exist_ok=True)
        return cloud.CloudStore(str(tmp / "workspace"), request=request)

    def test_keys_are_persisted_on_device_and_never_in_the_view(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            tmp = Path(tmp)
            s = self._store(tmp)
            s.save_key("daytona", "  dk-secret  ")
            view = s.view()
            self.assertNotIn("keys", json.dumps(view))
            self.assertNotIn("dk-secret", json.dumps(view))
            self.assertTrue(view["keySaved"]["daytona"])
            self.assertEqual(view["status"]["daytona"],
                             "API key saved in your workspace. Connect to prepare cloud Codex.")
            raw = json.loads((tmp / "workspace" / "cloud-settings.json").read_text("utf-8"))
            self.assertEqual(raw["keys"]["daytona"], "dk-secret")
            # The install ID is the device identity; it survives a reload.
            again = self._store(tmp)
            self.assertEqual(again.install_id, s.install_id)
            self.assertEqual(again.keys.get("daytona"), "dk-secret")

    def test_empty_key_is_rejected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            s = self._store(tmp)
            with self.assertRaises(cloud.CloudError):
                s.save_key("daytona", "   ")

    def test_provider_choice_is_validated_and_persisted(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            s = self._store(tmp)
            with self.assertRaises(cloud.CloudError):
                s.set_provider("workerbee")
            self.assertEqual(s.set_provider("agent37")["provider"], "agent37")
            self.assertEqual(self._store(tmp).provider, "agent37")

    def test_run_target_is_validated(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            s = self._store(tmp)
            with self.assertRaises(cloud.CloudError):
                s.set_run_target("orbit")
            self.assertEqual(s.set_run_target("cloud")["runTarget"], "cloud")

    def test_remove_key_reports_preserved_sandboxes(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            s = self._store(tmp)
            s.save_key("agent37", "a37-secret")
            view = s.remove_key("agent37")
            self.assertEqual(view["status"]["agent37"],
                             "Key removed. Existing sandboxes are preserved.")
            self.assertFalse(view["keySaved"]["agent37"])

    def test_daytona_list_pages_bearers_and_ownership_filter(self):
        calls = []
        holder = {}

        def request(method, url, headers, body, timeout=30.0):
            calls.append({"method": method, "url": url, "headers": headers, "body": body})
            install = holder["install"]
            if method == "GET" and url.startswith("https://app.daytona.io/api/sandbox"):
                if "cursor=page2" in url:
                    payload = {"items": [], "nextCursor": None}
                else:
                    payload = {"items": [
                        {"id": "box-owned-1", "name": "research-1", "state": "running",
                         "labels": {"autoresearch-install": install}},
                        {"id": "box-other", "name": "foreign", "state": "running",
                         "labels": {"autoresearch-install": "another-install"}},
                        {"id": "box-owned-2", "name": "connection", "state": "stopped",
                         "labels": {"autoresearch-install": install,
                                    "autoresearch-role": "connection"}}],
                    "nextCursor": "page2"}
                return 200, json.dumps(payload).encode()
            return 404, b"{}"

        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            s = self._store(tmp, request=request)
            holder["install"] = s.install_id
            s.save_key("daytona", "dk-secret")
            view = s.connect("daytona")
            self.assertEqual([item["id"] for item in view["owned"]["daytona"]],
                             ["box-owned-1", "box-owned-2"])
            self.assertTrue(view["connected"]["daytona"])
            self.assertEqual(view["status"]["daytona"],
                             "Daytona connected · cloud Codex sign-in required")
            self.assertEqual(calls[0]["method"], "GET")
            self.assertEqual(calls[0]["headers"]["Authorization"], "Bearer dk-secret")
            self.assertTrue(calls[0]["url"].startswith("https://app.daytona.io/api/sandbox?limit=100"))
            self.assertTrue(any("cursor=page2" in call["url"] for call in calls),
                            "the next cursor page was not fetched")
            self.assertNotIn("dk-secret", json.dumps(view))

    def test_agent37_list_filters_by_metadata_and_id_shape(self):
        holder = {}

        def request(method, url, headers, body, timeout=30.0):
            install = holder["install"]
            if method == "GET" and url == "https://api.agent37.com/v1/instances":
                payload = {"data": [
                    {"id": "abcdefghij", "name": "research-1", "status": "running",
                     "metadata": {"autoresearch-install": install}},
                    {"id": "abcdefghijkl", "name": "bad-shape", "status": "running",
                     "metadata": {"autoresearch-install": install}},
                    {"id": "zzzzzzzzzz", "name": "foreign", "status": "running",
                     "metadata": {"autoresearch-install": "another-install"}},
                ]}
                return 200, json.dumps(payload).encode()
            return 404, b"{}"

        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            s = self._store(tmp, request=request)
            holder["install"] = s.install_id
            s.save_key("agent37", "a37-secret")
            view = s.connect("agent37")
            self.assertEqual([item["id"] for item in view["owned"]["agent37"]],
                             ["abcdefghij"])
            self.assertEqual(view["owned"]["agent37"][0]["state"], "running")

    def test_connect_without_a_saved_key_fails_readable(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            s = self._store(tmp)
            with self.assertRaises(cloud.CloudError):
                s.connect("daytona")

    def test_rejected_key_names_the_provider(self):
        def request(method, url, headers, body, timeout=30.0):
            return 401, b"{}"

        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            s = self._store(tmp, request=request)
            s.save_key("daytona", "dk-secret")
            with self.assertRaises(cloud.CloudError):
                s.connect("daytona")

    def test_stop_posts_the_provider_stop_endpoint(self):
        calls = []
        holder = {}

        def request(method, url, headers, body, timeout=30.0):
            calls.append((method, url))
            if method == "GET" and "daytona" in url:
                payload = {"items": [{"id": "box-1", "name": "research-1", "state": "running",
                                      "labels": {"autoresearch-install": holder["install"]}}],
                           "nextCursor": None}
                return 200, json.dumps(payload).encode()
            return 200, b""

        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            s = self._store(tmp, request=request)
            holder["install"] = s.install_id
            s.save_key("daytona", "dk-secret")
            view = s.stop_owned("daytona")
            self.assertEqual(view["status"]["daytona"],
                             "Owned compute stopped. Files and sign-in are retained.")
            self.assertIn(("POST", "https://app.daytona.io/api/sandbox/box-1/stop"), calls)
            self.assertEqual(view["stopped"], 1)
            body = json.loads(json.dumps(view))
            self.assertNotIn("dk-secret", json.dumps(body))

    def test_agent37_stop_uses_instances_endpoint(self):
        calls = []
        holder = {}

        def request(method, url, headers, body, timeout=30.0):
            calls.append((method, url))
            if method == "GET" and "agent37" in url:
                payload = {"data": [{"id": "abcdefghij", "name": "research-1",
                                     "status": "running",
                                     "metadata": {"autoresearch-install": holder["install"]}}]}
                return 200, json.dumps(payload).encode()
            return 200, b""

        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            s = self._store(tmp, request=request)
            holder["install"] = s.install_id
            s.save_key("agent37", "a37-secret")
            s.stop_owned("agent37")
            self.assertIn(("POST", "https://api.agent37.com/v1/instances/abcdefghij/stop"),
                          calls)

    def test_unreadable_answer_is_reported(self):
        def request(method, url, headers, body, timeout=30.0):
            return 200, b"<html>gateway error</html>"

        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            s = self._store(tmp, request=request)
            s.save_key("daytona", "dk-secret")
            with self.assertRaises(cloud.CloudError):
                s.connect("daytona")


class FakeProcess:
    """A process double for the Lean server lifecycle."""

    def __init__(self, lines=None, exit_code=None):
        self._lines = lines or []
        self.returncode = exit_code or 0
        self.stdout = io.StringIO("\n".join(self._lines) + "\n") if lines else None
        self.terminated = False
        self.killed = False

    def poll(self):
        return None if self.returncode == 0 else self.returncode

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True

    def wait(self, timeout=None):
        return self.returncode


class LeanStoreTests(unittest.TestCase):
    def _resources(self, tmp: Path, with_tools: bool = True) -> Path:
        resources = tmp / "res"
        if with_tools:
            (resources / "LeanSearch").mkdir(parents=True, exist_ok=True)
            (resources / "LeanSearch" / "cli.py").write_text("# synthetic\n", encoding="utf-8")
        return resources

    def _lean(self, tmp, with_tools: bool = True, runner=None, popen=None,
              health=None) -> lean.LeanStore:
        tmp = Path(tmp)
        (tmp / "workspace").mkdir(parents=True, exist_ok=True)
        return lean.LeanStore(str(tmp / "workspace"), self._resources(tmp, with_tools),
                              runner=runner, popen=popen,
                              health=health or (lambda url, token, timeout=10.0: 200),
                              sleep=lambda seconds: None)

    def test_updates_persist_and_defaults_are_detected(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            tmp = Path(tmp)
            s = self._lean(tmp)
            view = s.save_updates({"storagePath": str(tmp / "server"),
                                   "leanPath": "", "ngrokPath": ""})
            self.assertEqual(view["ngrokPath"], "ngrok")
            self.assertTrue(view["leanPath"])
            again = lean.LeanStore(str(tmp / "workspace"), tmp / "res")
            self.assertEqual(again.storage_path, str(tmp / "server"))

    def test_start_requires_storage_tools_and_lean(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            tmp = Path(tmp)
            s = self._lean(tmp)
            with self.assertRaises(lean.LeanError):
                s.start()
            s.save_updates({"storagePath": str(tmp / "server")})
            s._resources = tmp / "empty-res"
            with self.assertRaises(lean.LeanError):
                s.start()

    def test_start_surfaces_init_failure_honestly(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            tmp = Path(tmp)
            lean_binary = tmp / "lean"
            lean_binary.write_bytes(b"\xff")
            s = self._lean(tmp, runner=lambda command, cwd, timeout=120.0:
                           (1, "", "ImportError: No module named 'fcntl'"))
            s.save_updates({"storagePath": str(tmp / "server"), "leanPath": str(lean_binary)})
            with self.assertRaises(lean.LeanError) as ctx:
                s.start()
            self.assertIn("init-server failed", str(ctx.exception))
            self.assertIn("fcntl", str(ctx.exception))

    def test_start_reaches_ready_and_persists(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            tmp = Path(tmp)
            lean_binary = tmp / "lean"
            lean_binary.write_bytes(b"\xff")
            spawned = []

            def popen(command, cwd):
                spawned.append(command)
                config_path = Path(command[command.index("--config") + 1])
                (config_path.parent / "server.json").write_text(json.dumps({
                    "pid": 1234, "url": "http://127.0.0.1:8760"}), encoding="utf-8")
                return FakeProcess()

            s = self._lean(tmp, runner=lambda command, cwd, timeout=120.0: (0, "", ""),
                           popen=popen)
            s.save_updates({"storagePath": str(tmp / "server"), "leanPath": str(lean_binary)})
            view = s.start()
            self.assertEqual(view["status"], "Ready · localhost only")
            self.assertEqual(view["localURL"], "http://127.0.0.1:8760")
            self.assertTrue(any("--config" in command for command in spawned), spawned)
            persisted = json.loads((tmp / "workspace" / "lean-settings.json").read_text("utf-8"))
            self.assertEqual(persisted["status"], "Ready · localhost only")

    def test_early_server_exit_is_reported(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            tmp = Path(tmp)
            lean_binary = tmp / "lean"
            lean_binary.write_bytes(b"\xff")

            def popen(command, cwd):
                process = FakeProcess()
                process.returncode = 1
                return process

            s = self._lean(tmp, runner=lambda command, cwd, timeout=120.0: (0, "", ""),
                           popen=popen)
            s.save_updates({"storagePath": str(tmp / "server"), "leanPath": str(lean_binary)})
            with self.assertRaises(lean.LeanError) as ctx:
                s.start()
            self.assertIn("exited", str(ctx.exception))

    def test_stop_terminates_the_server(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            tmp = Path(tmp)
            s = self._lean(tmp)
            process = FakeProcess()
            s.server_process = process
            s.local_url = "http://127.0.0.1:8760"
            view = s.stop()
            self.assertEqual(view["status"], "Stopped")
            self.assertEqual(view["localURL"], "")
            self.assertTrue(process.terminated)

    def test_share_reads_the_ngrok_url(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            line = json.dumps({"cfg": {"url": "https://abc-def.ngrok.app"}})
            s = self._lean(tmp, popen=lambda command, cwd: FakeProcess(lines=[line]))
            s.local_url = "http://127.0.0.1:8760"
            view = s.share()
            self.assertEqual(view["publicURL"], "https://abc-def.ngrok.app")
            self.assertEqual(view["status"], "Shared through ngrok · bearer token required")

    def test_share_requires_a_running_server(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            s = self._lean(tmp)
            with self.assertRaises(lean.LeanError):
                s.share()

    def test_stop_sharing_closes_public_access(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            tmp = Path(tmp)
            s = self._lean(tmp)
            s.ngrok_process = FakeProcess()
            s.status = "Shared through ngrok · bearer token required"
            s.public_url = "https://abc-def.ngrok.app"
            s.local_url = "http://127.0.0.1:8760"
            view = s.stop_sharing()
            self.assertEqual(view["publicURL"], "")
            self.assertEqual(view["status"], "Ready · localhost only")

    def test_token_needs_a_started_server(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            tmp = Path(tmp)
            s = self._lean(tmp)
            with self.assertRaises(lean.LeanError):
                s.token_text()
            storage = tmp / "server"
            storage.mkdir()
            (storage / "server-token").write_text("tok-1\n", encoding="utf-8")
            s.storage_path = str(storage)
            self.assertEqual(s.token_text(), "tok-1")


class _ApiHandler(BaseHTTPRequestHandler):
    seen: dict = {}

    def do_GET(self):
        if self.path == "/missing":
            self.send_response(404)
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok": true}')

    def do_POST(self):
        length = int(self.headers.get("Content-Length", "0") or 0)
        payload = self.rfile.read(length)
        _ApiHandler.seen["authorization"] = self.headers.get("Authorization")
        self.send_response(201)
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args):
        pass


class TransportRequestTests(unittest.TestCase):
    def _spin(self):
        bound = ThreadingHTTPServer(("127.0.0.1", 0), _ApiHandler)
        thread = threading.Thread(target=bound.serve_forever, daemon=True)
        thread.start()
        return bound

    def test_get_returns_status_and_body(self):
        bound = self._spin()
        try:
            port = bound.server_address[1]
            status, payload = transport.request_bytes(f"http://127.0.0.1:{port}/api",
                                                      headers={"Authorization": "Bearer unit-key"})
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(payload.decode("utf-8"))["ok"], True)
            status, _ = transport.request_bytes(f"http://127.0.0.1:{port}/missing")
            self.assertEqual(status, 404)
        finally:
            bound.shutdown()
            bound.server_close()

    def test_post_sends_body_and_headers(self):
        bound = self._spin()
        try:
            port = bound.server_address[1]
            status, payload = transport.request_bytes(
                f"http://127.0.0.1:{port}/api", method="POST", body='{"stop": true}',
                headers={"Authorization": "Bearer unit-key"})
            self.assertEqual(status, 201)
            self.assertEqual(json.loads(payload.decode("utf-8"))["stop"], True)
            self.assertEqual(_ApiHandler.seen.get("authorization"), "Bearer unit-key")
        finally:
            bound.shutdown()
            bound.server_close()

    def test_curl_fallback_hits_the_server(self):
        if not transport.curl_available():
            self.skipTest("curl.exe is only available on Windows")
        bound = self._spin()
        try:
            port = bound.server_address[1]
            url = f"http://127.0.0.1:{port}/api"
            status, payload = transport._request_via_curl(
                url, 10.0, {"Authorization": "Bearer unit-key"}, "GET", None)
            self.assertEqual(status, 200)
            self.assertEqual(json.loads(payload.decode("utf-8"))["ok"], True)
        finally:
            bound.shutdown()
            bound.server_close()


class CloudRouteTests(unittest.TestCase):
    def _server(self, tmp: Path, token=None):
        resources = resources_copy(tmp)
        root = tmp / "workspace"
        s = store.Store(resources)
        s.load_workspace(str(root))
        s._port = 0
        s._access = "lan" if token else "local"
        bound = server.Server(s, token, s._access, "127.0.0.1", 0)
        s._port = bound.httpd.server_address[1]
        return s, bound

    def _request(self, port: int, path: str, method: str = "GET", body: dict | None = None,
                 token: str | None = None) -> tuple[int, dict]:
        url = f"http://127.0.0.1:{port}{path}"
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(url, data=data, method=method,
                                         headers={"X-Auth-Token": token or ""})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            try:
                return error.code, json.loads(error.read().decode("utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                return error.code, {}

    def test_cloud_key_roundtrip_never_leaks(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            tmp = Path(tmp)
            s, bound = self._server(tmp)
            threading.Thread(target=bound.serve_forever, daemon=True).start()
            try:
                port = bound.httpd.server_address[1]
                status, view = self._request(port, "/api/cloud/status")
                self.assertEqual(status, 200)
                self.assertFalse(view["keySaved"]["daytona"])
                status, view = self._request(port, "/api/cloud/key", "POST",
                                             {"provider": "daytona", "key": "dk-secret"})
                self.assertEqual(status, 200)
                self.assertTrue(view["keySaved"]["daytona"])
                status, view = self._request(port, "/api/cloud/status")
                self.assertNotIn("dk-secret", json.dumps(view))
                raw = json.loads((tmp / "workspace" / "cloud-settings.json").read_text("utf-8"))
                self.assertEqual(raw["keys"]["daytona"], "dk-secret")
                status, _ = self._request(port, "/api/cloud/provider", "POST",
                                          {"provider": "workerbee"})
                self.assertEqual(status, 400)
                status, view = self._request(port, "/api/cloud/run-target", "POST",
                                             {"target": "cloud"})
                self.assertEqual(view["runTarget"], "cloud")
                status, _ = self._request(port, "/api/cloud/run-target", "POST",
                                          {"target": "orbit"})
                self.assertEqual(status, 400)
            finally:
                bound.stop()
                s.shutdown()

    def test_lean_routes_roundtrip(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            tmp = Path(tmp)
            s, bound = self._server(tmp)
            threading.Thread(target=bound.serve_forever, daemon=True).start()
            try:
                port = bound.httpd.server_address[1]
                status, view = self._request(port, "/api/lean/settings")
                self.assertEqual(status, 200)
                self.assertEqual(view["status"], "Stopped")
                status, view = self._request(port, "/api/lean/settings", "POST",
                                             {"storagePath": "", "leanPath": "lean.exe",
                                              "ngrokPath": "ngrok"})
                self.assertEqual(view["leanPath"], "lean.exe")
                with self.assertRaises(lean.LeanError):
                    s.lean_token()
            finally:
                bound.stop()
                s.shutdown()

    def test_codex_device_login_script_is_written(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            tmp = Path(tmp)
            cli = fake_cli_directory(tmp)
            s, bound = self._server(tmp)
            threading.Thread(target=bound.serve_forever, daemon=True).start()
            s.settings.codexPath = str(cli / "bin" / ("codex.cmd" if os.name == "nt" else "codex"))
            try:
                script = s.codex_device_login_script()
                text = script.read_text(encoding="utf-8")
                self.assertIn("login --device-auth", text)
                self.assertIn("terminal", str(script.parent))
            finally:
                bound.stop()
                s.shutdown()


if __name__ == "__main__":
    unittest.main()
