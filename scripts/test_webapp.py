"""Tests for the Auto Research web edition, without provider calls.

The suite follows the repository's harness conventions: synthetic fixtures, a
fake CLI executable that emits synthetic provider events, no network, and no
account data. Run it with the other Python tests:

    python3 -B -m unittest discover -s scripts -p 'test_*.py'
"""
from pathlib import Path
import importlib
import json
import os
import sys
import tempfile
import threading
import time
import unittest
import secrets
import socket
import urllib.request
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
(ROOT / "tmp").mkdir(exist_ok=True)

from webapp import catalog, cli_runner, dashboard, models, server, store, workspace  # noqa: E402
from launcher import launcher  # noqa: E402

REFERENCE = 978307200.0
UTC = timezone.utc


def resources_copy(tmp: Path) -> Path:
    """A Resources directory with the small files the store reads."""
    resources = tmp / "Resources"
    prompts = resources / "StarterPrompts"
    prompts.mkdir(parents=True, exist_ok=True)
    (resources / "BuiltinProjects.json").write_text(json.dumps([
        {"id": "test_kernel", "name": "Test kernel", "summary": "Test.",
         "category": "Kernel challenges", "website": "https://example.com/kernel",
         "leaderboard": "https://example.com/kernel",
         "promptFile": "StarterPrompts/test_kernel.md",
         "suggestedPrompts": ["Explain the starter."]}]), encoding="utf-8")
    (prompts / "test_kernel.md").write_text("Improve the kernel starter.", encoding="utf-8")
    (prompts / "OfficialJudge.md").write_text("The official judge evaluates remotely.", encoding="utf-8")
    (resources / "Defaults.example.json").write_text(json.dumps({
        "workspaceRoot": "", "catalogURL": "https://example.com/auto-research",
        "codexPath": "", "claudePath": "", "codexModel": "", "claudeModel": "",
        "codexSandbox": "workspace-write", "claudePermissionMode": "auto",
        "opencodePath": "", "opencodeModel": "", "opencodePermission": "reject",
        "antigravityPath": "", "antigravityModel": "", "antigravityMode": "accept-edits",
        "antigravityPermission": "review"}), encoding="utf-8")
    (resources / "ResearchPolicy.md").write_text("Provide evidence accurately.", encoding="utf-8")
    (resources / "ContinuePolicy.md").write_text("Consider continuing.", encoding="utf-8")
    (resources / "CatalogPolicy.json").write_text(json.dumps({"excludedOwners": []}), encoding="utf-8")
    return resources


def fake_cli_directory(tmp: Path, python: str = sys.executable) -> Path:
    """A fake codex CLI: emits synthetic events and reports a signed-in state."""
    directory = tmp / "fake-cli"
    (directory / "bin").mkdir(parents=True, exist_ok=True)
    script = directory / "fake_codex.py"
    script.write_text(
        "import json,sys,time\n"
        "args=sys.argv[1:]\n"
        "if 'mcp' in args:\n"
        "    print('[]');sys.exit(0)\n"
        "if 'login' in args:\n"
        "    print('logged in');sys.exit(0)\n"
        "if 'status' in args:\n"
        "    print('ok');sys.exit(0)\n"
        "time.sleep(float(__import__('os').environ.get('FAKE_CODEX_DELAY','0')))\n"
        "sys.stdin.read()\n"
        "for line in [\"{'type':'thread.started','thread_id':'TEST-THREAD'}\",\n"
        "             \"{'type':'item','item':{'id':'i1','type':'agent_message','text':'Finding.\\\\nRESEARCH_STATUS: finished'}}\",\n"
        "             \"{'type':'turn.completed'}\"]:\n"
        "    print(line.replace(\"'\",'\\\"'))\n"
        "sys.stdout.flush()\n", encoding="utf-8")
    if os.name == "nt":
        wrapper = directory / "bin" / "codex.cmd"
        wrapper.write_text(f'@echo off\r\n"{python}" "{script}" %*\r\n', encoding="utf-8")
    else:
        wrapper = directory / "bin" / "codex"
        wrapper.write_text(f'#!/bin/sh\nexec "{python}" "{script}" "$@"\n', encoding="utf-8")
        wrapper.chmod(0o755)
    return directory


class CatalogParseTests(unittest.TestCase):
    FIXTURE = """
    <html><body>
    <article class="project-entry" data-category="Mathematics">
      <h3>Unit distance grid refinement</h3>
      <a href="https://github.com/rohanarun/other-first">x</a>
      <div class="project-description"><p>Grids &amp; edges</p></div>
      <a href="https://github.com/rohanarun/unit-distance">repo</a>
    </article>
    <script>var t = { repo: "rohanarun/matrix-multiplication" };</script>
    <a href="https://github.com/rohanarun/lean-search">plain</a>
    </body></html>
    """

    def test_parse_extracts_cards_tracks_and_anchors(self):
        projects = catalog.parse(self.FIXTURE)
        ids = [p.id for p in projects]
        self.assertIn("rohanarun/unit-distance", ids)
        self.assertIn("rohanarun/matrix-multiplication", ids)
        self.assertIn("rohanarun/lean-search", ids)
        card = next(p for p in projects if p.id == "rohanarun/unit-distance")
        self.assertEqual(card.name, "Unit distance grid refinement")
        self.assertEqual(card.summary, "Grids & edges")
        self.assertEqual(card.category, "Mathematics")
        track = next(p for p in projects if p.id == "rohanarun/matrix-multiplication")
        self.assertEqual(track.category, "Mathematics")

    def test_merge_keeps_custom_and_session_projects(self):
        fetched = [models.ResearchProject("new/project")]
        existing = [models.ResearchProject("custom/kept", is_custom=True),
                    models.ResearchProject("stale/dropped")]
        session = models.ResearchProject("with/history")
        builtins = [models.ResearchProject(repository="", name="B", builtin_id="k")]
        merged = catalog.merge(fetched, existing + [session], builtins, {session.id})
        ids = [p.id for p in merged]
        self.assertEqual(ids[0], "builtin:k")  # fresh entries win, built-ins lead
        self.assertIn("custom/kept", ids)
        self.assertIn("with/history", ids)
        self.assertNotIn("stale/dropped", ids)


class CommandParityTests(unittest.TestCase):
    def setUp(self):
        self.settings = models.Settings({})

    def test_codex_command_matches_native_app(self):
        command = cli_runner.chat_command(
            "codex", self.settings, None, "P", "POLICY",
            disabled_mcp_servers=["alpha"], enabled_mcp_servers=["beta"])
        self.assertEqual(command.executable, "")
        self.assertEqual(command.arguments, [
            "exec", "--disable", "plugins", "-c", 'approval_policy="never"',
            "-c", 'sandbox_mode="workspace-write"',
            "-c", "mcp_servers.alpha.enabled=false",
            "-c", "mcp_servers.beta.enabled=true",
            "--json", "--skip-git-repo-check", "-"])
        self.assertTrue(command.input.startswith("Harness instructions:\nPOLICY"))
        self.assertIn("User request:\nP", command.input)

    def test_codex_resume_and_model(self):
        settings = models.Settings({"codexPath": "codex", "codexModel": "gpt-5.1-codex"})
        command = cli_runner.chat_command("codex", settings, "S", "P", "POLICY")
        self.assertEqual(command.executable, "codex")
        self.assertEqual(command.arguments, [
            "exec", "--disable", "plugins", "-c", 'approval_policy="never"',
            "-c", 'sandbox_mode="workspace-write"', "resume", "S",
            "--json", "--skip-git-repo-check", "--model", "gpt-5.1-codex", "-"])

    def test_claude_command_matches_native_app(self):
        settings = models.Settings({"claudePath": "claude", "claudePermissionMode": "plan"})
        command = cli_runner.chat_command("claude", settings, None, "P", "POLICY")
        self.assertEqual(command.arguments, [
            "--print", "--verbose", "--output-format", "stream-json",
            "--permission-mode", "plan", "--append-system-prompt", "POLICY"])
        self.assertEqual(command.input, "P")

    def test_opencode_command_matches_native_app(self):
        settings = models.Settings({"opencodePath": "opencode", "opencodePermission": "auto",
                                    "opencodeModel": "provider/model"})
        command = cli_runner.chat_command("opencode", settings, "S", "P", "POLICY")
        self.assertEqual(command.arguments, [
            "run", "--format", "json", "--auto", "--session", "S",
            "--model", "provider/model"])
        self.assertIn("User request:\nP", command.input)

    def test_antigravity_command_matches_native_app(self):
        settings = models.Settings({"antigravityPath": "agy", "antigravityPermission": "auto"})
        command = cli_runner.chat_command("antigravity", settings, "C", "P", "POLICY")
        self.assertEqual(command.arguments[-1], "--print=Harness instructions:\nPOLICY\n\nUser request:\nP")
        self.assertIn("--dangerously-skip-permissions", command.arguments)
        self.assertIn("--conversation", command.arguments)
        self.assertIsNone(command.input)

    def test_mcp_discovery_and_configured_servers(self):
        command = cli_runner.mcp_discovery("codex")
        self.assertEqual(command.arguments, ["--disable", "plugins", "mcp", "list", "--json"])
        self.assertEqual(cli_runner.configured_mcp_servers(
            json.dumps([{"name": "b"}, {"name": "a"}]).encode()), ["a", "b"])
        with self.assertRaises(cli_runner.ResearchError):
            cli_runner.configured_mcp_servers(json.dumps([{"name": "a.b"}]).encode())


class EventParityTests(unittest.TestCase):
    def test_codex_events(self):
        events = cli_runner.decode('{"type":"thread.started","thread_id":"T"}', "codex")
        self.assertEqual(events, [{"session": "T"}])
        events = cli_runner.decode(
            '{"type":"item","item":{"id":"i1","type":"command_execution",'
            '"command":"ls","status":"in_progress","aggregated_output":"x"}}', "codex")
        self.assertEqual(events, [{"message": ("i1", "tool", "ls\nin_progress\nx")}])
        events = cli_runner.decode(
            '{"type":"item","item":{"id":"i2","type":"agent_message","text":"Hi"}}', "codex")
        self.assertEqual(events, [{"message": ("i2", "assistant", "Hi")}])
        events = cli_runner.decode(
            '{"type":"turn.failed","error":{"message":"boom"}}', "codex")
        self.assertEqual(events, [{"failure": "boom"}])
        events = cli_runner.decode('{"type":"turn.completed"}', "codex")
        self.assertEqual(events, [{"completed": True}])

    def test_claude_events(self):
        events = cli_runner.decode(
            '{"type":"assistant","session_id":"S","message":{"id":"M","content":'
            '[{"type":"text","text":"A"},{"type":"tool_use","id":"T1","name":"Bash",'
            '"input":{"c":"ls"}}]}}', "claude")
        self.assertIn({"session": "S"}, events)
        self.assertIn({"message": ("M-0", "assistant", "A")}, events)
        self.assertIn({"message": ("T1", "tool", "Bash\n" + json.dumps({"c": "ls"}, indent=2,
                                                                       sort_keys=True))}, events)
        events = cli_runner.decode('{"type":"result","is_error":true,"errors":["x"]}', "claude")
        self.assertEqual(events, [{"failure": "x"}])
        events = cli_runner.decode('{"type":"result"}', "claude")
        self.assertEqual(events, [{"completed": True}])

    def test_opencode_events(self):
        events = cli_runner.decode(
            '{"type":"text","sessionID":"S","part":{"id":"P1","text":"A"}}', "opencode")
        self.assertIn({"session": "S"}, events)
        self.assertIn({"message": ("P1", "assistant", "A")}, events)
        events = cli_runner.decode(
            '{"type":"step_finish","part":{"reason":"stop"}}', "opencode")
        self.assertEqual(events, [{"completed": True}])
        events = cli_runner.decode(
            '{"type":"error","error":{"message":"nope"}}', "opencode")
        self.assertEqual(events, [{"failure": "nope"}])

    def test_antigravity_events(self):
        events = cli_runner.decode('{"type":"init","conversation_id":"C"}', "antigravity")
        self.assertEqual(events, [{"session": "C"}])
        events = cli_runner.decode(
            '{"type":"result","result":{"status":"SUCCESS","conversation_id":"C",'
            '"response":"Done","num_turns":1}}', "antigravity")
        self.assertEqual(events, [
            {"session": "C"},
            {"message": ("antigravity-result-C-1", "assistant", "Done")},
            {"completed": True}])

    def test_continuation_decision_reads_the_last_status(self):
        self.assertEqual(cli_runner.continuation_decision("a\nRESEARCH_STATUS: continue"), "continue")
        self.assertIsNone(cli_runner.continuation_decision("RESEARCH_STATUS: nope"))


class StateInteropTests(unittest.TestCase):
    def test_state_file_uses_native_keys_and_dates(self):
        message = models.ChatMessage(role="assistant", text="Hi")
        session = models.ResearchSession("p", "Research · Demo", "codex")
        session.messages.append(message)
        session.provider_session_id = "THREAD"
        state = models.SavedState()
        state.projects = [models.ResearchProject("demo/app")]
        state.sessions = [session]
        data = json.loads(json.dumps(state.to_json()))
        self.assertEqual(data["sessions"][0]["projectID"], "p")
        self.assertEqual(data["sessions"][0]["providerSessionID"], "THREAD")
        self.assertEqual(data["sessions"][0]["messages"][0]["id"], message.id)
        self.assertIsInstance(data["sessions"][0]["createdAt"], float)
        restored = models.SavedState.from_json(data)
        self.assertEqual(restored.sessions[0].messages[0].text, "Hi")
        self.assertEqual(restored.sessions[0].provider_session_id, "THREAD")

    def test_dates_match_swift_reference_encoding(self):
        date = models.now()
        self.assertGreater(date, 0)
        self.assertEqual(models.decode_date(date + REFERENCE), date)
        self.assertEqual(models.decode_date("2026-01-01T00:00:00Z"),
                         datetime(2026, 1, 1, tzinfo=UTC).timestamp() - REFERENCE)

    def test_settings_decode_with_missing_keys(self):
        settings = models.Settings({"workspaceRoot": "/x"})
        self.assertEqual(settings.opencodePermission, "reject")
        self.assertEqual(settings.antigravityMode, "accept-edits")
        self.assertEqual(settings.catalogURL, "https://app.getsupers.com/auto-research")


class WorkspaceTests(unittest.TestCase):
    def test_validate_accepts_writable_directory(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            self.assertEqual(workspace.validate(tmp), Path(tmp).resolve())

    def test_directory_rejects_escaping_destination(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            outside = Path(tmp).parent / "outside.txt"
            outside.write_text("private")
            try:
                with self.assertRaises(workspace.ResearchError):
                    workspace.directory(tmp, "../outside")
            finally:
                outside.unlink(missing_ok=True)

    def test_instruction_file_prefers_literal_entry_point(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            root = Path(tmp)
            (root / "docs").mkdir()
            (root / "docs" / "auto-research.md").write_text("nested", encoding="utf-8")
            (root / "AUTO-RESEARCH.md").write_text("top", encoding="utf-8")
            found = workspace.instruction_file(root, ["docs/auto-research.md", "AUTO-RESEARCH.md"])
            self.assertEqual(found, root / "AUTO-RESEARCH.md")


class BuiltinTests(unittest.TestCase):
    def test_prepare_seeds_starter_once(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            tmp = Path(tmp)
            resources = resources_copy(tmp)
            builtins = importlib.import_module("webapp.builtin")
            definitions = builtins.load(resources)
            self.assertEqual(len(definitions), 1)
            directory = definitions[0].prepare(str(tmp / "workspace"))
            guide = directory / "auto-research.md"
            self.assertIn("official judge evaluates remotely", guide.read_text(encoding="utf-8"))
            guide.write_text("MY EDITS", encoding="utf-8")
            definitions[0].prepare(str(tmp / "workspace"))
            self.assertEqual(guide.read_text(encoding="utf-8"), "MY EDITS")


class RunLifecycleTests(unittest.TestCase):
    def _store(self, tmp: Path, provider_paths: dict):
        resources = resources_copy(tmp)
        root = tmp / "workspace"
        s = store.Store(resources)
        s.load_workspace(str(root))
        s.settings.codexPath = str(provider_paths / "bin" / ("codex.cmd" if os.name == "nt" else "codex"))
        s.settings.workspaceRoot = str(root)
        # A pre-existing checkout keeps the suite offline; the app preserves
        # existing checkouts instead of re-cloning.
        checkout = root / "projects" / "octocat" / "research-demo"
        (checkout / ".git").mkdir(parents=True, exist_ok=True)
        s.refresh_catalog(fetched=[models.ResearchProject("octocat/research-demo")])
        s.bump()
        return s

    def _wait(self, s: store.Store, session_id: str, states: set, timeout: float = 60):
        deadline = time.time() + timeout
        while time.time() < deadline:
            session = s._session(session_id)
            if session and session.state in states:
                return session
            time.sleep(0.1)
        raise AssertionError("Session never left the running states; last state: "
                             + (s._session(session_id).state if s._session(session_id) else "missing"))

    def test_completed_turn_produces_messages_and_state(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            tmp = Path(tmp)
            cli = fake_cli_directory(tmp)
            s = self._store(tmp, cli)
            project = s._project("octocat/research-demo")
            session_id = s.play(project.id)
            session = self._wait(s, session_id, {"Completed", "Needs attention"})
            self.assertEqual(session.state, "Completed", s._session(session_id).messages)
            self.assertEqual(session.provider_session_id, "TEST-THREAD")
            texts = [m.text for m in session.messages]
            self.assertTrue(any(t.startswith("Finding.") for t in texts), texts)
            self.assertTrue(any(t.startswith("Loaded ") or t.startswith("auto-research.md was not found")
                                for t in texts))

    def test_stop_marks_stopped_and_guard_and_close(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            tmp = Path(tmp)
            cli = fake_cli_directory(tmp)
            os.environ["FAKE_CODEX_DELAY"] = "30"
            try:
                s = self._store(tmp, cli)
                project = s._project("octocat/research-demo")
                session_id = s.play(project.id)
                self._wait(s, session_id, {"Running"})
                # A running chat cannot be closed; history is preserved.
                with self.assertRaises(workspace.ResearchError):
                    s.close(session_id)
                s.stop(session_id)
                self._wait(s, session_id, {"Stopped"})
                deadline = time.time() + 30
                while time.time() < deadline and session_id in s._runners:
                    time.sleep(0.1)
                self.assertNotIn(session_id, s._runners)
                s.close(session_id)
                self.assertIsNotNone(s._session(session_id).closed_at)
                s.reopen(session_id)
                self.assertIsNone(s._session(session_id).closed_at)
            finally:
                os.environ.pop("FAKE_CODEX_DELAY", None)

    def test_missing_cli_reports_needs_attention(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            tmp = Path(tmp)
            s = self._store(tmp, Path(tmp))
            s.settings.codexPath = ""
            project = s._project("octocat/research-demo")
            session_id = s.play(project.id)
            deadline = time.time() + 30
            while time.time() < deadline:
                session = s._session(session_id)
                if session and session.state == "Needs attention" and session.messages:
                    break
                time.sleep(0.1)
            self.assertEqual(session.state, "Needs attention")
            self.assertIn("CLI is not installed", session.messages[-1].text)


class DashboardViewTests(unittest.TestCase):
    HISTORY = {
        "as_of": "2026-10-15T12:00:00Z", "refresh_after_seconds": 300.0,
        "scopes": {"t1": {"id": "t1", "name": "Unit distances",
                          "repo": "rohanarun/unit-distance"}},
        "groups": {"g1": {"id": "g1", "track": "t1", "name": "General case",
                          "direction": "lower",
                          "reconstructed": {"timeline": [
                              {"from": 1760000000.0, "to": 1760100000.0, "value": 36,
                               "sources": ["https://github.com/p/1"], "leaders": []},
                              {"from": 1760100000.0, "to": 1760200000.0, "value": 30,
                               "sources": [], "leaders": []}], "value": 30},
                          "observed": {"timeline": [], "value": None}}},
    }
    RESEARCH = {
        "tracks": {"t1": {"pulls": [
            {"number": 5, "title": "Add guide", "html_url": "https://github.com/p/5",
             "state": "closed", "merged_at": "2026-10-01", "created_at": "2026-10-01",
             "updated_at": "2026-10-02", "author": "a"}], "checked_at": "2026-10-15"}},
        "error": None,
    }

    def test_project_view_assembles_groups_and_pulls(self):
        view = dashboard.project_view(self.HISTORY, self.RESEARCH, "rohanarun/unit-distance")
        self.assertEqual(view["scope"]["id"], "t1")
        group = view["groups"][0]
        self.assertEqual(group["estimated"]["value"], 30)
        self.assertEqual(len(group["estimated"]["points"]), 2)
        self.assertEqual(view["pulls"][0]["status"], "Merged")
        self.assertEqual(view["claimStatus"], "Source claim")

    def test_scope_is_none_for_unknown_repository(self):
        self.assertIsNone(dashboard.project_view(self.HISTORY, self.RESEARCH, "no/such"))


class ServerApiTests(unittest.TestCase):
    def _server(self, tmp: Path, token: str | None):
        cli = fake_cli_directory(tmp)
        resources = resources_copy(tmp)
        root = tmp / "workspace"
        s = store.Store(resources)
        s.load_workspace(str(root))
        s.settings.codexPath = str(cli / "bin" / ("codex.cmd" if os.name == "nt" else "codex"))
        s.settings.workspaceRoot = str(root)
        checkout = root / "projects" / "octocat" / "research-demo"
        (checkout / ".git").mkdir(parents=True, exist_ok=True)
        s.refresh_catalog(fetched=[models.ResearchProject("octocat/research-demo")])
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
            body_text = error.read().decode("utf-8")
            try:
                return error.code, json.loads(body_text)
            except json.JSONDecodeError:
                return error.code, {}

    def test_state_settings_and_run_over_http(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            tmp = Path(tmp)
            s, bound = self._server(tmp, token=None)
            threading.Thread(target=bound.serve_forever, daemon=True).start()
            try:
                port = bound.httpd.server_address[1]
                status, state = self._request(port, "/api/state")
                self.assertEqual(status, 200)
                self.assertTrue(state["server"]["workspaceRoot"])
                status, payload = self._request(port, "/api/settings")
                self.assertEqual(payload["codexSandbox"], "workspace-write")
                self.assertIn("codexPath", payload["detectedPaths"])
                status, created = self._request(port, "/api/play", "POST",
                                                {"projectID": "octocat/research-demo"})
                self.assertEqual(status, 200)
                session_id = created["sessionID"]
                session = None
                for _ in range(100):
                    _, state = self._request(port, "/api/state")
                    session = next(x for x in state["sessions"] if x["id"] == session_id)
                    if session["state"] in ("Completed", "Needs attention"):
                        break
                    time.sleep(0.1)
                self.assertEqual(session["state"], "Completed")
                texts = [m["text"] for m in session["messages"]]
                self.assertTrue(any(t.startswith("Finding.") for t in texts), texts)
                status, _ = self._request(port, f"/api/logs?sessionID={session_id}")
                self.assertEqual(status, 200)
                self._request(port, "/api/chats/close", "POST", {"sessionID": session_id})
                self._request(port, "/api/chats/reopen", "POST", {"sessionID": session_id})
                self._request(port, "/api/chats/stop", "POST", {"sessionID": session_id})
            finally:
                bound.stop()
                s.shutdown()

    def test_token_guard_rejects_untokened_api_calls(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            tmp = Path(tmp)
            s, bound = self._server(tmp, token="unit-secret")
            threading.Thread(target=bound.serve_forever, daemon=True).start()
            try:
                port = bound.httpd.server_address[1]
                status, health = self._request(port, "/healthz")
                self.assertEqual((status, health["ok"]), (200, True))
                status, _ = self._request(port, "/api/state")
                self.assertEqual(status, 401)
                status, _ = self._request(port, "/api/state", token="wrong")
                self.assertEqual(status, 401)
                status, state = self._request(port, "/api/state", token="unit-secret")
                self.assertEqual(status, 200)
            finally:
                bound.stop()
                s.shutdown()


class TriageFixTests(unittest.TestCase):
    """The web edition's setup drafts, editing endpoints, snapshot fallback
    and parallel agents, mirroring the native app's flows."""

    # ---------- fixtures ----------

    def _resources(self, tmp: Path) -> Path:
        resources = tmp / "Resources"
        prompts = resources / "StarterPrompts"
        prompts.mkdir(parents=True, exist_ok=True)
        (resources / "BuiltinProjects.json").write_text(json.dumps([
            {"id": "test_kernel", "name": "Test kernel", "summary": "Test.",
             "category": "Kernel challenges", "website": "https://example.com/kernel",
             "leaderboard": "https://example.com/kernel",
             "promptFile": "StarterPrompts/test_kernel.md",
             "suggestedPrompts": ["Explain the starter.", "Second suggestion."]}]), encoding="utf-8")
        (prompts / "test_kernel.md").write_text("Improve the kernel starter.", encoding="utf-8")
        (prompts / "OfficialJudge.md").write_text("The official judge evaluates remotely.", encoding="utf-8")
        (prompts / "OKCLogin.md").write_text("Sign in with your own account.", encoding="utf-8")
        (prompts / "SetupTemplate.md").write_text(
            "# OKC kernel research request\n\n{{PROJECT_INPUTS_JSON}}\n\n## Account setup\n",
            encoding="utf-8")
        (resources / "Defaults.example.json").write_text(json.dumps({
            "workspaceRoot": "", "catalogURL": "https://example.com/auto-research",
            "codexPath": "", "claudePath": "", "codexModel": "", "claudeModel": "",
            "codexSandbox": "workspace-write", "claudePermissionMode": "auto",
            "opencodePath": "", "opencodeModel": "", "opencodePermission": "reject",
            "antigravityPath": "", "antigravityModel": "", "antigravityMode": "accept-edits",
            "antigravityPermission": "review"}), encoding="utf-8")
        (resources / "ResearchPolicy.md").write_text("Provide evidence accurately.", encoding="utf-8")
        (resources / "ContinuePolicy.md").write_text("Consider continuing.", encoding="utf-8")
        (resources / "SharedAgentPolicy.md").write_text("Shared agents keep handoffs current.", encoding="utf-8")
        (resources / "IndependentAgentPolicy.md").write_text("Independent agents share nothing.", encoding="utf-8")
        (resources / "CatalogPolicy.json").write_text(json.dumps({"excludedOwners": []}), encoding="utf-8")
        return resources

    def _fake_cli(self, tmp: Path, mcp_json: str = "[]") -> Path:
        """A fake codex CLI with configurable MCP listing."""
        directory = tmp / "fake-cli"
        (directory / "bin").mkdir(parents=True, exist_ok=True)
        script = directory / "fake_codex.py"
        script.write_text(
            "import json,sys,time\n"
            "args=sys.argv[1:]\n"
            "if 'mcp' in args:\n"
            "    print(" + json.dumps(mcp_json) + ");sys.exit(0)\n"
            "if 'login' in args:\n"
            "    print('logged in');sys.exit(0)\n"
            "if 'status' in args:\n"
            "    print('ok');sys.exit(0)\n"
            "time.sleep(float(__import__('os').environ.get('FAKE_CODEX_DELAY','0')))\n"
            "sys.stdin.read()\n"
            "for line in [\"{'type':'thread.started','thread_id':'TEST-THREAD'}\",\n"
            "             \"{'type':'item','item':{'id':'i1','type':'agent_message','text':'Finding.\\\\nRESEARCH_STATUS: finished'}}\",\n"
            "             \"{'type':'turn.completed'}\"]:\n"
            "    print(line.replace(\"'\",'\\\"'))\n"
            "sys.stdout.flush()\n", encoding="utf-8")
        if os.name == "nt":
            wrapper = directory / "bin" / "codex.cmd"
            wrapper.write_text(f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n', encoding="utf-8")
        else:
            wrapper = directory / "bin" / "codex"
            wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
            wrapper.chmod(0o755)
        return directory

    def _fake_okc(self, tmp: Path, account: str = "tester") -> Path:
        """A fake okc CLI whose whoami prints the signed-in identity line."""
        directory = tmp / "fake-okc"
        (directory / "bin").mkdir(parents=True, exist_ok=True)
        script = directory / "fake_okc.py"
        script.write_text(
            "import sys,time\n"
            "args=sys.argv[1:]\n"
            "if 'whoami' in args:\n"
            "    print('signed in as @" + account + "');sys.exit(0)\n"
            "if 'login' in args:\n"
            "    print('login flow');sys.exit(0)\n"
            "time.sleep(0)\n", encoding="utf-8")
        if os.name == "nt":
            wrapper = directory / "bin" / "okc.cmd"
            wrapper.write_text(f'@echo off\r\n"{sys.executable}" "{script}" %*\r\n', encoding="utf-8")
        else:
            wrapper = directory / "bin" / "okc"
            wrapper.write_text(f'#!/bin/sh\nexec "{sys.executable}" "{script}" "$@"\n', encoding="utf-8")
            wrapper.chmod(0o755)
        return directory / "bin" / ("okc.cmd" if os.name == "nt" else "okc")

    def _store(self, tmp: Path, cli: Path, okc: Path | None = None) -> "tuple[store.Store, Path]":
        from webapp import kernel as kernel_module
        root = tmp / "workspace"
        s = store.Store(self._resources(tmp))
        s.load_workspace(str(root))
        s.settings.codexPath = str(cli / "bin" / ("codex.cmd" if os.name == "nt" else "codex"))
        s.settings.workspaceRoot = str(root)
        checkout = root / "projects" / "octocat" / "research-demo"
        (checkout / ".git").mkdir(parents=True, exist_ok=True)
        s.refresh_catalog(fetched=[models.ResearchProject("octocat/research-demo")])
        if okc is not None:
            self._patch(kernel_module, "okc_executable", lambda: okc)
        s.bump()
        return s, root

    def _patch(self, module, name, replacement):
        original = getattr(module, name)
        setattr(module, name, replacement)
        self.addCleanup(setattr, module, name, original)

    # ---------- kernel drafts ----------

    def test_draft_validation_mirrors_native_rules(self):
        from webapp import kernel
        with self.assertRaises(kernel.ResearchError):
            kernel.validate_draft({"expectedAccount": "bad name", "researchPrompt": "p",
                                   "checkBudget": "0"})
        with self.assertRaises(kernel.ResearchError):
            kernel.validate_draft({"expectedAccount": "tester", "researchPrompt": "",
                                   "checkBudget": "0"})
        with self.assertRaises(kernel.ResearchError):
            kernel.validate_draft({"expectedAccount": "tester", "researchPrompt": "p",
                                   "checkBudget": "-1"})
        with self.assertRaises(kernel.ResearchError):
            kernel.validate_draft({"expectedAccount": "tester", "researchPrompt": "p",
                                   "checkBudget": "0", "allowScoredSubmissions": True,
                                   "scoreBudget": "2", "scoreCriterion": ""})
        with self.assertRaises(kernel.ResearchError):
            kernel.validate_draft({"expectedAccount": "tester", "researchPrompt": "p",
                                   "checkBudget": "0", "allowScoredSubmissions": True,
                                   "scoreBudget": "0"})
        draft = kernel.validate_draft({"expectedAccount": "@Tester ", "researchPrompt": " p ",
                                       "checkBudget": "12", "allowScoredSubmissions": True,
                                       "scoreBudget": "3", "scoreCriterion": "after merged PR"})
        self.assertEqual(draft["expectedAccount"], "tester")
        self.assertEqual(draft["checkBudget"], "12")
        self.assertEqual(draft["scoreBudget"], "3")

    def test_render_prompt_inserts_sorted_json_once(self):
        from webapp import kernel
        template = "# OKC kernel research request\n\n{{PROJECT_INPUTS_JSON}}\n\n## Account setup\n"
        draft = kernel.validate_draft({"expectedAccount": "tester", "researchPrompt": "Explain",
                                       "checkBudget": "0", "sourcePath": ""})
        prompt = kernel.render_prompt(template, draft, "test_kernel", "Test kernel")
        self.assertEqual(prompt.count("{{PROJECT_INPUTS_JSON}}"), 0)
        self.assertIn('"projectID": "test_kernel"', prompt)
        self.assertIn('"authorizeScoredSubmissions": false', prompt)
        self.assertIn('"superExperimentsURL": "https://app.getsupers.com/experiments"', prompt)
        self.assertNotIn('"sourcePath"', prompt)
        self.assertNotIn('"scoreCriterion"', prompt)

    def test_account_output_is_parsed_not_displayed(self):
        from webapp import kernel
        self.assertEqual(kernel.account_from_output("signed in as @OctCat\n"), "octcat")
        self.assertEqual(kernel.account_from_output("\x1b[31mtester\x1b[0m"), "tester")
        self.assertIsNone(kernel.account_from_output("unexpected banner\nnot an identity"))

    # ---------- catalog snapshot fallback ----------

    def test_catalog_fetch_uses_snapshot_when_offline(self):
        from webapp import transport
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            snapshot = Path(tmp) / "catalog.html"
            snapshot.write_text(
                '<a href="https://github.com/octo/snapshot-repo">x</a>', encoding="utf-8")

            def failing(address, timeout=20.0, accept="application/json"):
                raise transport.TransportError("certificate has expired")

            self._patch(transport, "fetch_text", failing)
            fetched, offline, note = catalog.fetch("https://example.com/auto-research",
                                                   snapshot=snapshot)
            self.assertTrue(offline)
            self.assertEqual([p.id for p in fetched], ["octo/snapshot-repo"])
            self.assertIn("snapshot", note.lower())

    def test_catalog_fetch_saves_snapshot_on_success(self):
        from webapp import transport
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            snapshot = Path(tmp) / "catalog.html"
            html = '<a href="https://github.com/octo/live-repo">x</a>'

            def passing(address, timeout=20.0, accept="application/json"):
                return html

            self._patch(transport, "fetch_text", passing)
            fetched, offline, note = catalog.fetch("https://example.com/auto-research",
                                                   snapshot=snapshot)
            self.assertFalse(offline)
            self.assertEqual([p.id for p in fetched], ["octo/live-repo"])
            self.assertEqual(snapshot.read_text(encoding="utf-8"), html)

    def test_transport_uses_curl_fallback_when_urllib_fails(self):
        from webapp import transport
        import subprocess as subprocess_module
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            fake = Path(tmp) / "curl.exe"
            fake.write_text("", encoding="utf-8")
            self._patch(transport, "CURL", fake)
            self._patch(transport, "curl_available", lambda: True)

            def runner(command, capture_output, timeout, creationflags=0):
                class Result:
                    returncode = 0
                    stdout = b"payload"
                    stderr = b""
                return Result()

            self._patch(subprocess_module, "run", runner)
            self.assertEqual(transport._fetch_via_curl("https://example.com/x"), b"payload")

    # ---------- store endpoints over HTTP ----------

    def _server(self, tmp: Path, cli: Path, okc: Path | None = None):
        from webapp import kernel as kernel_module
        resources = self._resources(tmp)
        root = tmp / "workspace"
        s = store.Store(resources)
        s.load_workspace(str(root))
        s.settings.codexPath = str(cli / "bin" / ("codex.cmd" if os.name == "nt" else "codex"))
        s.settings.workspaceRoot = str(root)
        checkout = root / "projects" / "octocat" / "research-demo"
        (checkout / ".git").mkdir(parents=True, exist_ok=True)
        s.refresh_catalog(fetched=[models.ResearchProject("octocat/research-demo")])
        if okc is not None:
            self._patch(kernel_module, "okc_executable", lambda: okc)
        s._port = 0
        s._access = "local"
        bound = server.Server(s, None, "local", "127.0.0.1", 0)
        s._port = bound.httpd.server_address[1]
        return s, bound

    def _request(self, port: int, path: str, method: str = "GET", body=None):
        url = f"http://127.0.0.1:{port}{path}"
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(url, data=data, method=method,
                                         headers={"X-Auth-Token": ""})
        try:
            with urllib.request.urlopen(request, timeout=30) as response:
                return response.status, json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as error:
            payload = error.read().decode("utf-8")
            try:
                return error.code, json.loads(payload)
            except json.JSONDecodeError:
                return error.code, {}

    def test_policy_instructions_setup_and_projects_over_http(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            tmp = Path(tmp)
            cli = self._fake_cli(tmp)
            okc = self._fake_okc(tmp)
            s, bound = self._server(tmp, cli, okc=okc)
            threading.Thread(target=bound.serve_forever, daemon=True).start()
            try:
                port = bound.httpd.server_address[1]
                # Policy editor round-trip.
                status, policy = self._request(port, "/api/policy")
                self.assertEqual((status, policy["text"]), (200, "Provide evidence accurately."))
                status, _ = self._request(port, "/api/policy", "POST", {"text": "Be honest.\n"})
                self.assertEqual(status, 200)
                self.assertEqual(s.policy_text(), "Be honest.\n")
                self.assertEqual((Path(s.workspace_root) / "research-policy.md")
                                 .read_text(encoding="utf-8"), "Be honest.\n")

                # Built-in setup: overview, save, validation, account verify.
                status, overview = self._request(port, "/api/setup?projectID=builtin:test_kernel")
                self.assertEqual(status, 200)
                self.assertEqual(overview["draft"]["researchPrompt"], "Explain the starter.")
                self.assertIn("promptError", overview)  # defaults have no account yet
                self.assertIn("Sign in with your own account.", overview["loginInstructions"])
                draft = dict(overview["draft"])
                draft.update({"expectedAccount": "tester", "checkBudget": "12",
                              "researchPrompt": "Explain the starter."})
                status, saved = self._request(port, "/api/setup", "POST",
                                              {"projectID": "builtin:test_kernel", "draft": draft})
                self.assertEqual(status, 200)
                self.assertEqual(saved["draft"]["expectedAccount"], "tester")
                status, again = self._request(port, "/api/setup?projectID=builtin:test_kernel")
                self.assertEqual(status, 200)
                self.assertIn('"projectID": "test_kernel"', again["prompt"])
                self.assertNotIn("promptError", again)
                status, account = self._request(port, "/api/setup/verify-account", "POST", {})
                self.assertEqual((status, account["account"]), (200, "tester"))

                # Built-in instructions exist after prepare and accept edits.
                status, guide = self._request(port, "/api/instructions?projectID=builtin:test_kernel")
                self.assertEqual(status, 200)
                self.assertIn("official judge evaluates remotely", guide["text"])
                status, _ = self._request(port, "/api/instructions", "POST",
                                          {"projectID": "builtin:test_kernel", "text": "MY EDITS"})
                self.assertEqual(status, 200)
                reloaded = (Path(s.workspace_root) / "projects" / "builtins" / "test_kernel"
                            / "auto-research.md").read_text(encoding="utf-8")
                self.assertEqual(reloaded, "MY EDITS\n")

                # Project edit + remove.
                status, edited = self._request(port, "/api/projects/edit", "POST",
                                               {"projectID": "octocat/research-demo",
                                                "name": "Renamed", "summary": "Mine."})
                self.assertEqual((status, edited["project"]["name"]), (200, "Renamed"))
                status, removed = self._request(port, "/api/projects/remove", "POST",
                                                {"projectID": "octocat/research-demo"})
                self.assertEqual(status, 200)
                self.assertFalse(any(p.id == "octocat/research-demo"
                                     for p in s.state.projects))

                # MCP selection persists.
                status, _ = self._request(port, "/api/mcp", "POST", {"enabled": ["beta"]})
                self.assertEqual(status, 200)
                status, mcp = self._request(port, "/api/mcp")
                self.assertEqual((status, mcp["enabled"]), (200, ["beta"]))
                saved_file = Path(s.workspace_root) / "mcp-selection.json"
                self.assertEqual(json.loads(saved_file.read_text(encoding="utf-8")), ["beta"])

                # Send with no open chat starts a titled chat.
                s.state.selected_project_id = "builtin:test_kernel"
                s.persist()
                status, sent = self._request(port, "/api/chats/send", "POST",
                                             {"sessionID": "", "text": "Explain the starter"})
                self.assertEqual(status, 200)
                title = s._session(sent["sessionID"]).title
                self.assertEqual(title, "Explain the starter")
            finally:
                bound.stop()
                s.shutdown()

    def test_mcp_refresh_lists_configured_servers(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            tmp = Path(tmp)
            cli = self._fake_cli(tmp, mcp_json='[{"name": "beta"}, {"name": "alpha"}]')
            s, bound = self._server(tmp, cli)
            threading.Thread(target=bound.serve_forever, daemon=True).start()
            try:
                port = bound.httpd.server_address[1]
                status, mcp = self._request(port, "/api/mcp/refresh", "POST", {})
                self.assertEqual(status, 200)
                self.assertEqual(mcp["available"], ["alpha", "beta"])
            finally:
                bound.stop()
                s.shutdown()

    def test_project_folder_and_terminal_routes_over_http(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            tmp = Path(tmp)
            cli = self._fake_cli(tmp)
            s, bound = self._server(tmp, cli)
            threading.Thread(target=bound.serve_forever, daemon=True).start()
            try:
                port = bound.httpd.server_address[1]
                project, _ = s.add_project("octocat/research-demo")
                session = self._selected_session(s, project.id)
                opened = self._capture_folder_open()
                status, res = self._request(port, "/api/project/folder", "POST", {})
                self.assertEqual((status, res["ok"]), (200, True))
                self.assertEqual(opened, [session.working_directory])
                # The terminal route opens the launcher on the host; fake the spawn.
                spawned = []
                class FakePopen:
                    def __init__(self, args, **kwargs):
                        spawned.append(args)
                self._patch(store.subprocess, "Popen", FakePopen)
                status, res = self._request(port, "/api/session/terminal", "POST",
                                            {"sessionID": session.id})
                self.assertEqual((status, res["spawned"]), (200, True))
                self.assertEqual(len(spawned), 1)
                text = Path(res["script"]).read_text(encoding="utf-8")
                self.assertIn(session.working_directory, text)
                self.assertIn("THREAD", text)
            finally:
                bound.stop()
                s.shutdown()

    def test_builtin_play_requires_verified_account_match(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            tmp = Path(tmp)
            cli = self._fake_cli(tmp)
            okc = self._fake_okc(tmp, account="octocat")
            s, root = self._store(tmp, cli, okc=okc)
            try:
                s.save_setup_draft("builtin:test_kernel",
                                   {"expectedAccount": "tester", "checkBudget": "0",
                                    "researchPrompt": "Explain the starter."})
                with self.assertRaisesRegex(store.ResearchError, "does not match"):
                    s.play("builtin:test_kernel")
            finally:
                s.shutdown()

    def test_builtin_play_runs_with_verified_account(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            tmp = Path(tmp)
            cli = self._fake_cli(tmp)
            okc = self._fake_okc(tmp, account="tester")
            s, root = self._store(tmp, cli, okc=okc)
            try:
                s.save_setup_draft("builtin:test_kernel",
                                   {"expectedAccount": "tester", "checkBudget": "0",
                                    "researchPrompt": "Explain the starter."})
                session_id = s.play("builtin:test_kernel")
                session = None
                for _ in range(100):
                    session = s._session(session_id)
                    if session and session.state in ("Completed", "Needs attention"):
                        break
                    time.sleep(0.1)
                self.assertEqual(session.state, "Completed", session.messages)
                first_user = next(m for m in session.messages if m.role == "user")
                self.assertIn('"projectID": "test_kernel"', first_user.text)
                self.assertIn("Improve the kernel starter.", first_user.text)
            finally:
                s.shutdown()

    def test_parallel_agents_get_own_checkouts_and_capture(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            tmp = Path(tmp)
            cli = self._fake_cli(tmp)
            s, root = self._store(tmp, cli)
            try:
                first = s.play("octocat/research-demo", parallel=True, agent_count=2,
                               share_context=False)
                sessions = [x for x in s.state.sessions
                            if x.shared_agent and x.shared_agent["sharesContext"] is False]
                self.assertEqual(len(sessions), 2)
                titles = sorted(x.title for x in sessions)
                self.assertEqual(titles, ["Agent 1 · independent", "Agent 2 · independent"])
                for session in sessions:
                    directory = Path(session.working_directory)
                    self.assertTrue((directory / ".git").exists(), directory)
                for _ in range(150):
                    if all(x.state in ("Completed", "Needs attention") for x in sessions):
                        break
                    time.sleep(0.1)
                self.assertTrue(all(x.state == "Completed" for x in sessions),
                                [x.state for x in sessions])
                self.assertEqual(first, sessions[0].id)
                texts = [m.text for x in sessions for m in x.messages if m.role == "user"]
                self.assertTrue(any("Independent parallel agent 1 of 2" in t for t in texts),
                                texts)

                # Shared context: one round per Play; the bundle is captured.
                s2 = s
                s2.state.selected_project_id = "octocat/research-demo"
                s2.play("octocat/research-demo", parallel=True, agent_count=2,
                        share_context=True)
                shared = [x for x in s2.state.sessions if x.shared_agent
                          and x.shared_agent["sharesContext"] is True]
                self.assertEqual(len(shared), 2)
                for _ in range(150):
                    if all(x.state in ("Completed", "Needs attention") for x in shared):
                        break
                    time.sleep(0.1)
                group = shared[0].shared_agent["groupID"]
                bundle = Path(s2.workspace_root) / "shared-context" / group.lower() / "round-1"
                self.assertTrue((bundle / "manifest.json").is_file(), bundle)
                self.assertTrue((bundle / "shared-context.md").is_file(), bundle)
                manifest = json.loads((bundle / "manifest.json").read_text(encoding="utf-8"))
                self.assertEqual([a["number"] for a in manifest["agents"]], [1, 2])
            finally:
                s.shutdown()

    def test_keep_researching_is_tracked_for_play_and_send(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            tmp = Path(tmp)
            cli = self._fake_cli(tmp)
            s, root = self._store(tmp, cli)
            try:
                session_id = s.play("octocat/research-demo", keep_researching=True)
                self.assertIn(session_id, s._keep_researching)
                for _ in range(100):
                    session = s._session(session_id)
                    if session and session.state == "Completed":
                        break
                    time.sleep(0.1)
                self.assertEqual(session.state, "Completed")
                s._keep_researching.clear()
                s.state.selected_project_id = "octocat/research-demo"
                resumed = s.send("", "Follow up", keep_researching=True)
                self.assertIn(resumed, s._keep_researching)
                s.stop(resumed)
            finally:
                s.shutdown()

    def test_drafts_and_selection_survive_reload(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            tmp = Path(tmp)
            cli = self._fake_cli(tmp)
            s, root = self._store(tmp, cli)
            try:
                s.save_setup_draft("builtin:test_kernel",
                                   {"expectedAccount": "tester", "checkBudget": "40",
                                    "researchPrompt": "Explain", "allowScoredSubmissions": True,
                                    "scoreBudget": "2", "scoreCriterion": "after merged PR"})
                s.set_mcp_selection(["beta"])
                s2 = store.Store(s.resources_dir)
                s2.load_workspace(str(root))
                self.assertEqual(s2.setup_draft("builtin:test_kernel")["checkBudget"], "40")
                self.assertEqual(s2.mcp_servers()["enabled"], ["beta"])
            finally:
                s.shutdown()


    # ---------- removal and first-time instructions ----------

    def test_remove_allows_after_chats_closed_and_preserves_history(self):
        from webapp.models import ResearchSession
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            tmp = Path(tmp)
            cli = self._fake_cli(tmp)
            s, root = self._store(tmp, cli)
            try:
                # An open chat blocks removal.
                open_session = ResearchSession(project_id="octocat/research-demo",
                                               title="t", provider="codex")
                s.state.sessions.append(open_session)
                with self.assertRaisesRegex(store.ResearchError, "open chats"):
                    s.remove_project("octocat/research-demo")
                # A closed chat does not block; its history stays in state.
                open_session.closed_at = 123.0
                s.remove_project("octocat/research-demo")
                self.assertFalse(any(p.id == "octocat/research-demo"
                                     for p in s.state.projects))
                self.assertTrue(any(x.id == open_session.id for x in s.state.sessions))
            finally:
                s.shutdown()

    def test_instructions_creates_template_when_missing(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            tmp = Path(tmp)
            cli = self._fake_cli(tmp)
            s, root = self._store(tmp, cli)
            try:
                file_path, text = s.instructions("octocat/research-demo")
                self.assertEqual(file_path.name, "auto-research.md")
                self.assertIn("Research instructions", text)
                # A later call must not rewrite the user's edits.
                file_path.write_text("MY EDITS", encoding="utf-8")
                _path, again = s.instructions("octocat/research-demo")
                self.assertEqual(again, "MY EDITS")
            finally:
                s.shutdown()

    def test_add_project_seeds_pasted_instructions(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            tmp = Path(tmp)
            cli = self._fake_cli(tmp)
            s, root = self._store(tmp, cli)
            try:
                # Repository project whose checkout already exists: seeded in place.
                project, seeded = s.add_project("octocat/research-demo",
                                                instructions="MY GUIDE")
                self.assertTrue(seeded)
                guide = (root / "projects" / "octocat" / "research-demo"
                         / "auto-research.md")
                self.assertEqual(guide.read_text(encoding="utf-8"), "MY GUIDE\n")
                # Re-adding selects the existing entry and never overwrites.
                project2, seeded2 = s.add_project("octocat/research-demo",
                                                  instructions="OTHER")
                self.assertIs(project2, project)
                self.assertFalse(seeded2)
                self.assertEqual(guide.read_text(encoding="utf-8"), "MY GUIDE\n")
                self.assertEqual(len([p for p in s.state.projects
                                      if p.id == "octocat/research-demo"]), 1)
                # A local folder project gets the file written directly.
                folder = tmp / "local-proj"
                folder.mkdir()
                _p3, seeded3 = s.add_project(local_path=str(folder),
                                             instructions="LOCAL GUIDE")
                self.assertTrue(seeded3)
                self.assertEqual((folder / "auto-research.md").read_text(encoding="utf-8"),
                                 "LOCAL GUIDE\n")
            finally:
                s.shutdown()

    def _capture_folder_open(self):
        """Patch the host folder opener; returns the list of opened paths."""
        from webapp import store as store_module
        opened = []
        if os.name == "nt":
            self._patch(os, "startfile", lambda path: opened.append(str(path)))
        else:
            class FakePopen:
                def __init__(self, args, **kwargs):
                    opened.append(str(args[1]))
            self._patch(store_module.subprocess, "Popen", FakePopen)
        return opened

    def _selected_session(self, s, project_id, provider="codex"):
        session = models.ResearchSession(project_id, "Research · Demo", provider)
        session.working_directory = str(Path(s.workspace_root) / "projects"
                                        / "octocat" / "research-demo")
        session.provider_session_id = "THREAD"
        s.state.sessions.append(session)
        s.state.selected_project_id = project_id
        s.state.selected_session_id = session.id
        return session

    def test_open_project_folder_opens_the_selected_checkout(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            tmp = Path(tmp)
            cli = self._fake_cli(tmp)
            s, root = self._store(tmp, cli)
            try:
                project, _ = s.add_project("octocat/research-demo")
                session = self._selected_session(s, project.id)
                opened = self._capture_folder_open()
                result = s.open_project_folder(session.id)
                self.assertEqual(result["ok"], True)
                self.assertEqual(opened, [session.working_directory])
                # Without a chat, a project's own local checkout is used.
                folder = tmp / "local-proj"
                folder.mkdir()
                local_project, _ = s.add_project(local_path=str(folder))
                s.state.selected_project_id = local_project.id
                s.state.selected_session_id = None
                result = s.open_project_folder()
                self.assertEqual(result["ok"], True)
                self.assertEqual(opened, [session.working_directory, str(folder)])
            finally:
                s.shutdown()

    def test_open_project_folder_reports_missing_checkouts(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            tmp = Path(tmp)
            cli = self._fake_cli(tmp)
            s, root = self._store(tmp, cli)
            try:
                project, _ = s.add_project("octocat/research-demo")
                session = self._selected_session(s, project.id)
                self._capture_folder_open()
                session.working_directory = None
                s.state.selected_session_id = None
                # A repository project without a checkout has no folder yet.
                with self.assertRaises(models.ResearchError):
                    s.open_project_folder()
                # Unknown ids say so instead of opening the workspace.
                session.working_directory = str(root / "projects" / "octocat" / "research-demo")
                s.state.selected_session_id = session.id
                session.working_directory = None
                with self.assertRaises(models.ResearchError):
                    s.open_project_folder("missing-session")
            finally:
                s.shutdown()

    def test_continue_in_terminal_writes_the_resume_script(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            tmp = Path(tmp)
            cli = self._fake_cli(tmp)
            s, root = self._store(tmp, cli)
            try:
                project, _ = s.add_project("octocat/research-demo")
                session = self._selected_session(s, project.id)
                script = s.continue_in_terminal(session.id)
                text = script.read_text(encoding="utf-8")
                self.assertTrue(script.name.endswith(".cmd" if os.name == "nt" else ".command"))
                self.assertIn(session.working_directory, text)
                self.assertIn("THREAD", text)
                self.assertIn("resume", text)
                # The launcher never embeds the policy or a research prompt.
                self.assertNotIn("Harness instructions", text)
            finally:
                s.shutdown()

    def test_continue_in_terminal_guards_mirror_native(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            tmp = Path(tmp)
            cli = self._fake_cli(tmp)
            s, root = self._store(tmp, cli)
            try:
                project, _ = s.add_project("octocat/research-demo")
                session = self._selected_session(s, project.id)
                # No provider conversation yet.
                session.provider_session_id = None
                with self.assertRaises(models.ResearchError) as ctx:
                    s.continue_in_terminal(session.id)
                self.assertIn("Start a chat first", str(ctx.exception))
                session.provider_session_id = "THREAD"
                # The selected chat running.
                s._runners[session.id] = object()
                with self.assertRaises(models.ResearchError) as ctx:
                    s.continue_in_terminal(session.id)
                self.assertIn("Stop the running research", str(ctx.exception))
                del s._runners[session.id]
                # A shared-context group with a running agent.
                other = models.ResearchSession(project.id, "Research · Agent 2", "codex")
                other.shared_agent = {"groupID": "G", "index": 2}
                s.state.sessions.append(other)
                s._shared_groups["G"] = {"projectID": project.id, "sessionIDs": [other.id],
                                         "sharesContext": True, "generation": 1,
                                         "agentCount": 2, "directoryName": "demo",
                                         "keepResearching": False}
                s._runners[other.id] = object()
                with self.assertRaises(models.ResearchError) as ctx:
                    s.continue_in_terminal(session.id)
                self.assertIn("Stop the shared-context group", str(ctx.exception))
                del s._runners[other.id]
                # Unknown sessions say the same thing the native app does.
                with self.assertRaises(models.ResearchError) as ctx:
                    s.continue_in_terminal("missing-session")
                self.assertIn("Start a chat first", str(ctx.exception))
            finally:
                s.shutdown()

    def test_resume_arguments_mirror_native_cli(self):
        self.assertEqual(cli_runner.resume_arguments("codex", "T"), ["resume", "T"])
        self.assertEqual(cli_runner.resume_arguments("claude", "T"), ["--resume", "T"])
        self.assertEqual(cli_runner.resume_arguments("opencode", "T"), ["--session", "T"])
        self.assertEqual(cli_runner.resume_arguments("antigravity", "T"),
                         ["--conversation", "T"])

    def test_git_command_keeps_the_repository_slash(self):
        from webapp import workspace
        command, url = workspace.git_command("mw00/project-maya", Path("dest"))
        self.assertIn("https://github.com/mw00/project-maya.git", command)
        self.assertEqual(url, "https://github.com/mw00/project-maya.git")

    def test_repository_parsing_mirrors_native_add(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            tmp = Path(tmp)
            cli = self._fake_cli(tmp)
            s, root = self._store(tmp, cli)
            try:
                self.assertEqual(s._parse_repository("https://github.com/owner/repo.git"),
                                 "owner/repo")
                self.assertEqual(s._parse_repository("github.com/owner/repo"), "owner/repo")
                self.assertEqual(s._parse_repository("owner/repo"), "owner/repo")
                with self.assertRaisesRegex(store.ResearchError, "GitHub URL"):
                    s._parse_repository("../etc/passwd")
                with self.assertRaisesRegex(store.ResearchError, "GitHub URL"):
                    s._parse_repository("owner/repo/extra")
            finally:
                s.shutdown()


class LauncherHelperTests(unittest.TestCase):
    """The Windows desktop launcher's pure helpers, without opening a window."""

    def test_device_defaults_keep_this_install_copy_and_token(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as home:
            home = Path(home)
            (home / "Auto-research-agent" / "Runtime").mkdir(parents=True)
            defaults = launcher.device_defaults(home=home)
            self.assertEqual(defaults["workspaceRoot"], str(home / "Auto-research-agent"))
            self.assertEqual(defaults["token"], launcher.DEVICE_TOKEN)
            self.assertEqual(defaults["port"], 8765)
            self.assertEqual(defaults["access"], "lan")

    def test_device_defaults_generate_a_token_on_a_fresh_machine(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as home:
            first = launcher.device_defaults(home=Path(home))
            second = launcher.device_defaults(home=Path(home))
            self.assertNotEqual(first["token"], launcher.DEVICE_TOKEN)
            self.assertRegex(first["token"], r"[A-Za-z0-9_-]{24}")
            self.assertRegex(second["token"], r"[A-Za-z0-9_-]{24}")

    def test_load_config_seeds_parses_and_repairs(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            path = Path(tmp) / "config.json"
            defaults = {"workspaceRoot": "D:\\research", "port": 8765,
                        "token": "device-token", "access": "lan"}
            self.assertTrue(launcher.seed_config(path, defaults))
            self.assertFalse(launcher.seed_config(path, defaults))
            self.assertEqual(launcher.load_config(path, defaults)["token"], "device-token")
            # Hand-edited values survive; unknown fields fall back per field.
            path.write_text(json.dumps({"workspaceRoot": "E:\\work", "port": 8766,
                                        "access": "local", "token": ""}),
                            encoding="utf-8")
            loaded = launcher.load_config(path, defaults)
            self.assertEqual(loaded["workspaceRoot"], "E:\\work")
            self.assertEqual(loaded["port"], 8766)
            self.assertEqual(loaded["access"], "local")
            self.assertEqual(loaded["token"], "device-token")
            # A corrupt file is re-seeded instead of crashing startup.
            path.write_text("not json", encoding="utf-8")
            repaired = launcher.load_config(path, defaults)
            self.assertEqual(repaired["port"], 8765)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), defaults)

    def test_port_busy_matches_a_real_listener(self):
        listener = socket.socket()
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        port = listener.getsockname()[1]
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        free = probe.getsockname()[1]
        probe.close()
        try:
            self.assertTrue(launcher.port_busy(port))
            self.assertFalse(launcher.port_busy(free))
        finally:
            listener.close()

    @unittest.skipUnless(os.name == "nt", "the single-instance mutex is Windows-only")
    def test_instance_mutex_detects_an_existing_owner(self):
        import uuid
        name = "AutoResearch.test." + str(uuid.uuid4()).replace("-", "")
        try:
            handle, held = launcher.acquire_instance_lock(name)
            self.assertFalse(held)
            again, held_again = launcher.acquire_instance_lock(name)
            self.assertTrue(held_again)
            launcher.release_instance_lock(handle)
            launcher.release_instance_lock(again)
            after, held_after = launcher.acquire_instance_lock(name)
            self.assertFalse(held_after)
            launcher.release_instance_lock(after)
        finally:
            pass

    def test_generate_token_matches_the_server_format(self):
        token = launcher.generate_token()
        self.assertEqual(len(token), len(secrets.token_urlsafe(18)))
        self.assertNotEqual(token, launcher.generate_token())
        self.assertRegex(token, r"[A-Za-z0-9_-]+")

    def test_resources_dir_prefers_the_workspace_checkout(self):
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            # A checkout root with its own Resources: used directly, like the
            # manual server reading that checkout's generated Defaults.json.
            checkout = Path(tmp) / "repo"
            (checkout / "Resources").mkdir(parents=True)
            self.assertEqual(launcher.resources_dir(str(checkout)),
                             checkout / "Resources")
            # A workspace configured as the Runtime directory resolves to
            # the checkout's Resources.
            runtime = checkout / "Runtime"
            runtime.mkdir()
            self.assertEqual(launcher.resources_dir(str(runtime)),
                             checkout / "Resources")
            # A plain workspace with no checkout nearby: the bundled
            # Resources (repository_root()/Resources in unfrozen runs).
            plain = Path(tmp) / "plain"
            plain.mkdir()
            self.assertEqual(launcher.resources_dir(str(plain)),
                             launcher.repository_root() / "Resources")

    def test_window_url_and_workspace_resolution(self):
        config = {"port": 8765, "token": "lTvSAN8IVidADF2yrAXKPKK0"}
        self.assertEqual(launcher.window_url(config),
                         "http://127.0.0.1:8765/?token=lTvSAN8IVidADF2yrAXKPKK0")
        with tempfile.TemporaryDirectory(dir=ROOT / "tmp") as tmp:
            root = Path(tmp) / "ws"
            (root / "Runtime").mkdir(parents=True)
            self.assertEqual(Path(launcher.resolve_workspace(str(root))).name, "Runtime")
            plain = Path(tmp) / "plain"
            plain.mkdir()
            self.assertEqual(launcher.resolve_workspace(str(plain)), str(plain))


if __name__ == "__main__":
    unittest.main()
