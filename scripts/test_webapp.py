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
import urllib.request
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
(ROOT / "tmp").mkdir(exist_ok=True)

from webapp import catalog, cli_runner, dashboard, models, server, store, workspace  # noqa: E402

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


if __name__ == "__main__":
    unittest.main()
