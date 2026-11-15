"""The research store: the web edition's counterpart of the native app's
AppStore. It manages state.json, settings, the catalog, and one worker thread
per running chat, with the same lifecycle states and event handling.
"""
import copy
import json
import os
import threading
import time
import uuid
from pathlib import Path

from . import builtin as builtin_module
from . import catalog as catalog_module
from . import cli_runner
from . import dashboard as dashboard_module
from . import workspace as workspace_module
from .models import (ChatMessage, ResearchError, ResearchProject,
                     ResearchSession, SavedState, Settings, dump_swift_json, now)


def _write_atomic(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    try:
        path.chmod(0o600)
    except OSError:
        pass


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


class Store:
    def __init__(self, resources_dir: Path):
        self.resources_dir = resources_dir
        self._lock = threading.RLock()
        self._condition = threading.Condition(threading.Lock())
        self.version = 1

        self.workspace_root = ""
        self.settings = Settings()
        self.state = SavedState()
        self.builtins: list[builtin_module.BuiltinProject] = []
        self.catalog_status = "Loading catalog"
        self._policy = ""
        self._catalog_policy = catalog_module.CatalogPolicy([])
        self._runners: dict[str, cli_runner.ProcessRunner] = {}
        self._threads: dict[str, threading.Thread] = {}
        self._stopped: set[str] = set()
        self._completed: set[str] = set()
        self._failed: set[str] = set()
        self._event_messages: dict[str, dict[str, str]] = {}
        self._active_paths: dict[str, str] = {}
        self._keep_researching: set[str] = set()
        self._enabled_mcp_servers: list[str] = []
        self._available_mcp_servers: list[str] = []
        self.auth: dict[str, str] = {}
        self._dashboard_cache: dict[str, tuple[float, dict]] = {}
        self._port = 8765
        self._access = "local"
        self.started_at = time.time()

    # ---------- lifecycle ----------

    def load_workspace(self, workspace_root: str):
        with self._lock:
            self.workspace_root = str(Path(workspace_root).resolve())
            root = Path(self.workspace_root)
            for component in ("projects", "sessions", "cache", "tmp"):
                (root / component).mkdir(parents=True, exist_ok=True)
            self.settings = Settings(_read_json(root / "settings.json")
                                     or _read_json(self.resources_dir / "Defaults.json")
                                     or _read_json(self.resources_dir / "Defaults.example.json"))
            if not self.settings.workspaceRoot:
                self.settings.workspaceRoot = self.workspace_root
            self.state = SavedState.from_json(_read_json(root / "state.json") or {})
            self.builtins = builtin_module.load(self.resources_dir)
            self._catalog_policy = catalog_module.load_policy(self.resources_dir)
            policy_file = root / "research-policy.md"
            if policy_file.is_file():
                self._policy = policy_file.read_text(encoding="utf-8")
            else:
                resource = self.resources_dir / "ResearchPolicy.md"
                self._policy = resource.read_text(encoding="utf-8") if resource.is_file() else ""
            if not self.state.projects:
                self.state.projects = [b.project for b in self.builtins]
                self.state.selected_project_id = (self.state.projects[0].id
                                                  if self.state.projects else None)
            self.version += 1

    def persist(self):
        with self._lock:
            _write_atomic(Path(self.workspace_root) / "state.json",
                          dump_swift_json(self.state.to_json()))
        self.bump()

    def bump(self):
        with self._condition:
            self.version += 1
            self._condition.notify_all()

    def wait_for_version(self, since: int, timeout: float) -> int:
        with self._condition:
            if self.version > since:
                return self.version
            self._condition.wait(timeout)
            return self.version

    def shutdown(self):
        with self._lock:
            ids = list(self._runners.keys())
        for session_id in ids:
            self.stop(session_id)

    # ---------- helpers ----------

    def _session(self, session_id: str) -> ResearchSession | None:
        for session in self.state.sessions:
            if session.id == session_id:
                return session
        return None

    def _project(self, project_id: str) -> ResearchProject | None:
        for project in self.state.projects:
            if project.id == project_id:
                return project
        for candidate in (b.project for b in self.builtins):
            if candidate.id == project_id:
                return candidate
        return None

    def append(self, session_id: str, role: str, text: str):
        with self._lock:
            session = self._session(session_id)
            if session is None:
                return
            session.messages.append(ChatMessage(role=role, text=text))
        self.persist()

    def set_state(self, session_id: str, value: str):
        with self._lock:
            session = self._session(session_id)
            if session:
                session.state = value
        self.persist()

    def running(self, project_id: str) -> bool:
        with self._lock:
            return any(session.project_id == project_id and session.id in self._runners
                       for session in self.state.sessions)

    def environment(self) -> dict:
        with self._lock:
            settings = copy.copy(self.settings)
        root = Path(self.workspace_root)
        cache = root / "cache"
        tmp = root / "tmp"
        env = dict(os.environ)
        # Avoid inheriting nested-session guards from a parent coding agent.
        env.pop("CLAUDECODE", None)
        paths = [str(Path(getattr(settings, key, "")).parent)
                 for key in ("codexPath", "claudePath", "opencodePath", "antigravityPath")
                 if getattr(settings, key, "")]
        paths += ["/opt/homebrew/bin", "/usr/local/bin",
                  os.environ.get("PATH", os.environ.get("Path", ""))]
        env["PATH"] = (os.pathsep if os.name == "nt" else ":").join(paths)
        env["TMPDIR"] = str(tmp) + os.sep
        env["XDG_CACHE_HOME"] = str(cache)
        env["PIP_CACHE_DIR"] = str(cache / "pip")
        env["npm_config_cache"] = str(cache / "npm")
        env["CLANG_MODULE_CACHE_PATH"] = str(cache / "clang")
        env["SWIFT_MODULECACHE_PATH"] = str(cache / "swift")
        return env

    def _full_policy(self, checkout: str) -> str:
        with self._lock:
            policy = self._policy
        return (policy + "\n\nConfigured workspace: " + self.workspace_root
                + ". Current checkout: " + checkout + ".")

    # ---------- catalog ----------

    def refresh_catalog(self, fetched: list[ResearchProject] | None = None) -> dict:
        with self._lock:
            settings = copy.copy(self.settings)
            existing = list(self.state.projects)
        if fetched is None:
            try:
                fetched = catalog_module.fetch(settings.catalogURL)
            except (ResearchError, OSError, ValueError) as error:
                self.catalog_status = "Offline · saved catalog"
                self.bump()
                return {"status": self.catalog_status, "error": str(error)}
        merged = catalog_module.merge(fetched, existing,
                                      [b.project for b in self.builtins],
                                      {s.project_id for s in self.state.sessions})
        self.catalog_status = f"{len(fetched)} repositories · {len(self.builtins)} built-in projects"
        with self._lock:
            self.state.projects = merged
            if not self.state.selected_project_id and merged:
                self.state.selected_project_id = merged[0].id
        self.persist()
        return {"status": self.catalog_status, "projects": len(merged)}

    def add_project(self, repository: str = "", local_path: str = "") -> ResearchProject:
        if local_path:
            if not Path(local_path).exists():
                raise ResearchError("The folder does not exist.")
            project = ResearchProject(repository="", local_path=str(Path(local_path)),
                                      is_custom=True)
        else:
            repository = repository.strip()
            if not repository or "/" not in repository:
                raise ResearchError("Enter a public GitHub repository as owner/name.")
            project = ResearchProject(repository=repository, is_custom=True)
        with self._lock:
            if not any(p.id == project.id for p in self.state.projects):
                self.state.projects.append(project)
            self.state.selected_project_id = project.id
        self.persist()
        return project

    # ---------- chats ----------

    def select(self, project_id: str | None = None, session_id: str | None = None):
        with self._lock:
            if session_id:
                session = self._session(session_id)
                if session:
                    session.closed_at = None
                    self.state.selected_project_id = session.project_id
                    self.state.selected_session_id = session.id
                    self.state.provider = session.provider
            elif project_id:
                self.state.selected_project_id = project_id
                self.state.selected_session_id = None
            else:
                self.state.selected_session_id = None
        self.persist()

    def create_chat(self, project_id: str, provider: str | None = None) -> ResearchSession:
        project = self._project(project_id)
        if project is None:
            raise ResearchError("Choose a research project before opening a chat.")
        with self._lock:
            self.state.provider = provider or self.state.provider
            session = ResearchSession(project_id=project.id,
                                      title="Research · " + project.name,
                                      provider=self.state.provider)
            self.state.sessions.append(session)
            self.state.selected_project_id = project.id
            self.state.selected_session_id = session.id
        self.persist()
        return session

    def play(self, project_id: str, prompt: str | None = None,
             provider: str | None = None) -> str:
        """Mirror AppStore.play for the local, single-agent path."""
        project = self._project(project_id)
        if project is None:
            raise ResearchError("Choose a research project before starting research.")
        with self._lock:
            if provider and provider != self.state.provider:
                self.state.provider = provider
            active = [s for s in self.state.sessions
                      if s.project_id == project.id and s.id in self._runners]
        if active:
            with self._lock:
                self.state.selected_session_id = active[0].id
            self.persist()
            return active[0].id
        with self._lock:
            session = ResearchSession(project_id=project.id,
                                      title="Research · " + project.name,
                                      provider=self.state.provider)
            self.state.sessions.append(session)
            self.state.selected_project_id = project.id
            self.state.selected_session_id = session.id
        self._launch(session.id, project, prompt)
        return session.id

    def send(self, session_id: str, text: str) -> str:
        text = (text or "").strip()
        if not text:
            raise ResearchError("Type a follow-up before sending.")
        with self._lock:
            session = self._session(session_id)
            if session is None:
                raise ResearchError("This chat is not open.")
            project = self._project(session.project_id)
            if session.id in self._runners:
                raise ResearchError("Stop research before sending a manual follow-up.")
            session.closed_at = None
            self.state.selected_session_id = session.id
        self._launch(session_id, project, text)
        return session_id

    def stop(self, session_id: str):
        with self._lock:
            runner = self._runners.get(session_id)
            if runner is None:
                return
            self._stopped.add(session_id)
            session = self._session(session_id)
            if session:
                session.state = "Stopping"
        self.bump()
        runner.stop()

    def close(self, session_id: str):
        with self._lock:
            session = self._session(session_id)
            if session is None or session.id in self._runners:
                raise ResearchError("Stop research before closing this chat.")
            session.closed_at = now()
            if self.state.selected_session_id == session_id:
                self.state.selected_session_id = None
        self.persist()

    def reopen(self, session_id: str):
        with self._lock:
            session = self._session(session_id)
            if session:
                session.closed_at = None
                self.state.selected_project_id = session.project_id
                self.state.selected_session_id = session.id
        self.persist()

    def logs(self, session_id: str) -> str:
        directory = Path(self.workspace_root) / "sessions" / session_id
        chunks: list[str] = []
        if directory.is_dir():
            for file in sorted(directory.glob("*.jsonl")):
                try:
                    chunks.append(file.read_text(encoding="utf-8"))
                except OSError:
                    pass
        return "\n".join(chunks)

    # ---------- the run ----------

    def _launch(self, session_id: str, project: ResearchProject, request: str | None):
        """Mirror AppStore.launch for the local single-agent path. The web
        edition's first milestone covers the core workflows: managed clones,
        built-in kernels, provider sessions, follow-ups and stopping."""
        with self._lock:
            if session_id in self._runners:
                return
            session = self._session(session_id)
            if session is None:
                return
            session.closed_at = None
            target_path = (session.working_directory or project.local_path
                           or str(Path(self.workspace_root) / project.managed_workspace_component))
            target = str(Path(os.path.abspath(target_path)))
            if target in self._active_paths.values():
                self.append(session_id, "error",
                            "Another chat is already using this checkout. Stop that run before "
                            "starting another writer.")
                session.state = "Needs attention"
                return
            self._active_paths[session_id] = target
            runner = cli_runner.ProcessRunner()
            self._runners[session_id] = runner
            self._stopped.discard(session_id)
            self._failed.discard(session_id)
            self._completed.discard(session_id)
            self._event_messages[session_id] = {}
            session.state = "Preparing"
        self.bump()
        thread = threading.Thread(target=self._run_worker,
                                  args=(session_id, project, request),
                                  name=f"research-{session_id}", daemon=True)
        self._threads[session_id] = thread
        thread.start()

    def _builtin(self, builtin_id: str) -> builtin_module.BuiltinProject:
        for candidate in self.builtins:
            if candidate.id == builtin_id:
                return candidate
        raise ResearchError("This built-in project's resources are unavailable.")

    def _run_worker(self, session_id: str, project: ResearchProject, request: str | None):
        next_decision: str | None = None
        try:
            env = self.environment()
            with self._lock:
                current = self._session(session_id)
                settings = copy.copy(self.settings)
            if current is None:
                return
            executable = {"codex": settings.codexPath, "claude": settings.claudePath,
                          "opencode": settings.opencodePath,
                          "antigravity": settings.antigravityPath}[current.provider]
            if not workspace_module.executable_check(executable):
                raise ResearchError(
                    cli_runner.PROVIDER_TITLES[current.provider]
                    + " CLI is not installed at the configured path. Open Connections to locate it "
                      "or view installation instructions.")
            if project.builtin_id:
                directory = Path(self._builtin(project.builtin_id).prepare(self.workspace_root))
                workspace_module.validate(str(directory))
            elif current.working_directory or project.local_path:
                directory = workspace_module.validate(current.working_directory or project.local_path)
            else:
                parent = workspace_module.directory(self.workspace_root, "projects")
                directory = parent / project.repository
                if not directory.exists():
                    workspace_module.directory(self.workspace_root,
                                               "projects/" + project.repository.split("/", 1)[0])
                    self.append(session_id, "status",
                                "Cloning " + project.repository + " into the workspace…")
                    command, _url = workspace_module.git_command(project.repository, directory)
                    code, output, _ = workspace_module.clone(
                        command, parent, env,
                        on_line=lambda line: self.append(session_id, "tool", line))
                    if code != 0:
                        raise ResearchError("Clone failed. " + output.strip())
                directory = workspace_module.validate(str(directory))
                if not (directory / ".git").exists():
                    raise ResearchError("An incomplete checkout already exists at " + str(directory)
                                        + ". Inspect it before retrying; it has been preserved.")
            with self._lock:
                if session_id in self._stopped:
                    raise ResearchError("Cancelled")
                current.working_directory = str(directory)
            self.bump()
            prompt = request or ""
            if project.builtin_id:
                instruction = workspace_module.instruction_file(directory, [])
                if instruction is None:
                    raise ResearchError("The starter's auto-research.md is missing or points outside "
                                        "its workspace. Restore the guide before continuing.")
                body = instruction.read_text(encoding="utf-8")
                if not body.strip():
                    raise ResearchError("auto-research.md is empty. Add research instructions before starting.")
                prompt = ((request or "Start this research project using its starter guide.") +
                          f"\n\nProject: {project.name} ({project.builtin_id}). Work in {directory}." +
                          f" Read applicable AGENTS.md and the editable guide at {instruction}.\n\n{body}")
                self.append(session_id, "status",
                            "Loaded editable starter guide · local agent with official remote judge")
            elif request is None:
                tracked = workspace_module.tracked_files(directory, env)
                instruction = workspace_module.instruction_file(directory, tracked)
                if instruction is not None:
                    body = instruction.read_text(encoding="utf-8")
                    if not body.strip():
                        raise ResearchError(str(instruction.name)
                                            + " is empty. Add research instructions before starting.")
                    prompt = (f"Start an autonomous research run in {directory}. Read applicable AGENTS.md "
                              f"and follow this repository instruction file: {instruction}.\n\n{body}")
                    self.append(session_id, "status", "Loaded " + str(instruction))
                else:
                    prompt = (f"Start a research run in {directory}. The harness did not find a tracked "
                              "auto-research.md file. Inspect the repository and its AGENTS.md, locate its "
                              "actual research instructions (including README-linked guides or program.md), "
                              "explain the source you choose, and carry out the documented research workflow. "
                              "If there is no actionable research objective, ask me instead of inventing "
                              "instructions.")
                    self.append(session_id, "status",
                                "auto-research.md was not found. The agent will inspect the repository "
                                "for its research guide.")
            with self._lock:
                if session_id in self._stopped:
                    raise ResearchError("Cancelled")
            self.append(session_id, "user", prompt)
            log_dir = workspace_module.directory(self.workspace_root, "sessions/" + session_id)
            log_file = log_dir / (str(uuid.uuid4()).upper() + ".jsonl")
            log_file.touch()
            log_lock = threading.Lock()

            def log_line(line: str):
                with log_lock:
                    try:
                        with log_file.open("a", encoding="utf-8") as handle:
                            handle.write(line + "\n")
                    except OSError:
                        pass

            full_policy = self._full_policy(str(directory))
            disabled_mcp_servers: list[str] = []
            if current.provider == "codex":
                discovered = cli_runner.ProcessRunner().run(
                    cli_runner.mcp_discovery(settings.codexPath), str(directory), env)
                if discovered.code != 0:
                    raise ResearchError("Could not inspect Codex MCP settings. No research process "
                                        "was started. " + discovered.stderr)
                names = cli_runner.configured_mcp_servers(discovered.stdout.encode("utf-8"))
                with self._lock:
                    self._available_mcp_servers = names
                    disabled_mcp_servers = [n for n in names
                                            if n not in self._enabled_mcp_servers]
            command = cli_runner.chat_command(current.provider, settings,
                                              current.provider_session_id, prompt, full_policy,
                                              disabled_mcp_servers=disabled_mcp_servers,
                                              enabled_mcp_servers=[n for n in self._available_mcp_servers
                                                                   if n in self._enabled_mcp_servers])
            with self._lock:
                runner = self._runners.get(session_id)
            if runner is None:
                raise ResearchError("Cancelled")
            self.set_state(session_id, "Running")

            def on_line(line: str, is_error: bool):
                log_line(line)
                self._consume(line, is_error, current.provider, session_id)

            result = runner.run(command, str(directory), env, on_line=on_line)
            for line in result.stdout.split("\n"):
                if not line:
                    continue
                for event in cli_runner.decode(line, current.provider):
                    if event.get("completed"):
                        self._completed.add(session_id)
                    elif "failure" in event:
                        self._failed.add(session_id)
                    elif "session" in event:
                        with self._lock:
                            session = self._session(session_id)
                            if session:
                                session.provider_session_id = event["session"]
                    elif "message" in event and event["message"][1] == "assistant":
                        next_decision = cli_runner.continuation_decision(event["message"][2])
            if session_id in self._stopped:
                self.set_state(session_id, "Stopped")
            elif result.code != 0 or session_id in self._failed:
                self.set_state(session_id, "Needs attention")
                if result.code != 0:
                    self.append(session_id, "error",
                                f"{cli_runner.PROVIDER_TITLES[current.provider]} exited with status "
                                f"{result.code}. {result.stderr}")
            elif session_id not in self._completed:
                self.set_state(session_id, "Needs attention")
                self.append(session_id, "error",
                            "The CLI exited without a completed-turn event. Review the run log "
                            "before resuming.")
            elif next_decision == "needs_input":
                self.set_state(session_id, "Needs attention")
            else:
                self.set_state(session_id, "Completed")
        except Exception as error:  # one run failing must never kill the server
            with self._lock:
                stopped = session_id in self._stopped
            if stopped:
                self.set_state(session_id, "Stopped")
            else:
                self.set_state(session_id, "Needs attention")
                self.append(session_id, "error", str(error))
        finally:
            with self._lock:
                self._runners.pop(session_id, None)
                self._active_paths.pop(session_id, None)
                keep = session_id in self._keep_researching
                session = self._session(session_id)
                completed = bool(session and session.state == "Completed")
            self.bump()
            self.persist()
            if keep:
                self._finish_keep_researching(session_id, project, next_decision, completed)

    def _finish_keep_researching(self, session_id: str, project: ResearchProject,
                                next_decision: str | None, completed: bool):
        if not completed or next_decision != "continue":
            with self._lock:
                self._keep_researching.discard(session_id)
            self.append(session_id, "status",
                        "Continuous research paused. The agent did not request another experiment, "
                        "or this run needs attention.")
            return
        resource = self.resources_dir / "ContinuePolicy.md"
        next_prompt = resource.read_text(encoding="utf-8") if resource.is_file() else ""
        if next_prompt.strip():
            self._launch(session_id, project, next_prompt)
        else:
            self.append(session_id, "status",
                        "Continuous research paused. The agent did not request another experiment, "
                        "or this run needs attention.")

    def _consume(self, line: str, stderr: bool, provider: str, session_id: str):
        if not line:
            return
        with self._lock:
            session = self._session(session_id)
            if session is None:
                return
            if stderr:
                mapping = self._event_messages.setdefault(session_id, {})
                existing_id = mapping.get("cli-diagnostics")
                if existing_id:
                    for index, message in enumerate(session.messages):
                        if message.id == existing_id:
                            session.messages[index].text += "\n" + line
                            self.persist()
                            return
                message = ChatMessage(role="tool", text="CLI diagnostics\n" + line)
                session.messages.append(message)
                mapping["cli-diagnostics"] = message.id
                self.persist()
                return
            for event in cli_runner.decode(line, provider):
                if "session" in event:
                    session.provider_session_id = event["session"]
                if event.get("completed"):
                    self._completed.add(session_id)
                    self._failed.discard(session_id)
                if "failure" in event:
                    self._failed.add(session_id)
                    self._completed.discard(session_id)
                    self._append_locked(session_id, "error", event["failure"])
                elif "message" in event:
                    key, role, text = event["message"]
                    self._message_locked(session_id, key, role, text)
            self.persist()

    def _append_locked(self, session_id: str, role: str, text: str):
        session = self._session(session_id)
        if session:
            session.messages.append(ChatMessage(role=role, text=text))

    def _message_locked(self, session_id: str, key: str, role: str, text: str):
        session = self._session(session_id)
        if session is None:
            return
        mapping = self._event_messages.setdefault(session_id, {})
        existing_id = mapping.get(key)
        if existing_id:
            for index, message in enumerate(session.messages):
                if message.id == existing_id:
                    session.messages[index].text = text
                    return
        message = ChatMessage(role=role, text=text)
        session.messages.append(message)
        mapping[key] = message.id

    # ---------- auth, settings ----------

    def refresh_auth(self):
        for provider in ("codex", "claude", "opencode", "antigravity"):
            with self._lock:
                settings = copy.copy(self.settings)
            executable = {"codex": settings.codexPath, "claude": settings.claudePath,
                          "opencode": settings.opencodePath,
                          "antigravity": settings.antigravityPath}[provider]
            if not workspace_module.executable_check(executable):
                self.auth[provider] = "Not installed"
                continue
            try:
                auth_dir = workspace_module.directory(self.workspace_root, "auth")
                result = cli_runner.ProcessRunner().run(
                    cli_runner.auth_command(provider, settings, login=False),
                    str(auth_dir), self.environment())
                if provider == "claude":
                    try:
                        payload = json.loads(result.stdout or "{}")
                        self.auth[provider] = ("Connected" if payload.get("loggedIn") is True
                                               else "Sign in required")
                    except json.JSONDecodeError:
                        self.auth[provider] = "Sign in required"
                elif provider == "opencode":
                    self.auth[provider] = ("Connected" if result.code == 0
                                           and cli_runner.opencode_signed_in(result.stdout)
                                           else "Sign in required")
                else:
                    self.auth[provider] = ("Connected" if result.code == 0
                                           else "Sign in required")
            except Exception:
                self.auth[provider] = "Unavailable"
        self.bump()

    def save_settings(self, updates: dict):
        with self._lock:
            if self._runners:
                raise ResearchError("Stop active runs before changing connection settings.")
            for key, value in (updates or {}).items():
                if hasattr(self.settings, key):
                    setattr(self.settings, key, value)
            _write_atomic(Path(self.workspace_root) / "settings.json",
                          dump_swift_json(self.settings.to_json()))
        policy_file = Path(self.workspace_root) / "research-policy.md"
        policy_file.write_text(self._policy, encoding="utf-8")
        try:
            policy_file.chmod(0o600)
        except OSError:
            pass
        self.bump()
        threading.Thread(target=self.refresh_auth, daemon=True).start()

    def login(self, provider: str) -> str:
        with self._lock:
            settings = copy.copy(self.settings)
        command = cli_runner.auth_command(provider, settings, login=True)
        if not workspace_module.executable_check(command.executable):
            raise ResearchError(
                cli_runner.PROVIDER_TITLES[provider]
                + " CLI is not installed at the configured path. Open Connections to locate it "
                  "or view installation instructions.")
        auth_dir = workspace_module.directory(self.workspace_root, "auth")
        try:
            result = cli_runner.ProcessRunner().run(command, str(auth_dir),
                                                    self.environment(), interactive=True)
            return (f"{cli_runner.PROVIDER_TITLES[provider]} login finished with status "
                    f"{result.code}. " + result.stderr)
        finally:
            threading.Thread(target=self.refresh_auth, daemon=True).start()

    # ---------- dashboard ----------

    def dashboard(self, project_id: str) -> dict | None:
        project = self._project(project_id)
        if project is None:
            raise ResearchError("Select a project first.")
        if project.builtin_id:
            raise ResearchError("Built-in kernel projects are judged remotely. The chart belongs to "
                                "GitHub catalog projects; see the challenge website instead.")
        with self._lock:
            settings = copy.copy(self.settings)
            cached = self._dashboard_cache.get(project.repository)
        if cached and time.time() - cached[0] < max(cached[1].get("refreshAfter") or 60, 30):
            return cached[1]
        try:
            feeds = dashboard_module.fetch(settings.catalogURL)
            view = dashboard_module.project_view(feeds["history"], feeds["research"],
                                                 project.repository)
        except (dashboard_module.DashboardError, OSError, ValueError, KeyError) as error:
            raise ResearchError(f"The research backend is unavailable: {error}")
        if view is None:
            return None
        with self._lock:
            self._dashboard_cache[project.repository] = (time.time(), view)
        return view

    # ---------- snapshot ----------

    def snapshot(self) -> dict:
        with self._lock:
            state = self.state.to_json()
            websites = {b.id: (b.website, b.leaderboard) for b in self.builtins}
            for project in state["projects"]:
                if project.get("builtinID") and project["builtinID"] in websites:
                    # Extra keys are ignored by the native app's decoder and let
                    # the UI link the challenge website.
                    project["website"] = websites[project["builtinID"]][0]
                    project["leaderboard"] = websites[project["builtinID"]][1]
            return {
                "version": self.version,
                "projects": state["projects"],
                "sessions": state["sessions"],
                "selectedProjectID": state["selectedProjectID"],
                "selectedSessionID": state["selectedSessionID"],
                "provider": state["provider"],
                "settings": self.settings.to_json(),
                "auth": dict(self.auth),
                "runningSessions": [session["id"] for session in state["sessions"]
                                    if session["id"] in self._runners],
                "server": {
                    "name": "Auto Research",
                    "workspaceRoot": self.workspace_root,
                    "catalogStatus": self.catalog_status,
                    "port": self._port,
                    "access": self._access,
                    "platform": os.name,
                    "editors": ("The native macOS app shares this workspace; "
                                "chats interchange between editions."),
                },
            }
