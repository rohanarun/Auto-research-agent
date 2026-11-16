"""Provider CLI invocation and event decoding, mirroring ResearchCore's
CLI.swift and EventDecoder.

The command builders produce the exact argument vectors the native app uses,
so a chat started here can be continued in the native app and in Terminal,
and the other way around.
"""
import json
import os
import subprocess
import threading
import time

from .models import ResearchError

PROVIDER_TITLES = {"codex": "Codex", "claude": "Claude", "opencode": "OpenCode",
                   "antigravity": "Antigravity"}


class Command:
    def __init__(self, executable: str, arguments: list[str], input: str | None = None):
        self.executable = executable
        self.arguments = arguments
        self.input = input


def chat_command(provider: str, settings, session_id: str | None, prompt: str,
                 policy: str, disabled_mcp_servers: list[str] | None = None,
                 enabled_mcp_servers: list[str] | None = None) -> Command:
    """Mirror CLI.chat(provider:settings:sessionID:prompt:policy:...)."""
    disabled_mcp_servers = disabled_mcp_servers or []
    enabled_mcp_servers = enabled_mcp_servers or []
    if provider == "codex":
        args = ["exec", "--disable", "plugins",
                "-c", 'approval_policy="never"',
                "-c", f'sandbox_mode="{settings.codexSandbox}"']
        for name in disabled_mcp_servers:
            args += ["-c", f"mcp_servers.{name}.enabled=false"]
        for name in enabled_mcp_servers:
            args += ["-c", f"mcp_servers.{name}.enabled=true"]
        if session_id:
            args += ["resume", session_id]
        args += ["--json", "--skip-git-repo-check"]
        if settings.codexModel:
            args += ["--model", settings.codexModel]
        args += ["-"]
        return Command(settings.codexPath, args,
                       input=f"Harness instructions:\n{policy}\n\nUser request:\n{prompt}")
    if provider == "antigravity":
        request = f"Harness instructions:\n{policy}\n\nUser request:\n{prompt}"
        args = ["--output-format", "stream-json", "--disable-slash-commands"]
        if settings.antigravityPermission == "auto":
            args.append("--dangerously-skip-permissions")
        if settings.antigravityMode:
            args += ["--mode", settings.antigravityMode]
        if session_id:
            args += ["--conversation", session_id]
        if settings.antigravityModel:
            args += ["--model", settings.antigravityModel]
        args.append("--print=" + request)
        return Command(settings.antigravityPath, args)
    if provider == "opencode":
        args = ["run", "--format", "json"]
        if settings.opencodePermission == "auto":
            args.append("--auto")
        if session_id:
            args += ["--session", session_id]
        if settings.opencodeModel:
            args += ["--model", settings.opencodeModel]
        return Command(settings.opencodePath, args,
                       input=f"Harness instructions:\n{policy}\n\nUser request:\n{prompt}")
    args = ["--print", "--verbose", "--output-format", "stream-json",
            "--permission-mode", settings.claudePermissionMode,
            "--append-system-prompt", policy]
    if session_id:
        args += ["--resume", session_id]
    if settings.claudeModel:
        args += ["--model", settings.claudeModel]
    return Command(settings.claudePath, args, input=prompt)


def resume_arguments(provider: str, session_id: str) -> list[str]:
    """CLI.resumeArguments: the arguments that reopen a provider conversation
    interactively in a terminal."""
    if provider == "codex":
        return ["resume", session_id]
    if provider == "antigravity":
        return ["--conversation", session_id]
    if provider == "opencode":
        return ["--session", session_id]
    return ["--resume", session_id]


def mcp_discovery(executable: str) -> Command:
    return Command(executable, ["--disable", "plugins", "mcp", "list", "--json"])


def configured_mcp_servers(data: bytes) -> list[str]:
    try:
        servers = json.loads(data.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        raise ResearchError("Codex returned an unreadable MCP configuration. No research process was started.")
    if not isinstance(servers, list):
        raise ResearchError("Codex returned an unreadable MCP configuration. No research process was started.")
    names = sorted(str(s.get("name", "")) for s in servers if isinstance(s, dict))
    if any("." in name for name in names):
        raise ResearchError("A Codex MCP server name contains a dot, which this CLI cannot address with a "
                            "per-run override. Rename that server in Codex before starting research.")
    return names


def auth_command(provider: str, settings, login: bool) -> Command:
    if provider == "codex":
        return Command(settings.codexPath, ["login"] if login else ["login", "status"])
    if provider == "claude":
        return Command(settings.claudePath, ["auth", "login" if login else "status"])
    if provider == "opencode":
        return Command(settings.opencodePath, ["auth", "login" if login else "list"])
    return Command(settings.antigravityPath, ["models"])


def opencode_signed_in(output: str) -> bool:
    text = output.strip()
    if not text or "No authenticated integrations" in text:
        return False
    import re
    if re.search(r"(?m)\b0 credentials\b", text) and "Environment" not in text:
        return False
    return True


def resume_arguments(provider: str, session_id: str) -> list[str]:
    return {"codex": ["resume", session_id], "claude": ["--resume", session_id],
            "opencode": ["--session", session_id],
            "antigravity": ["--conversation", session_id]}[provider]


def shared_context_ready(text: str) -> bool:
    lines = text.split("\n")
    return lines[-1].strip() == "SHARED_CONTEXT_STATUS: ready"


def continuation_decision(text: str) -> str | None:
    for line in reversed(text.split("\n")):
        trimmed = line.strip()
        if trimmed.startswith("RESEARCH_STATUS: "):
            decision = trimmed[len("RESEARCH_STATUS: "):]
            if decision in ("continue", "needs_input", "finished"):
                return decision
    return None


def decode(line: str, provider: str) -> list[dict]:
    """Decode one JSON line into CLI event dicts, mirroring EventDecoder.decode.

    Event dicts carry one of: {"session": id}, {"completed": True},
    {"failure": text}, or {"message": (key, role, text)}.
    """
    try:
        object = json.loads(line)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return []
    if not isinstance(object, dict):
        return []
    kind = object.get("type") or object.get("event") or ""
    events: list[dict] = []
    if provider == "codex":
        if kind == "thread.started":
            thread = object.get("thread_id")
            if isinstance(thread, str):
                events.append({"session": thread})
        if kind == "turn.completed":
            events.append({"completed": True})
        if kind == "error":
            error = object.get("error") or {}
            text = error.get("message") or object.get("message") or "Codex diagnostic"
            events.append({"message": ("provider-diagnostic", "tool", text)})
        if kind == "turn.failed":
            error = object.get("error") or {}
            events.append({"failure": error.get("message") or object.get("message")
                           or "Codex reported a failed turn."})
        item = object.get("item")
        if not isinstance(item, dict):
            return events
        key = str(item.get("id") or _uuid())
        kind = item.get("type") or ""
        if kind == "agent_message":
            text = item.get("text")
            if isinstance(text, str):
                events.append({"message": (key, "assistant", text)})
        elif kind == "command_execution":
            command = item.get("command") or "Command"
            output = item.get("aggregated_output") or ""
            status = item.get("status") or "running"
            events.append({"message": (key, "tool",
                                       f"{command}\n{status}" + (f"\n{output}" if output else ""))})
        elif kind in ("file_change", "mcp_tool_call", "web_search"):
            payload = json.dumps(item, indent=2, sort_keys=True)
            events.append({"message": (key, "tool", payload)})
        return events
    if provider == "antigravity":
        return _decode_antigravity(object, kind)
    if provider == "opencode":
        return _decode_opencode(object, kind)
    # Claude
    if isinstance(object.get("session_id"), str):
        events.append({"session": object["session_id"]})
    if kind == "assistant" and isinstance(object.get("message"), dict):
        message = object["message"]
        identifier = message.get("id") or _uuid()
        content = message.get("content") or []
        for index, block in enumerate(content if isinstance(content, list) else []):
            if isinstance(block, dict):
                if isinstance(block.get("text"), str):
                    events.append({"message": (f"{identifier}-{index}", "assistant", block["text"])})
                if block.get("type") == "tool_use":
                    payload = json.dumps(block.get("input") or {}, indent=2, sort_keys=True)
                    events.append({"message": (block.get("id") or f"{identifier}-{index}", "tool",
                                               f"{block.get('name') or 'Tool'}\n{payload}")})
    if kind == "user" and isinstance(object.get("message"), dict):
        content = object["message"].get("content") or []
        for block in content if isinstance(content, list) else []:
            if isinstance(block, dict) and block.get("type") == "tool_result":
                text = block.get("content") if isinstance(block.get("content"), str) \
                    else json.dumps(block.get("content"))
                events.append({"message": (f"result-{block.get('tool_use_id') or _uuid()}",
                                           "tool", text or "")})
    if kind == "result":
        if object.get("is_error") is True:
            errors = object.get("errors")
            text = "\n".join(errors) if isinstance(errors, list) and errors \
                else object.get("result") or "Claude reported a failed turn."
            events.append({"failure": text})
        else:
            events.append({"completed": True})
        denied = object.get("permission_denials")
        if isinstance(denied, list) and denied:
            events.append({"failure": f"Claude denied {len(denied)} tool request(s). Review permission mode "
                                      "in Connections or continue this session in Terminal."})
    return events


def _decode_antigravity(object, kind: str) -> list[dict]:
    events: list[dict] = []
    if kind == "init":
        identifier = object.get("conversation_id")
        if isinstance(identifier, str) and identifier:
            events.append({"session": identifier})
        return events
    if kind == "step_update":
        step = object.get("step_update")
        if not isinstance(step, dict):
            return events
        identifier = step.get("conversation_id")
        if isinstance(identifier, str) and identifier:
            events.append({"session": identifier})
        if step.get("step_type") != "tool":
            return events
        index = step.get("step_index") or 0
        info = step.get("tool_info") or {}
        text = step.get("tool_name") or info.get("name") or "Tool"
        parameters = info.get("parameters")
        if parameters is not None:
            text += "\n" + json.dumps(parameters, indent=2, sort_keys=True)
        output = info.get("output")
        if isinstance(output, str) and output:
            text += "\n" + output
        events.append({"message": (f"antigravity-tool-{index}", "tool", text)})
        return events
    if kind == "result":
        result = object.get("result")
        if not isinstance(result, dict):
            return events
        conversation = result.get("conversation_id")
        if conversation:
            events.append({"session": conversation})
        status = result.get("status") or ""
        if status != "SUCCESS":
            reason = (result.get("error") or result.get("response") or "").strip()
            events.append({"failure": reason or "Antigravity reported a failed turn."})
            return events
        response = (result.get("response") or "").strip()
        if response:
            turns = result.get("num_turns") or 0
            events.append({"message": (f"antigravity-result-{conversation}-{turns}",
                                       "assistant", response)})
        events.append({"completed": True})
    return events


def _decode_opencode(object, kind: str) -> list[dict]:
    events: list[dict] = []
    identifier = object.get("sessionID")
    if isinstance(identifier, str) and identifier:
        events.append({"session": identifier})
    part = object.get("part") if isinstance(object.get("part"), dict) else {}
    part_id = part.get("id") or part.get("partID") or _uuid()
    if kind == "text":
        text = part.get("text")
        if isinstance(text, str) and text.strip():
            events.append({"message": (part_id, "assistant", text)})
    elif kind == "tool_use":
        state = part.get("state") if isinstance(part.get("state"), dict) else {}
        status = state.get("status") or "running"
        payload = json.dumps(state.get("input") or {}, indent=2, sort_keys=True)
        output = state.get("output") or state.get("error") or ""
        if isinstance(state.get("error"), dict):
            output = state["error"].get("message") or ""
        events.append({"message": (f"tool-{part_id}", "tool",
                                   f"{part.get('tool') or 'Tool'}\n{status}\n{payload}"
                                   + (f"\n{output}" if output else ""))})
    elif kind == "step_finish":
        reason = part.get("reason") or ""
        if reason not in ("tool-calls", "error"):
            events.append({"completed": True})
    elif kind == "error":
        error = object.get("error")
        if isinstance(error, dict):
            message = error.get("message") or (error.get("data") or {}).get("message") \
                or error.get("name") or "OpenCode reported a failed turn."
        else:
            message = "OpenCode reported a failed turn."
        events.append({"failure": message})
    return events


def _uuid() -> str:
    import uuid
    return str(uuid.uuid4()).upper()


class ProcessResult:
    def __init__(self, code: int, stdout: str, stderr: str):
        self.code = code
        self.stdout = stdout
        self.stderr = stderr


def platform_launch(command: Command, directory: str, environment: dict) -> subprocess.Popen:
    """Launch with a platform-appropriate shell wrapper for .cmd shims."""
    executable = command.executable
    args = [executable] + list(command.arguments)
    creationflags = 0
    if os.name == "nt":
        if executable.lower().endswith((".cmd", ".bat")):
            args = ["cmd", "/c"] + args
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP
    return subprocess.Popen(
        args, cwd=directory, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, env=environment, creationflags=creationflags,
        text=True, encoding="utf-8", errors="replace")


class ProcessRunner:
    """Runs provider CLIs, streams lines, and stops this run's process tree.

    Stop semantics: on POSIX the whole process group is signalled; on Windows
    ``taskkill /T`` walks the child tree. Only this run's descendants are
    touched, never other CLI sessions on the machine.
    """

    def __init__(self):
        self._process: subprocess.Popen | None = None
        self._lock = threading.Lock()
        self._cancelled = False

    def stop(self):
        with self._lock:
            self._cancelled = True
            process = self._process
        if process is None or process.poll() is not None:
            return
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(process.pid)],
                               capture_output=True, timeout=10)
            else:
                import signal
                try:
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                except (ProcessLookupError, PermissionError, OSError):
                    process.terminate()
        except (OSError, subprocess.SubprocessError):
            try:
                process.terminate()
            except OSError:
                pass

    def send_input(self, text: str):
        with self._lock:
            process = self._process
        if process is not None and process.stdin is not None:
            try:
                process.stdin.write(text + "\n")
                process.stdin.flush()
            except (OSError, ValueError):
                pass

    def run(self, command: Command, directory: str, environment: dict | None = None,
            interactive: bool = False, on_line=None) -> ProcessResult:
        environment = environment or os.environ
        with self._lock:
            if self._cancelled:
                raise ResearchError("The run was cancelled before it started.")
        try:
            process = platform_launch(command, directory, environment)
        except FileNotFoundError as error:
            raise ResearchError(f"The CLI could not be started: {error}") from None
        with self._lock:
            self._process = process
        stdout_lines: list[str] = []
        stderr_lines: list[str] = []
        stdout_lock = threading.Lock()

        def pump(stream, sink: list[str], is_error: bool):
            try:
                for line in iter(stream.readline, ""):
                    line = line.rstrip("\n")
                    with stdout_lock:
                        sink.append(line)
                    if on_line:
                        on_line(line, is_error)
            except (OSError, ValueError):
                pass

        out_thread = threading.Thread(target=pump, args=(process.stdout, stdout_lines, False), daemon=True)
        err_thread = threading.Thread(target=pump, args=(process.stderr, stderr_lines, True), daemon=True)
        out_thread.start()
        err_thread.start()
        if command.input:
            try:
                process.stdin.write(command.input)
                process.stdin.flush()
            except (OSError, ValueError):
                pass
        if not interactive and process.stdin is not None:
            try:
                process.stdin.close()
            except (OSError, ValueError):
                pass
        while True:
            try:
                process.wait(timeout=0.5)
                break
            except subprocess.TimeoutExpired:
                pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        out_thread.join(timeout=10)
        err_thread.join(timeout=10)
        for stream in (process.stdin, process.stdout, process.stderr):
            try:
                if stream is not None:
                    stream.close()
            except (OSError, ValueError):
                pass
        code = process.returncode if process.returncode is not None else -1
        return ProcessResult(code, "\n".join(stdout_lines), "\n".join(stderr_lines))
