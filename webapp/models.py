"""Data model for the web edition, matching ResearchCore's Codable shape.

State files interop with the native macOS app: the same ``Runtime/state.json``
keys, uppercase UUID strings, and dates encoded as seconds since
2001-01-01T00:00:00Z (Swift's ``timeIntervalSinceReferenceDate``). Optional
fields decode as None when absent, as Swift's synthesized ``Codable`` does.
"""
import json
import uuid
from datetime import datetime, timezone

REFERENCE_DATE_UNIX = 978307200.0  # 2001-01-01T00:00:00Z


class ResearchError(Exception):
    """Errors surfaced to the chat, matching the native app's ResearchError."""

    def __init__(self, message: str):
        super().__init__(message)


def now() -> float:
    return datetime.now(timezone.utc).timestamp() - REFERENCE_DATE_UNIX


def new_uuid() -> str:
    return str(uuid.uuid4()).upper()


def encode_date(value: float) -> float:
    return round(value, 9)


def decode_date(value) -> float:
    """Accept Swift reference dates, unix seconds, or ISO-8601 strings."""
    if isinstance(value, (int, float)):
        seconds = float(value)
        if seconds > 1.6e9:  # unix seconds; Swift reference dates are smaller today
            return seconds - REFERENCE_DATE_UNIX
        return seconds
    if isinstance(value, str):
        text = value.strip()
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.timestamp() - REFERENCE_DATE_UNIX
        except ValueError:
            pass
    raise ValueError(f"Unsupported date value: {value!r}")


PROVIDERS = ("codex", "claude", "opencode", "antigravity")
PROVIDER_TITLES = {"codex": "Codex", "claude": "Claude", "opencode": "OpenCode", "antigravity": "Antigravity"}


def project_id(repository: str, name: str | None = None, builtin_id: str | None = None,
               local_path: str | None = None) -> str:
    if builtin_id:
        return "builtin:" + builtin_id
    if local_path:
        return local_path
    return repository.lower()


class ResearchProject:
    def __init__(self, repository: str, name: str | None = None, summary: str = "",
                 category: str = "Research", local_path: str | None = None,
                 is_custom: bool = False, builtin_id: str | None = None):
        self.repository = repository
        self.name = name or (repository.rsplit("/", 1)[-1] if repository else "Local workspace")
        self.summary = summary
        self.category = category
        self.local_path = local_path
        self.is_custom = is_custom
        self.builtin_id = builtin_id
        self.id = project_id(repository, self.name, builtin_id, local_path)

    @property
    def managed_workspace_component(self) -> str:
        if self.builtin_id:
            return "projects/builtins/" + self.builtin_id
        return "projects/" + self.repository

    def to_json(self):
        data = {
            "id": self.id,
            "name": self.name,
            "repository": self.repository,
            "summary": self.summary,
            "category": self.category,
            "isCustom": self.is_custom,
        }
        if self.local_path is not None:
            data["localPath"] = self.local_path
        if self.builtin_id is not None:
            data["builtinID"] = self.builtin_id
        return data

    @classmethod
    def from_json(cls, data):
        return cls(
            repository=data.get("repository", ""),
            name=data.get("name"),
            summary=data.get("summary", ""),
            category=data.get("category", "Research"),
            local_path=data.get("localPath"),
            is_custom=bool(data.get("isCustom", False)),
            builtin_id=data.get("builtinID"),
        )


class ChatMessage:
    ROLES = ("user", "assistant", "tool", "error", "status")

    def __init__(self, role: str, text: str, date: float | None = None,
                 message_id: str | None = None, source_key: str | None = None,
                 speaker: str | None = None):
        self.id = message_id or new_uuid()
        self.role = role
        self.text = text
        self.date = date if date is not None else now()
        self.source_key = source_key
        self.speaker = speaker

    def to_json(self):
        data = {"id": self.id, "role": self.role, "text": self.text,
                "date": encode_date(self.date)}
        if self.source_key is not None:
            data["sourceKey"] = self.source_key
        if self.speaker is not None:
            data["speaker"] = self.speaker
        return data

    @classmethod
    def from_json(cls, data):
        return cls(role=data.get("role", "assistant"), text=data.get("text", ""),
                   date=decode_date(data.get("date")) if data.get("date") is not None else None,
                   message_id=data.get("id"), source_key=data.get("sourceKey"),
                   speaker=data.get("speaker"))


class ResearchSession:
    def __init__(self, project_id: str, title: str, provider: str,
                 session_id: str | None = None):
        self.id = session_id or new_uuid()
        self.project_id = project_id
        self.title = title
        self.provider = provider
        self.provider_session_id: str | None = None
        self.working_directory: str | None = None
        self.messages: list[ChatMessage] = []
        self.created_at = now()
        self.state = "Ready"
        self.scripted: bool | None = None
        self.closed_at: float | None = None

    def to_json(self):
        return {
            "id": self.id,
            "projectID": self.project_id,
            "title": self.title,
            "provider": self.provider,
            "providerSessionID": self.provider_session_id,
            "workingDirectory": self.working_directory,
            "messages": [message.to_json() for message in self.messages],
            "createdAt": encode_date(self.created_at),
            "state": self.state,
            "scripted": self.scripted,
            "closedAt": encode_date(self.closed_at) if self.closed_at is not None else None,
        }

    @classmethod
    def from_json(cls, data):
        session = cls(data.get("projectID", ""), data.get("title", "Chat"),
                      data.get("provider", "codex"), session_id=data.get("id"))
        session.provider_session_id = data.get("providerSessionID")
        session.working_directory = data.get("workingDirectory")
        session.messages = [ChatMessage.from_json(m) for m in data.get("messages", [])]
        session.created_at = decode_date(data.get("createdAt")) if data.get("createdAt") is not None else now()
        session.state = data.get("state", "Ready")
        session.scripted = data.get("scripted")
        session.closed_at = decode_date(data.get("closedAt")) if data.get("closedAt") is not None else None
        return session


SETTINGS_KEYS = (
    "workspaceRoot", "catalogURL", "codexPath", "claudePath", "codexModel",
    "claudeModel", "codexSandbox", "claudePermissionMode", "opencodePath",
    "opencodeModel", "opencodePermission", "antigravityPath",
    "antigravityModel", "antigravityMode", "antigravityPermission",
)

SETTINGS_DEFAULTS = {
    "workspaceRoot": "",
    "catalogURL": "https://app.getsupers.com/auto-research",
    "codexPath": "",
    "claudePath": "",
    "codexModel": "",
    "claudeModel": "",
    "codexSandbox": "workspace-write",
    "claudePermissionMode": "auto",
    "opencodePath": "",
    "opencodeModel": "",
    "opencodePermission": "reject",
    "antigravityPath": "",
    "antigravityModel": "",
    "antigravityMode": "accept-edits",
    "antigravityPermission": "review",
}


class Settings:
    """Saved settings decode with safe defaults when a key is missing, like the
    native app's ``Settings.init(from:)``."""

    def __init__(self, data: dict | None = None):
        merged = dict(SETTINGS_DEFAULTS)
        merged.update({k: v for k, v in (data or {}).items() if k in SETTINGS_KEYS})
        for key in SETTINGS_KEYS:
            setattr(self, key, merged[key])

    def to_json(self):
        return {key: getattr(self, key) for key in SETTINGS_KEYS}


class SavedState:
    def __init__(self):
        self.projects: list[ResearchProject] = []
        self.sessions: list[ResearchSession] = []
        self.selected_project_id: str | None = None
        self.selected_session_id: str | None = None
        self.provider: str = "codex"

    def to_json(self):
        return {
            "projects": [p.to_json() for p in self.projects],
            "sessions": [s.to_json() for s in self.sessions],
            "selectedProjectID": self.selected_project_id,
            "selectedSessionID": self.selected_session_id,
            "provider": self.provider,
        }

    @classmethod
    def from_json(cls, data):
        state = cls()
        state.projects = [ResearchProject.from_json(p) for p in data.get("projects", [])]
        state.sessions = [ResearchSession.from_json(s) for s in data.get("sessions", [])]
        state.selected_project_id = data.get("selectedProjectID")
        state.selected_session_id = data.get("selectedSessionID")
        state.provider = data.get("provider", "codex")
        return state


def dump_swift_json(value) -> str:
    """Pretty-print with sorted keys, like the native app's JSONEncoder."""
    return json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
