"""Daytona and Agent37 cloud settings for the web edition.

Mirrors the native app's Cloud tab (ConnectionsView.swift + Daytona.swift +
Agent37.swift): the provider API key is stored on the device — workspace
storage here, macOS Keychain in the native app — and sent only to the
provider's management API. Connect validates the key and lists the sandboxes
or instances this installation owns; stopping owned compute stops them. This
edition never creates billable resources and never runs research in a sandbox;
the sandbox bridge lifecycle stays native-app only.
"""
from __future__ import annotations

import json
import os
import re
import threading
import uuid
from pathlib import Path

from . import transport
from .models import ResearchError

PROVIDERS = ("daytona", "agent37")
TITLES = {"daytona": "Daytona", "agent37": "Agent37"}
API_BASES = {"daytona": "https://app.daytona.io/api",
             "agent37": "https://api.agent37.com/v1"}
DASHBOARDS = {"daytona": "https://app.daytona.io/dashboard/sandboxes",
              "agent37": "https://www.agent37.com/dashboard/cloud"}
INSTALL_LABEL = "autoresearch-install"
INSTANCE_ID = re.compile(r"^[a-z0-9]{10}$")
MAX_PAGES = 5

STATUS_KEY_SAVED = "API key saved in your workspace. Connect to prepare cloud Codex."
STATUS_CONNECTING = "Preparing private cloud connection…"
STATUS_CONNECTED = "{title} connected · cloud Codex sign-in required"
STATUS_KEY_REMOVED = "Key removed. Existing sandboxes are preserved."
STATUS_STOPPED = "Owned compute stopped. Files and sign-in are retained."


class CloudError(ResearchError):
    """A readable, user-facing cloud settings failure."""


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None


def _write_atomic(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)
    try:
        path.chmod(0o600)
    except OSError:
        pass


def default_request(method: str, url: str, headers: dict, body: str | None,
                    timeout: float = 30.0) -> tuple[int, bytes]:
    status, payload = transport.request_bytes(
        url, headers=headers, method=method, body=body, timeout=timeout)
    return status, payload


class CloudStore:
    """Persisted provider choice, keys, and owned-compute bookkeeping."""

    def __init__(self, workspace_root: str, request=None):
        self._lock = threading.RLock()
        self._file = Path(workspace_root) / "cloud-settings.json"
        self._request = request or default_request
        data = _read_json(self._file) or {}
        provider = data.get("provider") or "daytona"
        self.provider = provider if provider in PROVIDERS else "daytona"
        self.install_id = data.get("installID") or str(uuid.uuid4())
        self.keys = {k: v for k, v in (data.get("keys") or {}).items()
                     if k in PROVIDERS and isinstance(v, str) and v}
        self.connected = {k: bool(v) for k, v in (data.get("connected") or {}).items()}
        self.status = {k: v for k, v in (data.get("status") or {}).items()
                       if k in PROVIDERS and isinstance(v, str)}
        target = data.get("runTarget") or "mac"
        self.run_target = target if target in ("mac", "cloud") else "mac"
        self.owned: dict[str, list[dict]] = {}
        self.busy = ""
        self._persist()

    # ---------- persistence ----------

    def _persist(self):
        with self._lock:
            _write_atomic(self._file, json.dumps({
                "provider": self.provider,
                "installID": self.install_id,
                "keys": self.keys,
                "connected": self.connected,
                "status": self.status,
                "runTarget": self.run_target,
            }, indent=2, sort_keys=True) + "\n", )

    # ---------- views ----------

    def view(self) -> dict:
        """Everything the dashboard may see — never the stored keys."""
        with self._lock:
            provider = self.provider
            return {
                "provider": provider,
                "titles": TITLES,
                "runTarget": self.run_target,
                "keySaved": {p: bool(self.keys.get(p)) for p in PROVIDERS},
                "connected": {p: bool(self.connected.get(p)) for p in PROVIDERS},
                "status": {p: (self.status.get(p) or "Not connected") for p in PROVIDERS},
                "owned": {p: list(self.owned.get(p) or []) for p in PROVIDERS},
                "dashboardURL": DASHBOARDS[provider],
                "deviceLoginURL": "https://auth.openai.com/codex/device",
                "busy": self.busy,
            }

    # ---------- settings updates ----------

    def set_provider(self, provider: str) -> dict:
        if provider not in PROVIDERS:
            raise CloudError("Choose Daytona or Agent37 as the sandbox provider.")
        with self._lock:
            self.provider = provider
        self._persist()
        return self.view()

    def set_run_target(self, target: str) -> dict:
        if target not in ("mac", "cloud"):
            raise CloudError("Choose where research runs.")
        with self._lock:
            self.run_target = target
        self._persist()
        return self.view()

    def save_key(self, provider: str, key: str) -> dict:
        if provider not in PROVIDERS:
            raise CloudError("Choose Daytona or Agent37 as the sandbox provider.")
        trimmed = (key or "").strip()
        if not trimmed:
            raise CloudError("Enter a " + TITLES[provider] + " API key.")
        with self._lock:
            self.keys[provider] = trimmed
            self.status[provider] = STATUS_KEY_SAVED
        self._persist()
        return self.view()

    def remove_key(self, provider: str) -> dict:
        if provider not in PROVIDERS:
            raise CloudError("Choose Daytona or Agent37 as the sandbox provider.")
        with self._lock:
            self.keys.pop(provider, None)
            self.connected[provider] = False
            self.status[provider] = STATUS_KEY_REMOVED
            self.owned.pop(provider, None)
        self._persist()
        return self.view()

    # ---------- management API ----------

    def _headers(self, provider: str, key: str) -> dict:
        return {"Authorization": "Bearer " + key,
                "Content-Type": "application/json"}

    def _api_call(self, method: str, provider: str, path: str,
                  body: str | None = None) -> tuple[int, bytes]:
        key = self.keys.get(provider)
        if not key:
            raise CloudError("Save a " + TITLES[provider] + " API key first.")
        url = API_BASES[provider] + path
        try:
            status, payload = self._request(method, url, self._headers(provider, key), body)
        except Exception as error:
            raise CloudError(TITLES[provider] + " could not be reached: " + str(error)) from None
        if status in (401, 403):
            raise CloudError(TITLES[provider] + " rejected the saved API key (" + str(status)
                             + "). Save a valid key and connect again.")
        if status >= 400:
            raise CloudError(TITLES[provider] + " answered " + str(status) + ".")
        return status, payload

    def _parse(self, payload: bytes, provider: str):
        try:
            return json.loads(payload.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise CloudError(TITLES[provider] + " answered with unreadable data.") from None

    def _list_owned(self, provider: str) -> list[dict]:
        """List the sandboxes/instances labeled with this installation's ID."""
        owned: list[dict] = []
        if provider == "daytona":
            cursor = ""
            for _ in range(MAX_PAGES):
                _, payload = self._api_call("GET", provider,
                                            "/sandbox?limit=100" + (cursor and ("&cursor=" + cursor) or ""))
                page = self._parse(payload, provider)
                for box in page.get("items") or []:
                    labels = box.get("labels") or {}
                    if labels.get(INSTALL_LABEL) == self.install_id:
                        owned.append({"id": box.get("id") or "", "name": box.get("name") or "",
                                      "state": box.get("state") or ""})
                cursor = page.get("nextCursor") or ""
                if not cursor:
                    break
        else:
            _, payload = self._api_call("GET", provider, "/instances")
            page = self._parse(payload, provider)
            for instance in page.get("data") or []:
                metadata = instance.get("metadata") or {}
                identifier = instance.get("id") or ""
                if (metadata.get(INSTALL_LABEL) == self.install_id
                        and INSTANCE_ID.match(identifier)):
                    owned.append({"id": identifier, "name": instance.get("name") or "",
                                  "state": instance.get("status") or ""})
        return owned

    def connect(self, provider: str) -> dict:
        """Validate the key against the management API and list owned compute.

        Native parity: the same list call that backs "Connect"; the sandbox
        bridge lifecycle (research in sandboxes) stays native-app only.
        """
        if provider not in PROVIDERS:
            raise CloudError("Choose Daytona or Agent37 as the sandbox provider.")
        with self._lock:
            self.busy = "connecting"
            self.status[provider] = STATUS_CONNECTING
        try:
            owned = self._list_owned(provider)
        except CloudError:
            with self._lock:
                self.busy = ""
            self._persist()
            raise
        with self._lock:
            self.busy = ""
            self.connected[provider] = True
            self.status[provider] = STATUS_CONNECTED.format(title=TITLES[provider])
            self.owned[provider] = owned
        self._persist()
        return self.view()

    def stop_owned(self, provider: str) -> dict:
        """Stop the compute this installation owns; never touch another's."""
        if provider not in PROVIDERS:
            raise CloudError("Choose Daytona or Agent37 as the sandbox provider.")
        with self._lock:
            self.busy = "stopping"
        try:
            owned = self._list_owned(provider)
            stopped = 0
            for item in owned:
                path = ("/sandbox/" + item["id"] + "/stop" if provider == "daytona"
                        else "/instances/" + item["id"] + "/stop")
                try:
                    self._api_call("POST", provider, path, "{}")
                    stopped += 1
                except CloudError as error:
                    if "404" not in str(error):
                        raise
        finally:
            with self._lock:
                self.busy = ""
        with self._lock:
            self.connected[provider] = True
            self.status[provider] = STATUS_STOPPED
            self.owned[provider] = owned
        self._persist()
        view = self.view()
        view["stopped"] = stopped
        return view
