"""HTTPS fetch with a Windows schannel fallback.

Python's OpenSSL build cannot construct a certificate chain for hosts that
serve their issuing certificate only through the AIA URL: the handshake then
fails with "certificate has expired" even though the certificate is valid.
curl.exe uses Windows schannel, which follows AIA and succeeds. The helper
tries urllib first and falls back to the system curl on Windows; the caller
keeps the original error so the failure stays explainable.
"""
from __future__ import annotations

import os
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

from . import spawn as spawn_module

CURL = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "curl.exe"


class TransportError(Exception):
    pass


def curl_available() -> bool:
    return os.name == "nt" and CURL.is_file()


def fetch_bytes(url: str, timeout: float = 20.0, accept: str = "application/json") -> bytes:
    """Fetch over HTTPS with urllib, falling back to curl.exe (schannel)."""
    try:
        request = urllib.request.Request(url, headers={"Accept": accept,
                                                       "User-Agent": "AutoResearchWeb/1.0"})
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise TransportError(f"The server answered {response.status}.")
            return response.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
            OSError, ValueError) as error:
        if not curl_available():
            raise TransportError(str(error)) from None
        # SSL failures keep a chain usable through schannel; other errors
        # (404, redirects to dead hosts) usually repeat under curl too, but a
        # retry costs little and keeps one code path.
        try:
            return _fetch_via_curl(url, timeout)
        except TransportError:
            raise TransportError(str(error)) from None


def _fetch_via_curl(url: str, timeout: float = 20.0) -> bytes:
    if not url.lower().startswith("https://"):
        raise TransportError("Only HTTPS addresses are supported.")
    result = subprocess.run(
        [str(CURL), "-sS", "-L", "--max-time", str(int(timeout) + 5),
         "--fail-with-body", "-H", "Accept: application/json",
         "-A", "AutoResearchWeb/1.0", url],
        capture_output=True, timeout=timeout + 10,
        creationflags=spawn_module.NO_WINDOW)
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise TransportError(detail or f"curl exited {result.returncode}.")
    return result.stdout


def fetch_text(url: str, timeout: float = 20.0, accept: str = "application/json") -> str:
    return fetch_bytes(url, timeout, accept).decode("utf-8")


def request_bytes(url: str, timeout: float = 20.0, headers: dict | None = None,
                  method: str = "GET", body: str | None = None) -> tuple[int, bytes]:
    """General HTTPS request returning (status, body), urllib then curl.exe.

    Unlike fetch_bytes, non-200 answers are returned, not raised: callers
    (the cloud management API) distinguish 401 from 404 and read error bodies.
    HTTPError answers keep their status so failed logins stay explainable.
    """
    merged = {"Accept": "application/json", "User-Agent": "AutoResearchWeb/1.0"}
    if headers:
        merged.update(headers)
    data = None if body is None else body.encode("utf-8")
    try:
        request = urllib.request.Request(url, headers=merged, method=method, data=data)
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return int(response.status), response.read()
        except urllib.error.HTTPError as error:
            try:
                return int(error.code), error.read()
            except OSError:
                return int(error.code), b""
    except (urllib.error.URLError, TimeoutError, OSError, ValueError) as error:
        if not curl_available():
            raise TransportError(str(error)) from None
        try:
            return _request_via_curl(url, timeout, merged, method, body)
        except TransportError:
            raise TransportError(str(error)) from None


def _request_via_curl(url: str, timeout: float, headers: dict, method: str,
                      body: str | None) -> tuple[int, bytes]:
    if not url.lower().startswith(("https://", "http://")):
        raise TransportError("Only HTTPS addresses are supported.")
    args = [str(CURL), "-sS", "-L", "--max-time", str(int(timeout) + 5),
            "-X", method, "--write-out", "\n%{http_code}"]
    for key, value in headers.items():
        args += ["-H", f"{key}: {value}"]
    if body is not None:
        args += ["-d", body]
    args.append(url)
    result = subprocess.run(args, capture_output=True, timeout=timeout + 10,
                            creationflags=spawn_module.NO_WINDOW)
    payload = result.stdout
    if result.returncode != 0 and not payload:
        detail = result.stderr.decode("utf-8", "replace").strip()
        raise TransportError(detail or f"curl exited {result.returncode}.")
    status = 0
    if payload.endswith(b"\n") or b"\n" in payload:
        text = payload.decode("utf-8", "replace")
        head, _, tail = text.rpartition("\n")
        if tail.strip().isdigit():
            status = int(tail.strip())
            payload = head.encode("utf-8", "replace") if head else b""
    if not status:
        status = result.returncode if result.returncode != 0 else 200
    return status, payload


def save_snapshot(path: Path, data: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(data, encoding="utf-8")


def read_snapshot(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8")
    except (FileNotFoundError, OSError):
        return None
