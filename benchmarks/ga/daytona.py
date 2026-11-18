"""Daytona H200 adapter for the portable GA.

This adapter wraps the repository's existing trusted GPU harness:

* ``benchmarks/gpu/dispatcher.py`` creates one short-lived H200 lease;
* ``benchmarks/gpu/session.py`` uploads one candidate per job, runs the judge,
  exports the result and keeps the lease until the queue is closed;
* ``benchmarks/gpu/judge.py`` performs correctness checks and paired CUDA-graph
  timing against the public baseline.

The GA treats each public benchmark case as a cell and each candidate source
file as a gene.  A candidate file is the same ``kernel_fn`` module that the
existing ``gpu_test.py`` / ``okc`` workflow uses.  The adapter never reads
credentials itself: ``DAYTONA_API_KEY`` is read only by
``benchmarks/gpu/daytona_control.py`` when a provider request is made.

No API key belongs in this file, an adapter file, a command line argument, a
bundle or a checked-in config.  Put it in the environment for the process that
runs the GA, and keep the local result directory private.  The adapter only
sends a candidate to the sandbox; the sandbox has networking blocked.

Example adapter file (keep it outside the repository if it contains private
paths)::

    from benchmarks.ga.daytona import DaytonaAdapter

    def make_adapter():
        return DaytonaAdapter(
            problem="muse_bf16_proj",
            candidates=["./submission.py"],
            cases=[0, 1, 2],
            rounds=4,
            seconds=900,
        )

Then::

    DAYTONA_API_KEY=... python3 benchmarks/tools/run_ga.py run \\
      --adapter ./my_daytona_adapter.py:make_adapter \\
      --state tmp/ga-daytona/state.json \\
      --per-round 4 --mutation-blocks 4 --confirm-blocks 6

Every measured gene/case pair is one H200 job.  Start with one or two cases and
a small number of candidates.  A new lease is deleted in ``close()``.  To avoid
rebuilding an image that cannot be snapshotted, pass ``sandbox_id`` to restart
an existing box (stopped in ``close()``) or ``snapshot`` to create from a GPU
snapshot (deleted in ``close()``); ``close_action="leave"`` opts out:

    DaytonaAdapter(..., sandbox_id="<sandbox-id>")
    DaytonaAdapter(..., snapshot="<snapshot-uuid>")
"""

from __future__ import annotations

import ast
import datetime
import importlib.util
import json
import math
import os
import re
import secrets
import statistics
import sys
import tempfile
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set

from .engine import BuildResult, Cell, ConfirmResult, GAState, Gene

_GPU_DIR = Path(__file__).resolve().parents[1] / "gpu"
_BENCHMARKS_DIR = Path(__file__).resolve().parents[1]
_REPO_ROOT = _BENCHMARKS_DIR.parent
_MAX_CANDIDATE_BYTES = 2_000_000


@dataclass
class _Eval:
    candidate: List[float]
    baseline: List[float]
    job: str
    status: str


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value)).strip("._-")
    return cleaned[:80] or "gene"


def _parse_cases(raw: Any) -> List[int]:
    if raw is None or raw == "":
        return [0]
    if isinstance(raw, int):
        return [int(raw)]
    if isinstance(raw, str):
        return [int(part) for part in raw.replace(";", ",").split(",") if part.strip()]
    return [int(part) for part in raw]


def _load_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load %s from %s" % (name, path))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class _HarnessTransport:
    """Lazy access to the existing provider transport in ``benchmarks/gpu``."""

    def __init__(self, session: Optional[Any] = None) -> None:
        self._session = session

    def _load(self) -> Any:
        if self._session is None:
            for root in (str(_REPO_ROOT), str(_BENCHMARKS_DIR), str(_GPU_DIR)):
                if root not in sys.path:
                    sys.path.insert(0, root)
            harness = _load_module("ga_daytona_dispatcher", _GPU_DIR / "dispatcher.py")
            self._session = getattr(harness, "gpu", None)
            if self._session is None:
                self._session = _load_module("ga_daytona_session", _GPU_DIR / "session.py")
        return self._session

    def request(
        self,
        plane: str,
        path: str,
        method: str = "GET",
        body: Optional[Any] = None,
        timeout: int = 45,
    ) -> Dict[str, Any]:
        control = getattr(self._load(), "dc", None)
        if control is None:
            raise RuntimeError("GPU transport does not expose daytona_control")
        return control.request(plane, path, method, body, timeout)

    def call(self, sid: str, command: str, timeout: int = 45) -> Dict[str, Any]:
        return self._load().call(sid, command, timeout)


class DaytonaLease:
    """Attach to an existing sandbox, or create one from a GPU snapshot.

    The default :class:`benchmarks.gpu.dispatcher.Lease` always builds the
    pinned H200 image, which fails for images with more than 69 layers.  This
    lease is the escape hatch for the two supported reuse paths:

    * ``sandbox_id`` attaches to an existing box and starts it when stopped;
    * ``snapshot`` creates a box from an existing GPU snapshot UUID or name.

    A newly created snapshot box is deleted on ``close()``.  An attached box is
    stopped on ``close()`` so it does not keep consuming GPU time; pass
    ``close_action="leave"`` to keep it running.  No API key is read here; the
    provider transport reads it only when a request is made.
    """

    _BAD_STATES = {"error", "destroyed", "build_failed"}

    def __init__(
        self,
        sandbox_id: Optional[str] = None,
        snapshot: Optional[str] = None,
        *,
        close_action: Optional[str] = None,
        expected_gpu: Optional[str] = None,
        require_network_block: bool = True,
        readiness_seconds: int = 600,
        poll_seconds: float = 3.0,
        transport: Optional[Any] = None,
        record_dir: Optional[Path] = None,
    ) -> None:
        if bool(sandbox_id) == bool(snapshot):
            raise ValueError("pass exactly one of sandbox_id or snapshot")
        action = (close_action or ("delete" if snapshot else "stop")).lower()
        if action not in {"delete", "stop", "leave"}:
            raise ValueError("close_action must be delete, stop or leave")
        self.sandbox_id = str(sandbox_id) if sandbox_id else None
        self.snapshot = str(snapshot) if snapshot else None
        self.close_action = action
        self.expected_gpu = str(expected_gpu) if expected_gpu else None
        self.require_network_block = bool(require_network_block)
        self.readiness_seconds = max(1, int(readiness_seconds))
        self.poll_seconds = max(0.1, float(poll_seconds))
        self._transport = transport or _HarnessTransport()
        self._record_dir = Path(record_dir) if record_dir else None
        self._created = False
        self._started_here = False
        self._snapshot_record: Optional[Dict[str, Any]] = None
        self.sid: Optional[str] = None
        self.info: Dict[str, Any] = {}

    # -- lifecycle -------------------------------------------------------

    def start(self) -> None:
        if self.sid:
            return
        try:
            if self.sandbox_id:
                self._attach()
            else:
                self._create_from_snapshot()
            self._verify_runtime()
        except BaseException:
            try:
                self.close()
            except Exception:
                pass
            raise

    def close(self) -> None:
        sid = self.sid
        if not sid:
            return
        try:
            if self.close_action == "delete":
                self._delete(sid)
            elif self.close_action == "stop":
                self._stop(sid)
            else:
                self._record("closed", {"sandbox_id": sid, "action": "leave"})
        finally:
            self.sid = None

    # -- attach / create -------------------------------------------------

    def _attach(self) -> None:
        state = self._sandbox(self.sandbox_id or "")
        sid = str(state.get("id") or self.sandbox_id)
        self.sid = sid
        self._record("attached.json", {"sandbox_id": sid, "state": state.get("state")})
        if state.get("state") != "started":
            self._post("/sandbox/%s/start" % sid, None, "start existing sandbox")
            self._started_here = True
            state = self._wait_started(sid)
        self._remember_state(state)

    def _create_from_snapshot(self) -> None:
        snapshot = self._resolve_snapshot(self.snapshot or "")
        if not snapshot.get("id"):
            raise RuntimeError("snapshot %r has no id" % self.snapshot)
        self._snapshot_record = snapshot
        state = str(snapshot.get("state") or "")
        if state != "active":
            if state in {"error", "build failed", "removing"}:
                raise RuntimeError("snapshot %r is %s: %s" % (
                    self.snapshot, state, snapshot.get("errorReason") or "no reason"))
            self._post("/snapshots/%s/activate" % snapshot["id"], None, "activate snapshot")
            snapshot = self._wait_snapshot(snapshot["id"])
        gpu_type = self.expected_gpu or snapshot.get("gpuType")
        name = "ga-bench-" + datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(3)
        body = {
            "name": name,
            "snapshot": str(snapshot["id"]),
            "public": False,
            "networkBlockAll": True,
            "autoStopInterval": 5,
            "autoPauseInterval": 0,
            "autoDeleteInterval": 0,
            "labels": {"owner": "auto-research-benchmark", "purpose": "portable-ga"},
        }
        created = self._post("/sandbox", body, "create sandbox from snapshot")
        sid = str(created.get("id") or "")
        if not sid:
            sid = self._find_sandbox(name)
        if not sid:
            raise RuntimeError("sandbox creation from snapshot was not confirmed")
        self.sid = sid
        self._created = True
        self._record("created.json", {"sandbox_id": sid, "snapshot": snapshot["id"], "name": name})
        state = self._wait_started(sid)
        self._remember_state(state, gpu_type=gpu_type)

    def _remember_state(self, state: Dict[str, Any], *, gpu_type: Optional[str] = None) -> None:
        self.info = dict(state)
        self.info["expected_gpu"] = self.expected_gpu or gpu_type or state.get("gpuType")
        self._record("status.json", state)

    def _verify_runtime(self) -> None:
        state = self.info
        if self.require_network_block and state.get("networkBlockAll") is not True:
            raise RuntimeError(
                "sandbox %s does not have networkBlockAll=true; refusing to use it for a "
                "kernel job (set require_network_block=False only for an isolated test box)"
                % self.sid
            )
        smoke = self._call(
            "/sbin/ldconfig && nvidia-smi --query-gpu=name,memory.total --format=csv,noheader",
            30,
        )
        if int(smoke.get("exitCode", 0) or 0) != 0:
            raise RuntimeError("GPU smoke test failed: %s" % str(smoke)[:800])
        gpu_name = str(smoke.get("result") or "").strip()
        expected = str(self.info.get("expected_gpu") or "")
        if expected:
            compact = lambda value: re.sub(r"[^A-Z0-9]+", "", str(value).upper())
            if compact(expected) not in compact(gpu_name):
                raise RuntimeError(
                    "sandbox %s reports GPU %r, expected %r" % (self.sid, gpu_name, expected)
                )
        probe = self._call(
            "python3 -c 'import json,os,sys; print(json.dumps({\"uid\":os.geteuid(),"
            "\"python\":sys.executable,\"torch\":__import__(\"torch\").__version__}))'",
            60,
        )
        if int(probe.get("exitCode", 0) or 0) != 0:
            raise RuntimeError("sandbox Python/PyTorch probe failed: %s" % str(probe)[:800])
        try:
            runtime = json.loads(probe.get("result") or "{}")
        except json.JSONDecodeError as exc:
            raise RuntimeError("sandbox Python probe returned invalid JSON: %s" % exc) from exc
        if int(runtime.get("uid", -1)) != 0:
            raise RuntimeError(
                "sandbox %s runs as uid %s; the bundled run_job requires root. "
                "Use a root snapshot or extend session.run_job for a non-root user."
                % (self.sid, runtime.get("uid"))
            )
        self.info.update(
            gpu_name=gpu_name,
            python=runtime.get("python"),
            torch=runtime.get("torch"),
            uid=runtime.get("uid"),
        )
        self._record("runtime.json", self.info)

    # -- provider calls --------------------------------------------------

    def _request(self, path: str, method: str = "GET", body: Optional[Any] = None) -> Dict[str, Any]:
        response = self._transport.request("control", path, method, body, 45)
        if not response.get("ok"):
            raise RuntimeError(
                "Daytona %s %s failed: HTTP %s %s"
                % (method, path, response.get("http_status", "?"), str(response.get("error") or "")[:800])
            )
        data = response.get("data")
        return data if isinstance(data, dict) else {"value": data}

    def _post(self, path: str, body: Optional[Any], action: str) -> Dict[str, Any]:
        try:
            return self._request(path, "POST", body)
        except RuntimeError as exc:
            raise RuntimeError("%s failed: %s" % (action, exc)) from exc

    def _call(self, command: str, timeout: int) -> Dict[str, Any]:
        result = self._transport.call(self.sid or "", command, timeout)
        if not isinstance(result, dict):
            raise RuntimeError("sandbox command returned a non-object result")
        return result

    def _sandbox(self, sid: str) -> Dict[str, Any]:
        return self._request("/sandbox/%s" % sid)

    def _find_sandbox(self, name: str) -> Optional[str]:
        try:
            listing = self._request("/sandbox")
        except RuntimeError:
            return None
        items = listing.get("items") if isinstance(listing.get("items"), list) else listing.get("value")
        if not isinstance(items, list):
            return None
        for item in items:
            if isinstance(item, dict) and item.get("name") == name:
                return str(item.get("id") or "") or None
        return None

    def _resolve_snapshot(self, ref: str) -> Dict[str, Any]:
        if re.fullmatch(r"[0-9a-fA-F-]{16,}", ref):
            try:
                record = self._request("/snapshots/%s" % ref)
                if record.get("id"):
                    return record
            except RuntimeError:
                pass
        for page in range(1, 11):
            listing = self._request("/snapshots?name=%s&limit=100&page=%d" % (
                urllib.parse.quote(ref, safe=""), page))
            items = listing.get("items")
            if not isinstance(items, list):
                items = listing.get("value") if isinstance(listing.get("value"), list) else []
            for item in items:
                if isinstance(item, dict) and item.get("name") == ref:
                    return item
            if len(items) < 100:
                break
        raise RuntimeError("snapshot %r was not found; use the snapshot UUID or exact name" % ref)

    def _wait_snapshot(self, snapshot_id: str) -> Dict[str, Any]:
        deadline = time.monotonic() + self.readiness_seconds
        while True:
            record = self._request("/snapshots/%s" % snapshot_id)
            state = str(record.get("state") or "")
            self._record("snapshot-status.json", record)
            if state == "active":
                return record
            if state in {"error", "build failed", "removing"}:
                raise RuntimeError("snapshot %s is %s: %s" % (
                    snapshot_id, state, record.get("errorReason") or "no reason"))
            if time.monotonic() > deadline:
                raise RuntimeError("snapshot %s did not become active in %ss (state %r)" % (
                    snapshot_id, self.readiness_seconds, state))
            time.sleep(self.poll_seconds)

    def _wait_started(self, sid: str) -> Dict[str, Any]:
        deadline = time.monotonic() + self.readiness_seconds
        while True:
            state = self._sandbox(sid)
            current = str(state.get("state") or "")
            self._record("status.json", state)
            if current == "started":
                return state
            if current in self._BAD_STATES:
                raise RuntimeError("sandbox %s failed to start: %s (%s)" % (
                    sid, current, state.get("errorReason") or "no reason"))
            if time.monotonic() > deadline:
                raise RuntimeError("sandbox %s did not start in %ss (state %r)" % (
                    sid, self.readiness_seconds, current))
            time.sleep(self.poll_seconds)

    def _stop(self, sid: str) -> None:
        response = self._transport.request("control", "/sandbox/%s/stop" % sid, "POST", None, 45)
        if not response.get("ok"):
            if response.get("http_status") == 404:
                self._record("closed.json", {"sandbox_id": sid, "action": "stop", "state": "gone"})
                return
            raise RuntimeError("stopping sandbox %s failed: HTTP %s %s" % (
                sid, response.get("http_status", "?"), str(response.get("error") or "")[:800]))
        deadline = time.monotonic() + self.readiness_seconds
        while True:
            try:
                state = self._sandbox(sid)
            except RuntimeError:
                self._record("closed.json", {"sandbox_id": sid, "action": "stop", "state": "gone"})
                return
            current = str(state.get("state") or "")
            self._record("stopped.json", state)
            if current in {"stopped", "archived", "destroyed"}:
                self._record("closed.json", {"sandbox_id": sid, "action": "stop", "state": current})
                return
            if current in {"error", "build_failed"}:
                raise RuntimeError("sandbox %s entered %s while stopping" % (sid, current))
            if time.monotonic() > deadline:
                raise RuntimeError("sandbox %s did not stop in %ss (state %r)" % (
                    sid, self.readiness_seconds, current))
            time.sleep(self.poll_seconds)

    def _delete(self, sid: str) -> None:
        response = self._transport.request("control", "/sandbox/%s" % sid, "DELETE", None, 45)
        if not response.get("ok"):
            if response.get("http_status") == 404:
                self._record("closed.json", {"sandbox_id": sid, "action": "delete", "state": "gone"})
                return
            raise RuntimeError("deleting sandbox %s failed: HTTP %s %s" % (
                sid, response.get("http_status", "?"), str(response.get("error") or "")[:800]))
        deadline = time.monotonic() + self.readiness_seconds
        while True:
            try:
                state = self._sandbox(sid)
            except RuntimeError:
                self._record("closed.json", {"sandbox_id": sid, "action": "delete", "state": "gone"})
                return
            current = str(state.get("state") or "")
            if current == "destroyed":
                self._record("closed.json", {"sandbox_id": sid, "action": "delete", "state": current})
                return
            if time.monotonic() > deadline:
                raise RuntimeError("deletion of sandbox %s was not confirmed in %ss" % (
                    sid, self.readiness_seconds))
            time.sleep(self.poll_seconds)

    def _record(self, name: str, value: Any) -> None:
        if self._record_dir is None:
            return
        try:
            self._record_dir.mkdir(parents=True, exist_ok=True)
            (self._record_dir / name).write_text(
                json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        except OSError:
            pass


class DaytonaAdapter:
    """H200-backed adapter using the repository's existing judge harness."""

    def __init__(
        self,
        problem: str,
        candidates: Optional[Sequence[Any]] = None,
        *,
        cases: Optional[Sequence[int]] = None,
        rounds: int = 4,
        seconds: int = 900,
        baseline_name: str = "t10",
        work_dir: Optional[Path] = None,
        lease_factory: Optional[Callable[[], Any]] = None,
        job_runner: Optional[Callable[..., Any]] = None,
        sandbox_id: Optional[str] = None,
        snapshot: Optional[str] = None,
        close_action: Optional[str] = None,
        expected_gpu: Optional[str] = None,
        require_network_block: bool = True,
        readiness_seconds: int = 600,
        uid_start: int = 10000,
        max_candidate_bytes: int = _MAX_CANDIDATE_BYTES,
    ) -> None:
        if sandbox_id and snapshot:
            raise ValueError("pass at most one of sandbox_id and snapshot")
        if lease_factory is not None and (sandbox_id or snapshot):
            raise ValueError("lease_factory cannot be combined with sandbox_id or snapshot")
        self.problem = str(problem)
        spec_path = _GPU_DIR / "specs" / (self.problem + ".json")
        if not spec_path.is_file():
            raise ValueError("unknown Daytona problem %r (no %s)" % (self.problem, spec_path))
        self.spec = json.loads(spec_path.read_text(encoding="utf-8"))
        self.benchmarks = list(self.spec.get("benchmarks") or [])
        if not self.benchmarks:
            raise ValueError("Daytona problem %r has no benchmark cases" % (self.problem,))
        self.cases = _parse_cases(cases)
        if not self.cases:
            raise ValueError("at least one public case index is required")
        for index in self.cases:
            if index < 0 or index >= len(self.benchmarks):
                raise ValueError(
                    "case index %d is outside 0..%d for %s"
                    % (index, len(self.benchmarks) - 1, self.problem)
                )
        if len(set(self.cases)) != len(self.cases):
            raise ValueError("case indices must be unique")
        self.rounds = max(2, int(rounds))
        self.seconds = max(1, int(seconds))
        self.baseline_name = str(baseline_name)
        self.max_candidate_bytes = max(1, int(max_candidate_bytes))
        self._lease_factory = lease_factory
        self._job_runner = job_runner
        self._sandbox_id = str(sandbox_id) if sandbox_id else None
        self._snapshot = str(snapshot) if snapshot else None
        self._close_action = close_action
        self._expected_gpu = str(expected_gpu) if expected_gpu else None
        self._require_network_block = bool(require_network_block)
        self._readiness_seconds = max(1, int(readiness_seconds))
        self._uid_start = int(uid_start)
        self._jobs = 0
        self._lease = None
        self._dispatcher = None
        self._session = None
        self._results: Dict[tuple, _Eval] = {}
        self._built: Dict[str, Dict[str, Path]] = {}

        self._tmp: Optional[tempfile.TemporaryDirectory] = None
        if work_dir is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="ga-daytona-")
            self._source_dir = Path(self._tmp.name)
        else:
            self._source_dir = Path(work_dir).expanduser()
            self._source_dir.mkdir(parents=True, exist_ok=True)
        self._source_dir = self._source_dir.resolve()
        self._baseline_path = self._write_spec_file("baseline.py")
        self._template_path = self._write_spec_file("template.py")

        self._candidate_paths = self._normalize_candidates(candidates)
        self._candidate_genes = [
            Gene("cand:%s" % _safe_name(path.stem), ["file", str(path)])
            for path in self._candidate_paths
        ]
        names = [gene.name for gene in self._candidate_genes]
        if len(set(names)) != len(names):
            raise ValueError("candidate file names collide; rename one of them")

        self._cells = [self._make_cell(index) for index in self.cases]
        self._initial_by_cell: Dict[str, List[Gene]] = {}
        self._mutants_by_cell: Dict[str, List[Gene]] = {}
        for cell in self._cells:
            initial = [
                Gene(self.baseline_name, ["file", str(self._baseline_path)]),
                Gene("template", ["file", str(self._template_path)], selectable=False),
            ]
            if self._candidate_genes:
                initial.append(self._candidate_genes[0])
            self._initial_by_cell[cell.key] = initial
            # The remaining candidate files are the mutation pool.  There is no
            # source-code mutation operator here; a gene is a real candidate
            # implementation supplied by the person running the search.
            self._mutants_by_cell[cell.key] = list(self._candidate_genes[1:])

    # -- setup -----------------------------------------------------------

    def _write_spec_file(self, name: str) -> Path:
        files = self.spec.get("files") or {}
        if name not in files:
            raise ValueError("Daytona problem %r has no %s" % (self.problem, name))
        path = self._source_dir / name
        path.write_text(str(files[name]), encoding="utf-8")
        return path

    def _normalize_candidates(self, candidates: Optional[Sequence[Any]]) -> List[Path]:
        if candidates is None:
            return []
        if isinstance(candidates, (str, os.PathLike)):
            candidates = [candidates]
        paths: List[Path] = []
        for item in candidates:
            path = Path(item).expanduser()
            if path.is_dir():
                paths.extend(sorted(candidate for candidate in path.glob("*.py") if candidate.is_file()))
            else:
                paths.append(path)
        resolved: List[Path] = []
        seen = set()
        for path in paths:
            path = path.resolve()
            if path in seen:
                continue
            if not path.is_file():
                raise ValueError("candidate file does not exist: %s" % path)
            seen.add(path)
            resolved.append(path)
        return resolved

    def _make_cell(self, index: int) -> Cell:
        record = self.benchmarks[index]
        layer = str(record.get("layer", "?"))
        m = int(record.get("m", 1))
        return Cell(
            key="case %d|%s|m=%d" % (index, layer, m),
            layer=layer,
            m=m,
            data={"case": index, "record": record},
        )

    # -- adapter protocol ------------------------------------------------

    def cells(self) -> List[Cell]:
        return list(self._cells)

    def initial_genes(self, cell: Cell) -> List[Gene]:
        return list(self._initial_by_cell.get(cell.key, []))

    def mutants(
        self,
        cell: Cell,
        incumbent: str,
        known: Set[str],
        state: GAState,
        rng: Any,
    ) -> List[Gene]:
        pool = [
            gene
            for gene in self._mutants_by_cell.get(cell.key, [])
            if gene.name not in known and "%s|%s" % (cell.key, gene.name) not in state.blacklist
        ]
        rng.shuffle(pool)
        return pool

    def build(self, cell: Cell, gene: Gene) -> BuildResult:
        path = self._gene_path(gene)
        try:
            data = path.read_bytes()
        except OSError as exc:
            return BuildResult(False, False, "cannot read candidate: %s" % exc)
        if len(data) > self.max_candidate_bytes:
            return BuildResult(False, False, "candidate exceeds %d bytes" % self.max_candidate_bytes)
        try:
            source = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            return BuildResult(False, False, "candidate is not UTF-8: %s" % exc)
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            return BuildResult(False, False, "syntax error: %s" % exc)
        names = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                names.add(node.name)
            elif isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        names.add(target.id)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                names.add(node.target.id)
            elif isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    names.add(alias.asname or alias.name)
        if "kernel_fn" not in names:
            return BuildResult(False, False, "candidate does not define kernel_fn")
        self._built.setdefault(cell.key, {})[gene.name] = path
        # GPU kernels are not expected to be bit-reproducible against the
        # baseline; --det-only is therefore not meaningful for this adapter.
        return BuildResult(True, False, None)

    def drop(self, cell: Cell, name: str) -> None:
        self._built.get(cell.key, {}).pop(name, None)

    def measure(self, cell: Cell, names: Sequence[str], blocks: int) -> Dict[str, List[float]]:
        names = list(dict.fromkeys(str(name) for name in names))
        target = max(2, int(blocks))
        out: Dict[str, List[float]] = {}
        for name in names:
            evaluation = self._evaluate(cell, name, target, fresh=False)
            out[name] = list(evaluation.candidate[:target])
        return out

    def confirm(self, cell: Cell, champion: str, base: str, blocks: int) -> ConfirmResult:
        target = max(2, int(blocks))
        if champion == base:
            evaluation = self._evaluate(cell, champion, target, fresh=True)
            median = statistics.median(evaluation.candidate)
            return ConfirmResult(rel=1.0, lo=1.0, hi=1.0, t_champion=median, t_base=median)
        champion_eval = self._evaluate(cell, champion, target, fresh=True)
        if base == self.baseline_name:
            # The judge times the official baseline.py in the same job, on the
            # same fresh inputs and in randomized paired order.  Reusing those
            # blocks is the paired comparison the engine's gate expects.
            candidate = champion_eval.candidate
            reference = champion_eval.baseline
        else:
            base_eval = self._evaluate(cell, base, target, fresh=True)
            candidate = champion_eval.candidate
            reference = base_eval.candidate
        if len(candidate) != len(reference):
            raise RuntimeError("paired confirmation lengths differ")
        ratios = [
            math.log(float(left) / float(right))
            for left, right in zip(candidate, reference)
        ]
        mean = sum(ratios) / len(ratios)
        error = statistics.stdev(ratios) / math.sqrt(len(ratios)) if len(ratios) > 1 else 0.0
        return ConfirmResult(
            rel=math.exp(mean),
            lo=math.exp(mean - 1.96 * error),
            hi=math.exp(mean + 1.96 * error),
            t_champion=statistics.median(candidate),
            t_base=statistics.median(reference),
        )

    def close(self) -> None:
        lease = self._lease
        self._lease = None
        try:
            if lease is not None and getattr(lease, "sid", None):
                lease.close()
        finally:
            if self._tmp is not None:
                self._tmp.cleanup()
                self._tmp = None

    # -- provider transport ----------------------------------------------

    def _gene_path(self, gene: Gene) -> Path:
        spec = gene.spec
        if not isinstance(spec, (list, tuple)) or len(spec) < 2 or spec[0] != "file":
            raise ValueError("Daytona gene spec must be ['file', '/path/to/candidate.py']")
        return Path(str(spec[1])).expanduser()

    def _load_harness(self) -> Any:
        if self._dispatcher is None:
            for root in (str(_REPO_ROOT), str(_BENCHMARKS_DIR), str(_GPU_DIR)):
                if root not in sys.path:
                    sys.path.insert(0, root)
            self._dispatcher = _load_module("ga_daytona_dispatcher", _GPU_DIR / "dispatcher.py")
            self._session = getattr(self._dispatcher, "gpu", None)
            if self._session is None:
                self._session = _load_module("ga_daytona_session", _GPU_DIR / "session.py")
        return self._dispatcher

    def _ensure_lease(self) -> Any:
        if self._lease is None:
            if self._lease_factory is not None:
                self._lease = self._lease_factory()
            else:
                self._load_harness()
                if self._sandbox_id or self._snapshot:
                    self._lease = DaytonaLease(
                        self._sandbox_id,
                        self._snapshot,
                        close_action=self._close_action,
                        expected_gpu=self._expected_gpu,
                        require_network_block=self._require_network_block,
                        readiness_seconds=self._readiness_seconds,
                        transport=_HarnessTransport(self._session),
                        record_dir=self._source_dir / "lease",
                    )
                else:
                    self._lease = self._dispatcher.Lease()
        if not getattr(self._lease, "sid", None):
            self._lease.start()
        return self._lease

    def _run_job(
        self,
        sid: str,
        problem: str,
        candidate: Path,
        jobid: str,
        uid: int,
        *,
        seconds: int,
        cases: Sequence[int],
        rounds: int,
    ) -> Any:
        if self._job_runner is not None:
            return self._job_runner(
                sid,
                problem,
                candidate,
                jobid,
                uid,
                seconds=seconds,
                cases=list(cases),
                rounds=rounds,
                cancel_event=None,
            )
        self._load_harness()
        return self._session.run_job(
            sid,
            problem,
            candidate,
            jobid,
            uid,
            seconds=seconds,
            cases=list(cases),
            rounds=rounds,
            cancel_event=None,
        )

    def _evaluate(self, cell: Cell, name: str, target: int, *, fresh: bool) -> _Eval:
        key = (cell.key, name)
        cached = self._results.get(key)
        if not fresh and cached is not None and len(cached.candidate) >= target:
            return cached
        path = self._built.get(cell.key, {}).get(name)
        if path is None:
            raise ValueError("measure/confirm called before build for %r" % name)
        case_index = int(cell.data["case"])
        jobid = "ga-%s-%d-%s-%s" % (
            _safe_name(self.problem),
            case_index,
            _safe_name(name),
            secrets.token_hex(4),
        )
        uid = self._uid_start + (self._jobs % 40000)
        self._jobs += 1
        lease = self._ensure_lease()
        state = self._run_job(
            lease.sid,
            self.problem,
            path,
            jobid,
            uid,
            seconds=self.seconds,
            cases=[case_index],
            rounds=target,
        )
        evaluation = self._parse_state(state, case_index, target, jobid)
        self._results[key] = evaluation
        return evaluation

    def _parse_state(self, state: Any, case_index: int, target: int, jobid: str) -> _Eval:
        result = state.get("result") if isinstance(state, dict) else None
        if not isinstance(result, dict):
            log = ""
            if isinstance(state, dict):
                log = str(state.get("log") or "")[-1200:]
            raise RuntimeError("H200 job returned no result for %s: %s" % (jobid, log))
        if result.get("status") != "pass":
            raise RuntimeError(
                "H200 job rejected %s: %s"
                % (jobid, str(result.get("error") or result.get("status"))[:1200])
            )
        rows = result.get("cases") or []
        row = None
        for item in rows:
            if isinstance(item, dict) and int(item.get("index", -1)) == int(case_index):
                row = item
                break
        if row is None:
            raise RuntimeError("H200 result for %s has no case %d" % (jobid, case_index))
        pairs = row.get("paired_rounds")
        if not isinstance(pairs, list) or len(pairs) < 2:
            raise RuntimeError("H200 result for %s has too few paired rounds" % jobid)
        candidate: List[float] = []
        baseline: List[float] = []
        for pair in pairs:
            if not isinstance(pair, dict):
                raise RuntimeError("H200 result for %s has a malformed round" % jobid)
            try:
                value = float(pair["candidate"])
                reference = float(pair["baseline"])
            except (KeyError, TypeError, ValueError) as exc:
                raise RuntimeError("H200 result for %s has a malformed round: %s" % (jobid, exc)) from exc
            if not (math.isfinite(value) and math.isfinite(reference)) or value <= 0 or reference <= 0:
                raise RuntimeError("H200 result for %s has a non-positive or non-finite round" % jobid)
            candidate.append(value)
            baseline.append(reference)
        if len(candidate) < target:
            raise RuntimeError(
                "H200 result for %s has %d rounds, wanted %d" % (jobid, len(candidate), target)
            )
        return _Eval(
            candidate=candidate[:target],
            baseline=baseline[:target],
            job=jobid,
            status=str(result.get("status")),
        )


def make_adapter() -> DaytonaAdapter:
    """Default factory; delegates to the environment-driven constructor."""

    return make_adapter_from_env()


def make_adapter_from_env() -> DaytonaAdapter:
    """Construct from ``GA_DAYTONA_*`` environment variables.

    ``GA_DAYTONA_CANDIDATES`` is a ``os.pathsep``-separated list of candidate
    source files or directories.  ``GA_DAYTONA_CASES`` defaults to ``0``.

    Reuse an existing box with ``GA_DAYTONA_SANDBOX_ID``, or create from a
    snapshot with ``GA_DAYTONA_SNAPSHOT`` (UUID or exact name).  The two are
    mutually exclusive.  ``GA_DAYTONA_CLOSE_ACTION`` is ``delete`` for a new
    snapshot box and ``stop`` for an attached box by default; set ``leave`` to
    keep the box running.  ``GA_DAYTONA_EXPECTED_GPU`` overrides the GPU type
    check (for example ``RTX-4090``); otherwise the snapshot or sandbox
    metadata is used.  ``DAYTONA_API_KEY`` is deliberately not read here; the
    provider transport reads it only when a request is made.
    """

    problem = os.environ.get("GA_DAYTONA_PROBLEM", "muse_bf16_proj")
    raw_candidates = os.environ.get("GA_DAYTONA_CANDIDATES", "").strip()
    if not raw_candidates:
        raise ValueError(
            "set GA_DAYTONA_CANDIDATES to one or more candidate .py files or directories "
            "(separated by %r), or write a small adapter file that constructs DaytonaAdapter"
            % os.pathsep
        )
    candidates = [part for part in raw_candidates.split(os.pathsep) if part]
    cases = _parse_cases(os.environ.get("GA_DAYTONA_CASES", "0"))
    return DaytonaAdapter(
        problem,
        candidates,
        cases=cases,
        rounds=int(os.environ.get("GA_DAYTONA_ROUNDS", "4")),
        seconds=int(os.environ.get("GA_DAYTONA_SECONDS", "900")),
        sandbox_id=os.environ.get("GA_DAYTONA_SANDBOX_ID") or None,
        snapshot=os.environ.get("GA_DAYTONA_SNAPSHOT") or None,
        close_action=os.environ.get("GA_DAYTONA_CLOSE_ACTION") or None,
        expected_gpu=os.environ.get("GA_DAYTONA_EXPECTED_GPU") or None,
        require_network_block=os.environ.get("GA_DAYTONA_ALLOW_NETWORK", "") not in {"1", "true", "yes"},
        readiness_seconds=int(os.environ.get("GA_DAYTONA_READY_SECONDS", "600")),
    )


__all__ = ["DaytonaAdapter", "DaytonaLease", "make_adapter", "make_adapter_from_env"]
