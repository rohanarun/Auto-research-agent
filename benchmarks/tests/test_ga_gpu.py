"""Tests for the optional GPU adapters.

The Torch tests are skipped when PyTorch or a CUDA/MPS device is unavailable.
The Daytona tests use a fake lease and a fake job runner; they never read
``DAYTONA_API_KEY`` and never contact the provider.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, List, Sequence
from unittest import mock

BENCHMARKS = Path(__file__).resolve().parents[1]
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from ga.daytona import DaytonaAdapter, DaytonaLease, make_adapter_from_env  # noqa: E402
from ga.engine import Gene  # noqa: E402

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def _gpu_available() -> bool:
    if not TORCH_AVAILABLE:
        return False
    try:
        import torch
    except Exception:
        return False
    if torch.cuda.is_available():
        return True
    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available())


class _FakeLease:
    def __init__(self) -> None:
        self.sid = None
        self.started = 0
        self.closed = 0

    def start(self) -> None:
        self.started += 1
        self.sid = "fake-sandbox"

    def close(self) -> None:
        self.closed += 1
        self.sid = None


class _FakeJobs:
    """Return deterministic paired rounds without touching Daytona."""

    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []

    def __call__(
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
        cancel_event: Any = None,
    ) -> Dict[str, Any]:
        case = int(cases[0])
        name = Path(candidate).name
        baseline = 100.0 + case
        value = 95.0 + case if "fast" in name else baseline
        pairs = [
            {"baseline": baseline, "candidate": value}
            for _ in range(max(2, int(rounds)))
        ]
        self.calls.append(
            {
                "sid": sid,
                "problem": problem,
                "candidate": name,
                "jobid": jobid,
                "uid": uid,
                "seconds": seconds,
                "cases": list(cases),
                "rounds": rounds,
            }
        )
        return {
            "exit_code": 0,
            "log": "",
            "result": {
                "status": "pass",
                "cases": [
                    {
                        "index": case,
                        "correct": True,
                        "baseline_us": baseline,
                        "candidate_us": value,
                        "paired_rounds": pairs,
                    }
                ],
            },
        }


class _FakeDaytona:
    """Minimal provider double for lease lifecycle tests; no key is read."""

    def __init__(
        self,
        *,
        sandbox_state: str = "stopped",
        snapshot_state: str = "active",
        gpu: str = "NVIDIA H200",
        uid: int = 0,
        network_block: bool = True,
    ) -> None:
        self.sandbox = {
            "id": "box-1",
            "name": "existing-box",
            "state": sandbox_state,
            "gpuType": "H200",
            "networkBlockAll": network_block,
        }
        self.snapshot = {
            "id": "snap-uuid",
            "name": "snap-1",
            "state": snapshot_state,
            "gpuType": "H200",
        }
        self.gpu = gpu
        self.uid = uid
        self.requests: List[Any] = []
        self.deleted = False
        self.created_body: Dict[str, Any] = {}

    def request(
        self,
        plane: str,
        path: str,
        method: str = "GET",
        body: Any = None,
        timeout: int = 45,
    ) -> Dict[str, Any]:
        self.requests.append((plane, method, path, body))
        if plane != "control":
            return {"ok": False, "error": "unexpected plane"}
        if path == "/sandbox" and method == "POST":
            self.created_body = dict(body or {})
            self.sandbox = {
                "id": "new-box",
                "name": self.created_body.get("name"),
                "state": "started",
                "gpuType": "H200",
                "networkBlockAll": True,
            }
            return {"ok": True, "data": {"id": "new-box"}}
        if path == "/snapshots?name=snap-1&limit=100&page=1":
            return {"ok": True, "data": {"items": [self.snapshot]}}
        if path == "/snapshots/snap-uuid":
            return {"ok": True, "data": self.snapshot}
        if path == "/snapshots/snap-uuid/activate":
            self.snapshot["state"] = "active"
            return {"ok": True, "data": {}}
        if path.startswith("/sandbox/"):
            if self.deleted:
                return {"ok": False, "http_status": 404, "error": "gone"}
            if path.endswith("/start"):
                self.sandbox["state"] = "started"
                return {"ok": True, "data": self.sandbox}
            if path.endswith("/stop"):
                self.sandbox["state"] = "stopped"
                return {"ok": True, "data": self.sandbox}
            if method == "DELETE":
                self.deleted = True
                return {"ok": True, "data": {}}
            return {"ok": True, "data": self.sandbox}
        return {"ok": False, "error": "unexpected path " + path}

    def call(self, sid: str, command: str, timeout: int = 45) -> Dict[str, Any]:
        if "nvidia-smi" in command:
            return {"exitCode": 0, "result": self.gpu + ", 143771 MiB"}
        if "geteuid" in command:
            import json

            return {
                "exitCode": 0,
                "result": json.dumps(
                    {"uid": self.uid, "python": "python3.11", "torch": "2.8.0"}
                ),
            }
        return {"exitCode": 1, "result": "unexpected command"}


class DaytonaLeaseTests(unittest.TestCase):
    def test_attaches_to_stopped_box_and_stops_on_close(self):
        fake = _FakeDaytona(sandbox_state="stopped")
        lease = DaytonaLease(
            sandbox_id="box-1",
            transport=fake,
            poll_seconds=0.01,
            readiness_seconds=5,
        )
        lease.start()
        self.assertEqual("box-1", lease.sid)
        self.assertEqual("started", fake.sandbox["state"])
        self.assertIn("H200", lease.info["gpu_name"])
        self.assertEqual(0, lease.info["uid"])
        lease.close()
        self.assertEqual("stopped", fake.sandbox["state"])
        self.assertFalse(fake.deleted)
        methods = [(method, path) for _, method, path, _ in fake.requests]
        self.assertIn(("POST", "/sandbox/box-1/start"), methods)
        self.assertIn(("POST", "/sandbox/box-1/stop"), methods)

    def test_snapshot_box_is_created_without_resources_and_deleted(self):
        fake = _FakeDaytona(snapshot_state="inactive")
        lease = DaytonaLease(
            snapshot="snap-1",
            transport=fake,
            poll_seconds=0.01,
            readiness_seconds=5,
        )
        lease.start()
        self.assertEqual("new-box", lease.sid)
        self.assertEqual("snap-uuid", fake.created_body.get("snapshot"))
        self.assertFalse(
            {"cpu", "gpu", "memory", "disk"} & set(fake.created_body),
            "snapshot creation must not send explicit resources",
        )
        self.assertTrue(fake.created_body.get("networkBlockAll"))
        self.assertIn(("POST", "/snapshots/snap-uuid/activate"),
                      [(method, path) for _, method, path, _ in fake.requests])
        lease.close()
        self.assertTrue(fake.deleted)

    def test_non_root_box_is_rejected(self):
        fake = _FakeDaytona(sandbox_state="started", uid=1001)
        lease = DaytonaLease(
            sandbox_id="box-1",
            transport=fake,
            poll_seconds=0.01,
            readiness_seconds=5,
        )
        with self.assertRaisesRegex(RuntimeError, "requires root"):
            lease.start()
        self.assertIsNone(lease.sid)
        self.assertEqual("stopped", fake.sandbox["state"])

    def test_network_is_required_by_default(self):
        fake = _FakeDaytona(sandbox_state="started", network_block=False)
        lease = DaytonaLease(
            sandbox_id="box-1",
            transport=fake,
            poll_seconds=0.01,
            readiness_seconds=5,
        )
        with self.assertRaisesRegex(RuntimeError, "networkBlockAll"):
            lease.start()

    def test_wrong_gpu_is_rejected(self):
        fake = _FakeDaytona(sandbox_state="started", gpu="NVIDIA H100")
        lease = DaytonaLease(
            sandbox_id="box-1",
            expected_gpu="H200",
            transport=fake,
            poll_seconds=0.01,
            readiness_seconds=5,
        )
        with self.assertRaisesRegex(RuntimeError, "expected 'H200'"):
            lease.start()


class DaytonaAdapterTests(unittest.TestCase):
    def _candidate(self, root: Path, name: str = "fast.py") -> Path:
        path = root / name
        path.write_text(
            "def kernel_fn(data, state):\n    return data\n", encoding="utf-8"
        )
        return path

    def test_case_gene_mapping_and_paired_confirmation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            candidate = self._candidate(root)
            lease = _FakeLease()
            jobs = _FakeJobs()
            adapter = DaytonaAdapter(
                "muse_bf16_proj",
                candidates=[candidate],
                cases=[0, 1],
                rounds=3,
                lease_factory=lambda: lease,
                job_runner=jobs,
            )
            try:
                cells = adapter.cells()
                self.assertEqual(2, len(cells))
                self.assertEqual([0, 1], [cell.data["case"] for cell in cells])
                initial = adapter.initial_genes(cells[0])
                names = [gene.name for gene in initial]
                self.assertIn("t10", names)
                self.assertIn("template", names)
                self.assertIn("cand:fast", names)
                self.assertFalse(next(gene for gene in initial if gene.name == "template").selectable)

                self.assertTrue(adapter.build(cells[0], initial[0]).ok)
                fast = next(gene for gene in initial if gene.name == "cand:fast")
                self.assertTrue(adapter.build(cells[0], fast).ok)

                times = adapter.measure(cells[0], ["t10", "cand:fast"], 3)
                self.assertEqual(3, len(times["t10"]))
                self.assertEqual(3, len(times["cand:fast"]))
                self.assertEqual([100.0, 100.0, 100.0], times["t10"])
                self.assertEqual([95.0, 95.0, 95.0], times["cand:fast"])
                self.assertEqual(1, lease.started)
                self.assertEqual(2, len(jobs.calls))
                self.assertTrue(all(call["sid"] == "fake-sandbox" for call in jobs.calls))

                confirmed = adapter.confirm(cells[0], "cand:fast", "t10", 4)
                self.assertAlmostEqual(0.95, confirmed.rel, places=9)
                self.assertAlmostEqual(0.95, confirmed.hi, places=9)
                self.assertAlmostEqual(100.0, confirmed.t_base, places=9)
                self.assertAlmostEqual(95.0, confirmed.t_champion, places=9)
                # Confirm used a fresh job, not the cached measure result.
                self.assertEqual(3, len(jobs.calls))
                self.assertEqual(4, jobs.calls[-1]["rounds"])
            finally:
                adapter.close()
            self.assertEqual(1, lease.closed)

    def test_build_rejects_missing_kernel_fn(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            candidate = self._candidate(root)
            bad = root / "bad.py"
            bad.write_text("value = 1\n", encoding="utf-8")
            adapter = DaytonaAdapter(
                "muse_bf16_proj",
                candidates=[candidate],
                cases=[0],
                lease_factory=_FakeLease,
                job_runner=_FakeJobs(),
            )
            try:
                result = adapter.build(adapter.cells()[0], Gene("bad", ["file", str(bad)]))
                self.assertFalse(result.ok)
                self.assertIn("kernel_fn", result.error or "")
            finally:
                adapter.close()

    def test_environment_factory_does_not_need_a_key(self):
        with tempfile.TemporaryDirectory() as folder:
            candidate = self._candidate(Path(folder))
            environment = {
                "GA_DAYTONA_PROBLEM": "muse_bf16_proj",
                "GA_DAYTONA_CANDIDATES": str(candidate),
                "GA_DAYTONA_CASES": "0,1",
                "GA_DAYTONA_ROUNDS": "3",
            }
            with mock.patch.dict(os.environ, environment, clear=False):
                os.environ.pop("DAYTONA_API_KEY", None)
                adapter = make_adapter_from_env()
                try:
                    self.assertEqual("muse_bf16_proj", adapter.problem)
                    self.assertEqual([0, 1], adapter.cases)
                    self.assertEqual(3, adapter.rounds)
                    self.assertIsNone(adapter._lease)
                finally:
                    adapter.close()

    def test_environment_factory_reads_reuse_configuration(self):
        with tempfile.TemporaryDirectory() as folder:
            candidate = self._candidate(Path(folder))
            environment = {
                "GA_DAYTONA_CANDIDATES": str(candidate),
                "GA_DAYTONA_SNAPSHOT": "snap-uuid",
                "GA_DAYTONA_CLOSE_ACTION": "leave",
                "GA_DAYTONA_EXPECTED_GPU": "RTX-4090",
            }
            with mock.patch.dict(os.environ, environment, clear=False):
                os.environ.pop("DAYTONA_API_KEY", None)
                adapter = make_adapter_from_env()
                try:
                    self.assertEqual("snap-uuid", adapter._snapshot)
                    self.assertEqual("leave", adapter._close_action)
                    self.assertEqual("RTX-4090", adapter._expected_gpu)
                finally:
                    adapter.close()

    def test_failed_job_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            candidate = self._candidate(Path(folder))
            adapter = DaytonaAdapter(
                "muse_bf16_proj",
                candidates=[candidate],
                cases=[0],
                lease_factory=_FakeLease,
                job_runner=_FakeJobs(),
            )
            try:
                with self.assertRaises(RuntimeError):
                    adapter._parse_state(
                        {"result": {"status": "failed", "error": "bad kernel"}},
                        0,
                        2,
                        "job",
                    )
            finally:
                adapter.close()


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class TorchGPUAdapterTests(unittest.TestCase):
    def test_cpu_fallback_is_real_and_labelled(self):
        from ga.torch_gpu import TorchGPUAdapter

        adapter = TorchGPUAdapter(
            device="cpu",
            shapes=[(2, 64, 64)],
            replays=2,
            warmup=0,
            flush_mb=1,
        )
        try:
            info = adapter.info()
            self.assertFalse(info["is_gpu"])
            self.assertEqual("cpu", info["device_type"])
            cell = adapter.cells()[0]
            for gene in adapter.initial_genes(cell):
                self.assertTrue(adapter.build(cell, gene).ok, gene.name)
            times = adapter.measure(cell, ["t10", "matmul"], 2)
            self.assertEqual(2, len(times["t10"]))
            self.assertTrue(all(value > 0 for value in times["t10"]))
            confirmed = adapter.confirm(cell, "matmul", "t10", 2)
            self.assertTrue(confirmed.rel > 0)
            self.assertTrue(confirmed.t_base > 0)
        finally:
            adapter.close()

    @unittest.skipUnless(_gpu_available(), "no CUDA or MPS device is available")
    def test_gpu_measurement_is_real(self):
        from ga.torch_gpu import TorchGPUAdapter

        adapter = TorchGPUAdapter(
            device="auto",
            shapes=[(4, 32, 256, 256)],
            replays=5,
            warmup=1,
            flush_mb=2,
        )
        try:
            info = adapter.info()
            self.assertTrue(info["is_gpu"])
            cell = adapter.cells()[0]
            for gene in adapter.initial_genes(cell):
                self.assertTrue(adapter.build(cell, gene).ok, gene.name)
            self.assertTrue(adapter.build(cell, Gene("bmm", ["bmm"], risky=True)).ok)
            times = adapter.measure(cell, ["t10", "matmul", "bmm"], 3)
            for name in ("t10", "matmul", "bmm"):
                self.assertEqual(3, len(times[name]))
                self.assertTrue(all(value > 0 for value in times[name]))
            confirmed = adapter.confirm(cell, "matmul", "t10", 3)
            self.assertTrue(confirmed.rel > 0)
            self.assertLessEqual(confirmed.lo, confirmed.hi)
        finally:
            adapter.close()


if __name__ == "__main__":
    unittest.main()
