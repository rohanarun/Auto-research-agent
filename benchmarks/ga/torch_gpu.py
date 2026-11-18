"""Real local GPU adapter for the portable GA (CUDA or Apple MPS).

This module is optional.  Importing it does **not** import PyTorch; the adapter
imports Torch when it is constructed and runs actual matrix multiplications on
the selected device.  The timings are real measurements of the local hardware,
not the synthetic cost model used by ``demo.py``.

The default target mixes a batched GEMM and a plain GEMM.  For the batched
cells the baseline is a Python loop of ``torch.mm`` calls and the candidates
include one ``torch.bmm`` / ``torch.matmul`` call.  Replacing that loop is a
genuine implementation choice with a measurable gain on CUDA and Apple MPS.
For the plain cells the candidates are different PyTorch matmul entry points.
The adapter is useful for checking that the search loop works against a real
GPU and for comparing those calls on a laptop.  It is not the official H200
judge: the judge uses CUDA graphs, frozen public inputs and an isolated
sandbox.  Use ``daytona.py`` for that target.

Install a CUDA or Apple MPS build of PyTorch to use this adapter::

    python3 -m pip install torch

CUDA users normally install the wheel published for their CUDA version.  Apple
Silicon users only need the standard macOS wheel.  The engine itself remains
standard-library-only.
"""

from __future__ import annotations

import math
import random
import statistics
import time
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from .engine import BuildResult, Cell, ConfirmResult, GAState, Gene

# Small shapes keep the demo quick on a laptop.  A 3-tuple is a plain GEMM
# ``(m, n, k)``; a 4-tuple is a batched GEMM ``(batch, m, n, k)``.  The batched
# cells are included because replacing a Python loop of ``torch.mm`` calls with
# one ``torch.bmm`` is a real implementation choice with a measurable gain.
DEFAULT_SHAPES = (
    (8, 64, 1024, 1024),
    (16, 32, 2048, 2048),
    (1, 1024, 1024),
)

# These are real implementation choices, not fake costs.  The baseline ``t10``
# is ``torch.mm`` for a plain GEMM and a loop of ``torch.mm`` for a batched
# GEMM.  A caller can add its own gene space; this is only a local demo target.
MUTANT_KINDS_2D = (
    ("bmm", ["bmm"]),
    ("out", ["out"]),
    ("einsum", ["einsum"]),
    ("addmm0", ["addmm0"]),
    ("split:2", ["split", 2]),
    ("split:4", ["split", 4]),
)
MUTANT_KINDS_BATCHED = (
    ("bmm", ["bmm"]),
    ("out", ["out"]),
    ("einsum", ["einsum"]),
    ("addmm0", ["addmm0"]),
)


def _torch():
    try:
        import torch  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on the machine
        raise RuntimeError(
            "PyTorch is required for the GPU adapter. Install a CUDA or Apple "
            "MPS build of torch, then rerun with --device auto."
        ) from exc
    return torch


def _dtype_name(dtype: Any) -> str:
    return str(dtype).replace("torch.", "")


class _Eval:
    """One measured evaluation of one gene on one cell."""

    __slots__ = ("candidate", "baseline", "job", "status")

    def __init__(self, candidate: List[float], baseline: List[float], job: str, status: str):
        self.candidate = candidate
        self.baseline = baseline
        self.job = job
        self.status = status


class TorchGPUAdapter:
    """In-process CUDA/MPS adapter with real paired timing.

    Parameters
    ----------
    device:
        ``auto``, ``cuda``, ``mps`` or ``cpu``.  ``auto`` requires a real GPU;
        ``cpu`` is an explicit fallback and is labelled as such.
    dtype:
        ``auto``, ``float16``, ``bfloat16`` or ``float32``.
    cells / shapes:
        Number of default shapes, or an explicit list of ``(m, n, k)`` tuples.
    replays:
        Kernel calls inside each timed block.  More replays reduce timer noise
        but make each block longer.
    flush_mb:
        Size of the cache-flushing buffer.  The flush is zeroed and synchronized
        outside the timed region.
    """

    def __init__(
        self,
        *,
        device: str = "auto",
        dtype: str = "auto",
        cells: Optional[int] = 3,
        shapes: Optional[Sequence[Sequence[int]]] = None,
        seed: int = 7,
        replays: int = 50,
        warmup: int = 5,
        flush_mb: int = 32,
    ) -> None:
        self.torch = _torch()
        self.device = self._resolve_device(device)
        self.dtype = self._resolve_dtype(dtype)
        self.seed = int(seed)
        self.replays = max(1, int(replays))
        self.warmup = max(0, int(warmup))
        self.flush_mb = max(0, int(flush_mb))
        self._rng = random.Random(self.seed)
        self._calls = 0
        self._kernels: Dict[str, Dict[str, Any]] = {}
        self._warmed: Set[Tuple[str, str]] = set()
        self._flush = self._make_flush()

        shape_list = [tuple(int(value) for value in shape) for shape in (shapes or DEFAULT_SHAPES)]
        for shape in shape_list:
            if len(shape) not in (3, 4) or min(shape) <= 0:
                raise ValueError("each shape must be a positive (m, n, k) or (batch, m, n, k) tuple")
        if cells is not None:
            shape_list = shape_list[: max(1, int(cells))]
        self._cells = [self._make_cell(shape) for shape in shape_list]

    @staticmethod
    def _make_cell(shape: Sequence[int]) -> Cell:
        if len(shape) == 4:
            batch, m, n, k = shape
            return Cell(
                key="batched gemm b=%d m=%d n=%d k=%d" % (batch, m, n, k),
                layer="batched_gemm",
                m=m,
                n=n,
                k=k,
                data={"batch": batch},
            )
        m, n, k = shape
        return Cell(key="gemm m=%d n=%d k=%d" % (m, n, k), layer="gemm", m=m, n=n, k=k)

    # -- device and dtype ------------------------------------------------

    def _resolve_device(self, requested: str) -> Any:
        torch = self.torch
        name = str(requested or "auto").lower()
        if name == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            mps = getattr(torch.backends, "mps", None)
            if mps is not None and mps.is_available():
                return torch.device("mps")
            raise RuntimeError(
                "no CUDA or Apple MPS device is available; pass --device cpu "
                "only for an explicitly labelled CPU fallback"
            )
        if name == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
            return torch.device("cuda")
        if name == "mps":
            mps = getattr(torch.backends, "mps", None)
            if mps is None or not mps.is_available():
                raise RuntimeError("MPS was requested but torch.backends.mps.is_available() is false")
            return torch.device("mps")
        if name == "cpu":
            return torch.device("cpu")
        raise ValueError("device must be auto, cuda, mps or cpu")

    def _resolve_dtype(self, requested: str) -> Any:
        torch = self.torch
        name = str(requested or "auto").lower()
        if name == "auto":
            if self.device.type == "cuda":
                try:
                    if torch.cuda.is_bf16_supported():
                        return torch.bfloat16
                except Exception:
                    pass
                return torch.float16
            if self.device.type == "mps":
                return torch.float16
            return torch.float32
        mapping = {
            "float16": torch.float16,
            "fp16": torch.float16,
            "half": torch.float16,
            "bfloat16": torch.bfloat16,
            "bf16": torch.bfloat16,
            "float32": torch.float32,
            "fp32": torch.float32,
            "float": torch.float32,
        }
        if name not in mapping:
            raise ValueError("dtype must be auto, float16, bfloat16 or float32")
        return mapping[name]

    def _device_name(self) -> str:
        if self.device.type == "cuda":
            return str(self.torch.cuda.get_device_name(self.device))
        if self.device.type == "mps":
            return "Apple MPS"
        return "CPU"

    def info(self) -> Dict[str, Any]:
        return {
            "device": str(self.device),
            "device_type": self.device.type,
            "device_name": self._device_name(),
            "dtype": _dtype_name(self.dtype),
            "torch": getattr(self.torch, "__version__", "?"),
            "is_gpu": self.device.type in ("cuda", "mps"),
            "replays": self.replays,
            "warmup": self.warmup,
            "flush_mb": self.flush_mb,
        }

    # -- adapter protocol ------------------------------------------------

    def cells(self) -> List[Cell]:
        return list(self._cells)

    def initial_genes(self, cell: Cell) -> List[Gene]:
        # ``cublas`` is an anchor: it may be raced but can never become champion.
        # ``matmul`` is a real alternative that can win generation 0.
        return [
            Gene("t10", ["mm"]),
            Gene("cublas", ["mm"], selectable=False),
            Gene("matmul", ["matmul"]),
        ]

    def mutants(
        self,
        cell: Cell,
        incumbent: str,
        known: Set[str],
        state: GAState,
        rng: random.Random,
    ) -> List[Gene]:
        pool: List[Gene] = []
        kinds = MUTANT_KINDS_BATCHED if self._is_batched(cell) else MUTANT_KINDS_2D
        for name, spec in kinds:
            if name not in known:
                pool.append(Gene(name, list(spec), risky=True))
        rng.shuffle(pool)
        return pool

    def build(self, cell: Cell, gene: Gene) -> BuildResult:
        try:
            kernel = self._make_kernel(cell, gene)
            ok, bit, error = self._check(cell, kernel)
        except Exception as exc:
            return BuildResult(False, False, repr(exc)[:300])
        if not ok:
            return BuildResult(False, False, error)
        self._kernels.setdefault(cell.key, {})[gene.name] = kernel
        return BuildResult(True, bool(bit), None)

    def drop(self, cell: Cell, name: str) -> None:
        self._kernels.get(cell.key, {}).pop(name, None)
        self._warmed = {item for item in self._warmed if item != (cell.key, name)}

    def measure(self, cell: Cell, names: Sequence[str], blocks: int) -> Dict[str, List[float]]:
        names = list(dict.fromkeys(str(name) for name in names))
        for name in names:
            if name not in self._kernels.get(cell.key, {}):
                raise ValueError("measure called before build for %r" % name)
        a, b = self._inputs(cell, seed=self.seed + 1009 * (self._calls + 1))
        self._calls += 1
        out: Dict[str, List[float]] = {name: [] for name in names}
        for _ in range(max(2, int(blocks))):
            order = list(names)
            self._rng.shuffle(order)
            for name in order:
                out[name].append(self._time(cell, name, a, b))
        return out

    def confirm(self, cell: Cell, champion: str, base: str, blocks: int) -> ConfirmResult:
        for name in (champion, base):
            if name not in self._kernels.get(cell.key, {}):
                raise ValueError("confirm called before build for %r" % name)
        a, b = self._inputs(cell, seed=self.seed + 7919 * (self._calls + 1))
        self._calls += 1
        names = [champion] if champion == base else [champion, base]
        series: Dict[str, List[float]] = {name: [] for name in names}
        for _ in range(max(2, int(blocks))):
            order = list(names)
            self._rng.shuffle(order)
            for name in order:
                series[name].append(self._time(cell, name, a, b))
        if champion == base:
            median = statistics.median(series[champion])
            return ConfirmResult(rel=1.0, lo=1.0, hi=1.0, t_champion=median, t_base=median)
        ratios = [
            math.log(float(candidate) / float(reference))
            for candidate, reference in zip(series[champion], series[base])
        ]
        mean = sum(ratios) / len(ratios)
        error = statistics.stdev(ratios) / math.sqrt(len(ratios)) if len(ratios) > 1 else 0.0
        return ConfirmResult(
            rel=math.exp(mean),
            lo=math.exp(mean - 1.96 * error),
            hi=math.exp(mean + 1.96 * error),
            t_champion=statistics.median(series[champion]),
            t_base=statistics.median(series[base]),
        )

    def close(self) -> None:
        self._kernels.clear()
        self._warmed.clear()
        self._flush = None
        try:
            if self.device.type == "cuda":
                self.torch.cuda.empty_cache()
            elif self.device.type == "mps":
                self.torch.mps.empty_cache()
        except Exception:
            pass

    # -- implementation --------------------------------------------------

    def _make_flush(self) -> Any:
        if self.flush_mb <= 0:
            return None
        megabytes = self.flush_mb
        if self.device.type == "cpu":
            megabytes = min(megabytes, 8)
        return self.torch.empty(
            megabytes * 1024 * 1024, dtype=self.torch.uint8, device=self.device
        )

    def _is_batched(self, cell: Cell) -> bool:
        return bool(isinstance(cell.data, dict) and int(cell.data.get("batch", 0) or 0) > 0)

    def _batch(self, cell: Cell) -> int:
        return int(cell.data.get("batch", 0) or 0) if isinstance(cell.data, dict) else 0

    def _inputs(self, cell: Cell, *, seed: int) -> Tuple[Any, Any]:
        torch = self.torch
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        batch = self._batch(cell)
        if batch:
            a = torch.randn(
                (batch, int(cell.m), int(cell.k)), generator=generator, dtype=torch.float32
            )
            b = torch.randn(
                (batch, int(cell.k), int(cell.n)), generator=generator, dtype=torch.float32
            )
        else:
            a = torch.randn((int(cell.m), int(cell.k)), generator=generator, dtype=torch.float32)
            b = torch.randn((int(cell.k), int(cell.n)), generator=generator, dtype=torch.float32)
        return (
            a.to(device=self.device, dtype=self.dtype),
            b.to(device=self.device, dtype=self.dtype),
        )

    def _make_kernel(self, cell: Cell, gene: Gene):
        torch = self.torch
        spec = gene.spec
        kind = spec[0] if isinstance(spec, (list, tuple)) and spec else str(spec)
        m, n, k = int(cell.m), int(cell.n), int(cell.k)
        batch = self._batch(cell)
        if kind == "mm":
            if batch:
                return lambda a, b: torch.stack(
                    [torch.mm(a[index], b[index]) for index in range(batch)]
                )
            return lambda a, b: torch.mm(a, b)
        if kind == "matmul":
            return lambda a, b: torch.matmul(a, b)
        if kind == "bmm":
            if batch:
                return lambda a, b: torch.bmm(a, b)
            return lambda a, b: torch.bmm(a.unsqueeze(0), b.unsqueeze(0))[0]
        if kind == "einsum":
            if batch:
                return lambda a, b: torch.einsum("bik,bkj->bij", a, b)
            return lambda a, b: torch.einsum("mk,kn->mn", a, b)
        if kind == "out":
            if batch:
                out = torch.empty((batch, m, n), dtype=self.dtype, device=self.device)

                def batched_out(a, b):
                    for index in range(batch):
                        torch.mm(a[index], b[index], out=out[index])
                    return out

                return batched_out
            out = torch.empty((m, n), dtype=self.dtype, device=self.device)
            return lambda a, b: torch.mm(a, b, out=out)
        if kind == "addmm0":
            if batch:
                bias = torch.zeros(n, dtype=self.dtype, device=self.device)

                def batched_addmm(a, b):
                    return torch.stack(
                        [torch.addmm(bias, a[index], b[index]) for index in range(batch)]
                    )

                return batched_addmm
            bias = torch.zeros(n, dtype=self.dtype, device=self.device)
            return lambda a, b: torch.addmm(bias, a, b)
        if kind == "split":
            chunks = max(2, int(spec[1])) if len(spec) > 1 else 2

            def split(a, b):
                out = torch.zeros(
                    (a.shape[0], b.shape[1]), dtype=a.dtype, device=a.device
                )
                step = (a.shape[1] + chunks - 1) // chunks
                for start in range(0, a.shape[1], step):
                    out += torch.mm(a[:, start : start + step], b[start : start + step, :])
                return out

            return split
        raise ValueError("unknown gene implementation %r" % (kind,))

    def _relative_l2(self, output: Any, reference: Any) -> float:
        difference = (output.float() - reference).norm()
        denominator = reference.norm().clamp_min(1e-12)
        return float(difference / denominator)

    def _check(self, cell: Cell, kernel: Any) -> Tuple[bool, bool, Optional[str]]:
        torch = self.torch
        a, b = self._inputs(cell, seed=self.seed + 17)
        batch = self._batch(cell)
        with torch.inference_mode():
            if batch:
                reference = torch.matmul(a.float(), b.float())
                baseline = torch.stack(
                    [torch.mm(a[index], b[index]) for index in range(batch)]
                )
                expected = (batch, int(cell.m), int(cell.n))
            else:
                reference = torch.mm(a.float(), b.float())
                baseline = torch.mm(a, b)
                expected = (int(cell.m), int(cell.n))
            output = kernel(a, b)
            self._sync()
        if tuple(output.shape) != expected:
            return False, False, "wrong output shape %r" % (tuple(output.shape),)
        if output.dtype != self.dtype:
            return False, False, "wrong output dtype %s" % _dtype_name(output.dtype)
        if output.device.type != self.device.type:
            return False, False, "wrong output device %s" % output.device
        if not bool(torch.isfinite(output.float()).all()):
            return False, False, "output contains NaN or Inf"
        error = self._relative_l2(output, reference)
        tolerance = {
            torch.float16: 2e-2,
            torch.bfloat16: 3e-2,
            torch.float32: 2e-4,
        }.get(self.dtype, 2e-2)
        if error > tolerance:
            return False, False, "relative L2 error %.3e > %.1e" % (error, tolerance)
        return True, bool(torch.equal(output, baseline)), None

    def _sync(self) -> None:
        if self.device.type == "cuda":
            self.torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            self.torch.mps.synchronize()

    def _flush_cache(self) -> None:
        if self._flush is not None:
            self._flush.zero_()
        self._sync()

    def _warm(self, cell: Cell, name: str, a: Any, b: Any) -> None:
        key = (cell.key, name)
        if key in self._warmed or self.warmup <= 0:
            return
        kernel = self._kernels[cell.key][name]
        with self.torch.inference_mode():
            for _ in range(self.warmup):
                kernel(a, b)
        self._sync()
        self._warmed.add(key)

    def _time(self, cell: Cell, name: str, a: Any, b: Any) -> float:
        kernel = self._kernels[cell.key][name]
        self._warm(cell, name, a, b)
        self._flush_cache()
        torch = self.torch
        if self.device.type == "cuda":
            begin = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            begin.record()
            with torch.inference_mode():
                for _ in range(self.replays):
                    kernel(a, b)
            end.record()
            end.synchronize()
            # elapsed_time is milliseconds.
            return max(1e-6, begin.elapsed_time(end) * 1000.0 / self.replays)
        with torch.inference_mode():
            self._sync()
            started = time.perf_counter()
            for _ in range(self.replays):
                kernel(a, b)
            self._sync()
            elapsed = time.perf_counter() - started
        return max(1e-6, elapsed * 1e6 / self.replays)


def make_adapter() -> TorchGPUAdapter:
    """Factory used by ``run --adapter benchmarks.ga.torch_gpu``."""

    return TorchGPUAdapter()


__all__ = [
    "TorchGPUAdapter",
    "make_adapter",
    "DEFAULT_SHAPES",
    "MUTANT_KINDS_2D",
    "MUTANT_KINDS_BATCHED",
]
