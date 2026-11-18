"""Synthetic adapter used by the local GA demo and tests.

The timings in this module are **synthetic**.  They exist to exercise the race,
confirmation, mutation and persistence paths on a laptop or CI runner.  They are
not kernel measurements and must never be reported as performance results.

The adapter mirrors the Open Kernel Challenge gene vocabulary so the demo
exercises the same mutation shapes as the production loop:

* ``t10`` baseline and ``cublas`` anchor;
* ``main:<n>`` implementation candidates from the target's own catalogue;
* ``rk:S<n>`` split-count variants (router cells above 32 tokens);
* ``cg:nth<n>`` thread-count variants (single-token cells);
* ``t10int:<tag>`` integer perturbations of a plan tuple.
"""

from __future__ import annotations

import hashlib
import math
import random
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from .engine import BuildResult, Cell, ConfirmResult, Gene, GAState


def _stable_fraction(text: str, buckets: int = 1000) -> float:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return (int.from_bytes(digest[:4], "big") % buckets) / float(buckets)


class DemoAdapter:
    """A deterministic, noisy, CPU-only adapter with hidden gene costs."""

    def __init__(
        self,
        *,
        cells: Optional[int] = 6,
        seed: int = 7,
        noise: float = 0.006,
        layers: Sequence[str] = ("qkv_gate", "o", "pooler", "unpooler", "router"),
        ms: Sequence[int] = (1, 8, 16, 32, 128, 256),
    ) -> None:
        self.seed = int(seed)
        self.noise = max(0.0, float(noise))
        self._calls = 0
        pairs = [(layer, m) for m in ms for layer in layers]
        if cells is not None:
            pairs = pairs[: max(1, int(cells))]
        self._cells = [
            Cell(
                key="decode %d|%s|%d" % (m, layer, m),
                layer=layer,
                m=int(m),
                n=1024 if layer == "unpooler" and m <= 8 else 4096,
                k=8192,
            )
            for layer, m in pairs
        ]

    def cells(self) -> List[Cell]:
        return list(self._cells)

    # -- hidden synthetic cost model -------------------------------------

    def _base_us(self, cell: Cell) -> float:
        return 100.0 + 0.5 * float(cell.m)

    def _multiplier(self, cell: Cell, name: str) -> float:
        """Hidden true speed ratio for a gene.  Lower is faster."""

        if name == "t10":
            return 1.0
        if name == "cublas":
            return 1.03
        # Small cell-specific variation makes the demo less uniform without
        # hiding the synthetic nature of the numbers.
        jitter = (_stable_fraction(cell.key) - 0.5) * 0.0016
        if name.startswith("main:"):
            try:
                index = int(name.split(":", 1)[1])
            except ValueError:
                return 1.02
            return 1.0 - 0.0015 * index + jitter
        if name.startswith("rk:S"):
            try:
                splits = int(name.split("S", 1)[1])
            except ValueError:
                return 1.02
            return 1.0 - 0.004 * math.log(max(2, splits), 2) - 0.002 + jitter
        if name.startswith("cg:nth"):
            try:
                threads = int(name.split("nth", 1)[1])
            except ValueError:
                return 1.02
            return 1.0 - 0.004 * math.log(max(1.0, threads / 128.0), 2) - 0.002 + jitter
        if name.startswith("t10int:"):
            tag = name.split(":", 1)[1]
            # Deliberately below the 0.3% mutation gate for most cells: the demo
            # should show rejected noise as well as accepted mutants.
            amount = 0.001 + 0.0006 * (_stable_fraction(cell.key + tag, 10))
            return 1.0 - amount + jitter
        return 1.0 + jitter

    # -- adapter protocol ------------------------------------------------

    def initial_genes(self, cell: Cell) -> List[Gene]:
        genes = [
            Gene("t10", ["baseline"]),
            Gene("cublas", ["anchor"], selectable=False),
        ]
        for index in range(1, 5):
            genes.append(Gene("main:%d" % index, ["main", index]))
        return genes

    def mutants(
        self,
        cell: Cell,
        incumbent: str,
        known: Set[str],
        state: GAState,
        rng: random.Random,
    ) -> Iterable[Gene]:
        pool: List[Gene] = []
        if cell.layer == "router" and 32 < cell.m <= 256:
            for splits in (2, 3, 4, 6, 8, 10, 12, 16):
                name = "rk:S%d" % splits
                if name not in known:
                    pool.append(Gene(name, ["rk", splits], risky=True))
        if cell.m == 1:
            for threads in (128, 192, 256, 384, 512):
                name = "cg:nth%d" % threads
                if name not in known:
                    pool.append(Gene(name, ["cg", threads], risky=True))
        for offset in (-2, -1, 1, 2):
            name = "t10int:i%+d" % offset
            if name not in known:
                pool.append(Gene(name, ["t10plan", [offset]], risky=True))
        rng.shuffle(pool)
        return pool

    def build(self, cell: Cell, gene: Gene) -> BuildResult:
        if "bad" in gene.name:
            return BuildResult(False, False, "synthetic build rejection")
        return BuildResult(True, True, None)

    def drop(self, cell: Cell, name: str) -> None:
        return None

    def measure(self, cell: Cell, names: Sequence[str], blocks: int) -> Dict[str, List[float]]:
        self._calls += 1
        rng = random.Random(self.seed * 100003 + self._calls)
        out: Dict[str, List[float]] = {name: [] for name in names}
        base = self._base_us(cell)
        for _ in range(int(blocks)):
            shared = math.exp(rng.gauss(0.0, self.noise))
            for name in names:
                gene_noise = math.exp(rng.gauss(0.0, self.noise * 0.35))
                out[name].append(base * self._multiplier(cell, name) * shared * gene_noise)
        return out

    def confirm(self, cell: Cell, champion: str, base: str, blocks: int) -> ConfirmResult:
        times = self.measure(cell, [champion, base], int(blocks))
        ratios = [
            math.log(float(candidate) / float(reference))
            for candidate, reference in zip(times[champion], times[base])
        ]
        mean = sum(ratios) / len(ratios)
        error = 0.0
        if len(ratios) > 1:
            import statistics

            error = statistics.stdev(ratios) / math.sqrt(len(ratios))
        return ConfirmResult(
            rel=math.exp(mean),
            lo=math.exp(mean - 1.96 * error),
            hi=math.exp(mean + 1.96 * error),
            t_champion=float(sorted(times[champion])[len(times[champion]) // 2]),
            t_base=float(sorted(times[base])[len(times[base]) // 2]),
        )

    def close(self) -> None:
        return None
