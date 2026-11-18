"""Dependency-free racing genetic optimizer for per-cell kernel genes.

Ported from the per-cell racing loop used for Open Kernel Challenge work.  The
port keeps the statistical decisions that matter:

* generation 0 races every gene per cell with successive elimination on paired
  log-time ratios, then confirms the winner on a fresh input;
* later generations try random mutants per cell and replace the incumbent only
  when the paired measurement is significant and a fresh-input confirmation
  agrees;
* risky genes are persisted as ``pending`` before they are built, so a process
  that dies while launching one can blacklist exactly that gene and resume;
* a champion is only replaced by a confirmed improvement, so the reported gain
  cannot fall.

The evaluator is supplied by an adapter.  See ``benchmarks/ga/README.md`` for
the protocol and ``benchmarks/ga/demo.py`` for a complete synthetic adapter.
This module uses only the Python standard library.
"""

from __future__ import annotations

import json
import math
import os
import random
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

STATE_SCHEMA = "auto-research-ga-state-v1"

# Confirmation gate from the original loop: a fresh-input champion/base ratio
# must be below 0.997 and the 95% upper confidence bound must be below 1.0.
CONFIRM_REL = 0.997
CONFIRM_CI_UPPER = 1.0

# Hill-climb gate: paired log-time improvement must exceed
# max(2.5 * SE, 0.3%).
MUTATION_Z = 2.5
MUTATION_MIN_EFFECT = 0.003

# Generation-0 elimination gate.
RACE_Z = 3.0
RACE_TIE = 0.002

# State compaction limits used by the original loop when it emits a checkpoint.
BLACKLIST_KEEP = 200
HISTORY_KEEP = 300


@dataclass(frozen=True)
class Gene:
    """One implementation choice for one cell.

    ``spec`` is JSON-serialisable and is opaque to the engine.  ``selectable``
    is false for anchors such as cuBLAS that may be raced but must never become
    a champion.  ``risky`` genes are persisted as pending before their build so
    a sticky CUDA or process fault can be recovered.
    """

    name: str
    spec: Any
    selectable: bool = True
    risky: bool = False


@dataclass
class Cell:
    """One independently hill-climbed region of a kernel."""

    key: str
    layer: str
    m: int
    n: Optional[int] = None
    k: Optional[int] = None
    data: Any = None


@dataclass(frozen=True)
class BuildResult:
    ok: bool
    bit: bool = False
    error: Optional[str] = None


@dataclass(frozen=True)
class ConfirmResult:
    rel: float
    lo: float
    hi: float
    t_champion: float
    t_base: float


class StickyFault(RuntimeError):
    """A required build faulted in a way that should blacklist one gene.

    A real CUDA launch fault can poison the process.  Adapters should raise this
    instead of a normal exception when the only safe recovery is a fresh
    process.  The engine leaves the gene id in ``state.pending`` and re-raises;
    the driver blacklists it before restarting.
    """

    def __init__(self, gene_id: str, message: str = "sticky gene fault"):
        super().__init__(message)
        self.gene_id = gene_id


@dataclass
class GAState:
    path: Path
    generation: int = 0
    pending: Optional[str] = None
    blacklist: List[str] = field(default_factory=list)
    champ: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    cells: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    history: List[Dict[str, Any]] = field(default_factory=list)

    @classmethod
    def load(cls, path: os.PathLike) -> "GAState":
        path = Path(path)
        if not path.exists():
            return cls(path=path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise ValueError(f"Cannot read GA state {path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"GA state {path} must contain a JSON object")
        schema = data.get("schema")
        if schema not in (None, STATE_SCHEMA):
            raise ValueError(f"Unsupported GA state schema {schema!r} in {path}")
        champ = data.get("champ") or {}
        cells = data.get("cells") or {}
        history = data.get("hist") or data.get("history") or []
        if not isinstance(champ, dict) or not isinstance(cells, dict) or not isinstance(history, list):
            raise ValueError(f"GA state {path} has invalid champ/cells/history fields")
        return cls(
            path=path,
            generation=int(data.get("generation", data.get("gen", 0)) or 0),
            pending=data.get("pending"),
            blacklist=list(data.get("blacklist") or []),
            champ=dict(champ),
            cells=dict(cells),
            history=list(history),
        )

    def to_record(self, compact: bool = True) -> Dict[str, Any]:
        return {
            "schema": STATE_SCHEMA,
            "generation": self.generation,
            "pending": self.pending,
            "blacklist": list(self.blacklist[-BLACKLIST_KEEP:] if compact else self.blacklist),
            "champ": self.champ,
            "cells": self.cells,
            "hist": list(self.history[-HISTORY_KEEP:] if compact else self.history),
        }

    def save(self) -> None:
        """Atomically replace the state file.

        The temporary file is created beside the destination on the same
        filesystem, flushed, then renamed.  A reader never observes a partial
        JSON document.
        """

        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_record(compact=False), indent=2, sort_keys=True) + "\n"
        tmp = self.path.with_name(self.path.name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, self.path)


def recover_pending(path: os.PathLike) -> Optional[str]:
    """Blacklist the gene that was pending when a process stopped.

    Returns the recovered gene id, or ``None`` when there was no pending gene.
    This is the operation the restart driver performs between child processes.
    """

    state = GAState.load(path)
    if not state.pending:
        return None
    gene_id = str(state.pending)
    if gene_id not in state.blacklist:
        state.blacklist.append(gene_id)
    state.pending = None
    state.save()
    return gene_id


def measured_gain(state: GAState) -> float:
    """Weighted measured gain over all cells that have a champion."""

    base = 0.0
    current = 0.0
    for record in state.champ.values():
        if not isinstance(record, dict):
            continue
        try:
            t10 = float(record.get("t10_us") or 0.0)
            rel = float(record.get("rel") or 0.0)
        except (TypeError, ValueError):
            continue
        if t10 > 0 and rel > 0:
            base += t10
            current += t10 * rel
    return 0.0 if base <= 0 else max(-1.0, 1.0 - current / base)


def paired_log_difference(candidate: Sequence[float], baseline: Sequence[float]) -> Tuple[float, float]:
    """Mean and standard error of paired log(candidate / baseline) blocks."""

    if len(candidate) != len(baseline):
        raise ValueError("paired timing arrays must have the same length")
    if len(candidate) < 2:
        raise ValueError("at least two paired timing blocks are required")
    diffs: List[float] = []
    for left, right in zip(candidate, baseline):
        left = float(left)
        right = float(right)
        if left <= 0 or right <= 0:
            raise ValueError("paired timings must be positive")
        diffs.append(math.log(left / right))
    mean = sum(diffs) / len(diffs)
    se = statistics.stdev(diffs) / math.sqrt(len(diffs)) if len(diffs) > 1 else 0.0
    return mean, se


def confirmation_passes(
    result: ConfirmResult,
    rel_threshold: float = CONFIRM_REL,
    ci_upper: float = CONFIRM_CI_UPPER,
) -> bool:
    """The fresh-input replication gate from the original loop."""

    try:
        return (
            math.isfinite(float(result.rel))
            and math.isfinite(float(result.hi))
            and float(result.rel) < rel_threshold
            and float(result.hi) < ci_upper
        )
    except (TypeError, ValueError):
        return False


def successive_elimination(
    adapter: Any,
    cell: Cell,
    names: Sequence[str],
    *,
    per_round: int = 100,
    rounds: int = 10,
    z: float = RACE_Z,
    tie: float = RACE_TIE,
    log: Callable[[str], None] = lambda _message: None,
) -> Tuple[List[str], Dict[str, float]]:
    """Race genes with successive elimination on paired log-time ratios.

    Every round measures all surviving genes in random order (the adapter owns
    the actual randomisation and cache flush).  For each block the reference is
    the median across survivors, so shared machine noise cancels.  A gene stays
    alive when it is not significantly worse than the current best:
    ``mean(log(gene / best)) < max(z * SE, tie)``.  The returned order is by
    mean log ratio, fastest first, and the mapping contains the accumulated
    means for every measured name.
    """

    ordered = list(dict.fromkeys(str(name) for name in names))
    if not ordered:
        raise ValueError("a race needs at least one gene")
    alive = list(ordered)
    accumulated: Dict[str, List[float]] = {name: [] for name in ordered}

    for round_index in range(max(1, int(rounds))):
        if len(alive) <= 1:
            break
        times = adapter.measure(cell, alive, int(per_round))
        if not isinstance(times, dict):
            raise ValueError("adapter.measure must return a mapping of gene name to timing blocks")
        lengths = set()
        for name in alive:
            blocks = times.get(name)
            if not isinstance(blocks, (list, tuple)):
                raise ValueError(f"adapter.measure did not return timing blocks for {name!r}")
            if len(blocks) < 2:
                raise ValueError(f"adapter.measure returned fewer than two blocks for {name!r}")
            lengths.add(len(blocks))
            for value in blocks:
                if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or float(value) <= 0:
                    raise ValueError(f"adapter.measure returned an invalid timing for {name!r}")
        if len(lengths) != 1:
            raise ValueError("adapter.measure returned different block counts per gene")
        block_count = lengths.pop()

        for block in range(block_count):
            reference = statistics.median(float(times[name][block]) for name in alive)
            if reference <= 0:
                raise ValueError("adapter.measure produced a non-positive block reference")
            for name in alive:
                accumulated[name].append(math.log(float(times[name][block]) / reference))

        means = {name: sum(accumulated[name]) / len(accumulated[name]) for name in alive}
        best = min(alive, key=lambda name: means[name])
        keep = [best]
        for name in alive:
            if name == best:
                continue
            diffs = [left - right for left, right in zip(accumulated[name], accumulated[best])]
            if len(diffs) < 2:
                difference, error = 0.0, 0.0
            else:
                difference = sum(diffs) / len(diffs)
                error = statistics.stdev(diffs) / math.sqrt(len(diffs))
            if difference < max(float(z) * error, float(tie)):
                keep.append(name)
        alive = keep
        log(
            "race round %d: %d alive, best %s (mean log ratio %+.5f)"
            % (round_index + 1, len(alive), best, means[best])
        )

    means = {
        name: (sum(accumulated[name]) / len(accumulated[name])) if accumulated[name] else 0.0
        for name in alive
    }
    return sorted(alive, key=lambda name: means[name]), means


class GeneticOptimizer:
    """Generation-0 race plus fresh-input-confirmed hill climbing.

    ``adapter`` must implement the methods documented in ``benchmarks/ga/README.md``.
    The engine does not import the target kernel; all target-specific work is in
    the adapter.  ``baseline_name`` must name the baseline gene (``t10`` in the
    original loop).
    """

    def __init__(
        self,
        adapter: Any,
        state: GAState,
        *,
        seed: int = 1000,
        baseline_name: str = "t10",
        per_round: int = 100,
        race_rounds: int = 10,
        race_z: float = RACE_Z,
        race_tie: float = RACE_TIE,
        mutation_blocks: int = 150,
        mutation_z: float = MUTATION_Z,
        mutation_min_effect: float = MUTATION_MIN_EFFECT,
        mutation_attempts: int = 3,
        confirm_blocks: int = 400,
        confirm_short_blocks: int = 50,
        confirm_rel: float = CONFIRM_REL,
        confirm_ci_upper: float = CONFIRM_CI_UPPER,
        det_only: bool = False,
        rerace: bool = False,
        seconds: Optional[float] = None,
        sleep_when_idle: float = 5.0,
        log: Callable[[str], None] = print,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.adapter = adapter
        self.state = state
        self.rng = random.Random(int(seed))
        self.baseline_name = str(baseline_name)
        self.per_round = max(2, int(per_round))
        self.race_rounds = max(1, int(race_rounds))
        self.race_z = float(race_z)
        self.race_tie = float(race_tie)
        self.mutation_blocks = max(2, int(mutation_blocks))
        self.mutation_z = float(mutation_z)
        self.mutation_min_effect = float(mutation_min_effect)
        self.mutation_attempts = max(1, int(mutation_attempts))
        self.confirm_blocks = max(2, int(confirm_blocks))
        self.confirm_short_blocks = max(2, int(confirm_short_blocks))
        self.confirm_rel = float(confirm_rel)
        self.confirm_ci_upper = float(confirm_ci_upper)
        self.det_only = bool(det_only)
        self.rerace = bool(rerace)
        self.sleep_when_idle = max(0.0, float(sleep_when_idle))
        self.log = log
        self.clock = clock
        self.started_at = clock()
        self.deadline = self.started_at + float(seconds) if seconds is not None else None
        self.cells: List[Cell] = list(adapter.cells())
        if not self.cells:
            raise ValueError("adapter returned no cells")
        keys = [cell.key for cell in self.cells]
        if len(set(keys)) != len(keys):
            raise ValueError("adapter returned duplicate cell keys")
        self._genes: Dict[str, Dict[str, Gene]] = {cell.key: {} for cell in self.cells}
        self._builds: Dict[str, Dict[str, BuildResult]] = {cell.key: {} for cell in self.cells}

    # -- bookkeeping -----------------------------------------------------

    @property
    def elapsed(self) -> float:
        return self.clock() - self.started_at

    def expired(self) -> bool:
        return self.deadline is not None and self.clock() >= self.deadline

    def _cell_for(self, key: str) -> Cell:
        for cell in self.cells:
            if cell.key == key:
                return cell
        raise KeyError(key)

    def _gene_id(self, cell: Cell, name: str) -> str:
        return f"{cell.key}|{name}"

    def gene(self, cell: Cell, name: str) -> Optional[Gene]:
        return self._genes.get(cell.key, {}).get(name)

    def build_result(self, cell: Cell, name: str) -> Optional[BuildResult]:
        return self._builds.get(cell.key, {}).get(name)

    @staticmethod
    def _coerce_gene(gene: Any) -> Gene:
        """Accept a Gene from an adapter that imported this file under another name.

        Python can load the same engine as both ``ga.engine`` and
        ``benchmarks.ga.engine``.  Adapter objects must not be rejected only
        because their dataclass identity differs.
        """

        if isinstance(gene, Gene):
            return gene
        if not hasattr(gene, "name") or not hasattr(gene, "spec"):
            raise ValueError("adapter yielded something that is not a Gene")
        return Gene(
            name=str(gene.name),
            spec=gene.spec,
            selectable=bool(getattr(gene, "selectable", True)),
            risky=bool(getattr(gene, "risky", False)),
        )

    @staticmethod
    def _coerce_build_result(result: Any) -> BuildResult:
        if isinstance(result, BuildResult):
            return result
        if hasattr(result, "ok"):
            return BuildResult(
                ok=bool(getattr(result, "ok")),
                bit=bool(getattr(result, "bit", False)),
                error=getattr(result, "error", None),
            )
        return BuildResult(bool(result), bool(result), None)

    def eligible(self, cell: Cell) -> List[str]:
        names: List[str] = []
        for name, result in self._builds.get(cell.key, {}).items():
            gene = self._genes[cell.key][name]
            if result.ok and gene.selectable and (result.bit or not self.det_only):
                names.append(name)
        return names

    def add_gene(self, cell: Cell, gene: Gene) -> Optional[BuildResult]:
        """Build and register one gene.  Returns ``None`` when blacklisted."""

        gene = self._coerce_gene(gene)
        gene_id = self._gene_id(cell, gene.name)
        if gene_id in self.state.blacklist:
            return None
        existing = self.build_result(cell, gene.name)
        if existing is not None:
            return existing
        self._genes[cell.key][gene.name] = gene
        if gene.risky:
            self.state.pending = gene_id
            self.state.save()
        try:
            result = self._coerce_build_result(self.adapter.build(cell, gene))
        except StickyFault:
            # Keep pending set: the driver blacklists this exact gene and starts
            # a fresh process.  Saving here is what makes the recovery exact.
            self.state.save()
            raise
        except Exception as exc:
            # An adapter loaded under a second package name may have its own
            # StickyFault class.  Match the documented shape as well as identity.
            if exc.__class__.__name__ == "StickyFault" and getattr(exc, "gene_id", None):
                self.state.save()
                raise StickyFault(str(exc.gene_id), str(exc)) from exc
            # A normal bad gene must not kill the race.
            result = BuildResult(False, False, repr(exc)[:300])
        if gene.risky:
            self.state.pending = None
            self.state.save()
        self._builds[cell.key][gene.name] = result
        if not result.ok:
            self.log("gene failed: %s (%s)" % (gene_id, result.error or "unknown error"))
        return result

    def drop_gene(self, cell: Cell, name: str) -> None:
        try:
            self.adapter.drop(cell, name)
        except Exception:
            pass
        self._builds.get(cell.key, {}).pop(name, None)
        self._genes.get(cell.key, {}).pop(name, None)

    def _confirm(self, cell: Cell, champion: str, base: str, blocks: int) -> ConfirmResult:
        result = self.adapter.confirm(cell, champion, base, int(blocks))
        if not isinstance(result, ConfirmResult):
            if isinstance(result, dict):
                result = ConfirmResult(
                    rel=float(result["rel"]),
                    lo=float(result.get("lo", result["rel"])),
                    hi=float(result.get("hi", 1.0)),
                    t_champion=float(result.get("t_champion", result.get("t_champ", 0.0))),
                    t_base=float(result.get("t_base", 0.0)),
                )
            elif all(hasattr(result, field) for field in ("rel", "lo", "hi", "t_base")):
                result = ConfirmResult(
                    rel=float(getattr(result, "rel")),
                    lo=float(getattr(result, "lo")),
                    hi=float(getattr(result, "hi")),
                    t_champion=float(getattr(result, "t_champion", getattr(result, "t_champ", 0.0))),
                    t_base=float(getattr(result, "t_base")),
                )
            else:
                raise ValueError("adapter.confirm must return ConfirmResult or a mapping")
        if result.t_base <= 0:
            raise ValueError("adapter.confirm returned a non-positive baseline time")
        return result

    # -- generation 0 ----------------------------------------------------

    def _restore_champion(self, cell: Cell) -> Optional[Dict[str, Any]]:
        record = self.state.champ.get(cell.key)
        if not isinstance(record, dict):
            return None
        name = record.get("gene")
        if not isinstance(name, str) or not name:
            return None
        if self._gene_id(cell, name) in self.state.blacklist:
            return None
        try:
            t10 = float(record.get("t10_us"))
            rel = float(record.get("rel"))
        except (TypeError, ValueError):
            return None
        if t10 <= 0 or rel <= 0:
            return None
        gene = self.gene(cell, name)
        if gene is None:
            gene = Gene(name=name, spec=record.get("spec"), risky=True)
            result = self.add_gene(cell, gene)
            if result is None or not result.ok:
                return None
        result = self.build_result(cell, name)
        if result is None or not result.ok or (self.det_only and not result.bit):
            return None
        if not self._genes[cell.key][name].selectable:
            return None
        return {
            "gene": name,
            "spec": self._genes[cell.key][name].spec,
            "rel": rel,
            "t10_us": t10,
            "det": bool(result.bit),
        }

    def run_generation_zero(self) -> bool:
        """Build genes and race every cell.  Returns False when the budget ends early."""

        self.log("gen 0: building initial genes for %d cells" % len(self.cells))
        kept = 0
        raced = 0
        for cell in self.cells:
            if self.expired():
                self.log("gen 0 paused at %d/%d cells (budget reached)" % (raced + kept, len(self.cells)))
                self.state.save()
                return False
            for gene in self.adapter.initial_genes(cell):
                self.add_gene(cell, gene)
            restored = None if self.rerace else self._restore_champion(cell)
            if restored is not None:
                self.state.champ[cell.key] = restored
                kept += 1
                self.log("%-30s kept champion %-34s rel %.3f" % (cell.key, restored["gene"], restored["rel"]))
                self.state.save()
                continue

            selectable = self.eligible(cell)
            anchors = [
                name
                for name, gene in self._genes[cell.key].items()
                if not gene.selectable and self.build_result(cell, name) is not None
                and self.build_result(cell, name).ok
            ]
            names = list(dict.fromkeys(selectable + anchors))
            if not names:
                raise RuntimeError(f"no usable genes for cell {cell.key}")
            if self.baseline_name not in names:
                raise RuntimeError(
                    f"cell {cell.key} has no built baseline gene {self.baseline_name!r}; "
                    "the adapter's initial_genes must include it"
                )
            order, means = successive_elimination(
                self.adapter,
                cell,
                names,
                per_round=self.per_round,
                rounds=self.race_rounds,
                z=self.race_z,
                tie=self.race_tie,
                log=self.log,
            )
            selectable_order = [name for name in order if self._genes[cell.key][name].selectable]
            top = selectable_order[:2]
            if not top:
                raise RuntimeError(f"cell {cell.key} has no selectable genes after the race")
            champion = top[0]
            if champion != self.baseline_name:
                confirmed = self._confirm(cell, champion, self.baseline_name, self.confirm_blocks)
                if not confirmation_passes(confirmed, self.confirm_rel, self.confirm_ci_upper):
                    champion = self.baseline_name
                    confirmed = self._confirm(cell, champion, champion, self.confirm_short_blocks)
            else:
                confirmed = self._confirm(cell, champion, champion, self.confirm_short_blocks)
            rel = confirmed.rel if champion != self.baseline_name else 1.0
            # Never let a numerically odd confirmation raise the stored champion.
            rel = min(1.0, max(1e-9, float(rel)))
            self.state.champ[cell.key] = {
                "gene": champion,
                "spec": self._genes[cell.key][champion].spec,
                "rel": rel,
                "t10_us": float(confirmed.t_base),
                "det": bool(self.build_result(cell, champion).bit),
            }
            top_table = []
            best_mean = means.get(order[0], 0.0)
            for name in order[:6]:
                if name in means:
                    top_table.append(
                        [
                            name,
                            round(float(math.exp(means[name] - best_mean)), 4),
                            bool(self.build_result(cell, name).bit) if self.build_result(cell, name) else False,
                        ]
                    )
            self.state.cells[cell.key] = {
                "n_genes": len(self._genes[cell.key]),
                "n_fail": sum(1 for value in self._builds[cell.key].values() if not value.ok),
                "top": top_table,
            }
            raced += 1
            self.log(
                "%-30s t10 %6.2fus  champion %-34s rel %.3f [%.3f, %.3f] det=%s"
                % (
                    cell.key,
                    confirmed.t_base,
                    champion,
                    rel,
                    confirmed.lo,
                    confirmed.hi,
                    self.build_result(cell, champion).bit,
                )
            )
            self.state.save()

        if any(cell.key not in self.state.champ for cell in self.cells):
            self.log("gen 0 incomplete; rerun to finish the remaining cells")
            self.state.save()
            return False

        self.state.generation = 1
        gain = measured_gain(self.state)
        self.state.history.append(
            {
                "t": time.time(),
                "gen": 0,
                "tried": 0,
                "gain_meas": gain,
                "improved": [],
                "kept": kept,
                "raced": raced,
            }
        )
        self.state.save()
        self.log(
            "gen 0 complete: %d kept, %d raced, measured gain %.2f%%"
            % (kept, raced, gain * 100.0)
        )
        return True

    # -- hill climbing ---------------------------------------------------

    def run_hillclimb_generation(self, generation: int) -> Tuple[List[List[Any]], int]:
        """Run one hill-climb generation and return ``(improvements, tried)``."""

        improved: List[List[Any]] = []
        tried = 0
        for cell in self.cells:
            if self.expired():
                break
            record = self.state.champ.get(cell.key)
            if not isinstance(record, dict):
                continue
            incumbent = record.get("gene")
            if not isinstance(incumbent, str):
                continue
            incumbent_build = self.build_result(cell, incumbent)
            if incumbent_build is None or not incumbent_build.ok:
                continue
            proposals = list(
                self.adapter.mutants(
                    cell,
                    incumbent,
                    set(self._genes[cell.key]),
                    self.state,
                    self.rng,
                )
            )
            attempts = 0
            for gene in proposals:
                if attempts >= self.mutation_attempts:
                    break
                attempts += 1
                gene = self._coerce_gene(gene)
                gene_id = self._gene_id(cell, gene.name)
                if gene.name in self._genes[cell.key] or gene_id in self.state.blacklist:
                    continue
                result = self.add_gene(cell, gene)
                if result is None or not result.ok:
                    continue
                tried += 1
                if self.det_only and not result.bit:
                    self.drop_gene(cell, gene.name)
                    continue
                times = self.adapter.measure(cell, [incumbent, gene.name], self.mutation_blocks)
                try:
                    mean, error = paired_log_difference(times[gene.name], times[incumbent])
                except (KeyError, TypeError, ValueError) as exc:
                    self.log("mutant %s measurement rejected: %s" % (gene_id, exc))
                    self.drop_gene(cell, gene.name)
                    continue
                threshold = max(self.mutation_z * error, self.mutation_min_effect)
                if mean < -threshold:
                    confirmed = self._confirm(cell, gene.name, incumbent, self.confirm_blocks)
                    if confirmation_passes(confirmed, self.confirm_rel, self.confirm_ci_upper):
                        previous = self.state.champ[cell.key]
                        new_rel = float(previous["rel"]) * float(confirmed.rel)
                        # A confirmed improvement must lower the stored ratio.
                        new_rel = min(float(previous["rel"]), max(1e-9, new_rel))
                        self.state.champ[cell.key] = {
                            "gene": gene.name,
                            "spec": gene.spec,
                            "rel": new_rel,
                            "t10_us": float(previous["t10_us"]),
                            "det": bool(result.bit),
                        }
                        incumbent = gene.name
                        improved.append([cell.key, gene.name, round(float(confirmed.rel), 4)])
                        self.log(
                            "%-30s accepted %-34s rel %.4f (stored %.4f)"
                            % (cell.key, gene.name, confirmed.rel, new_rel)
                        )
                        self.state.save()
                        continue
                self.drop_gene(cell, gene.name)
            self.state.save()

        self.state.generation = int(generation) + 1
        previous_gain = float(self.state.history[-1].get("gain_meas", 0.0)) if self.state.history else 0.0
        gain = measured_gain(self.state)
        # Confirmed replacements are strict improvements, so gain should rise.
        # Keep the published series monotonic even if a saved state was rounded.
        if gain + 1e-12 < previous_gain:
            self.log("warning: measured gain %.8f below previous %.8f; keeping the higher series value" % (gain, previous_gain))
            gain = previous_gain
        self.state.history.append(
            {
                "t": time.time(),
                "gen": int(generation),
                "tried": tried,
                "gain_meas": gain,
                "improved": improved,
            }
        )
        self.state.save()
        self.log(
            "gen %d: tried %d, improved %d %s, measured gain %.2f%%"
            % (generation, tried, len(improved), improved[:4], gain * 100.0)
        )
        return improved, tried

    def run(
        self,
        *,
        generations: Optional[int] = None,
        on_generation: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> GAState:
        """Run generation 0 if needed, then up to ``generations`` hill climbs."""

        if self.state.generation == 0 or not self.state.champ:
            completed = self.run_generation_zero()
            if on_generation and self.state.history:
                on_generation(dict(self.state.history[-1]))
            if not completed:
                return self.state

        remaining = generations
        while True:
            if remaining is not None and remaining <= 0:
                break
            if self.expired():
                self.log("budget reached at generation %d" % self.state.generation)
                break
            generation = int(self.state.generation)
            _improved, tried = self.run_hillclimb_generation(generation)
            if on_generation and self.state.history:
                on_generation(dict(self.state.history[-1]))
            if remaining is not None:
                remaining -= 1
            if tried == 0:
                if self.sleep_when_idle:
                    time.sleep(self.sleep_when_idle)
                # A generation with no proposals means the adapter exhausted its
                # finite mutant pool.  Do not spin forever.
                if remaining is None:
                    break
        self.state.save()
        return self.state

    def summary(self) -> Dict[str, Any]:
        gain = measured_gain(self.state)
        return {
            "schema": STATE_SCHEMA,
            "generation": self.state.generation,
            "gain": gain,
            "cells": len(self.cells),
            "championed": len(self.state.champ),
            "champions": self.state.champ,
        }
