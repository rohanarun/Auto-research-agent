"""JSON-lines command adapter for the portable GA.

This adapter lets a non-Mac user keep the search loop in Python while the
target-specific build and timing code lives in any executable.  Each operation
starts one subprocess, writes one JSON request to its standard input, and reads
one JSON response from its standard output.

The contract is intentionally small.  A real evaluator can compile a candidate,
run paired timings, call a remote GPU queue, or shell into a Linux box; it must
only return the fields documented below.  For in-process GPU graphs (where
build handles cannot survive a process boundary), use a Python adapter module
instead of this command adapter.

Request/response examples are in ``benchmarks/ga/README.md``.
"""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set

from .engine import BuildResult, Cell, ConfirmResult, Gene, GAState


class CommandError(RuntimeError):
    """The evaluator command failed or returned an invalid response."""


def _cell_record(cell: Cell) -> Dict[str, Any]:
    return {
        "key": cell.key,
        "layer": cell.layer,
        "m": cell.m,
        "n": cell.n,
        "k": cell.k,
        "data": cell.data,
    }


def _gene_record(gene: Gene) -> Dict[str, Any]:
    return {
        "name": gene.name,
        "spec": gene.spec,
        "selectable": gene.selectable,
        "risky": gene.risky,
    }


class CommandAdapter:
    """Run a user-supplied evaluator command for every GA operation."""

    def __init__(
        self,
        command: Sequence[str],
        cells: Sequence[Dict[str, Any]],
        *,
        timeout_seconds: float = 600.0,
        cwd: Optional[Path] = None,
        seed: int = 1000,
    ) -> None:
        if not command:
            raise ValueError("command adapter needs a non-empty command")
        self.command = [str(part) for part in command]
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self.cwd = Path(cwd) if cwd is not None else None
        self.seed = int(seed)
        self._cells = [self._parse_cell(record) for record in cells]
        if not self._cells:
            raise ValueError("command adapter config has no cells")
        self._genes: Dict[str, List[Gene]] = {}
        self._mutants: Dict[str, List[Gene]] = {}
        for record, cell in zip(cells, self._cells):
            self._genes[cell.key] = [self._parse_gene(item) for item in (record.get("genes") or [])]
            self._mutants[cell.key] = [self._parse_gene(item) for item in (record.get("mutants") or [])]
            if not self._genes[cell.key]:
                raise ValueError(f"cell {cell.key!r} has no genes")
        self._cell_by_key = {cell.key: cell for cell in self._cells}
        self._calls = 0

    @classmethod
    def from_config(cls, path: Path, *, seed: int = 1000) -> "CommandAdapter":
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("command adapter config must be a JSON object")
        command = data.get("command")
        if isinstance(command, str):
            command = shlex.split(command)
        if not isinstance(command, list) or not command:
            raise ValueError("command adapter config needs a command array")
        cells = data.get("cells")
        if not isinstance(cells, list):
            raise ValueError("command adapter config needs a cells array")
        cwd = data.get("cwd")
        return cls(
            command,
            cells,
            timeout_seconds=float(data.get("timeout_seconds", 600)),
            cwd=(path.parent / cwd) if cwd else path.parent,
            seed=seed,
        )

    @staticmethod
    def _parse_cell(record: Dict[str, Any]) -> Cell:
        if not isinstance(record, dict):
            raise ValueError("each cell must be a JSON object")
        for required in ("key", "layer", "m"):
            if required not in record:
                raise ValueError(f"cell is missing {required!r}")
        return Cell(
            key=str(record["key"]),
            layer=str(record["layer"]),
            m=int(record["m"]),
            n=record.get("n"),
            k=record.get("k"),
            data=record.get("data"),
        )

    @staticmethod
    def _parse_gene(record: Dict[str, Any]) -> Gene:
        if not isinstance(record, dict) or not record.get("name"):
            raise ValueError("each gene needs a name")
        return Gene(
            name=str(record["name"]),
            spec=record.get("spec"),
            selectable=bool(record.get("selectable", True)),
            risky=bool(record.get("risky", False)),
        )

    # -- command transport ----------------------------------------------

    def _call(self, request: Dict[str, Any]) -> Dict[str, Any]:
        self._calls += 1
        request = dict(request)
        request.setdefault("seed", self.seed)
        request.setdefault("call", self._calls)
        try:
            completed = subprocess.run(
                self.command,
                input=json.dumps(request, sort_keys=True),
                capture_output=True,
                text=True,
                timeout=self.timeout_seconds,
                cwd=str(self.cwd) if self.cwd else None,
            )
        except (OSError, subprocess.SubprocessError) as exc:
            raise CommandError(f"evaluator command failed to start: {exc}") from exc
        if completed.returncode != 0:
            tail = (completed.stderr or completed.stdout or "").strip()[-2000:]
            raise CommandError(
                f"evaluator command exited {completed.returncode} for {request.get('op')}: {tail}"
            )
        text = (completed.stdout or "").strip()
        if not text:
            raise CommandError(f"evaluator command returned no JSON for {request.get('op')}")
        # Accept a trailing newline and nothing else.  Reject accidental logs on
        # stdout instead of trying to parse around them.
        try:
            response = json.loads(text)
        except ValueError as exc:
            raise CommandError(f"evaluator returned invalid JSON: {text[-1000:]}") from exc
        if not isinstance(response, dict):
            raise CommandError("evaluator response must be a JSON object")
        if response.get("error"):
            raise CommandError(str(response["error"]))
        return response

    # -- adapter protocol ------------------------------------------------

    def cells(self) -> List[Cell]:
        return list(self._cells)

    def initial_genes(self, cell: Cell) -> List[Gene]:
        return list(self._genes.get(cell.key, []))

    def mutants(
        self,
        cell: Cell,
        incumbent: str,
        known: Set[str],
        state: GAState,
        rng: Any,
    ) -> Iterable[Gene]:
        pool = [
            gene
            for gene in self._mutants.get(cell.key, [])
            if gene.name not in known and f"{cell.key}|{gene.name}" not in state.blacklist
        ]
        rng.shuffle(pool)
        return pool

    def build(self, cell: Cell, gene: Gene) -> BuildResult:
        response = self._call(
            {"op": "build", "cell": _cell_record(cell), "gene": _gene_record(gene)}
        )
        return BuildResult(
            ok=bool(response.get("ok", False)),
            bit=bool(response.get("bit", False)),
            error=response.get("error"),
        )

    def drop(self, cell: Cell, name: str) -> None:
        try:
            self._call({"op": "drop", "cell": _cell_record(cell), "name": str(name)})
        except CommandError:
            # Dropping is best-effort.  The engine already rejected the gene.
            return None

    def measure(self, cell: Cell, names: Sequence[str], blocks: int) -> Dict[str, List[float]]:
        response = self._call(
            {
                "op": "measure",
                "cell": _cell_record(cell),
                "names": [str(name) for name in names],
                "blocks": int(blocks),
            }
        )
        times = response.get("times")
        if not isinstance(times, dict):
            raise CommandError("measure response needs a 'times' object")
        out: Dict[str, List[float]] = {}
        for name in names:
            values = times.get(name)
            if not isinstance(values, list) or len(values) != int(blocks):
                raise CommandError(
                    f"measure response for {name!r} must contain exactly {blocks} blocks"
                )
            out[name] = [float(value) for value in values]
        return out

    def confirm(self, cell: Cell, champion: str, base: str, blocks: int) -> ConfirmResult:
        response = self._call(
            {
                "op": "confirm",
                "cell": _cell_record(cell),
                "champion": str(champion),
                "base": str(base),
                "blocks": int(blocks),
            }
        )
        try:
            return ConfirmResult(
                rel=float(response["rel"]),
                lo=float(response["lo"]),
                hi=float(response["hi"]),
                t_champion=float(response["t_champion"]),
                t_base=float(response["t_base"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CommandError(f"confirm response is missing required numeric fields: {exc}") from exc

    def close(self) -> None:
        return None
