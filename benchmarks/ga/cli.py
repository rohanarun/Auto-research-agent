"""Command line interface for the portable racing GA.

Examples::

    python3 -m benchmarks.ga demo
    python3 -m benchmarks.ga gpu-demo --device auto
    python3 -m benchmarks.ga run --config ga.json --state tmp/ga/state.json
    python3 -m benchmarks.ga run --adapter my_adapter:make_adapter --state tmp/ga/state.json
    python3 -m benchmarks.ga status --state tmp/ga/state.json
    python3 -m benchmarks.ga export --state tmp/ga/state.json --output tmp/ga/best.json
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

_BENCHMARKS = Path(__file__).resolve().parents[1]
_REPO_ROOT = _BENCHMARKS.parent
for _import_root in (str(_REPO_ROOT), str(_BENCHMARKS)):
    if _import_root not in sys.path:
        sys.path.insert(0, _import_root)

from .command import CommandAdapter
from .demo import DemoAdapter
from .engine import (
    GAState,
    GeneticOptimizer,
    STATE_SCHEMA,
    StickyFault,
    measured_gain,
    recover_pending,
)


def _summary(state: GAState, cells: int) -> Dict[str, Any]:
    base = 0.0
    current = 0.0
    for record in state.champ.values():
        try:
            t10 = float(record.get("t10_us") or 0.0)
            rel = float(record.get("rel") or 0.0)
        except (TypeError, ValueError):
            continue
        if t10 > 0 and rel > 0:
            base += t10
            current += t10 * rel
    return {
        "schema": STATE_SCHEMA,
        "generation": state.generation,
        "gain": measured_gain(state),
        "gain_series": [
            {"gen": entry.get("gen"), "gain": entry.get("gain_meas")}
            for entry in state.history
            if isinstance(entry, dict)
        ],
        "cells": cells,
        "championed": len(state.champ),
        "baseline_total_us": base,
        "champion_total_us": current,
        "champions": state.champ,
    }


def _write_json(path: Path, value: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _print_summary(summary: Dict[str, Any], stream: Any = None) -> None:
    stream = sys.stdout if stream is None else stream
    print(
        "generation %(generation)s, %(championed)s/%(cells)s cells championed, "
        "measured gain %(gain).2f%%" % {**summary, "gain": summary["gain"] * 100.0},
        file=stream,
    )
    print("%-32s %-30s %8s %10s" % ("cell", "champion", "rel", "t10_us"), file=stream)
    for key in sorted(summary["champions"]):
        record = summary["champions"][key]
        print(
            "%-32s %-30s %8.4f %10.2f"
            % (key, record.get("gene", "?"), float(record.get("rel", 1.0)), float(record.get("t10_us", 0.0))),
            file=stream,
        )


def _split_adapter_spec(spec: str) -> Any:
    """Split ``module:factory`` without breaking a Windows drive path."""

    if ":" in spec:
        module_name, factory_name = spec.rsplit(":", 1)
        if module_name and factory_name.isidentifier():
            return module_name, factory_name
    return spec, "make_adapter"


def _load_adapter(spec: str) -> Any:
    module_name, factory_name = _split_adapter_spec(spec)
    if not module_name:
        raise ValueError("--adapter must be module:factory or path/to/file.py:factory")
    if module_name.endswith(".py") or "/" in module_name or "\\" in module_name:
        path = Path(module_name).resolve()
        if not path.is_file():
            raise ValueError(f"adapter file does not exist: {path}")
        unique = "ga_adapter_%x" % (abs(hash(str(path))) & 0xFFFFFFFF)
        module_spec = importlib.util.spec_from_file_location(unique, path)
        if module_spec is None or module_spec.loader is None:
            raise ValueError(f"cannot load adapter module from {path}")
        module = importlib.util.module_from_spec(module_spec)
        sys.modules[unique] = module
        module_spec.loader.exec_module(module)
    else:
        module = importlib.import_module(module_name)
    factory = getattr(module, factory_name, None)
    if factory is None or not callable(factory):
        raise ValueError(f"adapter module {module_name!r} has no callable {factory_name!r}")
    return factory()


def _add_common_run_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--state", required=True, help="JSON state file; created if absent")
    parser.add_argument("--seconds", type=float, default=1800.0, help="wall-clock budget (0 means no budget)")
    parser.add_argument("--generations", type=int, default=None, help="stop after this many hill-climb generations")
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--baseline", default="t10", help="name of the baseline gene in every cell")
    parser.add_argument("--det-only", action="store_true", help="only accept bit-reproducible builds")
    parser.add_argument("--rerace", action="store_true", help="re-race cells instead of keeping saved champions")
    parser.add_argument("--per-round", type=int, default=100)
    parser.add_argument("--race-rounds", type=int, default=10)
    parser.add_argument("--mutation-attempts", type=int, default=3)
    parser.add_argument("--mutation-blocks", type=int, default=150)
    parser.add_argument("--confirm-blocks", type=int, default=400)
    parser.add_argument("--sleep-when-idle", type=float, default=5.0)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--json", action="store_true", help="print the final summary as JSON")


def _cmd_demo(args: argparse.Namespace) -> int:
    state_path = Path(args.state).expanduser() if args.state else Path.cwd() / "tmp" / "ga-demo" / "state.json"
    if args.fresh:
        for suffix in ("", ".tmp"):
            try:
                Path(str(state_path) + suffix).unlink()
            except FileNotFoundError:
                pass
    adapter = DemoAdapter(cells=args.cells, seed=args.seed, noise=args.noise)
    state = GAState.load(state_path)
    print("Synthetic demo: timings are fabricated to exercise the loop; they are not kernel measurements.", flush=True)
    print("State: %s" % state_path, flush=True)
    started = time.monotonic()

    def log(message: str) -> None:
        if not args.quiet:
            print("[ga %6.1fs] %s" % (time.monotonic() - started, message), flush=True)

    engine = GeneticOptimizer(
        adapter,
        state,
        seed=args.seed,
        baseline_name="t10",
        per_round=args.per_round,
        race_rounds=args.race_rounds,
        mutation_blocks=args.mutation_blocks,
        mutation_attempts=args.mutation_attempts,
        confirm_blocks=args.confirm_blocks,
        det_only=args.det_only,
        rerace=args.rerace,
        seconds=None if args.seconds <= 0 else args.seconds,
        sleep_when_idle=0.0,
        log=log,
    )
    try:
        engine.run(generations=args.generations)
    finally:
        adapter.close()
    summary = _summary(state, len(engine.cells))
    _write_json(state_path.parent / "summary.json", summary)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _print_summary(summary)
    print("Synthetic demo complete. This establishes only that the loop ran; it is not a performance claim.", flush=True)
    return 0


def _cmd_gpu_demo(args: argparse.Namespace) -> int:
    from .torch_gpu import TorchGPUAdapter

    state_path = Path(args.state).expanduser() if args.state else Path.cwd() / "tmp" / "ga-gpu" / "state.json"
    if args.fresh:
        for suffix in ("", ".tmp"):
            try:
                Path(str(state_path) + suffix).unlink()
            except FileNotFoundError:
                pass
    try:
        adapter = TorchGPUAdapter(
            device=args.device,
            dtype=args.dtype,
            cells=args.cells,
            seed=args.seed,
            replays=args.replays,
            warmup=args.warmup,
            flush_mb=args.flush_mb,
        )
    except RuntimeError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    info = adapter.info()
    if info["is_gpu"]:
        print(
            "Real GPU timing on %s (%s, %s); these are measured torch calls, not synthetic costs."
            % (info["device_name"], info["device"], info["dtype"]),
            flush=True,
        )
    else:
        print(
            "CPU fallback on %s (%s); this is real timing but it is not GPU execution."
            % (info["device_name"], info["dtype"]),
            flush=True,
        )
    print("Torch %s; replays %d per block; state %s" % (info["torch"], info["replays"], state_path), flush=True)
    state = GAState.load(state_path)
    started = time.monotonic()

    def log(message: str) -> None:
        if not args.quiet:
            print("[ga %6.1fs] %s" % (time.monotonic() - started, message), flush=True)

    engine = GeneticOptimizer(
        adapter,
        state,
        seed=args.seed,
        baseline_name="t10",
        per_round=args.per_round,
        race_rounds=args.race_rounds,
        mutation_blocks=args.mutation_blocks,
        mutation_attempts=args.mutation_attempts,
        confirm_blocks=args.confirm_blocks,
        det_only=args.det_only,
        rerace=args.rerace,
        seconds=None if args.seconds <= 0 else args.seconds,
        sleep_when_idle=0.0,
        log=log,
    )
    try:
        engine.run(generations=args.generations)
    finally:
        adapter.close()
    summary = _summary(state, len(engine.cells))
    summary["adapter"] = info
    _write_json(state_path.parent / "summary.json", summary)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _print_summary(summary)
    print(
        "Local GPU demo complete. This is a microbenchmark of torch calls on this machine, "
        "not the official H200 judge or a leaderboard result.",
        flush=True,
    )
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    if bool(args.config) == bool(args.adapter):
        print("error: pass exactly one of --config or --adapter", file=sys.stderr)
        return 2
    state_path = Path(args.state).expanduser()
    adapter = CommandAdapter.from_config(Path(args.config), seed=args.seed) if args.config else _load_adapter(args.adapter)
    state = GAState.load(state_path)
    started = time.monotonic()

    def log(message: str) -> None:
        if not args.quiet:
            print("[ga %6.1fs] %s" % (time.monotonic() - started, message), flush=True)

    engine = GeneticOptimizer(
        adapter,
        state,
        seed=args.seed,
        baseline_name=args.baseline,
        per_round=args.per_round,
        race_rounds=args.race_rounds,
        mutation_blocks=args.mutation_blocks,
        mutation_attempts=args.mutation_attempts,
        confirm_blocks=args.confirm_blocks,
        det_only=args.det_only,
        rerace=args.rerace,
        seconds=None if args.seconds <= 0 else args.seconds,
        sleep_when_idle=args.sleep_when_idle,
        log=log,
    )
    try:
        engine.run(generations=args.generations)
    except StickyFault as exc:
        print(
            "sticky fault at %s; pending gene retained for the restart driver: %s"
            % (exc.gene_id, exc),
            file=sys.stderr,
        )
        return 3
    finally:
        adapter.close()
    summary = _summary(state, len(engine.cells))
    _write_json(state_path.parent / "summary.json", summary)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _print_summary(summary)
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    path = Path(args.state).expanduser()
    if not path.exists():
        print("no GA state at %s" % path, file=sys.stderr)
        return 1
    state = GAState.load(path)
    summary = _summary(state, len(state.cells) or len(state.champ))
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    print("state: %s" % path)
    print("schema: %s" % STATE_SCHEMA)
    print("generation: %d" % state.generation)
    print("pending: %s" % (state.pending or "none"))
    print("blacklisted genes: %d" % len(state.blacklist))
    print("championed cells: %d" % len(state.champ))
    print("measured gain: %.2f%%" % (summary["gain"] * 100.0))
    if state.history:
        print("last generation: %s" % json.dumps(state.history[-1], sort_keys=True))
    _print_summary(summary)
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    path = Path(args.state).expanduser()
    state = GAState.load(path)
    summary = _summary(state, len(state.champ))
    output = Path(args.output).expanduser()
    _write_json(output, summary)
    print("wrote %s" % output)
    return 0


def _cmd_recover(args: argparse.Namespace) -> int:
    recovered = recover_pending(Path(args.state).expanduser())
    if recovered:
        print("blacklisted pending gene: %s" % recovered)
    else:
        print("no pending gene")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python3 -m benchmarks.ga",
        description="Portable per-cell racing genetic optimizer (no model calls).",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="run a synthetic CPU-only demonstration")
    demo.add_argument("--state", default=None, help="state path (default: ./tmp/ga-demo/state.json)")
    demo.add_argument("--fresh", action="store_true", help="delete an existing demo state first")
    demo.add_argument("--cells", type=int, default=6)
    demo.add_argument("--generations", type=int, default=5)
    demo.add_argument("--seconds", type=float, default=30.0, help="wall-clock budget (0 means no budget)")
    demo.add_argument("--seed", type=int, default=7)
    demo.add_argument("--noise", type=float, default=0.006, help="synthetic per-block log noise")
    demo.add_argument("--per-round", type=int, default=100)
    demo.add_argument("--race-rounds", type=int, default=10)
    demo.add_argument("--mutation-attempts", type=int, default=3)
    demo.add_argument("--mutation-blocks", type=int, default=150)
    demo.add_argument("--confirm-blocks", type=int, default=400)
    demo.add_argument("--det-only", action="store_true")
    demo.add_argument("--rerace", action="store_true")
    demo.add_argument("--quiet", action="store_true")
    demo.add_argument("--json", action="store_true")
    demo.set_defaults(func=_cmd_demo)

    gpu = subparsers.add_parser(
        "gpu-demo",
        help="run real paired timings on a local CUDA or Apple MPS device",
    )
    gpu.add_argument("--state", default=None, help="state path (default: ./tmp/ga-gpu/state.json)")
    gpu.add_argument("--fresh", action="store_true", help="delete an existing GPU demo state first")
    gpu.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    gpu.add_argument("--dtype", choices=("auto", "float16", "bfloat16", "float32"), default="auto")
    gpu.add_argument("--cells", type=int, default=3)
    gpu.add_argument("--generations", type=int, default=2)
    gpu.add_argument("--seconds", type=float, default=120.0, help="wall-clock budget (0 means no budget)")
    gpu.add_argument("--seed", type=int, default=7)
    gpu.add_argument("--replays", type=int, default=50, help="kernel calls inside each timed block")
    gpu.add_argument("--warmup", type=int, default=5)
    gpu.add_argument("--flush-mb", type=int, default=32, help="cache-flush buffer size in MiB")
    gpu.add_argument("--per-round", type=int, default=8)
    gpu.add_argument("--race-rounds", type=int, default=4)
    gpu.add_argument("--mutation-attempts", type=int, default=3)
    gpu.add_argument("--mutation-blocks", type=int, default=8)
    gpu.add_argument("--confirm-blocks", type=int, default=12)
    gpu.add_argument("--det-only", action="store_true")
    gpu.add_argument("--rerace", action="store_true")
    gpu.add_argument("--quiet", action="store_true")
    gpu.add_argument("--json", action="store_true")
    gpu.set_defaults(func=_cmd_gpu_demo)

    run = subparsers.add_parser("run", help="run against a target adapter")
    run.add_argument("--config", help="JSON config for the command adapter")
    run.add_argument("--adapter", help="Python adapter module:factory or path/to/file.py:factory")
    _add_common_run_arguments(run)
    run.set_defaults(func=_cmd_run)

    status = subparsers.add_parser("status", help="print a saved state summary")
    status.add_argument("--state", required=True)
    status.add_argument("--json", action="store_true")
    status.set_defaults(func=_cmd_status)

    export = subparsers.add_parser("export", help="write a compact champion export")
    export.add_argument("--state", required=True)
    export.add_argument("--output", required=True)
    export.set_defaults(func=_cmd_export)

    recover = subparsers.add_parser("recover", help="blacklist the gene left pending by a stopped process")
    recover.add_argument("--state", required=True)
    recover.set_defaults(func=_cmd_recover)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("stopped; state is saved", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
