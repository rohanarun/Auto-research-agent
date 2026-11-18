#!/usr/bin/env python3
"""Cross-platform launcher for the portable genetic optimizer.

The macOS app is not required.  This wrapper works with any Python 3.9+ on
macOS, Linux or Windows and adds an optional restart-on-sticky-fault driver::

    python3 benchmarks/tools/run_ga.py demo --fresh
    python3 benchmarks/tools/run_ga.py run --config ga.json --state tmp/ga/state.json --restarts 5

The engine itself has no third-party dependencies.  A target adapter may import
PyTorch/CUDA when the evaluator actually needs them.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARKS = REPO_ROOT / "benchmarks"


def _parse_restarts(argv: List[str]) -> Tuple[List[str], int]:
    restarts = 1
    cleaned: List[str] = []
    index = 0
    while index < len(argv):
        argument = argv[index]
        if argument == "--restarts":
            if index + 1 >= len(argv):
                raise SystemExit("--restarts needs a number")
            restarts = max(1, int(argv[index + 1]))
            index += 2
            continue
        if argument.startswith("--restarts="):
            restarts = max(1, int(argument.split("=", 1)[1]))
            index += 1
            continue
        cleaned.append(argument)
        index += 1
    return cleaned, restarts


def _state_path(argv: List[str]) -> Optional[Path]:
    for index, argument in enumerate(argv):
        if argument == "--state" and index + 1 < len(argv):
            return Path(argv[index + 1]).expanduser()
        if argument.startswith("--state="):
            return Path(argument.split("=", 1)[1]).expanduser()
    return None


def _run_child(argv: List[str]) -> int:
    environment = dict(os.environ)
    previous = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = str(REPO_ROOT) + (os.pathsep + previous if previous else "")
    return subprocess.call(
        [sys.executable, "-m", "benchmarks.ga", *argv],
        env=environment,
    )


def main(argv: Optional[List[str]] = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    arguments, restarts = _parse_restarts(arguments)
    if restarts <= 1:
        if str(BENCHMARKS) not in sys.path:
            sys.path.insert(0, str(BENCHMARKS))
        from ga.cli import main as cli_main

        return int(cli_main(arguments))

    state = _state_path(arguments)
    if state is None:
        raise SystemExit("--restarts needs --state so the driver can recover a pending gene")
    state = state.resolve()
    for attempt in range(1, restarts + 1):
        print("[ga-driver] attempt %d/%d" % (attempt, restarts), flush=True)
        code = _run_child(arguments)
        if code == 0:
            return 0
        if code != 3:
            return code
        if str(BENCHMARKS) not in sys.path:
            sys.path.insert(0, str(BENCHMARKS))
        from ga.engine import recover_pending

        recovered = recover_pending(state)
        print("[ga-driver] recovered pending gene: %s" % (recovered or "none"), flush=True)
        if not recovered:
            return code
    return 3


if __name__ == "__main__":
    raise SystemExit(main())
