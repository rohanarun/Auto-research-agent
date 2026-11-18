"""Local, no-GPU tests for the portable racing GA.

Run from the repository root with either::

    python3 -m unittest discover -s benchmarks/tests -p 'test_ga_*.py' -v
    python3 benchmarks/tests/test_ga_engine.py
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Set

BENCHMARKS = Path(__file__).resolve().parents[1]
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from ga.command import CommandAdapter  # noqa: E402
from ga.demo import DemoAdapter  # noqa: E402
from ga.engine import (  # noqa: E402
    CONFIRM_REL,
    Cell,
    GAState,
    Gene,
    GeneticOptimizer,
    StickyFault,
    confirmation_passes,
    measured_gain,
    paired_log_difference,
    recover_pending,
    successive_elimination,
)


class StaticAdapter:
    """Exact paired timings; no noise, no GPU."""

    def __init__(self, cells: Sequence[Dict[str, Any]], costs: Dict[str, Dict[str, float]]):
        self._cells = [
            Cell(key=item["key"], layer=item.get("layer", "test"), m=int(item.get("m", 1)))
            for item in cells
        ]
        self.costs = costs
        self.built: Dict[str, Set[str]] = {cell.key: set() for cell in self._cells}
        self._initial = {cell.key: [Gene("t10", ["baseline"])] for cell in self._cells}
        self._mutants = {cell.key: [] for cell in self._cells}
        for item in cells:
            self._initial[item["key"]].extend(
                Gene(name, ["static", name]) for name in item.get("initial", [])
            )
            self._mutants[item["key"]].extend(
                Gene(name, ["static", name], risky=item.get("risky", False))
                for name in item.get("mutants", [])
            )

    def cells(self):
        return list(self._cells)

    def initial_genes(self, cell):
        return list(self._initial[cell.key])

    def mutants(self, cell, incumbent, known, state, rng):
        return [gene for gene in self._mutants[cell.key] if gene.name not in known]

    def build(self, cell, gene):
        from ga.engine import BuildResult

        if gene.name == "boom":
            raise StickyFault(f"{cell.key}|{gene.name}")
        self.built[cell.key].add(gene.name)
        return BuildResult(True, True, None)

    def drop(self, cell, name):
        return None

    def measure(self, cell, names, blocks):
        return {name: [self.costs[cell.key][name]] * int(blocks) for name in names}

    def confirm(self, cell, champion, base, blocks):
        from ga.engine import ConfirmResult

        rel = self.costs[cell.key][champion] / self.costs[cell.key][base]
        return ConfirmResult(
            rel=rel,
            lo=rel,
            hi=rel,
            t_champion=self.costs[cell.key][champion],
            t_base=self.costs[cell.key][base],
        )

    def close(self):
        return None


def _state_path(root: Path) -> Path:
    return root / "ga" / "state.json"


class StatisticsTests(unittest.TestCase):
    def test_paired_log_difference(self):
        mean, error = paired_log_difference([99.0, 99.0, 99.0, 99.0], [100.0] * 4)
        self.assertAlmostEqual(mean, math.log(0.99), places=12)
        self.assertAlmostEqual(error, 0.0, places=12)
        with self.assertRaises(ValueError):
            paired_log_difference([1.0], [1.0])
        with self.assertRaises(ValueError):
            paired_log_difference([0.0, 1.0], [1.0, 1.0])

    def test_confirmation_gate_is_strict(self):
        from ga.engine import ConfirmResult

        self.assertTrue(confirmation_passes(ConfirmResult(0.996, 0.995, 0.997, 1.0, 1.0)))
        self.assertFalse(confirmation_passes(ConfirmResult(CONFIRM_REL, 0.996, 0.999, 1.0, 1.0)))
        self.assertFalse(confirmation_passes(ConfirmResult(0.99, 0.99, 1.0, 1.0, 1.0)))
        self.assertFalse(confirmation_passes(ConfirmResult(float("nan"), 0.9, 0.99, 1.0, 1.0)))

    def test_race_selects_fastest_and_keeps_statistical_ties(self):
        adapter = StaticAdapter(
            [{"key": "c", "initial": ["fast", "mid", "slow"]}],
            {"c": {"t10": 100.0, "fast": 90.0, "mid": 97.0, "slow": 105.0}},
        )
        cell = adapter.cells()[0]
        order, means = successive_elimination(
            adapter, cell, ["t10", "fast", "mid", "slow"], per_round=8, rounds=3
        )
        self.assertEqual(order, ["fast"])
        self.assertIn("fast", means)

        tie_adapter = StaticAdapter(
            [{"key": "t", "initial": ["same"]}],
            {"t": {"t10": 100.0, "same": 100.0}},
        )
        tie_cell = tie_adapter.cells()[0]
        order, _ = successive_elimination(
            tie_adapter, tie_cell, ["t10", "same"], per_round=4, rounds=2
        )
        self.assertEqual(set(order), {"t10", "same"})


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_gen0_then_confirmed_hillclimb_is_monotonic(self):
        adapter = StaticAdapter(
            [{"key": "cell", "initial": ["fast"], "mutants": ["better"]}],
            {"cell": {"t10": 100.0, "fast": 99.0, "better": 98.0}},
        )
        state = GAState.load(_state_path(self.root))
        engine = GeneticOptimizer(
            adapter,
            state,
            per_round=4,
            race_rounds=2,
            mutation_blocks=8,
            confirm_blocks=12,
            sleep_when_idle=0.0,
            log=lambda _message: None,
        )
        engine.run(generations=2)
        gains = [float(entry["gain_meas"]) for entry in state.history]
        self.assertEqual(state.generation, 3)
        self.assertEqual(state.champ["cell"]["gene"], "better")
        self.assertAlmostEqual(state.champ["cell"]["rel"], 0.98, places=9)
        self.assertTrue(all(right + 1e-12 >= left for left, right in zip(gains, gains[1:])))
        self.assertAlmostEqual(measured_gain(state), 0.02, places=9)
        self.assertIsNone(state.pending)

        # Re-read the atomically saved state and resume without re-racing.
        restored = GAState.load(_state_path(self.root))
        self.assertEqual(restored.champ["cell"]["gene"], "better")
        engine2 = GeneticOptimizer(
            adapter,
            restored,
            per_round=4,
            race_rounds=2,
            mutation_blocks=8,
            confirm_blocks=12,
            sleep_when_idle=0.0,
            log=lambda _message: None,
        )
        engine2.run(generations=1)
        self.assertEqual(restored.champ["cell"]["gene"], "better")
        self.assertGreaterEqual(measured_gain(restored), gains[-1] - 1e-12)

    def test_weak_or_unconfirmed_mutant_is_rejected(self):
        adapter = StaticAdapter(
            [{"key": "cell", "mutants": ["weak", "false_positive"]}],
            {"cell": {"t10": 100.0, "weak": 99.9, "false_positive": 99.0}},
        )
        # confirm() returns the true ratio; only weak is below the 0.3% gate and
        # false_positive must still pass the gate but is made to fail the fresh
        # input confirmation by an adapter that reports a bad upper bound.
        class ConfirmAware(StaticAdapter):
            def confirm(self, cell, champion, base, blocks):
                from ga.engine import ConfirmResult

                rel = self.costs[cell.key][champion] / self.costs[cell.key][base]
                hi = 1.2 if champion == "false_positive" else rel
                return ConfirmResult(rel=rel, lo=rel, hi=hi, t_champion=self.costs[cell.key][champion], t_base=self.costs[cell.key][base])

        aware = ConfirmAware(
            [{"key": "cell", "mutants": ["weak", "false_positive"]}],
            adapter.costs,
        )
        state = GAState.load(_state_path(self.root))
        engine = GeneticOptimizer(
            aware,
            state,
            per_round=4,
            race_rounds=2,
            mutation_blocks=8,
            confirm_blocks=12,
            sleep_when_idle=0.0,
            log=lambda _message: None,
        )
        engine.run(generations=2)
        self.assertEqual(state.champ["cell"]["gene"], "t10")
        self.assertAlmostEqual(measured_gain(state), 0.0, places=12)


class StateTests(unittest.TestCase):
    def test_atomic_state_and_pending_recovery(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            state = GAState.load(path)
            state.pending = "cell|risky"
            state.save()
            self.assertFalse(Path(str(path) + ".tmp").exists())
            self.assertEqual(json.loads(path.read_text())["pending"], "cell|risky")
            recovered = recover_pending(path)
            self.assertEqual(recovered, "cell|risky")
            reloaded = GAState.load(path)
            self.assertIsNone(reloaded.pending)
            self.assertIn("cell|risky", reloaded.blacklist)

    def test_sticky_fault_preserves_pending_then_recovers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            adapter = StaticAdapter([{"key": "cell", "risky": True, "mutants": ["boom"]}], {"cell": {"t10": 100.0, "boom": 90.0}})
            state = GAState.load(path)
            engine = GeneticOptimizer(adapter, state, log=lambda _message: None)
            cell = adapter.cells()[0]
            with self.assertRaises(StickyFault):
                engine.add_gene(cell, Gene("boom", ["static"], risky=True))
            self.assertEqual(GAState.load(path).pending, "cell|boom")
            self.assertEqual(recover_pending(path), "cell|boom")
            self.assertIn("cell|boom", GAState.load(path).blacklist)

    def test_corrupt_state_is_not_silently_reset(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "state.json"
            path.write_text("{not json")
            with self.assertRaises(ValueError):
                GAState.load(path)


class DemoAndCliTests(unittest.TestCase):
    def test_demo_gain_is_monotonic_and_summary_is_written(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            adapter = DemoAdapter(cells=3, seed=7, noise=0.004)
            state = GAState.load(state_path)
            engine = GeneticOptimizer(
                adapter,
                state,
                per_round=20,
                race_rounds=3,
                mutation_blocks=20,
                confirm_blocks=40,
                sleep_when_idle=0.0,
                log=lambda _message: None,
            )
            engine.run(generations=4)
            gains = [float(entry["gain_meas"]) for entry in state.history]
            self.assertTrue(gains)
            self.assertTrue(all(right + 1e-12 >= left for left, right in zip(gains, gains[1:])))
            self.assertTrue(all(record["rel"] <= 1.0 for record in state.champ.values()))

    def test_cli_demo_script_path(self):
        from ga.cli import main

        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            output = io.StringIO()
            with contextlib.redirect_stdout(output), contextlib.redirect_stderr(output):
                code = main(
                    [
                        "demo",
                        "--state",
                        str(state_path),
                        "--fresh",
                        "--cells",
                        "2",
                        "--generations",
                        "2",
                        "--per-round",
                        "10",
                        "--race-rounds",
                        "2",
                        "--mutation-blocks",
                        "10",
                        "--confirm-blocks",
                        "20",
                        "--quiet",
                    ]
                )
            self.assertEqual(code, 0)
            self.assertTrue(state_path.exists())
            self.assertTrue((state_path.parent / "summary.json").exists())


class CommandAdapterTests(unittest.TestCase):
    def _write_fixture(self, root: Path) -> Path:
        evaluator = root / "fake_eval.py"
        evaluator.write_text(
            "\n".join(
                [
                    "import json, sys",
                    "request = json.load(sys.stdin)",
                    "op = request['op']",
                    "def mult(name):",
                    "    return {'t10': 1.0, 'cublas': 1.03, 'main:1': 0.99, 'rk:S8': 0.98}.get(name, 1.0)",
                    "if op == 'build':",
                    "    print(json.dumps({'ok': True, 'bit': True}))",
                    "elif op == 'measure':",
                    "    blocks = int(request['blocks'])",
                    "    print(json.dumps({'times': {name: [100.0 * mult(name)] * blocks for name in request['names']}}))",
                    "elif op == 'confirm':",
                    "    rel = mult(request['champion']) / mult(request['base'])",
                    "    print(json.dumps({'rel': rel, 'lo': rel, 'hi': rel, 't_champion': 100.0 * mult(request['champion']), 't_base': 100.0 * mult(request['base'])}))",
                    "elif op == 'drop':",
                    "    print(json.dumps({}))",
                    "else:",
                    "    print(json.dumps({'error': 'unknown op'}))",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        config = root / "config.json"
        config.write_text(
            json.dumps(
                {
                    "command": [sys.executable, str(evaluator)],
                    "timeout_seconds": 30,
                    "cells": [
                        {
                            "key": "router|m=1",
                            "layer": "router",
                            "m": 1,
                            "n": 4096,
                            "k": 8192,
                            "genes": [
                                {"name": "t10", "spec": ["baseline"]},
                                {"name": "cublas", "spec": ["anchor"], "selectable": False},
                                {"name": "main:1", "spec": ["main", 1]},
                            ],
                            "mutants": [{"name": "rk:S8", "spec": ["rk", 8], "risky": True}],
                        }
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return config

    def test_command_protocol_and_hillclimb(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._write_fixture(root)
            adapter = CommandAdapter.from_config(config)
            state = GAState.load(root / "state.json")
            engine = GeneticOptimizer(
                adapter,
                state,
                per_round=4,
                race_rounds=2,
                mutation_blocks=8,
                confirm_blocks=12,
                sleep_when_idle=0.0,
                log=lambda _message: None,
            )
            engine.run(generations=2)
            self.assertEqual(state.champ["router|m=1"]["gene"], "rk:S8")
            self.assertAlmostEqual(state.champ["router|m=1"]["rel"], 0.98, places=9)
            self.assertAlmostEqual(measured_gain(state), 0.02, places=9)


if __name__ == "__main__":
    unittest.main()
