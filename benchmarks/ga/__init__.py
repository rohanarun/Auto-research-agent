"""Portable per-cell racing genetic optimizer.

The engine is a dependency-free port of the per-cell racing loop used for Open
Kernel Challenge work.  It never calls a model, and it has no GPU, CUDA, NumPy or
credential requirements.  A target-specific adapter supplies cells, genes,
builds and paired timing measurements; the bundled ``demo`` adapter uses
synthetic timings so the whole loop can be exercised on any operating system.

This module is deliberately separate from the macOS app.  Run it with the
Python interpreter of your choice::

    python3 -m benchmarks.ga demo

The only supported public surface is :class:`GeneticOptimizer`, :class:`GAState`
and the adapter protocol documented in ``benchmarks/ga/README.md``.
"""

__all__ = ["engine", "demo", "command", "cli"]
__version__ = "1.0.0"
