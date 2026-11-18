"""Portable per-cell racing genetic optimizer.

The engine is a dependency-free port of the per-cell racing loop used for Open
Kernel Challenge work.  It never calls a model, and it has no GPU, CUDA, NumPy or
credential requirements.  A target-specific adapter supplies cells, genes,
builds and paired timing measurements; the bundled ``demo`` adapter uses
synthetic timings so the whole loop can be exercised on any operating system.
The optional ``torch_gpu`` adapter runs real local CUDA/MPS timings, and the
optional ``daytona`` adapter uses the repository's existing H200 harness.

This module is deliberately separate from the macOS app.  Run it with the
Python interpreter of your choice::

    python3 -m benchmarks.ga demo
    python3 -m benchmarks.ga gpu-demo --device auto

The only supported public surface is :class:`GeneticOptimizer`, :class:`GAState`
and the adapter protocol documented in ``benchmarks/ga/README.md``.
"""

__all__ = ["engine", "demo", "command", "cli", "torch_gpu", "daytona"]
__version__ = "1.0.0"
