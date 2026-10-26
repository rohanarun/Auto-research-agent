# Dense projections

Problem: `muse_bf16_proj`

Task and leaderboard: https://openkernelchallenge.dev/problems/muse_bf16_proj

Map the public starter's projection calls, shapes, strides and output dtypes. The router has a different precision requirement from the BF16 projections; read its current correctness rules before changing it. Small decode batches and larger verification batches need not have the same cost, but any shape-based dispatch must remain valid on the additional test shapes.

Begin with one ordinary projection and compare the starter's matrix-multiply path with the public baseline. Look for a redundant transpose, materialized intermediate, or avoidable launch. Explain the expected benefit and preserve a correct fallback for untouched paths. Use official per-case results to decide whether the change improves the complete problem, rather than promoting a single fast shape.

Lesson from the first public layout-only check: the starter already uses a transpose view, and repacking one ordinary projection did not demonstrate a gain; the unchanged router dominated the observed cost. Before another layout change, use the official profile and weighted load costs to explain how it could improve the complete problem. Compare the public baseline's precision-preserving router path before choosing the next bounded change. This lesson guides a hypothesis, not a promised gain.

## Jump-Start Engineering Guidelines

1. **Pre-Allocate Reusable Workspaces**: Avoid allocating output tensors or temporary scratchpads dynamically inside `kernel_fn`. Pre-allocate buffers in `prepare(weights)` and write into them directly to bypass CUDA allocator overhead.
2. **Eliminate Runtime Transposes**: Dense projections are primarily memory-bandwidth bound. Pre-transpose weight matrices during `prepare()` so memory loads along the reduction axis are naturally coalesced into continuous 128-byte cache lines without strided layout overhead.
3. **Specialized Single-Token GEMV Path**: When batch size $M=1$ (single-token decode), general 2D matrix multiplication kernels suffer from high fixed launch latency. Specialize $M=1$ into a 1D streaming GEMV that accumulates dot products directly in registers without shared-memory staging.
4. **Branch-Free Loop Unrolling**: In shapes where reduction dimensions divide evenly, statically unroll inner loops and remove runtime boundary masking to allow compilers to emit optimal pipelined load sequences.

Suggested first prompt: “Read the dense-projection starter and baseline. Explain one simple source-level improvement, preserve router precision, and test it with the official check budget I provide.”
