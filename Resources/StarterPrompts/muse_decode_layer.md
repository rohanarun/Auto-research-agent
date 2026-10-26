# Decoder layer

Problem: `muse_decode_layer`

Task and leaderboard: https://openkernelchallenge.dev/problems/muse_decode_layer

Draw a short dependency list for the public decoder layer: projections, attention, residual updates, routing, experts and the final normalization. Record all returned tensors and in-place cache effects, including the FP32 residual stream and expert IDs. Check the task's small-token unpooler convention and routing tolerances rather than assuming each component matches its standalone board.

Begin with one boundary that materializes a redundant pointwise intermediate. Keep the rest of the public layer intact, preserve intermediate precision and reference semantics, and measure the whole layer. A component improvement does not establish an end-to-end gain. Do not replace required routing, normalization or full attention with an approximation to pass a selected case.

## Jump-Start Engineering Guidelines

1. **Cross-Stage Buffer Reuse**: The whole decoder layer connects Attention → Projections → MoE → Norms. Avoid creating separate intermediate tensors between these stages; reuse a single pre-allocated contiguous residual workspace across blocks to avoid DRAM roundtrips.
2. **Asynchronous Stream Overlap**: Independent stages (such as the MoE routing projection and attention output projection) can be overlapped on dedicated CUDA streams. Launching the router concurrently with the tail of attention hides routing overhead completely behind GEMM execution.
3. **Epilogue Fusion**: Fuse residual addition and RMSNorm scaling into the epilogue of preceding linear projections (such as the unpooler or attention output) rather than launching separate lightweight kernels that suffer from kernel launch latency.
4. **Branch-Free Control Flow**: Keep token routing and expert assignment entirely on-device; never call `.item()` or trigger CPU-GPU synchronization inside the decoder step.

Suggested first prompt: “Read the whole-layer starter and identify one conservative intermediate-tensor reduction. Explain its correctness obligations and use official full-layer checks to evaluate it.”
