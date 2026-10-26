# Routed experts

Problem: `muse_moe_w4a16`

Task and leaderboard: https://openkernelchallenge.dev/problems/muse_moe_w4a16

Read the public NVFP4 helpers and trace routing to local experts, gate/up projection, SwiGLU, down projection, per-expert normalization and weighted accumulation. Check where routing work and intermediate tensors are created. Preserve the computation order required by the reference, especially normalization before expert outputs are combined.

Choose one basic improvement such as removing a redundant conversion or reducing an intermediate around pointwise work. Inspect the public baseline before writing a new GEMM. Retain NVFP4 weights and their scale semantics; do not pre-expand them into a persistent higher-precision copy. Routing is runtime data, so handle skew and empty local contributions without host synchronization or cached routing results.

Suggested first prompt: “Map the routed-expert starter's intermediate tensors and choose one simple reduction in memory traffic that preserves NVFP4 storage and per-expert normalization.”
