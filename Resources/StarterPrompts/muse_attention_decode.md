# Attention decode

Problem: `muse_attention_decode`

Task and leaderboard: https://openkernelchallenge.dev/problems/muse_attention_decode

Trace the public path from QKV input through normalization, position handling, cache updates, attention and output gating. Write down the sliding-window and full-context bounds, causal behavior inside verification blocks, page-size handling and cache scales. Read configuration values instead of assuming the scored cases cover all tests.

Start with an unnecessary intermediate or a simple adjacent pointwise operation. Keep the public baseline's attention path while investigating that boundary. Preserve FP8 cache writes, allowed query precision and every required key and value; sparse attention or skipping small contributions is not a valid shortcut. Evaluate the complete attention call with the official judge.

## Jump-Start Engineering Guidelines

1. **Pre-Allocate Attention Buffers**: Pre-allocate output and temporary logsumexp/max scratch buffers in `prepare()`. Never call dynamic memory allocation functions inside the time-critical decode step.
2. **Fused RoPE and Cache Update**: Avoid writing separate passes for rotary embeddings (RoPE), QK normalization, and cache updates. Perform rotary positioning and FP8 quantization directly in a single fused pre-pass before writing to the paged KV cache.
3. **Contiguous Paged Memory Alignment**: Paged KV cache reads easily bottleneck on uncoalesced memory transactions. Align thread block load sizes so that memory reads along the head dimension match hardware transaction boundaries (e.g. 64-byte or 128-byte segments), eliminating cross-page strided fetch stalls.
4. **Avoid Divergent Intra-Loop Branches**: In the causal attention loop, replace dynamic `if/else` checks with predicated selections (`mask=...`) to ensure all warps execute identical instruction streams without divergence.

Suggested first prompt: “Explain the attention starter's cache and causality contract, then propose one basic way to reduce intermediate memory traffic without changing which tokens are attended to.”
