# Prefill layer

Problem: `muse_prefill_layer`

Task and leaderboard: https://openkernelchallenge.dev/problems/muse_prefill_layer

Inspect the public full-layer path at its longer query blocks. Identify how causal attention, paged cache writes, routing and grouped expert work scale with token count. Establish the peak size of intermediate tensors before proposing a change. Keep the FP32 residual stream, required cache effects and routing outputs intact.

Start with one avoidable allocation, layout conversion or adjacent pointwise intermediate in the public baseline. Preserve NVFP4 expert weights and decode them only where the current rules allow; untimed persistent BF16 expansion is not an optimization permitted by those rules. Evaluate complete prefill cases instead of extrapolating a component time or a decode result. This problem has its own leaderboard and does not contribute to the overall decode board.

Suggested first prompt: “Read the prefill starter, map its largest intermediates, and propose one basic memory-traffic reduction that preserves the full layer and legal weight storage.”
