# Dense projections

Problem: `muse_bf16_proj`

Task and leaderboard: https://openkernelchallenge.dev/problems/muse_bf16_proj

Map the public starter's projection calls, shapes, strides and output dtypes. The router has a different precision requirement from the BF16 projections; read its current correctness rules before changing it. Small decode batches and larger verification batches need not have the same cost, but any shape-based dispatch must remain valid on the additional test shapes.

Begin with one ordinary projection and compare the starter's matrix-multiply path with the public baseline. Look for a redundant transpose, materialized intermediate, or avoidable launch. Explain the expected benefit and preserve a correct fallback for untouched paths. Use official per-case results to decide whether the change improves the complete problem, rather than promoting a single fast shape.

Suggested first prompt: “Read the dense-projection starter and baseline. Explain one simple source-level improvement, preserve router precision, and test it with the official check budget I provide.”
