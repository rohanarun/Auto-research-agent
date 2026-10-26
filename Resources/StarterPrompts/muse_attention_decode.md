# Attention decode

Problem: `muse_attention_decode`

Task and leaderboard: https://openkernelchallenge.dev/problems/muse_attention_decode

Trace the public path from QKV input through normalization, position handling, cache updates, attention and output gating. Write down the sliding-window and full-context bounds, causal behavior inside verification blocks, page-size handling and cache scales. Read configuration values instead of assuming the scored cases cover all tests.

Start with an unnecessary intermediate or a simple adjacent pointwise operation. Keep the public baseline's attention path while investigating that boundary. Preserve FP8 cache writes, allowed query precision and every required key and value; sparse attention or skipping small contributions is not a valid shortcut. Evaluate the complete attention call with the official judge.

Suggested first prompt: “Explain the attention starter's cache and causality contract, then propose one basic way to reduce intermediate memory traffic without changing which tokens are attended to.”
