# Decoder layer

Problem: `muse_decode_layer`

Task and leaderboard: https://openkernelchallenge.dev/problems/muse_decode_layer

Draw a short dependency list for the public decoder layer: projections, attention, residual updates, routing, experts and the final normalization. Record all returned tensors and in-place cache effects, including the FP32 residual stream and expert IDs. Check the task's small-token unpooler convention and routing tolerances rather than assuming each component matches its standalone board.

Begin with one boundary that materializes a redundant pointwise intermediate. Keep the rest of the public layer intact, preserve intermediate precision and reference semantics, and measure the whole layer. A component improvement does not establish an end-to-end gain. Do not replace required routing, normalization or full attention with an approximation to pass a selected case.

Suggested first prompt: “Read the whole-layer starter and identify one conservative intermediate-tensor reduction. Explain its correctness obligations and use official full-layer checks to evaluate it.”
