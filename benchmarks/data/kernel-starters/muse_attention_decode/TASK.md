# Gated attention decode (`muse_attention_decode`)

One decode step or 8-token verify block of grouped-query attention over a paged FP8 cache: QK-norm, a 2,048-token sliding window with RoPE or full attention over the whole context, and a gated, normalized output.

Your job is to compute this faster than SGLang's kernel does. Beside this file, `okc init` wrote:

- `submission.py`, your kernel. It starts as a plain PyTorch version of the computation. Keep its first line, `# okc: problem=muse_attention_decode`: it tells `okc submit` the problem.
- `baseline.py`, SGLang's kernel: the one you're timed against.
- `reference.py`, the computation as the judge checks it: `generate_input`, the reference and the correctness checks, with `checks.py` beside it: the spec your output must meet. The judge uses its own copies, and runs them with steps of its own (the weights, routing fields, fresh inputs), so to check a change, use `okc check`.
- `AGENTS.md`, notes for a coding agent, if you use one.

## What to compute

```python
# per request b and block token j, at position p = positions[t]
q, k, v, gate = split(qkv[t])   # the fused projection output
q = rmsnorm(q) * 5.473;  k = rmsnorm(k)
q, k = rope(q, p), rope(k, p)   # sliding layers
cache[cache_loc[t]] = fp8(k), fp8(v)
a = softmax(q @ K.T / 8) @ V    # up to p: window or all
out = rmsnorm(a) * sigmoid(gate)
```

`kernel_fn(data)` is the timed call and returns bfloat16 [B, L, Hq·64]. The `Inputs` and `Config` classes in submission.py give every field's dtype and shape; return GPU tensors of exactly that shape and dtype.

## Shapes

- B is batch and L is q_len: 1 for a decode step, or 8 for a speculative verify block (causal inside it). Each GPU has 16 query heads and 2 key/value heads (TP8).
- Each load times a sliding-window layer (the last 2,048 tokens, with RoPE) and a full-attention layer (the whole context, no RoPE). Contexts are around 80K tokens.
- The inputs are what SGLang’s attention receives: the fused projection output, plus the step’s positions, cache rows and FA3 lengths, which SGLang builds once per step.
- `scheduler_metadata` is FA3's tile schedule for the step, built from its lengths as SGLang does: set on full-attention decode steps, None on sliding layers and verify blocks. Kernels that don't use FA3 can ignore it.
- The cache is FP8 E4M3. Its page size is 1 in the scored cases and 64 in most tests, so read config["page_size"]. Its scales are 1 in the scored cases and 1/64 in some tests.
- Decode loads are SGLang serving with speculative decoding off; verify loads are its default recipe, which checks each request's 8-token draft in one step.

| load | shape | weight | floor |
|---|---|---|---|
| decode 1 | batch 1, ctx 80000, heads tp8, layer sliding, page 1, dist soft | 46 | 0.1 µs |
| decode 1 | batch 1, ctx 80000, heads tp8, layer full, page 1, dist coherent | 16 | 4.3 µs |
| decode 16 | batch 16, ctx 80000, heads tp8, layer sliding, page 1, spread agentic, dist soft | 46 | 1.6 µs |
| decode 16 | batch 16, ctx 80000, heads tp8, layer full, page 1, spread agentic, dist coherent | 16 | 63.2 µs |
| decode 32 | batch 32, ctx 80000, heads tp8, layer sliding, page 1, spread agentic, dist soft | 46 | 3.3 µs |
| decode 32 | batch 32, ctx 80000, heads tp8, layer full, page 1, spread agentic, dist coherent | 16 | 126.4 µs |
| verify 1 | batch 1, ctx 80000, heads tp8, layer sliding, page 1, q_len 8, dist soft | 46 | 0.1 µs |
| verify 1 | batch 1, ctx 80000, heads tp8, layer full, page 1, q_len 8, dist coherent | 16 | 4.3 µs |
| verify 16 | batch 16, ctx 80000, heads tp8, layer sliding, page 1, q_len 8, spread agentic, dist soft | 46 | 1.9 µs |
| verify 16 | batch 16, ctx 80000, heads tp8, layer full, page 1, q_len 8, spread agentic, dist coherent | 16 | 63.4 µs |
| verify 32 | batch 32, ctx 80000, heads tp8, layer sliding, page 1, q_len 8, spread agentic, dist soft | 46 | 3.7 µs |
| verify 32 | batch 32, ctx 80000, heads tp8, layer full, page 1, q_len 8, spread agentic, dist coherent | 16 | 126.8 µs |

A shape's weight is how many times one step runs it. A load's time is the sum of its shapes' times, each times its weight. Your speedup on the problem is the geometric mean, over loads, of SGLang's load time over yours, so each load counts equally. The time shown for a run is SGLang's time per step, averaged over loads the same way, divided by that speedup.

The floor is the time to move the data at 4.8 TB/s or do the math at 989 TFLOP/s, whichever is longer.

Before timing, a run must pass these 16 tests, some harder than the scored shapes:

```
batch 1, ctx 10, heads tp8, layer sliding
batch 3, ctx 2047, heads tp8, layer sliding
batch 5, ctx 5000, heads tp8, layer sliding, q_len 8, spread mixed, page 1
batch 4, ctx 3000, heads tp8, layer full
batch 8, ctx 8192, heads dp, layer full, spread mixed
batch 2, ctx 100000, heads tp8, layer full, q_len 8, page 1
batch 4, ctx 6000, heads tp8, layer sliding, dist edges, page 1
batch 4, ctx 6000, heads tp8, layer full, q_len 8, dist edges
batch 1, ctx 1048576, heads tp8, layer full, q_len 8
batch 3, ctx 4000, heads tp8, layer sliding, q_len 8, page 1, kv_scale 0.015625
batch 2, ctx 9000, heads tp8, layer full, kv_scale 0.015625
batch 4, ctx 80000, heads tp8, layer full, spread mixed, dist soft
batch 4, ctx 6000, heads tp8, layer sliding, q_len 8, dist soft, page 1
batch 2, ctx 20000, heads dp, layer full, q_len 8, dist soft
batch 4, ctx 80000, heads tp8, layer full, spread mixed, dist coherent, page 1
batch 2, ctx 20000, heads dp, layer full, q_len 8, dist coherent
```

## Correctness

Every test and every timed call is checked against a reference. Errors are relative L2, |out − ref| / |ref|:

- The output: within 2.5× the reference's own error when its queries are rounded to FP8 E4M3, plus 1.5%, kept between 3% and 15%, against an FP32 reference.
- The new cache entries: within 8% of the reference's. Your output is checked against the reference attending to your own new entries.
- Some tests put keys at the window's edges.
- Each head's keys share a direction, so attention spreads over many keys. On full layers the low-attention keys' values share a direction too.
- Any single row: at most 5× its bound. A near-zero row is measured against a tenth of the average row.
- No NaN or Inf.

## How a run is timed

- Each shape is captured as a CUDA graph and timed by replaying it, in the same run and on the same GPU as SGLang's kernel, your replays and SGLang's taking turns on the same fresh inputs.
- Every timed replay starts with a cold L2: the judge reads a buffer larger than L2 just before it.
- A shape's speedup is the median of paired ratios: each repetition's SGLang time over yours, taken two repetitions at a time with the order swapped, so which side went first cancels. Your time on the shape is SGLang's median time over that speedup.
- Each run reports its speedup with a 95% interval (at least ±1.5%). Gains inside the interval might be noise.
- The judge subtracts a graph replay's fixed cost (about 4 µs) from both sides.

## Rules

- **CUDA graphs.** The timed calls replay `kernel_fn` as a CUDA graph: no host syncs (`.item()`, `.cpu()`, `torch.cuda.synchronize()`, printing a tensor), and no allocating or branching on input values. Branching on shapes and `config` is fine.
- **Fresh inputs.** Every call gets new inputs; don't cache anything derived from `data`.
- **Activations.** BF16 wherever SGLang's kernel keeps BF16. The KV cache stays FP8; queries may be rounded to FP8 E4M3, as FA3 does, and no coarser.
- **Every key.** Attend to the whole 2,048-token window on sliding layers and the whole context on full ones: no top-k, sparse or block-skipping attention, and no skipping values by their attention weight.
- **Checked before running.** A submission is rejected if it patches timing, reads harness frames, touches harness I/O or secrets, edits loaded modules, imports the reference, starts processes or uses the network.

Full rules: https://openkernelchallenge.dev/rules

## Submitting

```bash
okc submit submission.py      # the tests, then timing; waits for the result
okc check submission.py       # the same run, kept off the board
okc status <job>              # a run's result again
okc jobs                      # your recent runs
```

A failing run names its first failing case and why. A passing run shows each shape's time against SGLang's and your speedup, and okc saves a profile beside your file, in `profiles/submission.<job>.txt`, showing where the time goes.

`print()` is fine for diagnostics (just not a tensor's values inside `kernel_fn`, which syncs). The last 16,000 characters of what a run prints come back with it, passing or not: okc saves it to `profiles/submission.<job>.log`, and `okc status <job> --json` has it as `log_tail`.

Runs on this problem have a daily limit, and the same code again returns its earlier result. `okc withdraw <job>` takes a run off the board. A check run (`okc check`) is the same run kept off the board and out of your attempts, for experiments, with a daily limit of its own. `okc problems` shows today's limits and what you have left; they reset at 00:00 UTC.

A run has 15 minutes in all, compiling included; one that runs past that is charged as a timeout.

## Where you stand

```bash
okc leaderboard muse_attention_decode
```

Your best run is your row on the board, which is on the site too, at https://openkernelchallenge.dev/problems/muse_attention_decode. It also counts toward the overall kernel board (`okc leaderboard`), which ranks by how much your best runs together would speed up a decode step.
