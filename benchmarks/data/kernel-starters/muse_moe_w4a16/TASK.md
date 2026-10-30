# Routed experts (`muse_moe_w4a16`)

Top-8 of 256 experts at width 4096: SwiGLU, NVFP4 weights, and a norm on each expert’s output before it is weighted and summed.

Your job is to compute this faster than SGLang's kernel does. Beside this file, `okc init` wrote:

- `submission.py`, your kernel. It starts as a plain PyTorch version of the computation. Keep its first line, `# okc: problem=muse_moe_w4a16`: it tells `okc submit` the problem.
- `baseline.py`, SGLang's kernel: the one you're timed against.
- `reference.py`, the computation as the judge checks it: `generate_input`, the reference and the correctness checks, with `checks.py` beside it: the spec your output must meet. The judge uses its own copies, and runs them with steps of its own (the weights, routing fields, fresh inputs), so to check a change, use `okc check`.
- `nvfp4.py`, the NVFP4 helper (the judge supplies its own copy).
- `AGENTS.md`, notes for a coding agent, if you use one.

## What to compute

```python
# per token: its top 8 of 256 experts; only local ones here
for t, (e, g) in routes if e is local:
    h = x[t] @ dequant(w13[e]).T
    gate, up = split(h)
    y = (silu(gate) * up) @ dequant(w2[e]).T
    out[t] += g * rmsnorm(y * (1 + post_norm_w))
```

`kernel_fn(data, state)` is the timed call and returns bfloat16 [M, H]. `state` is whatever `prepare(weights)` returned: it runs once, untimed, for up to 300 s. The `Inputs`, `Config` and `Weights` classes in submission.py give every field's dtype and shape; return GPU tensors of exactly that shape and dtype.

## Shapes

- bs (M in the shapes) is the tokens per step: 1, 16 and 32 for decode, and 8, 128 and 256 for 8-token verify blocks. The weights arrive once, in prepare(), untimed.
- This GPU holds 32 of the 256 experts, as under 8-way expert parallelism. Every token arrives, and only its routes to local experts are computed.
- Routings follow real routers’ skewed expert popularity. A step waits for the busiest GPU, so each routing makes this GPU the busiest.
- Decode loads are SGLang serving with speculative decoding off; verify loads are its default recipe, which checks each request's 8-token draft in one step.

| load | shape | weight | floor |
|---|---|---|---|
| decode 1 | bs 1, sharding ep8 | 60 | 11.4 µs |
| decode 16 | bs 16, sharding ep8 | 60 | 52.5 µs |
| decode 32 | bs 32, sharding ep8 | 60 | 70.8 µs |
| verify 1 | bs 8, sharding ep8 | 60 | 37.3 µs |
| verify 16 | bs 128, sharding ep8 | 60 | 101.3 µs |
| verify 32 | bs 256, sharding ep8 | 60 | 108.9 µs |

A shape's weight is how many times one step runs it. A load's time is the sum of its shapes' times, each times its weight. Your speedup on the problem is the geometric mean, over loads, of SGLang's load time over yours, so each load counts equally. The time shown for a run is SGLang's time per step, averaged over loads the same way, divided by that speedup.

The floor is the time to move the data at 4.8 TB/s or do the math at 989 TFLOP/s, whichever is longer.

Before timing, a run must pass these 5 tests, some harder than the scored shapes:

```
bs 4, sharding ep8
bs 33, sharding ep8
bs 257, sharding ep8
bs 128, sharding ep8, dist skewed
bs 1, sharding ep8
```

## Correctness

Every test and every timed call is checked against a reference. Errors are relative L2, |out − ref| / |ref|:

- The output, against an FP32 reference: at most 1.5%.
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
- **Weights.** Repack them in `prepare()`, and compress them losslessly if you like, but don't store them at a different precision: NVFP4 weights stay NVFP4 and BF16 weights stay BF16, or smaller, losslessly. Dequantizing inside your kernels is fine.
- **Activations.** BF16 wherever SGLang's kernel keeps BF16.
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
okc leaderboard muse_moe_w4a16
```

Your best run is your row on the board, which is on the site too, at https://openkernelchallenge.dev/problems/muse_moe_w4a16. It also counts toward the overall kernel board (`okc leaderboard`), which ranks by how much your best runs together would speed up a decode step.
