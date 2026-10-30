# Dense projections (`muse_bf16_proj`)

The attention and MoE-block projections and the router, as one GPU runs them serving Muse Spark Open on 8×H200. They are BF16 (the router FP32).

Your job is to compute this faster than SGLang's kernel does. Beside this file, `okc init` wrote:

- `submission.py`, your kernel. It starts as a plain PyTorch version of the computation. Keep its first line, `# okc: problem=muse_bf16_proj`: it tells `okc submit` the problem.
- `baseline.py`, SGLang's kernel: the one you're timed against.
- `reference.py`, the computation as the judge checks it: `generate_input`, the reference and the correctness checks, with `checks.py` beside it: the spec your output must meet. The judge uses its own copies, and runs them with steps of its own (the weights, routing fields, fresh inputs), so to check a change, use `okc check`.
- `AGENTS.md`, notes for a coding agent, if you use one.

## What to compute

```python
# x: [m, k] BF16   W: [n, k] = weights[layer]
# qkv_gate, o, pooler, unpooler: BF16; router: FP32
out = x @ W.T  # [m, n], BF16 (router: FP32)
```

`kernel_fn(data, state)` is the timed call and returns bfloat16 / float32 [m, n]. `state` is whatever `prepare(weights)` returned: it runs once, untimed, for up to 300 s. The `Inputs`, `Config` and `Weights` classes in submission.py give every field's dtype and shape; return GPU tensors of exactly that shape and dtype.

## Shapes

- m is the tokens per step: 1, 16 and 32 for decode, and 8, 128 and 256 for 8-token verify blocks. The weights arrive once, in prepare(), untimed.
- Decode loads are SGLang serving with speculative decoding off; verify loads are its default recipe, which checks each request's 8-token draft in one step.

| load | shape | weight | floor |
|---|---|---|---|
| decode 1 | m 1, layer qkv_gate | 62 | 7.9 µs |
| decode 1 | m 1, layer o | 62 | 3.5 µs |
| decode 1 | m 1, layer pooler | 62 | 14.0 µs |
| decode 1 | m 1, layer unpooler | 62 | 1.7 µs |
| decode 1 | m 1, layer router | 62 | 1.8 µs |
| decode 16 | m 16, layer qkv_gate | 62 | 7.9 µs |
| decode 16 | m 16, layer o | 62 | 3.6 µs |
| decode 16 | m 16, layer pooler | 62 | 14.1 µs |
| decode 16 | m 16, layer unpooler | 62 | 14.1 µs |
| decode 16 | m 16, layer router | 62 | 1.8 µs |
| decode 32 | m 32, layer qkv_gate | 62 | 8.0 µs |
| decode 32 | m 32, layer o | 62 | 3.6 µs |
| decode 32 | m 32, layer pooler | 62 | 14.1 µs |
| decode 32 | m 32, layer unpooler | 62 | 14.1 µs |
| decode 32 | m 32, layer router | 62 | 1.9 µs |
| verify 1 | m 8, layer qkv_gate | 62 | 7.9 µs |
| verify 1 | m 8, layer o | 62 | 3.5 µs |
| verify 1 | m 8, layer pooler | 62 | 14.0 µs |
| verify 1 | m 8, layer unpooler | 62 | 1.8 µs |
| verify 1 | m 8, layer router | 62 | 1.8 µs |
| verify 16 | m 128, layer qkv_gate | 62 | 8.4 µs |
| verify 16 | m 128, layer o | 62 | 4.0 µs |
| verify 16 | m 128, layer pooler | 62 | 14.6 µs |
| verify 16 | m 128, layer unpooler | 62 | 14.6 µs |
| verify 16 | m 128, layer router | 62 | 2.2 µs |
| verify 32 | m 256, layer qkv_gate | 62 | 9.8 µs |
| verify 32 | m 256, layer o | 62 | 4.5 µs |
| verify 32 | m 256, layer pooler | 62 | 17.4 µs |
| verify 32 | m 256, layer unpooler | 62 | 17.4 µs |
| verify 32 | m 256, layer router | 62 | 2.7 µs |

A shape's weight is how many times one step runs it. A load's time is the sum of its shapes' times, each times its weight. Your speedup on the problem is the geometric mean, over loads, of SGLang's load time over yours, so each load counts equally. The time shown for a run is SGLang's time per step, averaged over loads the same way, divided by that speedup.

The floor is the time to move the data at 4.8 TB/s or do the math at 989 TFLOP/s, whichever is longer.

Before timing, a run must pass these 9 tests, some harder than the scored shapes:

```
m 1, layer qkv_gate
m 7, layer pooler
m 33, layer o
m 130, layer unpooler
m 5, layer unpooler
m 8, layer router
m 9, layer router, dist outliers
m 512, layer qkv_gate, dist outliers
m 1000, layer unpooler, dist outliers
```

## Correctness

Every test and every timed call is checked against a reference. Errors are relative L2, |out − ref| / |ref|:

- The output, against an FP64 reference: at most 1%, and 0.01% for the router, whose outputs choose the experts. Splitting the router's FP32 weights into a BF16 hi + lo pair with FP32 accumulation, as SGLang does above 8 tokens, passes at any m; BF16 router weights alone don't.
- Use the weights at their own precision (the router's hi + lo split aside). About one row in eight of x has a single nonzero entry, a power of two, so its output is a column of the weights, exact with the original weights and slightly off with the router's hi + lo split. Those rows must be within 0.1%.
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
- **Weights.** Repack them in `prepare()`, and compress them losslessly if you like, but don't store them at a different precision: NVFP4 weights stay NVFP4 and BF16 weights stay BF16, or smaller, losslessly. Dequantizing inside your kernels is fine. The router's FP32 weights may run as a BF16 hi + lo pair with FP32 accumulation, as SGLang does above 8 tokens.
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

A run has 10 minutes in all, compiling included; one that runs past that is charged as a timeout.

## Where you stand

```bash
okc leaderboard muse_bf16_proj
```

Your best run is your row on the board, which is on the site too, at https://openkernelchallenge.dev/problems/muse_bf16_proj. It also counts toward the overall kernel board (`okc leaderboard`), which ranks by how much your best runs together would speed up a decode step.
