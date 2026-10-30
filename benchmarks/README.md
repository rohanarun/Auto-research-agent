# Auto-research benchmark

Rewind an open research project to a pinned predecessor, give an offline research agent 30 minutes, and measure whether its sealed work recovers the next improvement we previously made. This folder contains the runner, task manifests, evaluation code, initial results and chart source.

[Initial scores](RESULTS.md) · [Protocol and limitations](PROTOCOL.md) · [Release validation](VALIDATION.md) · [Raw results](data/results.json) · [Charts](charts/) · [Source and timing consistency check](tools/check_results.py)

The initial comparison uses GPT‑6.1 Sol, GPT‑6 Astra and GPT‑5.6 Sol, each with high reasoning. It is one trial per model/task using a structured-action Codex CLI broker. These are pilot observations, not a statistically established model ranking. The alternative notebook harness and API transports are available for new experiments; they were not part of the initial comparison.

## Tasks

The scored core is integer multiplication and the five MUSE/Open Kernel challenges: dense projections, attention decode, routed experts, decoder layer and prefill layer. Additional frozen tasks cover quasi-RH, van der Waerden, unit distance and Sidorenko. Proof submissions retain their original mathematical assumptions; finite arithmetic checks do not prove the analytic inputs or an end-to-end theorem.

[scenarios.json](scenarios.json) pins predecessor commits, snapshot hashes, improvement targets and evidence scope. Actor snapshots contain no Git history, remotes, future commits, reference solutions, account files or access to the live repository. Model providers can still have memorized public material during training; runtime isolation cannot remove that contamination.

## Setup

The verified local command backend requires macOS, Python 3.12 and Xcode Command Line Tools. Linux is used inside the isolated H200 evaluator; a Linux research-shell backend is not shipped. Keep the workspace on a mounted, writable external volume outside Documents, Desktop and iCloud. Clone this repository onto that volume, enter its `benchmarks` folder, and set paths for your own machine:

```sh
export AUTO_RESEARCH_WORKSPACE="/your/mounted/volume/auto-research-benchmark-workspace"
export AUTO_RESEARCH_STORAGE_ROOT="/your/mounted/volume"
export AUTO_RESEARCH_REQUIRE_MOUNT="/your/mounted/volume"
python3.12 setup.py --workspace "$AUTO_RESEARCH_WORKSPACE"
python3.12 bench.py preflight
python3.12 -m unittest discover -s tests -v
# Unit tests use synthetic credentials and mocked provider lifecycle calls; no GPU/model account is needed.
```

Setup is curator-only and uses the network to fetch exact pinned snapshots and public source dependencies. It checks the complete content hash. Run it before starting any actor. A failed hash or isolation preflight must stop the run. No package installation or network access is available inside an actor's command shell.

The CLI transport was verified with Codex CLI `0.165.0-alpha.1.2` and requires the tool-disabling and structured-output flags used by that build. Unsupported CLI flags fail closed. Install and authenticate your own compatible Codex CLI for this transport. `codex` must be on the host PATH. Set `CODEX_AUTH_FILE` only if your login file is elsewhere. A temporary login copy is accessible to the tool-free inference client, inaccessible to the research shell, and removed when a trial exits. Never publish workspace folders or raw client logs.

```sh
python3.12 tools/run_harness.py --harness codex-cli-broker-v1 \
  --model YOUR_MODEL_ID --effort high --seconds 1800 \
  --task integer-mult-frame-retiming --cohort YOUR_UNIQUE_RUN
```

Use `codex-cli-notebook-broker-v1` for the last-eight-actions plus contestant notebook variant. For another model provider, copy an example in [examples/](examples/), set `AUTO_RESEARCH_PLANNER_CONFIG` to that JSON file and set its named API-key environment variable on the trusted host. Responses and OpenAI-compatible chat transports send no tools and reject tool calls. Their HTTP contract is tested with a local mock; the initial scores used Codex CLI, not these API adapters. Add a transport by preserving the `{command, done, note}` action contract and the same isolated execution broker. Record every change to prompts, context management, tools, budget and environment as a distinct harness version. `AUTO_RESEARCH_PROMPT_FILE` selects an editable prompt; the optional v2 lesson prompt is distinct from the original pilot policy.

For integer admission, run `python3.12 tools/replay_reference.py` once in the curator workspace, then `python3.12 tools/adjudicate_integer.py`. The specialized admission route checks exact equivalence to the independently replayed reference; genuinely different constructions need their own full independent replay. Quasi-RH certificates can be checked with `python3.12 tools/adjudicate_quasi.py`.

## H200 runs

Set your own `DAYTONA_API_KEY` securely on the trusted host. Never put it in a prompt, actor environment, Git file or kernel bundle. The provider adapter allocates one fresh private H200 only when a GPU job is ready, blocks all egress, exports the result, and deletes the lease when the queue empties. Deletion is checked. A five-minute idle stop and deletion-on-stop are additional fallbacks. Availability, compilation and provisioning failures are infrastructure failures.

```sh
# Curator setup: warm only public dependencies and verify confined GPU execution.
python3.12 gpu/prepare.py

python3.12 tools/run_kernels.py --model YOUR_MODEL_ID \
  --cohort YOUR_UNIQUE_KERNEL_RUN --workers 1 --seconds 1800

# Standalone correctness/timing inspection of one sealed file:
python3.12 gpu/check.py --problem muse_attention_decode \
  --candidate /absolute/path/to/sealed/submission.py \
  --banks 2 --output "$AUTO_RESEARCH_WORKSPACE/heldout-result.json"
```

For a completed cohort, audit each final source against its frozen TASK.md and save an approval mapping keyed by `MODEL_ID/TASK_ID`, with `sha256`, `approved`, `source_rules_passed`, `reviewer` and a written rationale. Approval must bind the exact sealed source; do not approve from numerical output alone. Then run the complete final-source admission path:

```sh
python3.12 gpu/validate.py --cohort YOUR_UNIQUE_KERNEL_RUN-YOUR_MODEL_ID \
  --approvals /absolute/path/to/source-audits.json
```

Repeat `--cohort` for additional completed models; directory model IDs are URL-escaped by the runner. This evaluates two fresh banks, applies the frozen bootstrap/target policy, writes `adjudicated-result.json` beside each trial and deletes its GPU. It refuses to overwrite earlier final measurements. Standalone `gpu/check.py` is an inspection tool; its output alone does not admit a research-time breakthrough.

The diagnostic policy is editable in `gpu/research-policy.json`, or via `AUTO_RESEARCH_GPU_POLICY`; changing it creates a different harness configuration recorded in cohort metadata. The actor uses the file-only `gpu_test.py` bridge for at most three diagnostic requests, up to two public cases each. It never receives provider or OKC credentials. Diagnostics and GPU queue waits count against the wall-clock cap. Full final evaluations use fresh hidden seeds after the research clock stops. To limit GPU use, evaluation selects the final sealed source version and reports its earliest byte-identical in-budget checkpoint; it does not exhaustively evaluate every earlier source version. Audit the sealed source against TASK.md before admitting a score. GPU correctness, timing, source-rule compliance and target recovery are separate fields.

The warmup cache contains only public SGLang dependency compilations; keep it in the private workspace and never commit its binaries. The runtime image is pinned by digest. Correctness in this local protocol covers the public timed workload shapes with fresh seeds, not the organizer’s entire separate edge-case suite or hidden leaderboard tests. The private evaluator checks all public timed shapes on two fresh weight/input banks, validates changed-input CUDA graph replays, preserves authoritative oracle weights and original dtype/shape contracts, and measures randomized paired CUDA-graph blocks. Each block starts after a 64 MiB zero-fill, then averages repeated replays; caches can warm inside a block. The local score uses weighted shape aggregation and no fixed graph-cost subtraction. See the exact protocol before comparing scores. This is a local benchmark protocol, not an official Open Kernel submission or leaderboard score. No official check/submission quota is consumed.

Our optimized reference kernel sources remain private under their existing project policy. Their hashes, calibration traces and numeric targets are published. Public starter/evaluator sources are included with provenance and separate ownership notices. Others can evaluate their kernels against the fixed targets, but reproducing the exact reference implementation requires that source separately. Use your own reference source to define a new versioned target; do not silently replace a published target.

## Reproduce the charts

```sh
python3.12 -m venv /your/mounted/volume/chart-env
/your/mounted/volume/chart-env/bin/pip install matplotlib==3.10.7
/your/mounted/volume/chart-env/bin/python tools/chart.py
```

Charts read the committed result data. Timeouts are censored observations; missing, invalid and infrastructure results are never plotted as zero performance. Keep the seed, trial count, exact model identifier, harness hash and validation status with any new score. See [CONTRIBUTING.md](CONTRIBUTING.md).
