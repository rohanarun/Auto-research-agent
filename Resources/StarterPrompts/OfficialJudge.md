# Public kernel research with the official judge

Work on the MUSE challenge identified below using its public starter and your own improvements. This project uses a local coding-agent session for source analysis and CPU checks; the official service compiles, checks and times kernels on its GPU. You do not need a local CUDA installation or a rented GPU. Do not provision compute or start additional submission coordinators.

## Establish the task and budget

1. Check whether `okc` is available. If it is absent from `PATH`, check the official installer's `~/.okc/bin/okc` location before suggesting another installation. The official installation guide is https://openkernelchallenge.dev/. If installation or login is needed, explain the required Terminal step. A person must run `okc login` and accept the challenge rules; an agent cannot accept them on their behalf. Use only the user's account. Never print, copy into project files, or commit tokens.
2. Read `okc show <problem>`, `okc problems --json`, `okc jobs` and `okc leaderboard <problem> --json`. Substitute the problem identifier below. Establish the user's current best, the available quota and the experiment budget. Respect any budget already supplied; otherwise ask before spending checks. Quotas and availability can change, so do not assume fixed daily limits or a reset grants permission.
3. In this assigned workspace, run `okc init <problem> kernel` only if a task has not already been initialized there. Preserve existing sources and results. Read the downloaded `kernel/TASK.md`, `kernel/AGENTS.md`, `kernel/submission.py`, `kernel/baseline.py`, `kernel/reference.py` and any helpers. Current official instructions and https://openkernelchallenge.dev/rules govern the computation. Keep the problem header in the submission file.

## Make a small, testable improvement

Start with the public computation and SGLang baseline. Explain one suspected cost and one bounded change before editing. Basic starting points include redundant intermediate tensors, avoidable launches, strides and data layout, or an appropriate existing library primitive. Prefer a small understandable change over a broad parameter sweep. Use source inspection, Python syntax checks and small CPU checks where applicable; these do not establish GPU correctness or speed.

Distinguish improvement over the public starter, the official speedup against SGLang, and comparison with the user's scored best. Evidence for one comparison does not establish either of the others.

Keep the submission self-contained in one Python file. Use only libraries installed in the official image and helpers explicitly supplied by the task; include your own helper definitions in the submission instead of importing sibling files such as `baseline.py`.

Preserve all required outputs, cache writes, precision, intermediate rounding and CUDA-graph compatibility. Fusion must retain the specified rounding and computation order. Respect the rules for fresh inputs and untimed weight preparation. In particular, an untimed expanded weight cache is not a substitute for legal weight storage. Do not cache answers, exploit test values, import the judge's reference into a submission, or change the scoring machinery. The evaluator has its own inputs and copies of the tests. Consult the current rules before choosing a library or a transformation.

Preserve official compatibility failures, including unsupported operations, missing symbols and version mismatches, with their source hash and receipt. Inspect the current public baseline and image interfaces, and record the compatibility lesson before another attempt; local syntax or CPU success does not resolve a remote runtime failure.

Freeze each candidate's exact source and compute its SHA-256 before a remote attempt. Within the agreed check budget, run `okc check <problem> kernel/submission.py --json` and retain the job ID, source hash, returned result and generated profile. The command waits by default. For a previously launched job, use `okc status <job> --wait --json`; do not launch another attempt to recover its status. Read failures and per-case profiles, preserve negative and inconclusive results, and update `RESEARCH_HANDOFF.md` with the lesson and next hypothesis. Identical source can return an earlier result; a duplicate is not a new measurement. Do not repeat unchanged code to seek favorable timing noise.

## Scoring is a separate action

An off-board check is not a leaderboard result. Do not make a scored submission merely because Play was pressed or a check passed. A scored attempt needs the user's explicit instruction and budget, a completed clean official check of the exact frozen source, and current evidence that it satisfies the user's improvement criterion. Review the reported interval as well as the point estimate; overlapping uncertainty is not proof of a gain. Refresh the board, quota, recent jobs and ownership before submitting, and ensure no other coordinator is spending the same account's budget.

If the user has no scored entry, establish a first-baseline criterion and first-score budget with them. A missing score is not zero, and an initial scored baseline is not evidence of an improvement over an earlier entry.

When authorized and qualified, use `okc submit <problem> kernel/submission.py --json`. Preserve its separate receipt, then verify the actual scored result and rank with `okc status <job> --json` and `okc leaderboard <problem> --json`. Do not present a projected rank or a local timing as an achieved leaderboard improvement. If the source changes after checking, it is a new candidate. If the gate fails, report the remaining obligation instead of submitting or weakening it.

## Example follow-ups

- “Inspect the public starter and propose one basic improvement. Do not spend any official checks yet.”
- “Use the check budget I specify to test one hypothesis at a time. Preserve failures and stop when that budget is exhausted.”
- “Check this candidate and submit it if the exact source passes, its official evidence supports an improvement over my current scored best under my stated criterion, and it fits my remaining check and scored budgets. Otherwise explain why it is not ready.”

These are starter research directions, not a tested fast kernel or a promise of a particular score or rank. Report source analysis, CPU validation, off-board checks and scored outcomes separately.
