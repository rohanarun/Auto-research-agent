# Codex CLI vs Auto-research parallel harness: matched controls

The published historical timing chart does not establish that Auto Research is a better research agent. Both execution paths use Codex. Those runs had different full prompts, CLI versions, runtimes, scheduling and validation queues. Their stopping rule also accepted the supplied baseline. They cannot establish discovery speedup.

## What the new control matches

The CLI control must receive the app's **entire model-visible request**, including its research policy, Linux environment instructions and agent policy. Matching only the user task is insufficient. This is a prompt-matched CLI control, not a comparison against Codex's default instructions.

An opt-in local test exercises the real `AppStore.launchCloud -> CLI.chat -> DaytonaRunner` path and intercepts its outgoing requests with an in-process URL protocol. It captures 37 initial-agent requests across all 13 configurations, plus 27 successor requests with a replay fixture. No request reaches Daytona or a model. The control consumes the captured initial prompt bytes and arguments directly.

The configurations remain five repeats of 1/2/3/4 independent agents, and five repeats of 2/3/4 agents sharing every 60/180/300 seconds, for each harness. One agent has no peer context to share.

| Control | Required equality or treatment |
| --- | --- |
| Model and effort | Same configured model and effort; actual resolved effort must be observed. Two missing values do not prove equality. |
| CLI | Same pinned package **and actual binary hash**, not just a matching version label. |
| Instructions | Full initial user messages, base instructions and developer context compared from actual transcripts. Only fresh private state UUID paths may be normalized. |
| Tools and permissions | Same tool schemas, model catalog, sandbox, network denial, approval settings, disabled memory, plugins and additional agents. |
| Starting state | Fresh auth-only state; no resumed provider session, saved GUI chats, prior solutions or other-trial context. Same baseline tree hash. |
| Runtime | Same container image digest, Python/compiler versions and worker resources. |
| Workers | 2 CPU, 4 GiB RAM and 8 GiB disk per worker in both arms. Actual allocation must be checked. |
| Validation | Same external mathematical oracle and shared 4 CPU/8 GiB validation service with four admissions. Queue time is recorded inside elapsed time. |
| Dispatch | One matched pair at a time, randomized configuration and dispatch order, both groups prepared before release. |
| Timing | One coordinator release timestamp before either first CLI invocation. Includes CLI/provider startup, sharing and checker queues. Preparation and 300-second cooldown are separate. |
| Success | All three original frozen targets, canonical nonbaseline words, and independent replay. Baseline or partial results do not stop research. |

Later checkpoint contents will naturally differ because agents produce different research. Both arms must use the same capture, literal combination and injection rules; actual successor receipts and responses are audited. Identical replay content tests the transport format, not scientific benefit.

## Current verification boundary

The local production request capture passed: **64 requests**, including **37 initial requests copied byte-for-byte to the CLI control**. The parity tests reject changed or extra user instructions, argument differences, saved-state reuse, runtime differences and missing resolved effort. These are local fixtures, not additional benchmark results.

The corrected 130-group study has **not run**. Cloud Codex reported a usage limit. Actual provider context/runtime parity and integration of the common launch barrier and corrected target endpoint into both versioned drivers remain launch requirements. Provider load, caching and model randomness cannot be guaranteed identical; matched dispatch and full reporting reduce those uncertainties rather than pretending to eliminate them.

The red subtitle removed at the user's request remains removed. This document changes no historical timing values.
