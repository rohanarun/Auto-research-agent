# CPU search and shared Lean verification

Auto Research bundles `Resources/LeanSearch`. Local agents and Daytona workers receive its skill and CLI when a run starts. Research choices belong to the current model and editable policy; the tools retain evidence and run the configured checks.

In **Connections & settings → Lean server**, select an installed Lean binary and start the server. It listens on a dynamically selected loopback port, verifies its authenticated health endpoint, and supplies a private connection file to new local research runs. Choose a dedicated server folder on the external or internal disk, outside synced folders. Changing folders does not migrate or delete the previous server's data.

The initial `kernel-check` profile is an end-to-end service check, **not QRH certification**. It checks the exact statement `1 + 1 = 2`, including an independent statement check and terminal axiom audit. Other profiles are operator-owned entries in `server-config.json`; stop the service before adding one. Each profile uses a dedicated prepared project directory, exact verification plan, candidate-file allowlist and macOS isolation configuration. Never reuse a checkout with an active writer. A single queue serializes builds so profiles can retain their `.lake` caches without overlapping source changes. Candidate files are backed up and restored. Jobs, original requests, plans, logs and receipts remain in the server folder. Interrupted jobs are not silently replayed.

**Share with ngrok** starts an owned tunnel using the installed ngrok CLI and its existing login. It disables HTTP inspection in the local agent and retains a tunnel log. Bearer authentication remains required through the HTTPS URL. New Daytona workers receive the remote connection only while sharing is enabled; other nodes can use the same client files. **Stop sharing** closes that tunnel without stopping localhost. **Stop server** drains the active proof; queued requests remain preserved. App quit waits for the owned service to finish. No other ngrok processes or sandboxes are stopped.

## QRH first

The Track A recipe refers to [QRH PR #9](https://github.com/rohanarun/quasi-rh-hillclimb/pull/9). Inspect the exact source and freeze its revision, certifier hash and theorem scope before a run. The included historical baseline has ell=20003/120000 and h=65003/80000. It is a regression point, not an optimized target.

A CPU plan names the pinned `certify.py`, SHA-256, exact rational candidates, the full parameter box, and subdivision depth. Run:

```sh
python3 -B cli.py qrh-search --plan search.json --output new-screen-directory
```

SymPy is needed on the CPU screener. This adapter calls the exact certificate functions, checks both polynomials and all side constraints, and saves the rational Bernstein boxes. It does not accept a successful Python exit code as mathematical evidence. `EXACT_CONDITIONAL` remains conditional on the underlying theorem; `NOT_CERTIFIED` does not prove infeasibility.

For a full QRH Lean profile, prepare a **dedicated** copy of the reconstructed formalization using the pinned public-source workflow in QRH's `lean-tau` branch. Use that workflow's exact dependency/tool revisions, challenge construction and Comparator restrictions. Have the research model produce and review the candidate Lean changes. The harness does not automatically rewrite tau numerals. Bind the independent challenge, audit file, dependency lockfile and trusted checker inputs in the plan's `pins`. List all three terminal roots and preserve the original pole/family conditions. `qrh-track-a.json` supplies the stage shape; adapt absolute Lean/Lake/Comparator paths and filenames to the prepared build. Pre-fetch dependencies before service execution, which denies network access.

A profile entry has this shape (paths and approved files come from the actual proof project):

```json
{
  "project": "/absolute/dedicated/build-project",
  "editable_files": ["Candidate.lean"],
  "isolation": {"kind": "macos", "compiler_root": "/absolute/installed/lean"},
  "plan": {
    "scope": "Exact terminal theorem statement and assumptions",
    "targets": ["Candidate.result"],
    "permitted_axioms": ["propext", "Classical.choice", "Quot.sound"],
    "pins": {"Audit.lean": "actual-sha256"},
    "stages": [
      {"kind": "build", "argv": ["/absolute/lake", "build", "+Candidate"]},
      {"kind": "axioms", "argv": ["/absolute/lake", "env", "lean", "Audit.lean"]},
      {"kind": "comparator", "argv": ["/absolute/lake", "env", "/absolute/comparator", "challenge.json"], "success_markers": ["Lean default kernel accepts the solution", "Your solution is okay!"]}
    ]
  }
}
```

The native host implements macOS isolation. Linux Comparator runs needing Landlock/systemd guards must still use the documented Linux verification environment; a macOS kernel/axiom pass must not be reported as that guarded Comparator result. Generic direct verification can run inside an already isolated Agent37/Daytona Linux workspace. Hosting the native queue on Linux is not implemented.

## Search-node protocol

A connection JSON contains `url` and `token_file`. A relative token path is resolved beside the connection file. Keep both outside version control and set token-file permissions to 0600. Use HTTPS for remote URLs.

```sh
python3 -B cli.py profiles --connection connection.json
python3 -B cli.py submit --connection connection.json --candidate candidate.json --handle new-job.json
python3 -B cli.py status --connection connection.json --handle new-job.json
```

`candidate.json` contains a registered `profile`, a `files` object mapping approved relative source paths to text, and optional metadata with exact candidate parameters/provenance. The server ignores client claims of success. The CLI writes the UUID before submission, never automatically retries a mutation, and polling retrieves the existing job. `GET /v1/jobs/{id}/logs/{stage-log-name}` retrieves a configured log tail; full logs stay on the host. Every endpoint requires bearer authentication. Reusing an ID with different source is rejected.

The service does not accept client-supplied commands, arbitrary file paths, or changes to the trusted challenge. Source compilation is still code execution: profiles must refer to dedicated build projects containing no account files. The sandbox allows reads of system/toolchain/project files, denies network and unrelated user files, and restricts writes to generated build locations. Prepared dependency caches are read-only. Profiles, tools and dependencies are trusted operator inputs; receipts are not signed attestations or an adversarial proof-auditing substitute for Comparator.

`remote.py` exposes adapters for existing Agent37 and Daytona instances using explicit provider URLs, instance IDs, environment-variable key names and transport budgets. It does not create paid sandboxes or manage model authentication. Consult [Agent37 exec](https://www.agent37.com/docs/agents-api/exec) and [Daytona process execution](https://www.daytona.io/docs/en/process-code-execution/). Agent37's API timeout does not stop its command: use a persistent worker/background process and poll the retained handle. Search nodes on either provider can submit to the shared HTTPS verifier without installing Lean.

## Other projects and evidence

Use `generic-plan.json` after inspecting the project's authoritative checker. The same pipeline supports integer-multiplication and other Lean projects once their exact terminal theorem, assumptions, challenge, tools and allowed candidate files are registered. No automatic theorem adapter is inferred from the repository name.

`LEAN_CHECKED` means the configured build and terminal axiom audit passed. `COMPARATOR_ACCEPTED` additionally means the configured Comparator exited successfully with its acceptance markers. Scope, exact source hashes, pinned plan, raw logs and dependency qualification accompany the receipt. Neither status claims a new record, publication or maintainer acceptance. A theorem-scope or input change invalidates reuse. Time limits, if needed, are explicit in the plan; interrupted checks stay incomplete.

Validation commands:

```sh
LEAN_TEST_BINARY=/absolute/installed/lean python3 -B scripts/test_lean_search.py
AUTORESEARCH_LEAN_LIVE_TEST=1 swift test --filter LeanServerTests
```
