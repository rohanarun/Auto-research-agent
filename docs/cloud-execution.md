# Cloud execution with Daytona and Agent37

Version 1.5.0 and current source builds support **Daytona and Agent37** as cloud sandbox providers.
Both run Codex through the same persistent bridge, with single-agent research,
parallel agents, checkpoint sharing, optional research methods, worker chat and
recovery exports. [Download version 1.5.0](../README.md#download) or
[build the current source](../README.md#build-from-source) to use either provider.

## Connect a provider

1. Open **Connections & settings → Cloud**.
2. Choose **Daytona** or **Agent37** in **Sandbox provider**.
3. Enter your own API key for that provider and click **Save key**.
4. Click **Connect** to prepare a private connection sandbox.
5. Click **Sign in to cloud Codex** and complete the official device login.
6. Select the provider's **sandboxes · Codex** execution option and start research.

Your provider account pays for cloud compute and storage. Codex model access uses
your separate Codex account. A local Codex installation is not required for cloud
execution. Claude, OpenCode and Antigravity run locally; built-in MUSE challenge
starters and CPU search also keep their local coordinators.

API keys use separate macOS Keychain services. Each provider has its own settings,
installation identity, connection sandbox and ownership ledger. The app never
uploads the Mac's Codex account files. Cloud Codex sign-in is transferred directly
between owned sandboxes; API keys and sign-in files are excluded from research
prompts and recovery archives.

## Provider differences

| Behavior | Daytona | Agent37 |
| --- | --- | --- |
| Management API | `https://app.daytona.io/api` | `https://api.agent37.com/v1` |
| Editable defaults | [`DaytonaDefaults.json`](../Resources/DaytonaDefaults.json) | [`Agent37Defaults.json`](../Resources/Agent37Defaults.json) |
| Initial image setting | `snapshot` selects a Daytona snapshot | `snapshot` selects an Agent37 template |
| Persistent files | Retained sandbox filesystem | Checkout, private state and tools on the `/workspace` mount |
| Idle behavior | Stops after the configured idle period | Sleeps after the configured idle period |
| Successful recovery export | Stops and archives the worker | Stops the instance and retains its disk |

Both default to a 60-minute idle period. For Agent37, `idleStopMinutes` configures
its auto-sleep timeout. Runtime settings are stored as `daytona.json` or
`agent37.json` in the selected workspace; changing bundled defaults does not
overwrite saved settings.

Agent37 has no Daytona-style archive operation. Its stopped instances continue to
incur storage charges until you delete them in the provider dashboard. The app
never deletes instances. Recovery first downloads a credential-excluded archive
and verifies its size, manifest and SHA-256. A failed export preserves the original
sandbox; inspect the error and retained files before retrying recovery.

**Stop owned cloud compute** reconciles and stops this installation's instances
for the selected provider. It preserves files and sign-in. Switch providers to
manage the other provider's owned instances. Opening the project button for a
cloud chat opens that chat's provider dashboard.

## Existing chats and shared verification

Changing the selected provider affects new cloud work. Saved cloud chats keep
their original provider; select that provider before resuming them. The app blocks
provider changes while runs or cleanup are active and preserves legacy Daytona
chats and settings.

The same research-method runner and saved operation monitoring work on either
provider. Observation failures preserve accepted remote work; reconnect to its
saved operation instead of creating a replacement. Explicit Stop cancels owned
work. The Mac controller must remain open for scheduled checkpoint handoffs.

Daytona and Agent37 workers receive the shared Lean verifier connection only when
HTTPS sharing is enabled. See [CPU search and shared Lean verification](lean-search.md)
for setup and the distinction between a completed model turn and checked proof
evidence.

## Validation

Offline tests cover provider API contracts, ownership, uncertain creation,
command failures, truncated reads, credential transfer and saved settings. The
release app build and native settings rendering are also checked. These checks
do not establish a live Agent37 deployment or authenticated Codex research run.

The optional Agent37 infrastructure smoke test requires your own funded account
and `AGENT37_API_KEY` set securely in the environment:

```sh
TMPDIR="$PWD/tmp/" RESEARCH_LIVE_AGENT37_TEST=1 \
  swift test --filter testLiveAgent37PersistentBridge
```

It creates one instance, checks bridge upload and file persistence across a
stop/start, packs a recovery archive, and stops the instance with its disk retained.
It prints the instance ID for dashboard cleanup. It does not invoke a model.
