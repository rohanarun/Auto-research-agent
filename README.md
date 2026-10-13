# Auto-research agent

The idea is to make it dead simple to point your Codex or Claude at open problems, put unused tokens to work, and help push the frontier.

Auto-research agent is a native macOS app with a Codex-style project sidebar and chat. It helps you find GitHub projects with `auto-research.md` agent guides and makes starting a contribution a one-button click: pick an open problem and press Play.

It runs the Codex or Claude Code CLI under the hood, using your own account. If your chosen CLI is already installed and logged in, you can start immediately after the app's initial workspace setup, with no separate login needed.

The sidebar discovers repositories from [Auto Research by Super](https://app.getsupers.com/auto-research), with owner exclusions configured in `Resources/CatalogPolicy.json`. Layr-Labs projects are excluded from the sidebar, including saved catalog entries. Existing chats and checkouts are preserved. It supports one-click research launches, persistent conversations, live tool output, stop controls, provider login, and an optional model-directed research loop.

![Auto-research agent on macOS, showing the research project sidebar, native backend chart, latest claim, and recent pull requests](docs/images/auto-research-agent-charts.png)

## Download

[Download for macOS — Apple Silicon (.dmg)](https://getsupers.com/downloads/Auto-research-agent-1.1.1-arm64.dmg) · [SHA-256 checksum](https://getsupers.com/downloads/Auto-research-agent-1.1.1-arm64.dmg.sha256)

Requires macOS 14 or newer. The app and disk image are Developer ID signed and Apple notarized. Open the DMG, drag **Auto Research** into Applications, and launch it. On first launch, choose a folder on a mounted external drive for research files, then connect your installed Codex or Claude Code CLI in **Connections & settings**. The download contains no account credentials or local chats.

## Build from source

Requirements: macOS 14 or newer, Xcode with its Swift toolchain, Python 3, Git, and at least one installed CLI: [Codex](https://developers.openai.com/codex/cli) or [Claude Code](https://code.claude.com/docs/en/overview). The current workflow requires a mounted, writable external volume outside iCloud-synced folders. The first build fetches the public catalog over HTTPS.

From a terminal in your external-volume development workspace:

```sh
git clone https://github.com/rohanarun/Auto-research-agent.git
cd Auto-research-agent
./scripts/build.sh
open "build/Auto Research.app"
```

Open **Connections & settings**, check the detected CLI paths, and sign in if needed. Select a provider and press a project's Play button. This is an independent app; it is not an official OpenAI or Anthropic product.

## Open the app

Open `build/Auto Research.app`. The build targets the host architecture and has been tested on Apple Silicon. Source builds are locally ad-hoc signed; the downloadable release above is signed and notarized. Keep the build and its runtime workspace on the mounted external volume.

The build generates an ignored `Resources/Defaults.json` from `Resources/Defaults.example.json`, discovering CLI executables from `PATH` and using this checkout's `Runtime/` folder. `Runtime/settings.json` stores settings after the first save. Model fields are intentionally empty so each CLI chooses its configured default. Existing runtime settings and chats are preserved.

Optional build environment variables are `RESEARCH_WORKSPACE` (an external-volume runtime directory), `RESEARCH_CODEX_PATH`, `RESEARCH_CLAUDE_PATH`, and `RESEARCH_APP_OUTPUT`. Configure these before the first build or edit CLI paths in the app. Moving an existing runtime requires preserving its files and settings; the build does not migrate active projects.

## Use it

- The top-left **Open research** list refreshes from the supplied Auto Research website. Search by project, owner or topic. A bundled source snapshot is available offline.
- Select a sidebar project to see a compact native chart from the backend, its latest claim and source, and recent PRs. Estimated and observed histories are separate; projects with multiple metrics have a metric picker. Data refreshes at the backend’s requested interval. Projects without backend data show an empty state. Existing chats remain available through **Open chat**, and **New chat** opens the message composer.
- Press the play button next to a repository to create a chat, clone into the workspace if needed, load its `auto-research.md` verbatim, and run with the selected provider. If that file is absent, the agent is explicitly told to find the repository's actual guide and report which source it uses. It must ask if no actionable objective exists.
- Type a follow-up and press Command-Return. Follow-ups resume the exact provider session, including after relaunch. Changing providers starts a new chat.
- **Connections & settings** contains CLI paths, optional model overrides, permission settings, login and the editable research policy. Login runs the provider's official command and uses its browser flow. The app does not copy credential files. Claude requires its own sign-in.
- **MCP tools** is an explicit allowlist for direct Codex MCP connections. The app discovers configured server names and disables unselected servers for each invocation. CLI plugins are not loaded in research runs. Nothing is enabled by default, so unrelated global integrations do not start with a research session. Global Codex configuration and credentials remain unchanged.
- **Keep researching** opts into another model turn after successful completion, only when the agent explicitly decides another experiment is justified. Missing decisions, blockers, finished work and errors pause the loop. It uses `Resources/ContinuePolicy.md`. Turn it off or press Stop to end continuation. It is off by default and does not survive app restart.
- The chat menu opens the repository, checkout, raw run logs, or exact CLI session in Terminal. Stop an in-app run before opening the same session in Terminal.
- Add public GitHub repositories or existing external-SSD folders with the plus button. Existing checkouts are preserved and are never reset or automatically pulled.

## Permissions and evidence

Codex defaults to the workspace-write sandbox; read-only is available. The headless bridge sets approval policy to never, so operations needing additional approval fail visibly instead of hanging. Claude defaults to its automatic permission review, with accept-edits, plan and deny-unapproved modes available. No bypass-permissions option is enabled by the app. Configured CLI permissions can limit network or tool operations; use the Terminal handoff for interactive approvals.

The UI distinguishes preparing, running, stopped, interrupted, completed and needs-attention states. Exit status alone is insufficient for success: the provider must emit a completed-turn event. A completed turn is not a verified research result; the editable research policy requires the agent to describe its evidence accurately.

Only one in-app run writes a given project at a time. Other independently launched tools cannot be globally coordinated by this app. Stop cancels the selected CLI process and its currently observed child processes; independently detached remote work may need separate cleanup. Closing the window leaves research running; quitting prompts to stop active runs and saves history.

## Data

`Runtime/projects/` contains managed clones; `Runtime/state.json` contains chat history; `Runtime/sessions/` contains raw event logs; `Runtime/cache/` and `Runtime/tmp/` hold disposable output. The app validates the mounted external volume and rejects paths that escape it. Existing CLI-owned credentials and provider histories remain in their normal locations outside this workspace; the app does not migrate them.

## Build and test

```sh
./scripts/build.sh
TMPDIR="$PWD/tmp/" \
CLANG_MODULE_CACHE_PATH="$PWD/build-cache/clang" \
SWIFT_MODULECACHE_PATH="$PWD/build-cache/swift" \
swift test --scratch-path "$PWD/.build" --cache-path "$PWD/build-cache/spm" --disable-sandbox
```

Set `RESEARCH_LIVE_CODEX_TEST=1` for the opt-in two-turn authenticated smoke test. This makes real provider requests. Core tests cover catalog extraction, session commands, protocol events, storage boundaries, exact instruction selection, UTF-8 process I/O and cancellation. The repository has no third-party Swift dependencies.

Set `RESEARCH_LIVE_DASHBOARD_TEST=1` to verify the public backend chart feeds; this test makes no provider CLI calls.

Provider integration references: [Codex non-interactive mode](https://developers.openai.com/codex/noninteractive/), [Claude programmatic usage](https://code.claude.com/docs/en/headless). Catalog: [Auto Research by Super](https://app.getsupers.com/auto-research).

## Package a release

Set `RESEARCH_SIGNING_IDENTITY` to your Developer ID Application identity and `RESEARCH_NOTARY_PROFILE` to an existing notarytool keychain profile, then run `./scripts/package_dmg.sh`. The script generates portable defaults, signs and notarizes the app and DMG, staples both tickets, verifies the image, and writes the release and checksum under `build/distribution/`. It does not upload files or interrupt running research.

## Contributing and license

Contributions are welcome. Include the problem, the expected behavior, and relevant test results. For protocol or process changes, add a regression test using synthetic events; real provider tests are opt-in and use the contributor's account. Do not include credentials, local account configuration, chat transcripts, research checkouts, or raw provider logs in issues or pull requests.

Licensed under the [MIT License](LICENSE). Codex, Claude Code, the fetched catalog, and the research repositories are separate projects with their own terms and licenses; their contents are not relicensed by this repository.
