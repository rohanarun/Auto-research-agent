# Windows and the live dashboard

Auto Research started as a native macOS app. The SwiftUI interface cannot run
on Windows — AppKit and SwiftUI exist only on macOS — so this edition runs the
same research harness as a **cross-platform server** and serves its dashboard
UI in the browser. The server and the native app share the same workspace
layout (`Runtime/state.json`, `Runtime/projects/`, `Runtime/sessions/`), so
chats interchange between editions.

The result on Windows:

- The same workflow: pick a project from the catalog, press Play, watch the
  agent work, send follow-ups, and stop or close the run.
- The same provider CLIs (Codex, Claude Code, OpenCode, Antigravity), invoked
  with the same arguments the native app uses, so a session started here can
  be continued in Terminal and the other way around.
- The dashboard is reachable from **other devices on your network**, including
  a Tailscale tailnet, so research can be watched and steered remotely.

## Install on Windows

1. Install the prerequisites:

   ```powershell
   winget install --id Git.Git -e
   winget install --id Python.Python.3.13 -e
   ```

2. Get the source and configure it:

   ```powershell
   git clone https://github.com/rohanarun/Auto-research-agent.git
   cd Auto-research-agent
   python scripts\setup_webapp.py
   ```

   `setup_webapp.py` creates the `Runtime` workspace and detects the provider
   CLIs from your PATH (`codex`, `claude`, `opencode`, `agy`). Install the CLI
   you want to use and sign in with its official command, exactly as the macOS
   app requires; the app never copies your account files.

3. Start the dashboard:

   ```powershell
   python -m webapp.server
   ```

   The address is printed: `http://127.0.0.1:8765`. Open it in a browser and
   connect your CLI in **Connections & settings** if the path was not detected.

## Reach the dashboard from other devices (Tailscale)

By default the server binds the loopback address. Start it with LAN access to
share the dashboard on your network; the API is then guarded by an access
token:

```powershell
python webapp\server.py --access lan
```

The server prints a URL with the token, for example
`http://<this device's address>:8765/?token=…`. On a Tailscale network, use
the device's Tailscale name or address from any other device in the same
tailnet:

```
http://my-windows-machine:8765/?token=…
```

Notes:

- Tailscale traffic is already encrypted end to end inside the tailnet; the
  token guards the API against other processes on shared networks. Keep the
  default unless a trusted device cannot open the URL.
- `--token my-secret` sets your own token, and `--port` changes the port.
- `--workspace D:\research\Runtime` places the workspace on another drive. The
  web edition runs from any writable directory and rejects synced folders
  (OneDrive, iCloud, Desktop, Documents); the native macOS app continues to
  require a mounted external SSD.
- If the dashboard shows "Offline · saved catalog" while your browser can
  open the catalog website, your machine's Python TLS stack could not build
  the certificate chain. The server then retries with the Windows system
  `curl.exe`, which uses schannel, and keeps the fetched page as a snapshot in
  `Runtime/cache/catalog.html`; the next offline attempt still shows the full
  repository list. The saved catalog and the built-in kernel projects remain
  available, and **Open research → refresh** retries later.

## The Windows app (exe)

The repo also packages the dashboard as a desktop app. `launcher\build.ps1`
builds `dist\AutoResearch\AutoResearch.exe`, a native window around the same
server. Double-click it — no terminal, no install steps, no admin rights. The
window shows the repo's icon and the dashboard the manual server serves.

- The exe serves the same workspace the manual server does, with zero
  migration. First run writes `%APPDATA%\AutoResearch\config.json` with the
  workspace root, port 8765, the device token and LAN access; edit that file
  to move the workspace or change the port. A machine without an existing
  install copy generates its own token in the format the server prints.
- Tailnet links keep working: the server binds all interfaces (`--access
  lan`) with the same token, so saved `http://<address>:8765/?token=…`
  bookmarks are unchanged.
- Provider CLIs are detected from PATH the same way as the manual server.
  Nothing is bundled or reinstalled, and signing in still opens a terminal
  window on the device that serves the dashboard. The exe reads the
  workspace's own checkout's `Resources` when the workspace is a checkout —
  the same generated `Defaults.json` with the detected CLI paths — so the
  provider setup matches the manual server with no re-setup.
- The catalog fetch uses the same `curl.exe` fallback when Python's TLS
  cannot build the certificate chain.
- A second launch — or a launch while the manual server already owns the
  port — does not start a second server. It notes "already running" and
  opens the dashboard in the browser instead.
- Closing the window stops the server and frees the port; the process exits
  with code 0. If research is still running, the app asks before quitting,
  like the native app's quit prompt, and stops the runs after confirmation.
- First run asks once — "Do you want a desktop shortcut for Auto Research?"
  Yes creates an icon on your real desktop (it honors Explorer's User Shell
  Folders redirect) pointing at the installed exe; No installs just the app.
  The answer is recorded in the config (`shortcutChoiceMade`,
  `desktopShortcut`) and the question is never asked again. If the shortcut
  is later deleted, the next run recreates it only when the answer was Yes.
- No terminal popups while the app works: the exe is a windowed process, and
  every console child it spawns — `git.exe`, `curl.exe`, `taskkill.exe`,
  `cmd.exe` wrappers, provider CLIs — allocates its own console window at
  launch. On this machine each spawn flashed a Windows Terminal window
  during the child's run (measured with a window-enumeration probe run from
  a console-less parent: a plain piped child spawn opened +1 visible
  console window; with `CREATE_NO_WINDOW` it opened +0 — the numbers are
  machine-specific: Windows 11 hosts consoles in a Windows Terminal window,
  class `CASCADIA_HOSTING_WINDOW_CLASS`, older hosts in
  `ConsoleWindowClass`). Every piped spawn in the webapp now passes
  `CREATE_NO_WINDOW` (`webapp/spawn.py`); the deliberate visible terminal
  that runs provider sign-ins keeps its window.
- If WebView2 cannot start, the tokenized dashboard URL opens in the default
  browser and the server keeps serving. The launcher logs to
  `%APPDATA%\AutoResearch\launcher.log`.

Build and test on the machine that runs the app:

```powershell
pip install -r launcher\requirements-packaging.txt
powershell -NoProfile -ExecutionPolicy Bypass -File launcher\build.ps1
powershell -NoProfile -ExecutionPolicy Bypass -File launcher\test_app.ps1
```

`build.ps1` uses the system Python 3, bundles `webapp/` and `Resources/` as
data, collects the pywebview shell, and zips `dist\AutoResearch` to
`dist\AutoResearch-windows-x64.zip`. `test_app.ps1` stops the manual server,
seeds the recorded shortcut choice into the config (backs it up first and
restores it at the end, so the smoke run never prompts and never touches
your choice), launches the exe twice, and asserts the token guard, workspace
parity with `Runtime/state.json`, the catalog status, a chart feed, the
window title and the tailnet binding over HTTP, then a clean shutdown and a
relaunch. The launcher's pure helpers (config seed, port check, mutex,
token) are covered by the Python suite in `scripts/test_webapp.py`, and
`scripts/test_desktop.py` covers the spawn flag and the shortcut-choice
roundtrip offline (the spawned processes are fakes).

Untested: a second, clean machine (first-run seeding there, WebView2 outside
Windows 11), and reaching the dashboard from other tailnet devices — only
this device's own tailnet address was verified. The exe is unsigned, so
Windows SmartScreen may ask on first run. Autostart, a tray icon, auto-update
and code signing are future work. The shortcut's creation through the
WScript.Shell COM object is verified only by the smoke run on this machine,
and the popup-probe numbers above are machine-specific — other consoles
(hosted vs. legacy) may count differently.

## Built-in kernels and the setup form

Built-in kernel projects (the MUSE challenges) use a setup form, mirroring
the native project page:

- **Your GitHub/OKC username** — verified with `okc whoami` before any run
  starts. **Refresh account** re-checks; **Choose account** opens a terminal
  window on the device that serves the dashboard and runs `okc login --force`
  there, exactly like the native app's Terminal launcher.
- **Kernel source** (optional), **official checks per day**, **Super
  Experiments pre-flight** on/off, optional **scored submissions** with their
  own budget and criterion, and the **research prompt** with the starter's
  suggested prompts.
- **Review request** shows the exact request the run will receive, with the
  user-supplied inputs rendered as JSON, and **Use in chat** copies it to the
  composer. **Start research** saves the setup, verifies the account, and
  starts one visible chat.

The setup drafts are saved in `Runtime/kernel-setups.json`, in the same shape
the native app writes, so drafts interchange between editions.

## Editing everything by hand

- **Edit instructions** (project "⋯" menu, or the guide button on a built-in
  project page) opens the checkout's `auto-research.md`. The starter guide is
  seeded once and never overwritten, so edits survive every run.
- **Research policy** lives in the settings sheet's own tab and is saved to
  `Runtime/research-policy.md`; both editions read it before every run.
- **Edit project details** and **Remove project** (catalog and local
  projects) rename, re-describe, re-target or remove the saved entry; chat
  history is preserved.
- **MCP tools** in the settings sheet lists the direct Codex MCP connections
  the CLI exposes; the selection is saved to `Runtime/mcp-selection.json` and
  passed to research runs the same way the native app passes it.

## Parallel agents

**Share context** on a project page starts several agents at once:

- Every agent gets **its own copy of the checkout** under
  `Runtime/parallel-agents/<group>/agent-N` (or `Runtime/shared-context/…`
  when sharing), so runs never write to the same files.
- Independent agents share nothing. With **Share context** enabled, each
  agent is asked to maintain `RESEARCH_HANDOFF.md` and checkpoints in its
  checkout; when all agents finish, their transcripts, notes and checkpoints
  are captured to `Runtime/shared-context/<group>/round-N/` and published as
  a `shared-context.md` bundle, mirroring the native format.
- The web edition runs **one round per Start research**. The native app also
  continues rounds on its own schedule; the web edition says so in each
  agent's status message.

## Signing in from a remote device

Interactive sign-in needs prompts and a browser, which a remote dashboard
cannot answer. The web edition therefore signs in **on the device that serves
the dashboard**:

1. Press **Sign in** on a provider card (or **Choose account** on a built-in
   project). The server writes a small launcher into `Runtime/terminal/` and
   opens it in a visible terminal window **on the host device**.
2. Complete the sign-in there — the official CLI stays in charge and
   credentials never touch the server.
3. Press **Refresh account** in the dashboard; the status turns green.

Pre-auth alternatives: sign in on the host device beforehand (the CLIs keep
their own account state, which the server reuses), or provide API keys in the
server's environment (for example `ANTHROPIC_API_KEY` for Claude Code).
**Run in the server session instead** keeps the old behavior for local use.

## What the native macOS app still does

The native app remains the macOS interface: same workspace files, same chat
history (`Runtime/state.json`), same provider sessions. A chat started on
Windows opens in the macOS app and vice versa, because both editions write
the same state format with the same provider session IDs. Cloud sandboxes
(Daytona and Agent37), the Lean server, shared-context **rounds** and CPU
search subtasks stay native-app features; the web edition's settings sheet
carries their tabs with notes where they are unavailable.
