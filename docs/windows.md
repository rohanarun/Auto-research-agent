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
  the certificate chain. The saved catalog and the built-in kernel projects
  remain available, and **Open research → refresh** retries later.

## What the native macOS app still does

The native app remains the macOS interface: same workspace files, same chat
history (`Runtime/state.json`), same provider sessions. A chat started on
Windows opens in the macOS app and vice versa, because both editions write
the same state format with the same provider session IDs.
