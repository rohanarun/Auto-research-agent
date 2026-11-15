# Web edition API (v1)

The web edition runs the same research harness as the native app and serves a
UI that mirrors the desktop layout. This document defines the HTTP API the
static UI (`webapp/static/`) talks to.

## Conventions

- JSON in and out unless stated. All requests and responses are UTF-8.
- Provider values are `codex`, `claude`, `opencode`, `antigravity`.
- Identifiers are UUID strings (uppercase, 8-4-4-4-12).
- Dates are numbers: seconds since `2001-01-01T00:00:00Z`
  (Swift's `timeIntervalSinceReferenceDate`), so `Runtime/state.json`
  written by the web edition can be read by the native app and vice versa.
- The optional auth token is sent on every API call as an `X-Auth-Token`
  header. Static files under `/` need no token; the token is passed to the UI
  as `?token=` on first open and stored in `localStorage`.
- A `version` integer increments on every state change. Poll `GET /api/state`
  to follow it; pass `?since=<version>` to long-poll (up to 25 s) until the
  version changes or the timeout returns the current state anyway.

## State shape

`GET /api/state` returns:

```json
{
  "version": 42,
  "projects": [ResearchProject],
  "sessions": [ResearchSession],
  "selectedProjectID": "builtin:muse_bf16_proj",
  "selectedSessionID": null,
  "provider": "codex",
  "settings": Settings,
  "auth": {"codex": "Connected", "claude": "Missing", "opencode": "Checking connection", "antigravity": "Signed out"},
  "server": {
    "name": "Auto Research",
    "workspaceRoot": "C:/Users/me/Auto-research-agent/Runtime",
    "catalogStatus": "14 repositories · live",
    "access": "lan",
    "port": 8765,
    "platform": "Windows",
    "editors": "The native macOS app shares this workspace; chats interchange."
  }
}
```

`ResearchProject` — `id`, `name`, `repository`, `summary`, `category`,
`localPath?`, `isCustom`, `builtinID?`. Project `id` is `builtin:<id>` for
built-in kernels, the local path for folder projects, otherwise the lowercased
repository (`owner/name`).

`ResearchSession` — `id`, `projectID`, `title`, `provider`,
`providerSessionID?`, `workingDirectory?`, `messages: [ChatMessage]`,
`createdAt`, `state`, `scripted?`, `closedAt?`.
States: `Ready`, `Preparing`, `Running`, `Stopped`, `Interrupted`,
`Completed`, `Needs attention`, `Closed`.

`ChatMessage` — `id`, `role` (`user`|`assistant`|`tool`|`error`|`status`),
`text`, `date`, `speaker?`.

`Settings` mirrors `Resources/Defaults.example.json` keys: `workspaceRoot`,
`catalogURL`, `codexPath`, `claudePath`, `codexModel`, `claudeModel`,
`codexSandbox`, `claudePermissionMode`, `opencodePath`, `opencodeModel`,
`opencodePermission`, `antigravityPath`, `antigravityModel`,
`antigravityMode`, `antigravityPermission`.

## Endpoints

- `GET /api/state?since=N` — full state, long-poll as described above.
- `GET /api/dashboard?projectID=<id>` — backend feeds for the project:
  `{as_of, refreshAfter, groups:[{id,name,direction,estimated:{value,points},observed:{value,points},latest:{sources,leaders}}], pulls:[{number,title,html_url,state,merged_at,status,author}], claimStatus}`.
  Charts render `estimated` and `observed` as step lines; `value` is the
  current metric. `404` when the project has no backend track.
- `POST /api/select` `{projectID?, sessionID?}` — set sidebar selection.
- `POST /api/projects/refresh` — refetch the live catalog and merge it;
  returns `{projects, status}`.
- `POST /api/projects/add` `{repository?|localPath?}` — add a public GitHub
  repository or an existing folder as a project.
- `POST /api/chats/create` `{projectID, provider?}` — open an empty chat.
- `POST /api/play` `{projectID, prompt?, provider?, parallel?, agentCount?, shareContext?, shareEvery?}`
  — create or reuse the project's open chat, prepare the workspace (clone or
  built-in starter), load `auto-research.md`, and start the agent. Returns
  `{sessionID}`.
- `POST /api/chats/send` `{sessionID, text}` — send a follow-up; resumes the
  exact provider session.
- `POST /api/chats/stop` `{sessionID}` — stop the run; the chat stays open.
- `POST /api/chats/close` `{sessionID}` / `POST /api/chats/reopen` `{sessionID}`.
- `GET /api/logs/<sessionID>` — raw run log text for the selected chat.
- `GET /api/settings` — settings plus `detectedPaths` and `auth`.
- `POST /api/settings` — merge the given keys into `Runtime/settings.json`.
- `POST /api/auth/login` `{provider}` — run the provider's login flow on the
  host device (interactive; the UI shows the returned instructions).
- `GET /healthz` — `{ok, uptimeSeconds, version}`; no token required.

## Behaviour parity with the desktop app

- Enter sends the composer text; Shift+Enter inserts a newline.
- The play buttons in the sidebar and header start the same saved run.
- Changing the provider in the header starts a new chat, as in the app.
- Closing a chat keeps messages, session IDs, logs and the checkout; it is
  reopened from the closed-chats list in the sidebar.
- Stop cancels the observed child process tree of that run only.
