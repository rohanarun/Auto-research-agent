"""Backend chart feeds, mirroring ResearchCore's ResearchDashboard.swift.

The catalog site publishes two JSON feeds next to the catalog page:
``api/contributors`` (history groups per track) and ``api/research``
(tracks with pull requests). The native app renders a compact chart from
these feeds; the web edition renders the same data in the project view.
"""
import json
import urllib.request


class DashboardError(Exception):
    pass


def _feed(url: str, timeout: float = 20.0):
    if not url.startswith("https://"):
        raise DashboardError("Use an HTTPS backend address.")
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise DashboardError("The research backend is unavailable.")
            return json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as error:
        raise DashboardError(f"The research backend is unavailable: {error}")


def fetch(catalog_url: str, timeout: float = 20.0) -> dict:
    if not catalog_url.rstrip("/").startswith("https://"):
        raise DashboardError("Use an HTTPS backend address.")
    base = catalog_url.rstrip("/")
    history = _feed(base + "/api/contributors", timeout)
    research = _feed(base + "/api/research", timeout)
    return {"history": history, "research": research}


def scope_for(history: dict, repository: str) -> dict | None:
    scopes = history.get("scopes") or {}
    for scope in scopes.values():
        if (scope.get("repo") or "").lower() == (repository or "").lower():
            return scope
    return None


def groups_for(history: dict, scope_id: str | None) -> list[dict]:
    if not scope_id:
        return []
    groups = history.get("groups") or {}
    matched = [g for g in groups.values() if g.get("track") == scope_id]
    matched.sort(key=lambda g: str(g.get("id", "")))
    return matched


def valid_intervals(history: dict) -> list[dict]:
    timeline = history.get("timeline") or []
    intervals = [i for i in timeline
                 if isinstance(i.get("from"), (int, float))
                 and isinstance(i.get("to"), (int, float))
                 and i["to"] >= i["from"]
                 and (i.get("value") is None or isinstance(i.get("value"), (int, float)))]
    intervals.sort(key=lambda i: i["from"])
    return intervals


def recent_pulls(track: dict) -> list[dict]:
    pulls = track.get("pulls") or []

    def key(pull):
        return pull.get("updated_at") or pull.get("created_at") or ""

    return sorted(pulls, key=key, reverse=True)


def interval_status(pulls: list[dict], interval: dict | None) -> str:
    """Mirrors DashboardTrack.status(for:)."""
    if interval is None:
        return "No claims"
    sources = interval.get("sources") or []
    matching = [p for p in pulls if p.get("html_url") in sources]
    if matching and all(p.get("state") == "open" and p.get("merged_at") is None for p in matching):
        return "Open proposal"
    if matching and all(p.get("merged_at") is not None for p in matching):
        return "Merged PR"
    return "Source claim"


def project_view(history: dict, research: dict, repository: str) -> dict | None:
    """Assemble the per-project dashboard payload for the UI."""
    scope = scope_for(history, repository)
    tracks = (research.get("tracks") or {})
    track = tracks.get(scope["id"]) if scope and scope.get("id") in tracks else None
    if scope is None:
        return None
    groups = groups_for(history, scope.get("id"))
    pulls = recent_pulls(track or {})
    metric_groups = []
    for group in groups:
        estimated = group.get("reconstructed") or {}
        observed = group.get("observed") or {}
        metric_groups.append({
            "id": group.get("id"),
            "name": group.get("name"),
            "direction": group.get("direction", "lower"),
            "estimated": {"value": estimated.get("value"),
                          "points": valid_intervals(estimated)},
            "observed": {"value": observed.get("value"),
                         "points": valid_intervals(observed)},
            "latest": latest_sources(estimated),
        })
    latest = metric_groups[-1]["latest"] if metric_groups else None
    claim_status = "No claims"
    if metric_groups and metric_groups[-1]["estimated"]["points"]:
        claim_status = interval_status(pulls, metric_groups[-1]["estimated"]["points"][-1])
    return {
        "scope": {"id": scope.get("id"), "name": scope.get("name"), "repo": scope.get("repo")},
        "groups": metric_groups,
        "pulls": [{"number": p.get("number"), "title": p.get("title"),
                   "html_url": p.get("html_url"), "state": p.get("state"),
                   "merged_at": p.get("merged_at"),
                   "created_at": p.get("created_at"), "updated_at": p.get("updated_at"),
                   "author": p.get("author"),
                   "status": "Merged" if p.get("merged_at") else (p.get("state") or "").capitalize()}
                  for p in pulls],
        "claimStatus": claim_status,
        "latestSources": latest,
        "asOf": history.get("as_of"),
        "refreshAfter": history.get("refresh_after_seconds"),
        "checkedAt": (track or {}).get("checked_at"),
    }


def latest_sources(history: dict) -> dict:
    intervals = valid_intervals(history)
    if not intervals:
        return {"sources": [], "leaders": [], "from": None, "to": None, "value": None}
    last = intervals[-1]
    return {"sources": last.get("sources") or [], "leaders": last.get("leaders") or [],
            "from": last.get("from"), "to": last.get("to"), "value": last.get("value")}
