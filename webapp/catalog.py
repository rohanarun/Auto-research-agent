"""Catalog fetch, parse and merge, mirroring ResearchCore's Catalog.swift.

Fetching uses the shared transport, which falls back to the Windows system
curl when Python's TLS stack cannot build the certificate chain. The fetched
page is saved as a snapshot in the workspace cache; when the site is not
reachable the snapshot still serves the full list.
"""
import json
import re

from . import transport
from .models import ResearchError, ResearchProject


class CatalogPolicy:
    def __init__(self, excluded_owners: list[str]):
        self.excluded_owners = excluded_owners

    def includes(self, project: ResearchProject) -> bool:
        owner = project.repository.split("/", 1)[0] if "/" in project.repository else None
        if owner is None:
            return True
        return owner.lower() not in (o.lower() for o in self.excluded_owners)


def load_policy(resources) -> CatalogPolicy:
    try:
        with (resources / "CatalogPolicy.json").open("rb") as handle:
            data = json.loads(handle.read().decode("utf-8"))
        return CatalogPolicy(data.get("excludedOwners", []))
    except FileNotFoundError:
        return CatalogPolicy([])


def _matches(pattern: str, text: str):
    """Regex matches with DOTALL, returning [full match, groups...] like the
    Swift code, which indexes the whole match as element 0."""
    try:
        compiled = re.compile(pattern, re.DOTALL)
    except re.error:
        return []
    out = []
    for match in compiled.finditer(text):
        out.append([match.group(0)] + [group if group is not None else "" for group in match.groups()])
    return out


_TAGS = re.compile(r"<[^>]+>")
_ENTITIES = (("&amp;", "&"), ("&#39;", "'"), ("&quot;", '"'))


def _clean(value: str) -> str:
    text = _TAGS.sub("", value)
    for entity, replacement in _ENTITIES:
        text = text.replace(entity, replacement)
    return text.strip()


_REPO_LINK = re.compile(r"https://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/?", re.IGNORECASE)


def parse(html: str) -> list[ResearchProject]:
    projects: list[ResearchProject] = []
    seen: set[str] = set()

    def add(project: ResearchProject):
        if project.id not in seen:
            seen.add(project.id)
            projects.append(project)

    # Source project cards contain their own canonical repository link and copy.
    for article in _matches(r"<article\b[^>]*class=[\"'][^\"']*project-entry[^\"']*[\"'][^>]*>(.*?)</article>", html):
        body = article[1]
        roots = _matches(r"href=[\"']https://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/?[\"']", body)
        if not roots or not roots[-1][1]:
            continue
        repo = roots[-1][1]
        titles = _matches(r"<h3[^>]*>(.*?)</h3>", body)
        title = titles[0][1] if titles else None
        summaries = _matches(r"class=[\"']project-description[\"'][^>]*>\s*<p>(.*?)</p>", body)
        summary = summaries[0][1] if summaries else ""
        categories = _matches(r"data-category=[\"']([^\"']+)", article[0])
        category = categories[0][1] if categories else "Research"
        add(ResearchProject(repository=repo, name=_clean(title) if title else None,
                            summary=_clean(summary), category=_clean(category)))

    # Chart tracks declared as page data, including tracks without static cards.
    for match in _matches(r"\brepo[\"']?\s*:\s*[\"']([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)[\"']", html):
        add(ResearchProject(repository=match[1], category="Mathematics"))

    # Canonical repository anchors used by the launch and source sections.
    for match in _matches(r"href=[\"']https://github\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/?[\"']", html):
        add(ResearchProject(repository=match[1]))

    if not projects:
        raise ResearchError("The catalog did not contain repository links. Your saved projects have been preserved.")
    return projects


def fetch(address: str, timeout: float = 20.0,
          snapshot: "object | None" = None) -> tuple[list[ResearchProject], bool, str]:
    """Fetch and parse the catalog page.

    Returns ``(projects, offline, note)``: ``offline`` is True when the saved
    snapshot was used, and ``note`` explains why. Raises when there is no
    snapshot either. ``snapshot`` is a cache path for the page HTML.
    """
    address = address.strip()
    if not address.startswith("https://"):
        raise ResearchError("Use an HTTPS catalog URL.")
    try:
        html = transport.fetch_text(address, timeout=timeout,
                                    accept="text/html,application/xhtml+xml")
    except transport.TransportError as error:
        if snapshot is not None:
            saved = transport.read_snapshot(snapshot)
            if saved:
                try:
                    return parse(saved), True, "Offline · saved catalog snapshot"
                except ResearchError:
                    pass
        raise ResearchError("The catalog server could not be reached. Showing saved projects: "
                            + str(error)) from None
    if snapshot is not None:
        transport.save_snapshot(snapshot, html)
    return parse(html), False, ""


def merge(fetched: list[ResearchProject], existing: list[ResearchProject],
          builtins: list[ResearchProject], session_project_ids: set[str]) -> list[ResearchProject]:
    """Fresh public entries win; custom projects and projects with session
    history stay accessible even when removed from a remote catalog."""
    seen: set[str] = set()
    retained = [p for p in existing if p.is_custom or p.id in session_project_ids]
    result: list[ResearchProject] = []
    for project in builtins + fetched + retained:
        if project.id not in seen:
            seen.add(project.id)
            result.append(project)
    return result
