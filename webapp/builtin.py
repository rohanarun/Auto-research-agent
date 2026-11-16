"""Bundled MUSE kernel projects, mirroring ResearchCore's BuiltinProjects.swift.

The starter ``auto-research.md`` is seeded once and is never overwritten,
preserving edits by the user or a previous model session.
"""
import json
import pathlib

from .models import ResearchError, ResearchProject


class BuiltinProject:
    def __init__(self, data: dict, resources: pathlib.Path):
        self.id = data["id"]
        self.name = data["name"]
        self.summary = data["summary"]
        self.category = data["category"]
        self.website = data["website"]
        self.leaderboard = data["leaderboard"]
        self.prompt_file = data["promptFile"]
        self.suggested_prompts = data.get("suggestedPrompts", [])
        self._resources = resources
        self._validate()

    def _validate(self):
        import re
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", self.id):
            raise ResearchError("A built-in project has an invalid identifier.")
        for field in (self.name, self.summary, self.category):
            if not field.strip():
                raise ResearchError(f"The built-in project {self.id} is missing its description.")
        if not self.suggested_prompts or any(not p.strip() for p in self.suggested_prompts):
            raise ResearchError(f"The built-in project {self.id} is missing starter prompts.")
        for url in (self.website, self.leaderboard):
            if not url.startswith("https://") or not url[len("https://"):].split("/", 1)[0]:
                raise ResearchError(f"The built-in project {self.id} must use HTTPS public links.")

    def _read_prompt(self, relative: str) -> str:
        if relative.startswith("/") or "\\" in relative or ".." in relative.split("/"):
            raise ResearchError("A built-in starter resource has an invalid relative path.")
        resolved = (self._resources / relative).resolve()
        root = self._resources.resolve()
        if not resolved.is_relative_to(root):
            raise ResearchError("A built-in starter resource points outside the application resources.")
        if not resolved.is_file():
            raise ResearchError(f"A built-in starter instruction file is missing: {relative}")
        body = resolved.read_text(encoding="utf-8").strip()
        if not body:
            raise ResearchError(f"A built-in starter instruction file is empty: {relative}")
        return body

    def instructions(self) -> str:
        return self._read_prompt("StarterPrompts/OfficialJudge.md") + "\n\n" + self._read_prompt(self.prompt_file) + "\n"

    @property
    def project(self) -> ResearchProject:
        return ResearchProject(repository="", name=self.name, summary=self.summary,
                               category=self.category, builtin_id=self.id)

    def prepare(self, workspace_root: str) -> pathlib.Path:
        """Seed a local starter once; existing files are retained."""
        from . import workspace as ws
        body = self.instructions()
        target = ws.directory(workspace_root, self.project.managed_workspace_component)
        instruction = target / "auto-research.md"
        ws.reject_symlink(instruction)
        if instruction.exists():
            if not instruction.is_file():
                raise ResearchError(
                    "The starter instruction path is not a regular file. Its contents were preserved: "
                    + str(instruction))
        else:
            instruction.write_text(body, encoding="utf-8")
            try:
                instruction.chmod(0o600)
            except OSError:
                pass
        return target


def load(resources: pathlib.Path) -> list[BuiltinProject]:
    file = resources / "BuiltinProjects.json"
    if not file.is_file():
        return []
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ResearchError(f"The built-in project catalog is unreadable: {error}")
    seen: set[str] = set()
    definitions: list[BuiltinProject] = []
    for entry in data:
        builtin = BuiltinProject(entry, resources)
        if builtin.id in seen:
            raise ResearchError(f"The built-in project catalog contains a duplicate identifier: {builtin.id}")
        seen.add(builtin.id)
        definitions.append(builtin)
    return definitions
