from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any, Type

from pydantic import BaseModel, Field, PrivateAttr

try:
    from langchain_core.tools import BaseTool
except ImportError:  # pragma: no cover
    BaseTool = object  # type: ignore[assignment]


_FRONTMATTER_RE = re.compile(r"^---\n(.*?)\n---\n(.*)", re.DOTALL)


@dataclass(frozen=True)
class Skill:
    name: str
    description: str
    path: Path
    body: str


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    match = _FRONTMATTER_RE.match(text)
    if not match:
        return {}, text.strip()

    metadata: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip().strip("'\"")
    return metadata, match.group(2).strip()


class SkillRegistry:
    def __init__(self, skills_dir: str | Path):
        self.skills_dir = Path(skills_dir)
        self._skills = self._load_skills()

    def _load_skills(self) -> dict[str, Skill]:
        if not self.skills_dir.exists():
            return {}

        skills: dict[str, Skill] = {}
        for path in sorted(self.skills_dir.rglob("SKILL.md")):
            text = path.read_text(encoding="utf-8")
            metadata, body = _parse_frontmatter(text)
            name = metadata.get("name") or path.parent.name
            description = metadata.get("description", "")
            skills[name] = Skill(
                name=name,
                description=description,
                path=path,
                body=body,
            )
        return skills

    @property
    def available(self) -> tuple[Skill, ...]:
        return tuple(self._skills[name] for name in sorted(self._skills))

    def get(self, name: str) -> Skill | None:
        return self._skills.get(name)

    def format_inventory(self) -> str | None:
        if not self._skills:
            return None

        lines = [
            "Available skills:",
            "Use `load_skill` only when one of these skills is directly relevant.",
        ]
        for skill in self.available:
            description = f": {skill.description}" if skill.description else ""
            lines.append(f"- {skill.name}{description}")
        return "\n".join(lines)

    def load_payload(self, name: str) -> dict[str, Any]:
        skill = self.get(name)
        if skill is None:
            return {
                "error": "unknown_skill",
                "requested": name,
                "available": [item.name for item in self.available],
            }
        return {
            "name": skill.name,
            "description": skill.description,
            "path": str(skill.path),
            "body": skill.body,
        }


class LoadSkillInput(BaseModel):
    name: str = Field(..., description="Name of the skill to load.")


class LoadSkillTool(BaseTool):
    name: str = "load_skill"
    description: str = "Load the full instructions for an available local skill by name."
    args_schema: Type[BaseModel] = LoadSkillInput

    _registry: SkillRegistry = PrivateAttr()

    def __init__(self, registry: SkillRegistry, **kwargs):
        super().__init__(**kwargs)
        self._registry = registry

    def invoke(self, input: dict[str, Any], **_: Any) -> str:
        return self._run(**input)

    def _run(self, name: str) -> str:
        return json.dumps(self._registry.load_payload(name), ensure_ascii=False)
