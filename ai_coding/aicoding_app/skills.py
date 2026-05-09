from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any


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
            skills[name] = Skill(
                name=name,
                description=metadata.get("description", ""),
                path=path,
                body=body,
            )
        return skills

    @property
    def available(self) -> tuple[Skill, ...]:
        return tuple(self._skills[name] for name in sorted(self._skills))

    def format_inventory(self) -> str:
        if not self._skills:
            return "Available skills: none"
        lines = [
            "Available skills:",
            "Use load_skill only when a skill is directly relevant to the coding task.",
        ]
        for skill in self.available:
            suffix = f": {skill.description}" if skill.description else ""
            lines.append(f"- {skill.name}{suffix}")
        return "\n".join(lines)

    def load_payload(self, name: str) -> dict[str, Any]:
        skill = self._skills.get(name)
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

    def load_json(self, name: str) -> str:
        return json.dumps(self.load_payload(name), ensure_ascii=False)
