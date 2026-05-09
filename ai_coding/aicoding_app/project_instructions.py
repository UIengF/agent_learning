from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .evidence_cache import compact_text


_INSTRUCTION_PATHS = ("AGENTS.md", ".aicoding/AGENTS.md")


@dataclass(frozen=True)
class ProjectInstructions:
    files: tuple[str, ...]
    content: str

    def as_context(self, max_chars: int = 3000) -> str:
        if not self.content:
            return "No project instructions found."
        return compact_text(self.content, max_chars)


def load_project_instructions(workspace: str | Path) -> ProjectInstructions:
    root = Path(workspace).resolve()
    chunks: list[str] = []
    files: list[str] = []
    for relative in _INSTRUCTION_PATHS:
        path = root / relative
        if not path.exists() or not path.is_file():
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        files.append(relative)
        chunks.append(f"# {relative}\n{text.strip()}")
    return ProjectInstructions(files=tuple(files), content="\n\n".join(chunks))
