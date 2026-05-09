from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import unicodedata


SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", ".ruff_cache", "runtime"}
TEXT_SUFFIXES = {".py", ".md", ".txt", ".toml", ".json", ".yaml", ".yml"}
REPLACEMENTS = {
    "\u2014": "-",
    "\u2013": "-",
    "\u2018": "'",
    "\u2019": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2026": "...",
    "\u2192": "->",
    "\u25cb": "open",
    "\u2713": "done",
    "\ufeff": "",
    "\xa0": " ",
    "тАФ": "-",
    "бк": "->",
    "б·": "->",
    "тЬУ": "done",
    "тЧЛ": "open",
    "тЖТ": "->",
}


@dataclass(frozen=True)
class TextIssue:
    path: str
    line: int
    column: int
    character: str
    codepoint: str
    name: str

    def format(self) -> str:
        return f"{self.path}:{self.line}:{self.column}: {self.codepoint} {self.name} {self.character!r}"


def scan_text_hygiene(workspace: str | Path, *, max_issues: int = 200) -> tuple[TextIssue, ...]:
    root = Path(workspace).resolve()
    issues: list[TextIssue] = []
    for path in sorted(root.rglob("*")):
        if path.is_dir() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        rel_parts = path.relative_to(root).parts
        if any(part in SKIP_DIRS for part in rel_parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        rel = path.relative_to(root).as_posix()
        for line_number, line in enumerate(text.splitlines(), start=1):
            for column, character in enumerate(line, start=1):
                if ord(character) <= 127:
                    continue
                issues.append(
                    TextIssue(
                        path=rel,
                        line=line_number,
                        column=column,
                        character=character,
                        codepoint=f"U+{ord(character):04X}",
                        name=unicodedata.name(character, "UNKNOWN"),
                    )
                )
                if len(issues) >= max_issues:
                    return tuple(issues)
    return tuple(issues)


def format_text_hygiene_report(workspace: str | Path) -> str:
    issues = scan_text_hygiene(workspace)
    if not issues:
        return "Text hygiene: no non-ASCII characters found."
    lines = ["Text hygiene: non-ASCII characters found."]
    lines.extend(f"- {issue.format()}" for issue in issues)
    lines.append("No files were modified. Use text hygiene clean to apply safe replacements.")
    return "\n".join(lines)


def clean_text_hygiene(workspace: str | Path) -> str:
    root = Path(workspace).resolve()
    changed: list[str] = []
    for path in sorted(root.rglob("*")):
        if path.is_dir() or path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        rel_parts = path.relative_to(root).parts
        if any(part in SKIP_DIRS for part in rel_parts):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        cleaned = text
        for old, new in REPLACEMENTS.items():
            cleaned = cleaned.replace(old, new)
        if cleaned != text:
            path.write_text(cleaned, encoding="utf-8")
            changed.append(path.relative_to(root).as_posix())
    if not changed:
        return "Text hygiene clean: no safe replacements applied."
    return "\n".join(["Text hygiene clean: changed files:", *[f"- {item}" for item in changed]])
