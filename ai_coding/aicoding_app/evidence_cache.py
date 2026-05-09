from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any


def compact_text(text: str, max_chars: int = 800) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_chars:
        return compact
    return compact[: max_chars - 3] + "..."


@dataclass(frozen=True)
class EvidenceItem:
    kind: str
    key: str
    summary: str
    content: str
    updated_at: str


class EvidenceCache:
    def __init__(self, items: list[EvidenceItem] | None = None, *, max_content_chars: int = 4000):
        self.max_content_chars = max_content_chars
        self._items: dict[tuple[str, str], EvidenceItem] = {}
        for item in items or []:
            self._items[(item.kind, item.key)] = item

    def add(self, kind: str, key: str, content: str, *, summary_chars: int = 500) -> EvidenceItem:
        item = EvidenceItem(
            kind=kind,
            key=key,
            summary=compact_text(content, summary_chars),
            content=content[: self.max_content_chars],
            updated_at=datetime.now(timezone.utc).isoformat(),
        )
        self._items[(kind, key)] = item
        return item

    def to_context(self, max_chars: int) -> str:
        lines: list[str] = []
        used = 0
        for item in sorted(self._items.values(), key=lambda value: (value.kind, value.key)):
            line = f"- {item.kind}:{item.key} => {item.summary}"
            if used + len(line) > max_chars:
                break
            lines.append(line)
            used += len(line)
        return "\n".join(lines)

    def to_jsonable(self) -> list[dict[str, Any]]:
        return [asdict(item) for item in self._items.values()]

    @classmethod
    def from_jsonable(cls, values: list[dict[str, Any]] | None) -> "EvidenceCache":
        items = [EvidenceItem(**value) for value in values or []]
        return cls(items)
