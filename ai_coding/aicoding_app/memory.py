from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import uuid
from typing import Any


_SECRET_RE = re.compile(
    r"(api[_-]?key|authorization|bearer\s+[a-z0-9._-]+|password|secret|token)\s*[:=]",
    re.IGNORECASE,
)


class SensitiveMemoryError(ValueError):
    pass


@dataclass(frozen=True)
class MemoryItem:
    id: str
    kind: str
    text: str
    created_at: str

    def format(self) -> str:
        return f"- {self.id} [{self.kind}] {self.text}"


class MemoryStore:
    def __init__(self, runtime_dir: str | Path):
        self.path = Path(runtime_dir) / "memory.json"

    def load(self) -> list[MemoryItem]:
        if not self.path.exists():
            return []
        data = json.loads(self.path.read_text(encoding="utf-8"))
        return [MemoryItem(**item) for item in data.get("items", [])]

    def save(self, items: list[MemoryItem]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {"items": [asdict(item) for item in items]}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def add(self, kind: str, text: str) -> MemoryItem:
        if is_sensitive_memory(text):
            raise SensitiveMemoryError("refusing to save memory that looks like a secret")
        items = self.load()
        item = MemoryItem(
            id=f"mem-{uuid.uuid4().hex[:8]}",
            kind=kind.strip() or "note",
            text=text.strip(),
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        items.append(item)
        self.save(items)
        return item

    def forget(self, memory_id: str) -> bool:
        items = self.load()
        kept = [item for item in items if item.id != memory_id]
        self.save(kept)
        return len(kept) != len(items)

    def format(self, *, max_items: int = 20) -> str:
        items = self.load()
        if not items:
            return "Memory: none"
        lines = ["Memory:"]
        lines.extend(item.format() for item in items[-max_items:])
        return "\n".join(lines)


def is_sensitive_memory(text: str) -> bool:
    return bool(_SECRET_RE.search(text))
