from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any


_SAFE_SESSION_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def safe_session_id(session_id: str) -> str:
    safe = _SAFE_SESSION_RE.sub("_", session_id.strip())
    return safe or "default"


@dataclass
class SessionState:
    session_id: str
    workspace: str
    history: list[dict[str, str]] = field(default_factory=list)
    summary: str = ""
    plan: dict[str, Any] = field(default_factory=dict)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    latest_diff: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def touch(self) -> None:
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def add_message(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        self.touch()

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)


class SessionStore:
    def __init__(self, runtime_dir: str | Path):
        self.runtime_dir = Path(runtime_dir)
        self.sessions_dir = self.runtime_dir / "sessions"

    def path_for(self, session_id: str) -> Path:
        return self.sessions_dir / f"{safe_session_id(session_id)}.json"

    def load(self, session_id: str) -> SessionState | None:
        path = self.path_for(session_id)
        if not path.exists():
            return None
        data = json.loads(path.read_text(encoding="utf-8"))
        return SessionState(**data)

    def get_or_create(self, session_id: str, workspace: str) -> SessionState:
        existing = self.load(session_id)
        if existing is not None:
            return existing
        return SessionState(session_id=safe_session_id(session_id), workspace=workspace)

    def save(self, session: SessionState) -> None:
        self.sessions_dir.mkdir(parents=True, exist_ok=True)
        session.touch()
        self.path_for(session.session_id).write_text(
            json.dumps(session.to_jsonable(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
