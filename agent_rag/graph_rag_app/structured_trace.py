from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


class StructuredTraceWriter:
    def __init__(self, trace_path: str | Path, *, enabled: bool = True):
        self.trace_path = Path(trace_path)
        self.enabled = enabled

    def append(self, event_type: str, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self.trace_path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            **payload,
        }
        with self.trace_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
