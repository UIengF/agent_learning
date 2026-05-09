from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any


class StructuredTraceWriter:
    def __init__(self, trace_dir: str | Path, session_id: str, *, enabled: bool = True):
        self.trace_dir = Path(trace_dir)
        self.session_id = session_id
        self.enabled = enabled
        self.trace_path = self.trace_dir / f"{session_id}.jsonl"

    def append(
        self,
        event_type: str,
        *,
        task_id: str | None = None,
        tool_name: str | None = None,
        input_summary: str = "",
        output_summary: str = "",
        status: str = "ok",
        payload: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": self.session_id,
            "task_id": task_id,
            "event_type": event_type,
            "tool_name": tool_name,
            "input_summary": input_summary,
            "output_summary": output_summary,
            "status": status,
            "payload": payload or {},
        }
        with self.trace_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
