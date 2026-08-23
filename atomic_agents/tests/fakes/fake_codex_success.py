#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    workspace = Path.cwd()
    output_path = workspace / "codex-output.md"
    raw_events_path = workspace / "codex-events.jsonl"

    output_path.write_text("codex ok\n", encoding="utf-8")
    events = [
        {"type": "message", "usage": {"total_tokens": 11}},
        {"type": "item.completed", "usage": {"total_tokens": 7}},
        {"type": "item.completed"},
        {"type": "item.completed", "payload": {"usage": {"total_tokens": 3}}},
    ]
    raw_events_path.write_text(
        "\n".join(json.dumps(event) for event in events) + "\n",
        encoding="utf-8",
    )

    print("session_id=sess-fake")
    print(f"output_path={output_path}")
    print(f"raw_events_path={raw_events_path}")


if __name__ == "__main__":
    main()
