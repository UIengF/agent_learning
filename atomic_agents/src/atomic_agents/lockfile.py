"""Append-only lockfile writer and reader for atomic-agents runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from atomic_agents.validation import tolerant_read_event


DEFAULT_BASE_DIR = Path(".atomic-agents/runs")


class LockWriter:
    def __init__(self, run_id: str, base_dir: str | Path = DEFAULT_BASE_DIR) -> None:
        self.run_id = run_id
        self.base_dir = Path(base_dir)
        self.run_dir = self.base_dir / run_id
        self.path = self.run_dir / "run.lock.jsonl"
        self._next_event_number = 1
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def emit(
        self,
        evt: str,
        payload: dict[str, Any],
        atom_id: str | None = None,
        attempt_id: int = 1,
        adapter_version: str | None = None,
    ) -> dict[str, Any]:
        event = {
            "schema_version": 1,
            "event_id": f"evt-{self._next_event_number:04d}",
            "run_id": self.run_id,
            "atom_id": atom_id,
            "attempt_id": attempt_id,
            "adapter_version": adapter_version,
            "ts": datetime.now(timezone.utc).isoformat(),
            "evt": evt,
            "payload": payload,
        }
        self._next_event_number += 1

        with self.path.open("a", encoding="utf-8") as lockfile:
            lockfile.write(json.dumps(event, ensure_ascii=False) + "\n")
            lockfile.flush()

        return event

    def run_started(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.emit("run_started", payload)

    def skeleton_hashed(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.emit("skeleton_hashed", payload)

    def binding_locked(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.emit("binding_locked", payload)

    def user_approved(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.emit("user_approved", payload)

    def atom_input(
        self,
        payload: dict[str, Any],
        atom_id: str | None = None,
        attempt_id: int = 1,
        adapter_version: str | None = None,
    ) -> dict[str, Any]:
        return self.emit("atom_input", payload, atom_id, attempt_id, adapter_version)

    def atom_started(
        self,
        payload: dict[str, Any],
        atom_id: str | None = None,
        attempt_id: int = 1,
        adapter_version: str | None = None,
    ) -> dict[str, Any]:
        return self.emit("atom_started", payload, atom_id, attempt_id, adapter_version)

    def atom_finished(
        self,
        payload: dict[str, Any],
        atom_id: str | None = None,
        attempt_id: int = 1,
        adapter_version: str | None = None,
    ) -> dict[str, Any]:
        return self.emit("atom_finished", payload, atom_id, attempt_id, adapter_version)

    def scope_drift(
        self,
        payload: dict[str, Any],
        atom_id: str | None = None,
        attempt_id: int = 1,
        adapter_version: str | None = None,
    ) -> dict[str, Any]:
        return self.emit("scope_drift", payload, atom_id, attempt_id, adapter_version)

    def review_finished(
        self,
        payload: dict[str, Any],
        atom_id: str | None = None,
        attempt_id: int = 1,
        adapter_version: str | None = None,
    ) -> dict[str, Any]:
        return self.emit("review_finished", payload, atom_id, attempt_id, adapter_version)

    def retry_scheduled(
        self,
        payload: dict[str, Any],
        atom_id: str | None = None,
        attempt_id: int = 1,
        adapter_version: str | None = None,
    ) -> dict[str, Any]:
        return self.emit("retry_scheduled", payload, atom_id, attempt_id, adapter_version)

    def run_stopped(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.emit("run_stopped", payload)

    def run_summary(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.emit("run_summary", payload)


def read_events(run_id: str, base_dir: str | Path = DEFAULT_BASE_DIR) -> list[dict[str, Any]]:
    lockfile_path = Path(base_dir) / run_id / "run.lock.jsonl"
    events: list[dict[str, Any]] = []
    with lockfile_path.open("r", encoding="utf-8") as lockfile:
        for line in lockfile:
            event = tolerant_read_event(line)
            if event is not None:
                events.append(event)
    return events


__all__ = ["DEFAULT_BASE_DIR", "LockWriter", "read_events"]
