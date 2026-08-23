from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from atomic_agents.models import AtomContract


@pytest.fixture
def make_contract(tmp_path: Path) -> Callable[..., AtomContract]:
    def _make_contract(**overrides: Any) -> AtomContract:
        values: dict[str, Any] = {
            "task": "do x",
            "inputs": [],
            "context_files": [],
            "workspace": str(tmp_path),
            "read_only": False,
            "write_scope": [],
            "required_capabilities": [],
            "status": "success",
            "result": "",
            "artifacts": [],
            "output_file": "",
            "output_schema_ref": None,
            "handoff": {"completed": [], "pending": [], "decisions": [], "risks": []},
            "consult": None,
            "atom_id": "atom-test",
            "correlation_id": "run-test",
            "logical_role": "implementer",
            "resolved_runner": None,
            "session_id": None,
            "hop_count": 1,
            "limits": {"max_cost": 2.0, "timeout_sec": 1800, "max_internal_turns": 30},
            "cost": 0.0,
            "duration_sec": 0.0,
            "timestamps": {"started_at": "2026-06-27T00:00:00Z", "finished_at": "2026-06-27T00:00:00Z"},
        }
        values.update(overrides)
        return AtomContract(**values)

    return _make_contract
