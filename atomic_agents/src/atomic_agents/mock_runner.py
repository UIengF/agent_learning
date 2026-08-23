"""Deterministic in-memory runner adapter for tests and P0 smoke flows."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, cast

from atomic_agents.models import Artifact, AtomContract, AtomResult, Status


class MockRunner:
    """Programmable RunnerAdapter-style test double.

    Scripts are keyed by either ``AtomContract.atom_id`` or
    ``AtomContract.logical_role``. ``atom_id`` is preferred when both exist.
    When a scripted response list is exhausted, the last response is reused so
    repeated calls stay deterministic.
    """

    adapter_version = "mock-1"

    def __init__(self, script: dict[str, list[dict[str, Any]]] | None = None) -> None:
        self.script = {key: [deepcopy(item) for item in value] for key, value in (script or {}).items()}
        self.call_count = 0
        self.call_counts: dict[str, int] = {}
        self.calls: list[AtomContract] = []
        self.timeouts: list[int] = []

    def invoke(self, contract: AtomContract, timeout_sec: int) -> AtomResult:
        self.call_count += 1
        self.calls.append(contract)
        self.timeouts.append(timeout_sec)

        script_key = self._script_key(contract)
        count_key = script_key or contract.atom_id
        call_index = self.call_counts.get(count_key, 0)
        self.call_counts[count_key] = call_index + 1

        response = self._scripted_response(script_key, call_index)
        artifacts = cast(list[Artifact], deepcopy(response.get("artifacts", [])))
        status = cast(Status, response.get("status", "success"))
        result = str(response.get("result", contract.task))

        return AtomResult(
            status=status,
            result=result,
            artifacts=artifacts,
            session_id=cast(str | None, response.get("session_id")),
            cost=float(response.get("cost", 0.1)),
            duration_sec=float(response.get("duration_sec", 1.0)),
            raw_events_path=cast(str | None, response.get("raw_events_path")),
            error=cast(str | None, response.get("error")),
            output_file=str(response.get("output_file", "")),
            output_sha256=cast(str | None, response.get("output_sha256")),
        )

    def _script_key(self, contract: AtomContract) -> str | None:
        if contract.atom_id in self.script:
            return contract.atom_id
        if contract.logical_role in self.script:
            return contract.logical_role
        return None

    def _scripted_response(self, script_key: str | None, call_index: int) -> dict[str, Any]:
        if script_key is None:
            return {}

        responses = self.script[script_key]
        if not responses:
            return {}

        response_index = min(call_index, len(responses) - 1)
        return deepcopy(responses[response_index])


__all__ = ["MockRunner"]
