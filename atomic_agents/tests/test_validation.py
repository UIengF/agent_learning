from __future__ import annotations

import copy
import json
from collections.abc import Callable
from typing import Any

import pytest

from atomic_agents.validation import (
    ValidationError,
    tolerant_read_event,
    validate_approval_summary,
    validate_atom_contract,
    validate_lockfile_event,
    validate_skeleton,
    validate_stop_report,
)


def valid_atom_contract() -> dict[str, Any]:
    return {
        "task": "Implement X",
        "inputs": [{"from": "plan", "field": "result"}],
        "context_files": ["src/a.py"],
        "workspace": ".",
        "read_only": False,
        "write_scope": ["src/a.py"],
        "required_capabilities": ["write_files"],
        "status": "success",
        "result": "",
        "artifacts": [{"path": "src/a.py", "type": "code", "sha256": "abc123"}],
        "output_file": "",
        "output_schema_ref": None,
        "handoff": {
            "completed": [],
            "pending": [],
            "decisions": [],
            "risks": [],
        },
        "consult": None,
        "atom_id": "atom-7f3a",
        "correlation_id": "run-91c2",
        "logical_role": "implementer",
        "resolved_runner": {"runner": "mock", "model": "mock", "permission": "full"},
        "session_id": None,
        "hop_count": 1,
        "limits": {"max_cost": 2.0, "timeout_sec": 1800, "max_internal_turns": 30},
        "cost": 0.0,
        "duration_sec": 0.0,
        "timestamps": {"started_at": "2026-06-27T00:00:00Z", "finished_at": "2026-06-27T00:00:01Z"},
    }


def valid_skeleton() -> dict[str, Any]:
    return {
        "name": "feature-with-review",
        "version": 1,
        "nodes": [
            {
                "id": "plan",
                "role": "planner",
                "task": "Plan implementation",
                "depends_on": [],
                "inputs": [],
                "write_scope": ["docs/plan.md"],
                "read_only": False,
                "required_capabilities": ["write_files"],
                "reviewer_criteria": [],
                "output_file": "docs/plan.md",
            },
            {
                "id": "review",
                "role": "reviewer",
                "task": "Review implementation",
                "depends_on": ["plan"],
                "inputs": [{"from": "plan", "field": "result"}],
                "write_scope": [],
                "read_only": True,
                "required_capabilities": [],
                "reviewer_criteria": ["All tests pass"],
                "output_file": "",
            },
        ],
        "edges": [{"from": "plan", "to": "review"}],
        "run_limits": {"max_repair_attempts": 2, "max_total_cost": 10.0},
        "irreversible_ops": ["git_push"],
    }


def valid_lockfile_event() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "event_id": "evt-0001",
        "run_id": "run-91c2",
        "atom_id": None,
        "attempt_id": 1,
        "adapter_version": None,
        "ts": "2026-06-27T00:00:00Z",
        "evt": "run_started",
        "payload": {"launcher": "mock"},
    }


def valid_approval_summary() -> dict[str, Any]:
    return {
        "task_restated": "Implement X",
        "stages": ["Plan", "Implement", "Review"],
        "files_may_change": ["src/feature.py"],
        "reviewers": [{"node": "review", "criteria": ["All tests pass"]}],
        "budget": {"max_total_cost": 10.0, "per_atom_timeout_sec": 1800},
        "stop_points": ["Repair fails more than 2 times"],
        "irreversible_ops": ["git_push"],
        "risk_flags": ["v1 has no worktree isolation"],
        "editable_hints": "Adjust stages, criteria, budget, or irreversible operations.",
    }


def valid_stop_report() -> dict[str, Any]:
    return {
        "reason": "repair_failed",
        "failed_atom": "impl",
        "attempts": 2,
        "reviewer_evidence": [{"criterion": "tests", "verdict": "fail", "evidence": "3 failing"}],
        "files_changed": ["src/feature.py"],
        "cost_consumed": 3.4,
        "likely_causes": ["Test environment missing dependency"],
        "options": ["Continue after manual fix", "Rerun with changed request"],
    }


VALIDATORS: list[tuple[Callable[[Any], None], Callable[[], dict[str, Any]], str]] = [
    (validate_atom_contract, valid_atom_contract, "task"),
    (validate_skeleton, valid_skeleton, "nodes"),
    (validate_lockfile_event, valid_lockfile_event, "event_id"),
    (validate_approval_summary, valid_approval_summary, "task_restated"),
    (validate_stop_report, valid_stop_report, "reason"),
]


@pytest.mark.parametrize(("validator", "factory", "required_key"), VALIDATORS)
def test_validate_legal_samples_pass(
    validator: Callable[[Any], None],
    factory: Callable[[], dict[str, Any]],
    required_key: str,
) -> None:
    del required_key
    validator(factory())


@pytest.mark.parametrize(("validator", "factory", "required_key"), VALIDATORS)
def test_validate_missing_required_fields_raise(
    validator: Callable[[Any], None],
    factory: Callable[[], dict[str, Any]],
    required_key: str,
) -> None:
    data = factory()
    del data[required_key]

    with pytest.raises(ValidationError):
        validator(data)


@pytest.mark.parametrize(("validator", "factory", "required_key"), VALIDATORS)
def test_validate_additional_properties_policy(
    validator: Callable[[Any], None],
    factory: Callable[[], dict[str, Any]],
    required_key: str,
) -> None:
    del required_key
    data = factory()
    data["unknown_field"] = "kept"

    if validator is validate_lockfile_event:
        validator(data)
    else:
        with pytest.raises(ValidationError):
            validator(data)


def test_tolerant_read_event_handles_empty_lines_and_unknown_fields() -> None:
    assert tolerant_read_event("\n") is None

    event = valid_lockfile_event()
    event["unknown_field"] = "kept"

    assert tolerant_read_event(json.dumps(event)) == event


def test_tolerant_read_event_rejects_missing_envelope_field() -> None:
    event = copy.deepcopy(valid_lockfile_event())
    del event["run_id"]

    with pytest.raises(ValidationError):
        tolerant_read_event(json.dumps(event))
