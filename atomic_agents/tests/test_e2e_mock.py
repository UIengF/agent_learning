from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path
from typing import Any

from atomic_agents.lockfile import LockWriter, read_events
from atomic_agents.mock_runner import MockRunner
from atomic_agents.models import AtomContract, AtomResult, Skeleton, SkeletonNode
from atomic_agents.validation import validate_lockfile_event, validate_skeleton


def make_contract(
    *,
    atom_id: str,
    logical_role: str,
    task: str,
    inputs: list[dict[str, str]] | None = None,
    write_scope: list[str] | None = None,
) -> AtomContract:
    return AtomContract(
        task=task,
        inputs=inputs or [],
        context_files=[],
        workspace=".",
        read_only=not bool(write_scope),
        write_scope=write_scope or [],
        required_capabilities=["write_files"] if write_scope else [],
        status="success",
        result="",
        artifacts=[],
        output_file=write_scope[0] if write_scope else "",
        output_schema_ref=None,
        handoff={"completed": [], "pending": [], "decisions": [], "risks": []},
        consult=None,
        atom_id=atom_id,
        correlation_id="run-e2e",
        logical_role=logical_role,
        resolved_runner={"runner": "mock", "model": "mock", "permission": "full"},
        session_id=None,
        hop_count=1,
        limits={"max_cost": 2.0, "timeout_sec": 1800, "max_internal_turns": 30},
        cost=0.0,
        duration_sec=0.0,
        timestamps={"started_at": "2026-06-27T00:00:00Z", "finished_at": "2026-06-27T00:00:00Z"},
    )


def result_payload(result: AtomResult) -> dict[str, Any]:
    return {
        "status": result.status,
        "result_sha256": hashlib.sha256(result.result.encode("utf-8")).hexdigest(),
        "artifacts": result.artifacts,
        "session_id": result.session_id,
        "cost": result.cost,
        "duration_sec": result.duration_sec,
        "raw_events_path": result.raw_events_path,
        "error": result.error,
    }


def test_e2e_mock_runner_static_plan_impl_review_lockfile_layers(tmp_path: Path) -> None:
    skeleton = Skeleton(
        name="plan-impl-review",
        version=1,
        nodes=[
            SkeletonNode(
                id="plan",
                role="planner",
                task="Plan implementation",
                depends_on=[],
                inputs=[],
                write_scope=["docs/plan.md"],
                read_only=False,
                required_capabilities=["write_files"],
                reviewer_criteria=[],
                output_file="docs/plan.md",
            ),
            SkeletonNode(
                id="impl",
                role="implementer",
                task="Implement from plan",
                depends_on=["plan"],
                inputs=[{"from": "plan", "field": "result"}],
                write_scope=["src/feature.py"],
                read_only=False,
                required_capabilities=["write_files"],
                reviewer_criteria=[],
                output_file="src/feature.py",
            ),
            SkeletonNode(
                id="review",
                role="reviewer",
                task="Review implementation",
                depends_on=["impl"],
                inputs=[{"from": "impl", "field": "artifacts"}],
                write_scope=[],
                read_only=True,
                required_capabilities=[],
                reviewer_criteria=["All tests pass", "No regressions"],
                output_file="docs/review.md",
            ),
        ],
        edges=[{"from": "plan", "to": "impl"}, {"from": "impl", "to": "review"}],
        run_limits={"max_repair_attempts": 2, "max_total_cost": 10.0},
        irreversible_ops=[],
    )
    validate_skeleton(skeleton.to_dict())

    runner = MockRunner(
        {
            "plan": [{"result": "Plan: update feature.py"}],
            "impl": [
                {
                    "result": "Implemented feature",
                    "artifacts": [{"path": "src/feature.py", "type": "code", "sha256": "abc123"}],
                }
            ],
            "review": [{"result": "pass"}],
        }
    )
    writer = LockWriter("run-e2e", base_dir=tmp_path)

    writer.run_started({"launcher": "mock"})
    writer.skeleton_hashed(
        {
            "sha256": hashlib.sha256(json.dumps(skeleton.to_dict(), sort_keys=True).encode("utf-8")).hexdigest(),
            "skeleton_ref": "skeleton.json",
        }
    )
    writer.binding_locked(
        {
            "bindings": {
                "planner": {"runner": "mock", "model": "mock", "permission": "full"},
                "implementer": {"runner": "mock", "model": "mock", "permission": "full"},
                "reviewer": {"runner": "mock", "model": "mock", "permission": "full"},
            }
        }
    )

    contracts = [
        make_contract(atom_id="plan", logical_role="planner", task="Plan implementation", write_scope=["docs/plan.md"]),
        make_contract(
            atom_id="impl",
            logical_role="implementer",
            task="Implement from plan",
            inputs=[{"from": "plan", "field": "result"}],
            write_scope=["src/feature.py"],
        ),
        make_contract(
            atom_id="review",
            logical_role="reviewer",
            task="Review implementation",
            inputs=[{"from": "impl", "field": "artifacts"}],
        ),
    ]
    path_taken: list[str] = []
    total_cost = 0.0

    for contract in contracts:
        writer.atom_input(
            {"snapshot": contract.to_dict()},
            atom_id=contract.atom_id,
            adapter_version=runner.adapter_version,
        )
        result = runner.invoke(contract, timeout_sec=contract.limits["timeout_sec"])
        writer.atom_finished(
            result_payload(result),
            atom_id=contract.atom_id,
            adapter_version=runner.adapter_version,
        )
        path_taken.append(contract.atom_id)
        total_cost += result.cost

    writer.run_summary(
        {
            "total_cost": total_cost,
            "path_taken": path_taken,
            "dynamic_atoms_added": [],
            "unplanned_files_touched": [],
            "env": {
                "python": platform.python_version(),
                "platform": platform.platform(),
                "adapter_version": runner.adapter_version,
            },
        }
    )

    events = read_events("run-e2e", base_dir=tmp_path)

    assert runner.call_count == 3
    assert any(event["evt"] == "binding_locked" for event in events)

    summary = next(event for event in events if event["evt"] == "run_summary")
    assert summary["payload"]["path_taken"] == ["plan", "impl", "review"]
    assert summary["payload"]["env"]["adapter_version"] == "mock-1"

    atom_inputs = [event for event in events if event["evt"] == "atom_input"]
    atom_finished = [event for event in events if event["evt"] == "atom_finished"]
    assert {event["atom_id"] for event in atom_inputs} == {"plan", "impl", "review"}
    assert {event["atom_id"] for event in atom_finished} == {"plan", "impl", "review"}

    for event in events:
        validate_lockfile_event(event)
