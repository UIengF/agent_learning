from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from atomic_agents.approval import (
    ApprovalCallback,
    ApprovalDecision,
    AutoApprove,
    AutoReject,
    render_approval_summary,
)
from atomic_agents.lockfile import LockWriter, read_events
from atomic_agents.meta import meta_compile
from atomic_agents.mock_runner import MockRunner
from atomic_agents.models import ApprovalSummary, Edge, InputRef, Skeleton, SkeletonNode
from atomic_agents.scheduler import StaticScheduler
from atomic_agents.templates import MockCompiler
from atomic_agents.validation import validate_stop_report


def test_scheduler_without_approval_keeps_existing_behavior(tmp_path: Path) -> None:
    states, events, _lock_base = _run_scheduler(
        _simple_skeleton(),
        tmp_path,
        run_id="run-no-approval",
    )

    assert {state.status for state in states.values()} == {"succeeded"}
    assert _events(events, "user_approved") == []
    assert _events(events, "user_rejected") == []


def test_plan_approval_allows_run_and_records_audit_event(tmp_path: Path) -> None:
    states, events, _lock_base = _run_scheduler(
        _simple_skeleton(),
        tmp_path,
        run_id="run-plan-approved",
        approval=AutoApprove(),
        approval_summary=_approval_summary(),
    )

    assert {state.status for state in states.values()} == {"succeeded"}
    approved = _events(events, "user_approved")
    assert any(event["payload"]["gate"] == "plan" and event["payload"]["approved"] is True for event in approved)


def test_plan_rejection_stops_before_any_atom_starts_and_writes_stop_report(tmp_path: Path) -> None:
    states, events, lock_base = _run_scheduler(
        _simple_skeleton(),
        tmp_path,
        run_id="run-plan-rejected",
        approval=AutoReject(),
        approval_summary=_approval_summary(),
    )

    assert _events(events, "atom_started") == []
    assert _single_event(events, "run_stopped")["payload"]["reason"] == "user_rejected"
    assert {state.status for state in states.values()} == {"blocked"}
    stop_report = _read_stop_report(lock_base, "run-plan-rejected")
    validate_stop_report(stop_report)
    assert stop_report["reason"] == "user_rejected"
    assert _single_event(events, "run_summary")["payload"]["path_taken"] == []


def test_irreversible_rejection_stops_before_any_atom_starts_and_writes_stop_report(tmp_path: Path) -> None:
    states, events, lock_base = _run_scheduler(
        _simple_skeleton(irreversible_ops=["git_push"]),
        tmp_path,
        run_id="run-irreversible-rejected",
        approval=_DenyIrreversibleApproval(),
        approval_summary=_approval_summary(),
    )

    assert _events(events, "atom_started") == []
    assert _single_event(events, "run_stopped")["payload"]["reason"] == "irreversible_denied"
    assert {state.status for state in states.values()} == {"blocked"}
    stop_report = _read_stop_report(lock_base, "run-irreversible-rejected")
    validate_stop_report(stop_report)
    assert stop_report["reason"] == "irreversible_denied"
    assert any("git_push" in cause for cause in stop_report["likely_causes"])


def test_irreversible_approval_allows_run_and_records_audit_event(tmp_path: Path) -> None:
    states, events, _lock_base = _run_scheduler(
        _simple_skeleton(irreversible_ops=["git_push"]),
        tmp_path,
        run_id="run-irreversible-approved",
        approval=AutoApprove(),
        approval_summary=_approval_summary(),
    )

    assert {state.status for state in states.values()} == {"succeeded"}
    approved = _events(events, "user_approved")
    irreversible = [event for event in approved if event["payload"]["gate"] == "irreversible"]
    assert len(irreversible) == 1
    assert irreversible[0]["payload"]["ops"] == ["git_push"]


def test_approval_without_summary_skips_plan_gate_and_can_still_run(tmp_path: Path) -> None:
    states, events, _lock_base = _run_scheduler(
        _simple_skeleton(),
        tmp_path,
        run_id="run-summary-none",
        approval=AutoApprove(),
        approval_summary=None,
    )

    assert {state.status for state in states.values()} == {"succeeded"}
    assert [event for event in _events(events, "user_approved") if event["payload"]["gate"] == "plan"] == []


class _DenyIrreversibleApproval(ApprovalCallback):
    def approve_plan(self, summary: ApprovalSummary) -> ApprovalDecision:
        del summary
        return ApprovalDecision(approved=True)

    def approve_irreversible(self, op: str) -> bool:
        del op
        return False


def _run_scheduler(
    skeleton: Skeleton,
    tmp_path: Path,
    *,
    run_id: str,
    approval: ApprovalCallback | None = None,
    approval_summary: ApprovalSummary | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
    workspace = tmp_path / f"{run_id}-workspace"
    workspace.mkdir()
    lock_base = tmp_path / f"{run_id}-locks"
    writer = LockWriter(run_id, base_dir=lock_base)
    scheduler = StaticScheduler(
        skeleton=skeleton,
        adapter=MockRunner(),
        lock=writer,
        run_id=run_id,
        workspace=str(workspace),
        approval=approval,
        approval_summary=approval_summary,
    )

    states = scheduler.run()
    return states, read_events(run_id, base_dir=lock_base), lock_base


def _simple_skeleton(*, irreversible_ops: list[str] | None = None) -> Skeleton:
    nodes = [
        _make_node("impl", "implementer", [], [], True),
        _make_node(
            "review",
            "verifier",
            ["impl"],
            [],
            True,
            inputs=[{"from": "impl", "field": "result"}],
            criteria=["result accepted"],
        ),
    ]
    return Skeleton(
        name="approval-gate-test",
        version=1,
        nodes=nodes,
        edges=_edges_from_dependencies(nodes),
        run_limits={"max_repair_attempts": 0, "max_total_cost": 10.0},
        irreversible_ops=list(irreversible_ops or []),
    )


def _make_node(
    id: str,
    role: str,
    deps: list[str],
    write_scope: list[str],
    read_only: bool,
    *,
    inputs: list[InputRef] | None = None,
    criteria: list[str] | None = None,
) -> SkeletonNode:
    return SkeletonNode(
        id=id,
        role=role,
        task=f"Run {id}",
        depends_on=list(deps),
        inputs=list(inputs or []),
        write_scope=list(write_scope),
        read_only=read_only,
        required_capabilities=["write_files"] if write_scope else [],
        reviewer_criteria=list(criteria or []),
        output_file=write_scope[0] if write_scope else "",
    )


def _approval_summary() -> ApprovalSummary:
    return render_approval_summary(meta_compile("实现登录功能", "ducc", MockCompiler()))


def _edges_from_dependencies(nodes: list[SkeletonNode]) -> list[Edge]:
    return [{"from": dependency, "to": node.id} for node in nodes for dependency in node.depends_on]


def _events(events: list[dict[str, Any]], event_name: str) -> list[dict[str, Any]]:
    return [event for event in events if event["evt"] == event_name]


def _single_event(events: list[dict[str, Any]], event_name: str) -> dict[str, Any]:
    matching = _events(events, event_name)
    assert len(matching) == 1
    return matching[0]


def _read_stop_report(lock_base: Path, run_id: str) -> dict[str, Any]:
    with (lock_base / run_id / "stop-report.json").open("r", encoding="utf-8") as report_file:
        report = json.load(report_file)
    assert isinstance(report, dict)
    return report
