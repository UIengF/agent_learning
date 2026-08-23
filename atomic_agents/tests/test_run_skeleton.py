from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from atomic_agents import RunResult as ExportedRunResult
from atomic_agents.adapters import RunnerFeatureProfile
from atomic_agents.approval import AutoApprove, AutoReject
from atomic_agents.lockfile import read_events
from atomic_agents.meta import meta_compile
from atomic_agents.models import ApprovalSummary, AtomContract, AtomResult, Edge, InputRef, Skeleton, SkeletonNode
from atomic_agents.reviewer import CriterionVerdict, ReviewResult
from atomic_agents.run import run_meta_plan, run_skeleton
from atomic_agents.templates import MockCompiler


class FileWritingAdapter:
    adapter_version = "file-writing-1"

    def __init__(self) -> None:
        self.feature_profile = RunnerFeatureProfile(
            name="file-writing",
            supports_session_resume=False,
            supports_cost_capture=True,
            supports_raw_events=False,
            supports_internal_turn_count=False,
            permission_modes=["read-only", "workspace-write"],
        )
        self.calls: list[str] = []

    def invoke(self, contract: AtomContract, timeout_sec: int) -> AtomResult:
        del timeout_sec
        self.calls.append(contract.atom_id)
        output_sha256 = None
        artifacts = []

        if contract.output_file:
            if contract.logical_role.strip().lower() == "reviewer":
                content = json.dumps(_passing_review().to_dict(), ensure_ascii=False, indent=2) + "\n"
            else:
                content = f"product:{contract.atom_id}\n"

            output_path = _resolve_output_path(contract.workspace, contract.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(content, encoding="utf-8")
            output_sha256 = _sha256_file(output_path)
            artifacts.append({"path": contract.output_file, "type": "file", "sha256": output_sha256})

        return AtomResult(
            status="success",
            result=f"ok:{contract.atom_id}",
            artifacts=artifacts,
            session_id=None,
            cost=0.1,
            duration_sec=0.01,
            raw_events_path=None,
            error=None,
            output_file=contract.output_file,
            output_sha256=output_sha256,
        )


def test_run_skeleton_runs_plan_impl_review_to_completion(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lock_base = tmp_path / "locks"

    result = run_skeleton(
        _plan_impl_review_skeleton(),
        FileWritingAdapter(),
        run_id="run-step7",
        workspace=str(workspace),
        lock_base_dir=lock_base,
    )

    assert isinstance(result, ExportedRunResult)
    assert result.succeeded is True
    assert result.failed_nodes == []
    assert result.blocked_nodes == []
    assert {state.status for state in result.states.values()} == {"succeeded"}
    assert Path(result.lock_dir) == lock_base / "run-step7"
    assert (Path(result.lock_dir) / "run.lock.jsonl").is_file()


def test_run_skeleton_generates_run_id_when_missing(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    result = run_skeleton(
        _plan_impl_review_skeleton(),
        FileWritingAdapter(),
        workspace=str(workspace),
        lock_base_dir=tmp_path / "locks",
    )

    assert result.run_id
    assert result.run_id.startswith("run-")
    assert Path(result.lock_dir).is_dir()
    assert (Path(result.lock_dir) / "run.lock.jsonl").is_file()


def test_run_skeleton_with_approval_reject_stops(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lock_base = tmp_path / "locks"

    result = run_skeleton(
        _plan_impl_review_skeleton(),
        FileWritingAdapter(),
        run_id="run-reject",
        workspace=str(workspace),
        lock_base_dir=lock_base,
        approval=AutoReject(),
        approval_summary=_approval_summary(),
    )
    events = read_events(result.run_id, base_dir=lock_base)

    assert result.succeeded is False
    assert result.failed_nodes or result.blocked_nodes
    assert _events(events, "atom_started") == []
    assert _single_event(events, "run_stopped")["payload"]["reason"] == "user_rejected"


def test_run_meta_plan_auto_generates_summary_for_approval(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lock_base = tmp_path / "locks"
    plan = meta_compile("实现登录功能", "ducc", MockCompiler())

    result = run_meta_plan(
        plan,
        FileWritingAdapter(),
        run_id="run-meta-approved",
        workspace=str(workspace),
        lock_base_dir=lock_base,
        approval=AutoApprove(),
    )
    events = read_events(result.run_id, base_dir=lock_base)

    assert result.succeeded is True
    assert any(
        event["payload"]["gate"] == "plan" and event["payload"]["approved"] is True
        for event in _events(events, "user_approved")
    )


def test_run_meta_plan_reject_blocks_all(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lock_base = tmp_path / "locks"
    plan = meta_compile("实现登录功能", "ducc", MockCompiler())

    result = run_meta_plan(
        plan,
        FileWritingAdapter(),
        run_id="run-meta-rejected",
        workspace=str(workspace),
        lock_base_dir=lock_base,
        approval=AutoReject(),
    )
    events = read_events(result.run_id, base_dir=lock_base)

    assert result.succeeded is False
    assert result.blocked_nodes == [node.id for node in plan.skeleton.nodes]
    assert _events(events, "atom_started") == []
    assert _single_event(events, "run_stopped")["payload"]["reason"] == "user_rejected"


def test_run_skeleton_last_node_over_budget_is_not_succeeded(tmp_path: Path) -> None:
    """A node that finishes with status=success but pushes total cost over budget
    must not make the overall run report succeeded=True: _stop_run("over_budget", ...)
    fires after the node's own state is already set to "succeeded", so succeeded must
    also check whether the run was terminated by _stop_run, not just node statuses."""

    workspace = tmp_path / "workspace"
    workspace.mkdir()
    lock_base = tmp_path / "locks"
    skeleton = Skeleton(
        name="over-budget-single-node",
        version=1,
        nodes=[_make_node("solo", "implementer", [], ["out.txt"], False)],
        edges=[],
        run_limits={"max_repair_attempts": 0, "max_total_cost": 0.01},
        irreversible_ops=[],
    )

    result = run_skeleton(
        skeleton,
        FileWritingAdapter(),
        run_id="run-over-budget-last-node",
        workspace=str(workspace),
        lock_base_dir=lock_base,
    )
    events = read_events(result.run_id, base_dir=lock_base)

    assert result.states["solo"].status == "succeeded"
    assert result.succeeded is False
    assert result.stop_reason == "over_budget"
    assert _single_event(events, "run_stopped")["payload"]["reason"] == "over_budget"


def _plan_impl_review_skeleton() -> Skeleton:
    nodes = [
        _make_node("plan", "planner", [], ["docs/plan.md"], False),
        _make_node(
            "impl",
            "implementer",
            ["plan"],
            ["src/feature.py"],
            False,
            inputs=[{"from": "plan", "field": "result"}],
        ),
        _make_node(
            "review",
            "reviewer",
            ["impl"],
            [],
            True,
            inputs=[{"from": "impl", "field": "artifacts"}],
            criteria=["implementation satisfies request", "no blocking findings"],
            output_file="reviews/verdict.json",
        ),
    ]
    return Skeleton(
        name="run-skeleton-test",
        version=1,
        nodes=nodes,
        edges=_edges_from_dependencies(nodes),
        run_limits={"max_repair_attempts": 0, "max_total_cost": 10.0},
        irreversible_ops=[],
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
    output_file: str | None = None,
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
        output_file=output_file if output_file is not None else (write_scope[0] if write_scope else ""),
    )


def _passing_review() -> ReviewResult:
    return ReviewResult(
        passed=True,
        criteria=[
            CriterionVerdict(
                criterion="all criteria",
                verdict="pass",
                evidence="fake adapter produced the expected output",
                confidence="high",
            )
        ],
        blocking_findings=[],
        reviewer_session=None,
        feedback="",
    )


def _approval_summary() -> ApprovalSummary:
    return ApprovalSummary(
        task_restated="Run skeleton test",
        stages=["plan", "impl", "review"],
        files_may_change=["docs/plan.md", "src/feature.py"],
        reviewers=[{"node": "review", "criteria": ["implementation satisfies request"]}],
        budget={"max_total_cost": 10.0, "per_atom_timeout_sec": 1800},
        stop_points=["stop on rejection"],
        irreversible_ops=[],
        risk_flags=[],
        editable_hints="N/A",
    )


def _edges_from_dependencies(nodes: list[SkeletonNode]) -> list[Edge]:
    return [{"from": dependency, "to": node.id} for node in nodes for dependency in node.depends_on]


def _events(events: list[dict[str, Any]], event_name: str) -> list[dict[str, Any]]:
    return [event for event in events if event["evt"] == event_name]


def _single_event(events: list[dict[str, Any]], event_name: str) -> dict[str, Any]:
    matching = _events(events, event_name)
    assert len(matching) == 1
    return matching[0]


def _resolve_output_path(workspace: str, output_file: str) -> Path:
    output_path = Path(output_file)
    if output_path.is_absolute():
        return output_path
    return Path(workspace) / output_path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
