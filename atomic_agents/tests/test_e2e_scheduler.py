from __future__ import annotations

from pathlib import Path
from typing import Any

from atomic_agents.adapters import RunnerFeatureProfile
from atomic_agents.lockfile import LockWriter, read_events
from atomic_agents.models import AtomContract, AtomResult, Edge, InputRef, Skeleton, SkeletonNode
from atomic_agents.scheduler import StaticScheduler


def make_node(
    id: str,
    role: str,
    deps: list[str],
    write_scope: list[str],
    read_only: bool,
    inputs: list[InputRef] | None = None,
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
        reviewer_criteria=[],
        output_file=write_scope[0] if write_scope else "",
    )


def make_skeleton(nodes: list[SkeletonNode], edges: list[Edge] | None = None) -> Skeleton:
    return Skeleton(
        name="drift-e2e",
        version=1,
        nodes=nodes,
        edges=edges if edges is not None else _edges_from_dependencies(nodes),
        run_limits={"max_repair_attempts": 0, "max_total_cost": 10.0},
        irreversible_ops=[],
    )


class OutOfScopeWritingAdapter:
    adapter_version = "drift-adapter-1"

    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self.feature_profile = RunnerFeatureProfile(
            name="drift-adapter",
            supports_session_resume=False,
            supports_cost_capture=True,
            supports_raw_events=False,
            supports_internal_turn_count=False,
            permission_modes=["workspace-write"],
        )
        self.call_count = 0
        self.calls: list[str] = []

    def invoke(self, contract: AtomContract, timeout_sec: int) -> AtomResult:
        del timeout_sec
        self.call_count += 1
        self.calls.append(contract.atom_id)
        if contract.output_file:
            output_path = self.workspace / contract.output_file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(f"product:{contract.atom_id}", encoding="utf-8")
        if contract.atom_id == "write":
            (self.workspace / "outside.txt").write_text("out of scope", encoding="utf-8")
        return AtomResult(
            status="success",
            result=f"ok:{contract.atom_id}",
            artifacts=[],
            session_id=None,
            cost=0.1,
            duration_sec=0.01,
            raw_events_path=None,
            error=None,
            output_file=contract.output_file,
            output_sha256=None,
        )


class StraySiblingWritingAdapter:
    """Two-writer batch adapter: only the node matching ``stray_atom_id`` writes an
    undeclared file. Both nodes always write their own declared output_file first."""

    adapter_version = "stray-sibling-adapter-1"

    def __init__(self, workspace: Path, *, stray_atom_id: str) -> None:
        self.workspace = workspace
        self.stray_atom_id = stray_atom_id
        self.feature_profile = RunnerFeatureProfile(
            name="stray-sibling-adapter",
            supports_session_resume=False,
            supports_cost_capture=True,
            supports_raw_events=False,
            supports_internal_turn_count=False,
            permission_modes=["workspace-write"],
        )
        self.calls: list[str] = []

    def invoke(self, contract: AtomContract, timeout_sec: int) -> AtomResult:
        del timeout_sec
        self.calls.append(contract.atom_id)
        if contract.output_file:
            output_path = self.workspace / contract.output_file
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(f"product:{contract.atom_id}", encoding="utf-8")
        if contract.atom_id == self.stray_atom_id:
            (self.workspace / "outside.txt").write_text("out of scope", encoding="utf-8")
        return AtomResult(
            status="success",
            result=f"ok:{contract.atom_id}",
            artifacts=[],
            session_id=None,
            cost=0.1,
            duration_sec=0.01,
            raw_events_path=None,
            error=None,
            output_file=contract.output_file,
            output_sha256=None,
        )


def test_scope_drift_is_recorded_without_stopping_by_default(tmp_path: Path) -> None:
    # 路1：未注入 drift_handler 时默认 record-not-stop。越界写只记录 scope_drift +
    # run_summary.unplanned_files_touched，节点照常成功、下游照常执行、run 不止损。
    skeleton = _drift_skeleton()
    workspace = tmp_path / "workspace-record"
    workspace.mkdir()
    lock_base = tmp_path / "locks-record"
    adapter = OutOfScopeWritingAdapter(workspace)

    states, events = _run_scheduler(skeleton, adapter, workspace, lock_base, run_id="run-drift-record")

    assert adapter.calls == ["write", "downstream"]
    assert states["write"].status == "succeeded"
    assert states["downstream"].status == "succeeded"
    assert [event for event in events if event["evt"] == "run_stopped"] == []

    drift_event = _single_event(events, "scope_drift")
    assert drift_event["atom_id"] == "write"
    assert drift_event["payload"]["drift"] == ["outside.txt"]
    assert drift_event["payload"]["user_choice"] == "continue"
    assert _path_taken(events) == ["write", "downstream"]

    summary = _single_event(events, "run_summary")
    assert "outside.txt" in summary["payload"]["unplanned_files_touched"]


def test_scope_drift_stops_run_when_handler_requests_stop(tmp_path: Path) -> None:
    # 兼容性：调用方显式注入 drift_handler 返回 "stop" 时仍止损并 block 下游。
    skeleton = _drift_skeleton()
    workspace = tmp_path / "workspace-stop"
    workspace.mkdir()
    lock_base = tmp_path / "locks-stop"
    adapter = OutOfScopeWritingAdapter(workspace)

    states, events = _run_scheduler(
        skeleton,
        adapter,
        workspace,
        lock_base,
        run_id="run-drift-stop",
        drift_handler=lambda _node_id, _drift: "stop",
    )

    assert adapter.calls == ["write"]
    assert states["write"].status == "failed"
    assert states["write"].result is not None
    assert states["write"].result.error is not None
    assert "scope_drift" in states["write"].result.error
    assert states["downstream"].status == "blocked"

    drift_event = _single_event(events, "scope_drift")
    assert drift_event["atom_id"] == "write"
    assert drift_event["payload"]["drift"] == ["outside.txt"]
    assert drift_event["payload"]["user_choice"] == "stop"
    assert _path_taken(events) == []


def test_batch_drift_stop_settles_surviving_siblings_instead_of_discarding_them(tmp_path: Path) -> None:
    # Regression: when a parallel write batch has a batch-level drift and the injected
    # drift_handler returns "stop", _detect_batch_drift force-fails only the topo-min
    # "owner" node (an anchor, not necessarily the actual culprit). The owner's siblings
    # in the same batch may have finished successfully and produced real files on disk.
    # run()'s main loop must still call _settle_node on those siblings — skipping them
    # would silently discard already-produced, paid-for results and mislabel a genuinely
    # successful node as "blocked". Only the owner itself (whose state is already
    # terminal by _detect_batch_drift) should be skipped.
    nodes = [
        make_node("aaa_first", "implementer", [], ["first.txt"], False),
        make_node("zzz_stray", "implementer", [], ["stray_owned.txt"], False),
    ]
    skeleton = make_skeleton(nodes)
    workspace = tmp_path / "workspace-batch-stop"
    workspace.mkdir()
    lock_base = tmp_path / "locks-batch-stop"
    adapter = StraySiblingWritingAdapter(workspace, stray_atom_id="zzz_stray")

    states, events = _run_scheduler(
        skeleton,
        adapter,
        workspace,
        lock_base,
        run_id="run-batch-drift-stop",
        drift_handler=lambda _node_id, _drift: "stop",
    )

    # aaa_first is the topo-min anchor: _detect_batch_drift force-fails it regardless of
    # who actually caused the drift (documented "not necessarily the culprit" behavior).
    assert states["aaa_first"].status == "failed"
    # zzz_stray is the actual culprit but finished its own work successfully; it must be
    # settled normally, not silently dropped to a stale/blocked state.
    assert states["zzz_stray"].status == "succeeded"
    assert states["zzz_stray"].result is not None
    assert states["zzz_stray"].result.status == "success"
    assert (workspace / "stray_owned.txt").exists()

    drift_event = _single_event(events, "scope_drift")
    assert drift_event["payload"]["user_choice"] == "stop"


def test_scope_drift_can_be_recorded_and_allowed_to_continue(tmp_path: Path) -> None:
    skeleton = _drift_skeleton()
    workspace = tmp_path / "workspace-continue"
    workspace.mkdir()
    lock_base = tmp_path / "locks-continue"
    adapter = OutOfScopeWritingAdapter(workspace)

    states, events = _run_scheduler(
        skeleton,
        adapter,
        workspace,
        lock_base,
        run_id="run-drift-continue",
        drift_handler=lambda _node_id, _drift: "continue",
    )

    assert adapter.calls == ["write", "downstream"]
    assert states["write"].status == "succeeded"
    assert states["downstream"].status == "succeeded"

    drift_event = _single_event(events, "scope_drift")
    assert drift_event["payload"]["drift"] == ["outside.txt"]
    assert drift_event["payload"]["user_choice"] == "continue"
    assert _path_taken(events) == ["write", "downstream"]


def _drift_skeleton() -> Skeleton:
    nodes = [
        make_node("write", "implementer", [], ["allowed.txt"], False),
        make_node(
            "downstream",
            "verifier",
            ["write"],
            [],
            True,
            inputs=[{"from": "write", "field": "result"}],
        ),
    ]
    return make_skeleton(nodes)


def _run_scheduler(
    skeleton: Skeleton,
    adapter: OutOfScopeWritingAdapter,
    workspace: Path,
    lock_base: Path,
    *,
    run_id: str,
    drift_handler: Any | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    writer = LockWriter(run_id, base_dir=lock_base)
    scheduler = StaticScheduler(
        skeleton=skeleton,
        adapter=adapter,
        lock=writer,
        run_id=run_id,
        drift_handler=drift_handler,
        workspace=str(workspace),
    )

    states = scheduler.run()
    return states, read_events(run_id, base_dir=lock_base)


def _edges_from_dependencies(nodes: list[SkeletonNode]) -> list[Edge]:
    return [{"from": dependency, "to": node.id} for node in nodes for dependency in node.depends_on]


def _single_event(events: list[dict[str, Any]], event_name: str) -> dict[str, Any]:
    matching = [event for event in events if event["evt"] == event_name]
    assert len(matching) == 1
    return matching[0]


def _path_taken(events: list[dict[str, Any]]) -> list[str]:
    summary = _single_event(events, "run_summary")
    return list(summary["payload"]["path_taken"])
