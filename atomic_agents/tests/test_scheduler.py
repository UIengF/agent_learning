from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import pytest

from atomic_agents.adapters import RunnerFeatureProfile
from atomic_agents.lockfile import LockWriter, read_events
from atomic_agents.mock_runner import MockRunner
from atomic_agents.models import AtomContract, AtomResult, Edge, InputRef, Skeleton, SkeletonNode
from atomic_agents.scheduler import StaticScheduler, _result_payload, build_contract
from atomic_agents.validation import validate_lockfile_event


def make_node(
    id: str,
    role: str,
    deps: list[str],
    write_scope: list[str],
    read_only: bool,
    inputs: list[InputRef] | None = None,
    output_file: str | None = None,
    context_files: list[str] | None = None,
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
        output_file=write_scope[0] if output_file is None and write_scope else (output_file or ""),
        context_files=list(context_files or []),
    )


def make_skeleton(
    nodes: list[SkeletonNode],
    edges: list[Edge] | None = None,
    *,
    name: str = "test-skeleton",
    version: int = 1,
    max_total_cost: float = 10.0,
    max_repair_attempts: int = 0,
    irreversible_ops: list[str] | None = None,
) -> Skeleton:
    return Skeleton(
        name=name,
        version=version,
        nodes=nodes,
        edges=edges if edges is not None else _edges_from_dependencies(nodes),
        run_limits={"max_repair_attempts": max_repair_attempts, "max_total_cost": max_total_cost},
        irreversible_ops=list(irreversible_ops or []),
    )


class InstrumentedAdapter:
    adapter_version = "instrumented-1"

    def __init__(self, *, sleep_sec: float = 0.05, barrier_parties: int | None = None) -> None:
        self.feature_profile = RunnerFeatureProfile(
            name="instrumented",
            supports_session_resume=False,
            supports_cost_capture=True,
            supports_raw_events=False,
            supports_internal_turn_count=False,
            permission_modes=["read-only", "workspace-write"],
        )
        self.sleep_sec = sleep_sec
        self.call_count = 0
        self.max_concurrent = 0
        self._current = 0
        self._lock = threading.Lock()
        self._barrier = threading.Barrier(barrier_parties) if barrier_parties is not None else None

    def invoke(self, contract: AtomContract, timeout_sec: int) -> AtomResult:
        del timeout_sec
        with self._lock:
            self.call_count += 1
            self._current += 1
            self.max_concurrent = max(self.max_concurrent, self._current)

        try:
            if self._barrier is not None:
                try:
                    self._barrier.wait(timeout=1.0)
                except threading.BrokenBarrierError:
                    pass
            time.sleep(self.sleep_sec)
            if contract.output_file:
                output_path = Path(contract.workspace) / contract.output_file
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(f"product:{contract.atom_id}", encoding="utf-8")
            return _result(result=contract.atom_id)
        finally:
            with self._lock:
                self._current -= 1


class ExtraWriteAdapter(InstrumentedAdapter):
    def __init__(
        self,
        extra_writes: dict[str, str],
        *,
        sleep_sec: float = 0.05,
        barrier_parties: int | None = None,
    ) -> None:
        super().__init__(sleep_sec=sleep_sec, barrier_parties=barrier_parties)
        self.extra_writes = dict(extra_writes)

    def invoke(self, contract: AtomContract, timeout_sec: int) -> AtomResult:
        result = super().invoke(contract, timeout_sec)
        extra_path = self.extra_writes.get(contract.atom_id)
        if extra_path:
            output_path = Path(contract.workspace) / extra_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(f"extra:{contract.atom_id}", encoding="utf-8")
        return result


def test_topological_progression_runs_serial_chain_in_order(tmp_path: Path) -> None:
    nodes = [
        make_node("a", "planner", [], [], True),
        make_node("b", "implementer", ["a"], [], True),
        make_node("c", "verifier", ["b"], [], True),
    ]
    skeleton = make_skeleton(nodes)
    runner = MockRunner()

    _states, events = _run_scheduler(skeleton, runner, tmp_path, run_id="run-chain")

    assert runner.call_count == 3
    assert _path_taken(events) == ["a", "b", "c"]


def test_diamond_allows_read_only_fanout_before_merge(tmp_path: Path) -> None:
    nodes = [
        make_node("plan", "planner", [], [], True),
        make_node("a", "analyst", ["plan"], [], True),
        make_node("b", "analyst", ["plan"], [], True),
        make_node(
            "merge",
            "verifier",
            ["a", "b"],
            [],
            True,
            inputs=[{"from": "a", "field": "result"}, {"from": "b", "field": "result"}],
        ),
    ]
    skeleton = make_skeleton(nodes)
    runner = MockRunner()

    states, events = _run_scheduler(skeleton, runner, tmp_path, run_id="run-diamond")
    path_taken = _path_taken(events)

    assert runner.call_count == 4
    assert {state.status for state in states.values()} == {"succeeded"}
    assert path_taken[0] == "plan"
    assert set(path_taken[1:3]) == {"a", "b"}
    assert path_taken[3] == "merge"


def test_read_only_siblings_run_in_parallel(tmp_path: Path) -> None:
    nodes = [
        make_node("a", "reader", [], [], True),
        make_node("b", "reader", [], [], True),
        make_node("c", "reader", [], [], True),
    ]
    adapter = InstrumentedAdapter(barrier_parties=3)

    _run_scheduler(make_skeleton(nodes), adapter, tmp_path, run_id="run-read-parallel")

    assert adapter.call_count == 3
    assert adapter.max_concurrent == 3


def test_disjoint_write_siblings_run_in_parallel(tmp_path: Path) -> None:
    nodes = [
        make_node("a", "writer", [], ["a.txt"], False),
        make_node("b", "writer", [], ["b.txt"], False),
        make_node("c", "writer", [], ["c.txt"], False),
    ]
    adapter = InstrumentedAdapter(barrier_parties=3)

    _run_scheduler(make_skeleton(nodes), adapter, tmp_path, run_id="run-write-parallel")

    assert adapter.call_count == 3
    assert adapter.max_concurrent == 3


def test_overlapping_write_siblings_run_serially(tmp_path: Path) -> None:
    nodes = [
        make_node("a", "writer", [], ["src/"], False, output_file="src/a.txt"),
        make_node("b", "writer", [], ["src/x.py"], False),
    ]
    adapter = InstrumentedAdapter()

    states, _events = _run_scheduler(make_skeleton(nodes), adapter, tmp_path, run_id="run-write-overlap")

    assert adapter.call_count == 2
    assert adapter.max_concurrent == 1
    assert {state.status for state in states.values()} == {"succeeded"}


def test_parallel_write_sibling_outputs_do_not_trigger_scope_drift(tmp_path: Path) -> None:
    nodes = [
        make_node("a", "writer", [], ["a.txt"], False),
        make_node("b", "writer", [], ["b.txt"], False),
    ]
    adapter = InstrumentedAdapter(barrier_parties=2)

    states, events = _run_scheduler(make_skeleton(nodes), adapter, tmp_path, run_id="run-write-no-cross-drift")

    assert adapter.max_concurrent == 2
    assert {state.status for state in states.values()} == {"succeeded"}
    assert [event for event in events if event["evt"] == "scope_drift"] == []


def test_parallel_write_out_of_scope_file_records_drift_without_collateral(tmp_path: Path) -> None:
    # 路1 + P1 回归守卫：并行写批次里 b 写了 scope 外的 d.txt（越界）。
    # 默认 record-not-stop：run 不停、合规的 a 仍 succeeded（成果保留）、肇事的 b 也仍
    # succeeded（不连坐、不误判失败），只 emit 批次级 scope_drift 事件，且越界文件进
    # run_summary.unplanned_files_touched 如实汇报。
    nodes = [
        make_node("a", "writer", [], ["a.txt"], False),
        make_node("b", "writer", [], ["b.txt"], False),
    ]
    adapter = ExtraWriteAdapter({"b": "d.txt"}, barrier_parties=2)

    states, events = _run_scheduler(make_skeleton(nodes), adapter, tmp_path, run_id="run-write-real-drift")

    assert adapter.max_concurrent == 2
    # 不连坐：批内两个写节点都成功，成果保留。
    assert {state.status for state in states.values()} == {"succeeded"}
    assert (tmp_path / "run-write-real-drift-workspace" / "a.txt").exists()
    # run 未止损。
    assert [event for event in events if event["evt"] == "run_stopped"] == []
    # 越界被记录为批次级（无法定位单个肇事 sibling）。
    drift_events = [event for event in events if event["evt"] == "scope_drift"]
    assert any(
        event["payload"]["drift"] == ["d.txt"]
        and event["payload"].get("attribution") == "batch"
        and event["payload"]["user_choice"] == "continue"
        for event in drift_events
    )
    # 计划外文件进 run_summary，事后可见。
    summary = next(event for event in events if event["evt"] == "run_summary")
    assert "d.txt" in summary["payload"]["unplanned_files_touched"]


def test_single_node_out_of_scope_write_records_drift_without_stopping(tmp_path: Path) -> None:
    # 路1：单个写节点越界（写了 scope 外文件）默认也是记录不停——节点 succeeded、
    # run 正常结束、scope_drift 事件记录、越界文件进 run_summary。
    nodes = [make_node("solo", "writer", [], ["solo.txt"], False)]
    adapter = ExtraWriteAdapter({"solo": "stray.txt"})

    states, events = _run_scheduler(make_skeleton(nodes), adapter, tmp_path, run_id="run-solo-drift")

    assert states["solo"].status == "succeeded"
    assert [event for event in events if event["evt"] == "run_stopped"] == []
    drift_events = [event for event in events if event["evt"] == "scope_drift"]
    assert any(
        event["atom_id"] == "solo" and event["payload"]["drift"] == ["stray.txt"]
        for event in drift_events
    )
    summary = next(event for event in events if event["evt"] == "run_summary")
    assert "stray.txt" in summary["payload"]["unplanned_files_touched"]


def test_injected_drift_handler_can_still_stop_on_out_of_scope_write(tmp_path: Path) -> None:
    # 兼容性守卫：默认 record-not-stop，但调用方注入 drift_handler 返回 "stop" 时仍止损。
    # 单节点越界的 stop 路径经由 failed result 走 atom_failed，error 里带 scope_drift 证据。
    nodes = [make_node("solo", "writer", [], ["solo.txt"], False)]
    adapter = ExtraWriteAdapter({"solo": "stray.txt"})

    states, events = _run_scheduler(
        make_skeleton(nodes),
        adapter,
        tmp_path,
        run_id="run-solo-stop",
        drift_handler=lambda node_id, drift: "stop",
    )

    assert states["solo"].status == "failed"
    assert states["solo"].result is not None
    assert states["solo"].result.error is not None and "scope_drift" in states["solo"].result.error
    assert [event for event in events if event["evt"] == "run_stopped"]


def test_dependency_failure_blocks_downstream_without_invoking_them(tmp_path: Path) -> None:
    nodes = [
        make_node("a", "planner", [], [], True),
        make_node("b", "implementer", ["a"], [], True),
        make_node("c", "verifier", ["b"], [], True),
    ]
    runner = MockRunner({"a": [{"status": "failed", "result": "boom", "error": "failed"}]})

    states, events = _run_scheduler(make_skeleton(nodes), runner, tmp_path, run_id="run-blocked")

    assert runner.call_count == 1
    assert states["a"].status == "failed"
    assert states["b"].status == "blocked"
    assert states["c"].status == "blocked"
    assert _path_taken(events) == []


def test_cycle_detection_raises_value_error(tmp_path: Path) -> None:
    nodes = [
        make_node("a", "planner", ["b"], [], True),
        make_node("b", "implementer", ["a"], [], True),
    ]
    skeleton = make_skeleton(nodes, edges=[{"from": "b", "to": "a"}, {"from": "a", "to": "b"}])

    with pytest.raises(ValueError, match="cycle"):
        StaticScheduler(
            skeleton=skeleton,
            adapter=MockRunner(),
            lock=LockWriter("run-cycle", base_dir=tmp_path / "locks"),
            run_id="run-cycle",
            workspace=str(tmp_path / "workspace"),
        )


def test_lockfile_events_cover_scheduler_run_layers(tmp_path: Path) -> None:
    nodes = [
        make_node("plan", "planner", [], [], True),
        make_node("impl", "implementer", ["plan"], [], True),
        make_node("review", "verifier", ["impl"], [], True),
    ]

    _states, events = _run_scheduler(make_skeleton(nodes), MockRunner(), tmp_path, run_id="run-lockfile")

    event_names = [event["evt"] for event in events]
    assert "run_started" in event_names
    assert "binding_locked" in event_names
    assert "run_summary" in event_names
    assert {event["atom_id"] for event in events if event["evt"] == "atom_input"} == {"plan", "impl", "review"}
    assert {event["atom_id"] for event in events if event["evt"] == "atom_finished"} == {"plan", "impl", "review"}

    summary = next(event for event in events if event["evt"] == "run_summary")
    assert summary["payload"]["path_taken"] == ["plan", "impl", "review"]
    assert summary["payload"]["env"]["adapter_version"] == "mock-1"

    for event in events:
        validate_lockfile_event(event)


def test_build_contract_passes_node_fields_and_input_refs(tmp_path: Path) -> None:
    upstream_results = {
        "plan": _result(result="plan result", output_file="docs/plan.md"),
        "impl": _result(
            result="impl result",
            artifacts=[{"path": "src/a.py", "type": "code", "sha256": "abc"}],
            output_file="src/a.py",
        ),
    }
    node = make_node(
        "review",
        "reviewer",
        ["plan", "impl"],
        ["reports/review.md"],
        False,
        inputs=[{"from": "plan", "field": "output_file"}, {"from": "impl", "field": "output_file"}],
    )
    skeleton = make_skeleton(
        [
            make_node("plan", "planner", [], [], True),
            make_node("impl", "implementer", ["plan"], ["src/a.py"], False),
            node,
        ]
    )

    contract = build_contract(node, skeleton, upstream_results, run_id="run-contract", workspace=str(tmp_path))

    assert contract.inputs == [{"from": "plan", "field": "output_file"}, {"from": "impl", "field": "output_file"}]
    assert contract.logical_role == "reviewer"
    assert contract.write_scope == ["reports/review.md"]
    assert contract.read_only is False
    assert contract.correlation_id == "run-contract"
    # Decision 2: upstream output_files are surfaced into downstream context_files.
    assert contract.context_files == ["docs/plan.md", "src/a.py"]
    assert contract.output_file == node.output_file


def test_build_contract_merges_static_context_files_with_upstream_outputs(tmp_path: Path) -> None:
    upstream_results = {
        "impl": AtomResult(
            status="success",
            result="ok",
            artifacts=[],
            session_id=None,
            cost=0.0,
            duration_sec=0.0,
            raw_events_path=None,
            error=None,
            output_file="src/a.py",
            output_sha256=None,
        )
    }
    node = make_node(
        "review",
        "reviewer",
        ["impl"],
        ["reports/review.md"],
        False,
        inputs=[{"from": "impl", "field": "output_file"}],
        context_files=["research.md", "src/a.py"],
    )
    skeleton = make_skeleton([make_node("impl", "implementer", [], ["src/a.py"], False), node])

    contract = build_contract(node, skeleton, upstream_results, run_id="run-contract", workspace=str(tmp_path))

    assert contract.context_files == ["research.md", "src/a.py"]


def test_result_payload_includes_last_activity() -> None:
    result = _result(result="ok")
    result.last_activity = "running: pytest -q"

    payload = _result_payload(result)

    assert payload["last_activity"] == "running: pytest -q"
    assert _result_payload(_result())["last_activity"] is None


def _run_scheduler(
    skeleton: Skeleton,
    adapter: Any,
    tmp_path: Path,
    *,
    run_id: str,
    drift_handler: Any | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    workspace = tmp_path / f"{run_id}-workspace"
    workspace.mkdir()
    lock_base = tmp_path / f"{run_id}-locks"
    writer = LockWriter(run_id, base_dir=lock_base)
    scheduler = StaticScheduler(
        skeleton=skeleton,
        adapter=adapter,
        lock=writer,
        run_id=run_id,
        workspace=str(workspace),
        drift_handler=drift_handler,
    )

    states = scheduler.run()
    return states, read_events(run_id, base_dir=lock_base)


def _edges_from_dependencies(nodes: list[SkeletonNode]) -> list[Edge]:
    return [{"from": dependency, "to": node.id} for node in nodes for dependency in node.depends_on]


def _path_taken(events: list[dict[str, Any]]) -> list[str]:
    summary = next(event for event in events if event["evt"] == "run_summary")
    return list(summary["payload"]["path_taken"])


def _result(
    *,
    result: str = "ok",
    artifacts: list[dict[str, str]] | None = None,
    output_file: str = "",
) -> AtomResult:
    return AtomResult(
        status="success",
        result=result,
        artifacts=artifacts or [],
        session_id=None,
        cost=0.1,
        duration_sec=0.01,
        raw_events_path=None,
        error=None,
        output_file=output_file,
        output_sha256=None,
    )
