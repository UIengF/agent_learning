from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal, cast

from atomic_agents.lockfile import LockWriter, read_events
from atomic_agents.models import AtomContract, AtomResult, Edge, InputRef, Skeleton, SkeletonNode, Status
from atomic_agents.reviewer import CriterionVerdict, ReviewResult
from atomic_agents.scheduler import StaticScheduler
from atomic_agents.validation import validate_stop_report


def make_node(
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
        output_file=write_scope[0] if output_file is None and write_scope else (output_file or ""),
    )


def make_skeleton(
    nodes: list[SkeletonNode],
    *,
    max_repair_attempts: int,
    max_total_cost: float = 10.0,
    edges: list[Edge] | None = None,
) -> Skeleton:
    return Skeleton(
        name="retry-test",
        version=1,
        nodes=nodes,
        edges=edges if edges is not None else _edges_from_dependencies(nodes),
        run_limits={"max_repair_attempts": max_repair_attempts, "max_total_cost": max_total_cost},
        irreversible_ops=[],
    )


def test_review_fail_retries_same_atom_then_passes(tmp_path: Path) -> None:
    runner = ScriptedFileRunner(verdicts=[_fail_review("tests", "3 failing"), _pass_review("tests")])

    states, events, _lock_base = _run_scheduler(
        _review_skeleton(max_repair_attempts=2),
        runner,
        tmp_path,
        run_id="run-review-retry-pass",
    )

    assert states["impl"].status == "succeeded"
    assert states["review"].status == "succeeded"
    assert runner.call_counts == {"impl": 2, "review": 2}

    retry_events = _events(events, "retry_scheduled")
    assert [(event["atom_id"], event["attempt_id"]) for event in retry_events] == [("impl", 2), ("review", 2)]

    reviews = _events(events, "review_finished")
    assert [event["payload"]["verdict"] for event in reviews] == ["fail", "pass"]
    assert [event["attempt_id"] for event in reviews] == [1, 2]


def test_review_fail_exhausts_attempts_and_stops_run(tmp_path: Path) -> None:
    nodes = _review_nodes()
    nodes.append(make_node("downstream", "verifier", ["review"], [], True))
    runner = ScriptedFileRunner(verdicts=[_fail_review("tests", "3 failing"), _fail_review("tests", "still failing")])

    states, events, lock_base = _run_scheduler(
        make_skeleton(nodes, max_repair_attempts=2),
        runner,
        tmp_path,
        run_id="run-review-retry-stop",
    )

    assert states["impl"].status == "succeeded"
    assert states["review"].status == "failed"
    assert states["downstream"].status == "blocked"
    assert runner.call_counts == {"impl": 2, "review": 2}

    stopped = _single_event(events, "run_stopped")
    assert stopped["payload"]["reason"] == "repair_failed"
    stop_report = _read_stop_report(lock_base, "run-review-retry-stop")
    validate_stop_report(stop_report)
    assert stop_report["reason"] == "repair_failed"
    assert stop_report["failed_atom"] == "review"
    assert stop_report["attempts"] == 2
    assert stop_report["reviewer_evidence"] == [
        {"criterion": "tests", "verdict": "fail", "evidence": "still failing"}
    ]


def test_retry_injects_reviewer_feedback_into_next_contract(tmp_path: Path) -> None:
    runner = ScriptedFileRunner(verdicts=[_fail_review("tests", "3 failing"), _pass_review("tests")])

    _states, _events, _lock_base = _run_scheduler(
        _review_skeleton(max_repair_attempts=2),
        runner,
        tmp_path,
        run_id="run-feedback",
    )

    assert len(runner.calls_by_id["impl"]) == 2
    assert runner.calls_by_id["impl"][1].task == (
        "Run impl\n\n[上一轮反馈]\nReviewer blocking feedback:\n- tests: 3 failing"
    )


def test_zero_max_repair_attempts_stops_without_retry(tmp_path: Path) -> None:
    runner = ScriptedFileRunner(verdicts=[_fail_review("tests", "3 failing")])

    states, events, _lock_base = _run_scheduler(
        _review_skeleton(max_repair_attempts=0),
        runner,
        tmp_path,
        run_id="run-zero-retry",
    )

    assert states["impl"].status == "succeeded"
    assert states["review"].status == "failed"
    assert runner.call_counts == {"impl": 1, "review": 1}
    assert _events(events, "retry_scheduled") == []
    assert _single_event(events, "run_stopped")["payload"]["reason"] == "repair_failed"


def test_adapter_failure_retries_same_atom_then_passes_review(tmp_path: Path) -> None:
    nodes = [
        make_node("impl", "implementer", [], [], True),
        make_node(
            "review",
            "reviewer",
            ["impl"],
            [],
            True,
            inputs=[{"from": "impl", "field": "result"}],
            criteria=["tests"],
            output_file="verdict.json",
        ),
    ]
    runner = ScriptedFileRunner(
        script={"impl": [{"status": "failed", "error": "boom"}, {"status": "success", "result": "fixed"}]},
        verdicts=[_pass_review("tests")],
    )

    states, events, _lock_base = _run_scheduler(
        make_skeleton(nodes, max_repair_attempts=2),
        runner,
        tmp_path,
        run_id="run-adapter-failure-retry",
    )

    assert states["impl"].status == "succeeded"
    assert states["review"].status == "succeeded"
    assert runner.call_counts == {"impl": 2, "review": 1}
    retry = _single_event(events, "retry_scheduled")
    assert retry["payload"]["feedback_summary"] == "boom"
    reviews = _events(events, "review_finished")
    assert len(reviews) == 1
    assert reviews[0]["payload"]["verdict"] == "pass"


def test_over_budget_stops_run(tmp_path: Path) -> None:
    nodes = [
        make_node("impl", "implementer", [], [], True, criteria=["tests"]),
        make_node("downstream", "verifier", ["impl"], [], True),
    ]
    runner = ScriptedFileRunner(script={"impl": [{"status": "success", "result": "ok", "cost": 0.2}]})

    states, events, lock_base = _run_scheduler(
        make_skeleton(nodes, max_repair_attempts=2, max_total_cost=0.1),
        runner,
        tmp_path,
        run_id="run-over-budget",
    )

    assert states["impl"].status == "succeeded"
    assert states["downstream"].status == "blocked"
    assert runner.call_counts == {"impl": 1}
    assert _single_event(events, "run_stopped")["payload"]["reason"] == "over_budget"
    stop_report = _read_stop_report(lock_base, "run-over-budget")
    validate_stop_report(stop_report)
    assert stop_report["reason"] == "over_budget"
    assert stop_report["cost_consumed"] == 0.2


def test_declared_output_file_missing_forces_failed(tmp_path: Path) -> None:
    node = make_node("impl", "implementer", [], ["out.md"], False)
    runner = ScriptedFileRunner(script={"impl": [{"write_output": False}]})

    states, events, _lock_base = _run_scheduler(
        make_skeleton([node], max_repair_attempts=0),
        runner,
        tmp_path,
        run_id="run-missing-output",
    )

    assert states["impl"].status == "failed"
    assert states["impl"].result is not None
    assert states["impl"].result.output_file == "out.md"
    assert states["impl"].result.output_sha256 is None
    assert states["impl"].result.error is not None
    assert "declared output_file not produced: out.md" in states["impl"].result.error
    assert runner.call_counts == {"impl": 1}
    assert _single_event(events, "run_stopped")["payload"]["reason"] == "atom_failed"


def test_output_file_present_verified_with_sha256(tmp_path: Path) -> None:
    nodes = [
        make_node("impl", "implementer", [], ["out.md"], False),
        make_node(
            "downstream",
            "verifier",
            ["impl"],
            [],
            True,
            inputs=[{"from": "impl", "field": "output_file"}],
        ),
    ]
    runner = FileWritingRunner(["verified output"])

    states, _events, _lock_base = _run_scheduler(
        make_skeleton(nodes, max_repair_attempts=0),
        runner,
        tmp_path,
        run_id="run-output-present",
    )

    impl_result = states["impl"].result
    assert states["impl"].status == "succeeded"
    assert states["downstream"].status == "succeeded"
    assert impl_result is not None
    assert impl_result.output_file == "out.md"
    assert impl_result.output_sha256 == hashlib.sha256(b"verified output").hexdigest()
    assert runner.calls[1].context_files == ["out.md"]


def test_retry_writes_new_attempt_file(tmp_path: Path) -> None:
    node = make_node("impl", "implementer", [], ["out.md"], False)
    runner = FileWritingRunner([None, "retry output"])

    states, events, _lock_base = _run_scheduler(
        make_skeleton([node], max_repair_attempts=2),
        runner,
        tmp_path,
        run_id="run-output-retry",
    )

    assert states["impl"].status == "succeeded"
    assert states["impl"].result is not None
    assert states["impl"].result.output_file == "out.attempt2.md"
    assert states["impl"].result.output_sha256 == hashlib.sha256(b"retry output").hexdigest()
    assert [call.output_file for call in runner.calls] == ["out.md", "out.attempt2.md"]
    assert (tmp_path / "run-output-retry-workspace" / "out.attempt2.md").read_text(encoding="utf-8") == "retry output"
    assert _single_event(events, "retry_scheduled")["attempt_id"] == 2
    assert _events(events, "scope_drift") == []


class TransientThenSuccessRunner:
    """Returns status=transient for the first N invokes, then succeeds.

    瞬时（基础设施）错误退避重试模拟：前 N 次返回 transient，第 N+1 次成功。
    用于验证调度器就地退避重试、不烧 repair 配额。
    """

    adapter_version = "transient-1"

    def __init__(self, transient_count: int) -> None:
        self.transient_count = transient_count
        self.calls = 0

    def invoke(self, contract: AtomContract, timeout_sec: int) -> AtomResult:
        del timeout_sec
        self.calls += 1
        if self.calls <= self.transient_count:
            return _atom_result(
                status="transient",
                result="",
                cost=0.0,
                error="503 All credentials exhausted; usually temporary",
            )
        if contract.output_file:
            _write_text(contract, "recovered output")
        return _atom_result(result=f"ok:{contract.atom_id}")


class AlwaysTransientRunner:
    adapter_version = "always-transient-1"

    def __init__(self) -> None:
        self.calls = 0

    def invoke(self, contract: AtomContract, timeout_sec: int) -> AtomResult:
        del timeout_sec
        self.calls += 1
        return _atom_result(
            status="transient",
            result="",
            cost=0.0,
            error="503 All credentials exhausted; server-side issue, usually temporary",
        )


def test_transient_error_backs_off_and_recovers_without_burning_repair(tmp_path: Path) -> None:
    # 网关抖动：前 2 次 transient、第 3 次成功。调度器就地退避重试，节点最终 succeeded，
    # 不消耗 max_repair_attempts（这里设 0，仍能恢复），不产生 retry_scheduled 事件。
    node = make_node("impl", "implementer", [], ["out.md"], False)
    runner = TransientThenSuccessRunner(transient_count=2)
    sleeps: list[float] = []

    states, events, _lock_base = _run_scheduler(
        make_skeleton([node], max_repair_attempts=0),
        runner,
        tmp_path,
        run_id="run-transient-recover",
        sleep_fn=sleeps.append,
    )

    assert states["impl"].status == "succeeded"
    assert runner.calls == 3  # 2 transient + 1 success
    assert sleeps == [5.0, 15.0]  # 退避序列，第3次成功无需再睡
    assert _events(events, "retry_scheduled") == []  # 未烧 repair 配额
    assert [event for event in events if event["evt"] == "run_stopped"] == []


def test_transient_error_exhausts_backoff_and_stops_with_infra_cause(tmp_path: Path) -> None:
    # 网关持续不可用：退避用尽仍 transient → 归一化 failed 并如实归因 infrastructure_unavailable，
    # stop-report 的 likely_causes 指向基础设施而非需求/任务。
    node = make_node("impl", "implementer", [], ["out.md"], False)
    runner = AlwaysTransientRunner()

    states, events, lock_base = _run_scheduler(
        make_skeleton([node], max_repair_attempts=0),
        runner,
        tmp_path,
        run_id="run-transient-exhaust",
    )

    assert states["impl"].status == "failed"
    assert runner.calls == 1 + 3  # 首次 + TRANSIENT_MAX_RETRIES 次退避
    stopped = _single_event(events, "run_stopped")
    assert stopped["payload"]["reason"] == "infrastructure_unavailable"

    stop_report = stopped["payload"]["stop_report"]
    validate_stop_report(stop_report)
    assert any("基础设施" in cause for cause in stop_report["likely_causes"])
    assert states["impl"].result is not None
    assert states["impl"].result.error is not None
    assert "transient_exhausted" in states["impl"].result.error


class ScriptedFileRunner:
    adapter_version = "scripted-file-1"

    def __init__(
        self,
        script: dict[str, list[dict[str, Any]]] | None = None,
        verdicts: list[ReviewResult] | None = None,
    ) -> None:
        self.script = {key: [deepcopy(item) for item in value] for key, value in (script or {}).items()}
        self.verdicts = list(verdicts or [])
        self.call_counts: dict[str, int] = {}
        self.calls: list[AtomContract] = []
        self.calls_by_id: dict[str, list[AtomContract]] = {}

    def invoke(self, contract: AtomContract, timeout_sec: int) -> AtomResult:
        del timeout_sec
        self.calls.append(contract)
        call_index = self.call_counts.get(contract.atom_id, 0)
        self.call_counts[contract.atom_id] = call_index + 1
        self.calls_by_id.setdefault(contract.atom_id, []).append(contract)

        if contract.logical_role.strip().lower() == "reviewer":
            verdict = self.verdicts[min(call_index, len(self.verdicts) - 1)] if self.verdicts else _pass_review("tests")
            _write_text(contract, json.dumps(verdict.to_dict()))
            return _atom_result(result=f"review:{contract.atom_id}")

        response = self._scripted_response(contract.atom_id, call_index)
        status = cast(Status, response.get("status", "success"))
        should_write = bool(response.get("write_output", status == "success"))
        if should_write and contract.output_file:
            _write_text(contract, str(response.get("content", f"product:{contract.atom_id}:{call_index + 1}")))

        return _atom_result(
            status=status,
            result=str(response.get("result", f"ok:{contract.atom_id}")),
            cost=float(response.get("cost", 0.1)),
            error=cast(str | None, response.get("error")),
        )

    def _scripted_response(self, atom_id: str, call_index: int) -> dict[str, Any]:
        responses = self.script.get(atom_id)
        if not responses:
            return {}
        return deepcopy(responses[min(call_index, len(responses) - 1)])


class FileWritingRunner:
    adapter_version = "file-writer-1"

    def __init__(self, contents: list[str | None]) -> None:
        self.contents = list(contents)
        self.calls: list[AtomContract] = []
        self.call_counts: dict[str, int] = {}

    def invoke(self, contract: AtomContract, timeout_sec: int) -> AtomResult:
        del timeout_sec
        self.calls.append(contract)
        self.call_counts[contract.atom_id] = self.call_counts.get(contract.atom_id, 0) + 1
        index = min(len(self.calls) - 1, len(self.contents) - 1)
        content = self.contents[index] if self.contents else None
        if content is not None and contract.output_file:
            _write_text(contract, content)

        return _atom_result(result=f"ok:{contract.atom_id}")


def _review_skeleton(*, max_repair_attempts: int) -> Skeleton:
    return make_skeleton(_review_nodes(), max_repair_attempts=max_repair_attempts)


def _review_nodes() -> list[SkeletonNode]:
    return [
        make_node("impl", "implementer", [], ["impl.txt"], False),
        make_node(
            "review",
            "reviewer",
            ["impl"],
            [],
            True,
            inputs=[{"from": "impl", "field": "output_file"}],
            criteria=["tests"],
            output_file="verdict.json",
        ),
    ]


def _run_scheduler(
    skeleton: Skeleton,
    adapter: Any,
    tmp_path: Path,
    *,
    run_id: str,
    sleep_fn: Any | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]], Path]:
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
        sleep_fn=sleep_fn if sleep_fn is not None else (lambda _seconds: None),
    )

    states = scheduler.run()
    return states, read_events(run_id, base_dir=lock_base), lock_base


def _write_text(contract: AtomContract, content: str) -> None:
    output_path = Path(contract.output_file)
    if not output_path.is_absolute():
        output_path = Path(contract.workspace) / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(content, encoding="utf-8")


def _atom_result(
    *,
    status: Status = "success",
    result: str = "ok",
    cost: float = 0.1,
    error: str | None = None,
) -> AtomResult:
    return AtomResult(
        status=status,
        result=result,
        artifacts=[],
        session_id=None,
        cost=cost,
        duration_sec=0.01,
        raw_events_path=None,
        error=error,
        output_file="",
        output_sha256=None,
    )


def _pass_review(criterion: str) -> ReviewResult:
    return ReviewResult(
        passed=True,
        criteria=[_criterion(criterion, "pass", "ok")],
        blocking_findings=[],
        reviewer_session=None,
        feedback="",
    )


def _fail_review(criterion: str, evidence: str) -> ReviewResult:
    return ReviewResult(
        passed=False,
        criteria=[_criterion(criterion, "fail", evidence)],
        blocking_findings=[criterion],
        reviewer_session=None,
        feedback="",
    )


def _criterion(
    criterion: str,
    verdict: Literal["pass", "fail"],
    evidence: str,
) -> CriterionVerdict:
    return CriterionVerdict(
        criterion=criterion,
        verdict=verdict,
        evidence=evidence,
        confidence="high",
    )


def _edges_from_dependencies(nodes: list[SkeletonNode]) -> list[Edge]:
    return [{"from": dependency, "to": node.id} for node in nodes for dependency in node.depends_on]


def _events(events: list[dict[str, Any]], event_name: str) -> list[dict[str, Any]]:
    return [event for event in events if event["evt"] == event_name]


def _single_event(events: list[dict[str, Any]], event_name: str) -> dict[str, Any]:
    matching = _events(events, event_name)
    assert len(matching) == 1
    return matching[0]


def _read_stop_report(lock_base: Path, run_id: str) -> dict[str, Any]:
    report_path = lock_base / run_id / "stop-report.json"
    with report_path.open("r", encoding="utf-8") as report_file:
        return json.load(report_file)
