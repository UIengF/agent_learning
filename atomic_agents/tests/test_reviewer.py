from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import pytest

from atomic_agents.lockfile import LockWriter, read_events
from atomic_agents.models import AtomContract, AtomResult, Edge, InputRef, Skeleton, SkeletonNode
from atomic_agents.reviewer import CriterionVerdict, ReviewResult, build_review_feedback, review_to_lock_payload
from atomic_agents.scheduler import StaticScheduler


def test_review_result_passed_follows_blocking_findings() -> None:
    passing = ReviewResult(
        passed=False,
        criteria=[_criterion("tests", "pass", "not blocking")],
        blocking_findings=[],
        reviewer_session="rev-pass",
        feedback="",
    )
    failing = ReviewResult(
        passed=True,
        criteria=[_criterion("tests", "fail", "3 failing")],
        blocking_findings=["tests"],
        reviewer_session="rev-fail",
        feedback="",
    )

    assert passing.passed is True
    assert failing.passed is False


def test_review_result_rejects_failed_criterion_missing_from_blocking_findings() -> None:
    with pytest.raises(ValueError, match="criteria marked fail but missing from blocking_findings"):
        ReviewResult.from_dict(
            {
                "passed": True,
                "criteria": [
                    {
                        "criterion": "tests",
                        "verdict": "fail",
                        "evidence": "3 failing",
                        "confidence": "high",
                    }
                ],
                "blocking_findings": [],
                "reviewer_session": None,
                "feedback": "",
            }
        )


def test_review_result_allows_extra_global_blocking_findings() -> None:
    review = ReviewResult(
        passed=True,
        criteria=[_criterion("tests", "fail", "3 failing")],
        blocking_findings=["tests", "verdict_file"],
        reviewer_session="rev-fail",
        feedback="",
    )

    assert review.passed is False
    assert review.blocking_findings == ["tests", "verdict_file"]


def test_build_review_feedback_summarizes_failed_criteria_and_evidence() -> None:
    review = ReviewResult(
        passed=False,
        criteria=[
            _criterion("tests", "fail", "3 failing"),
            _criterion("lint", "pass", "clean"),
            _criterion("requirements", "fail", "missing retry feedback"),
        ],
        blocking_findings=["tests", "requirements"],
        reviewer_session="rev-impl-0",
        feedback="",
    )

    feedback = build_review_feedback(review)

    assert "tests: 3 failing" in feedback
    assert "requirements: missing retry feedback" in feedback
    assert "lint" not in feedback


def test_review_to_lock_payload_matches_review_finished_shape() -> None:
    review = ReviewResult(
        passed=False,
        criteria=[
            _criterion("c1", "pass", "ok", "medium"),
            _criterion("c2", "fail", "3 tests failing", "high"),
        ],
        blocking_findings=["c2"],
        reviewer_session="rev-impl-0",
        feedback="",
    )

    assert review_to_lock_payload(review) == {
        "verdict": "fail",
        "criteria": [
            {"id": "c1", "verdict": "pass", "evidence": "ok", "confidence": "medium"},
            {"id": "c2", "verdict": "fail", "evidence": "3 tests failing", "confidence": "high"},
        ],
        "blocking_findings": ["c2"],
        "reviewer_session": "rev-impl-0",
    }


def test_reviewer_node_pass_verdict_lets_downstream_proceed(tmp_path: Path) -> None:
    skeleton = _skeleton_with_downstream(max_repair_attempts=1)
    adapter = VerdictRunner([_pass_review("tests")])

    states, events, _lock_base = _run_scheduler(skeleton, adapter, tmp_path, run_id="run-review-pass")

    assert states["impl"].status == "succeeded"
    assert states["review"].status == "succeeded"
    assert states["downstream"].status == "succeeded"
    assert adapter.call_counts == {"impl": 1, "review": 1, "downstream": 1}
    assert _path_taken(events) == ["impl", "review", "downstream"]
    assert [event["payload"]["verdict"] for event in _events(events, "review_finished")] == ["pass"]


def test_reviewer_node_fail_verdict_triggers_upstream_retry(tmp_path: Path) -> None:
    skeleton = _skeleton_with_downstream(max_repair_attempts=2)
    adapter = VerdictRunner([_fail_review("tests", "3 failing"), _pass_review("tests")])

    states, events, _lock_base = _run_scheduler(skeleton, adapter, tmp_path, run_id="run-review-retry")

    assert {state.status for state in states.values()} == {"succeeded"}
    assert adapter.call_counts == {"impl": 2, "review": 2, "downstream": 1}
    assert [call.output_file for call in adapter.calls_by_id["impl"]] == ["impl.txt", "impl.attempt2.txt"]
    assert adapter.calls_by_id["impl"][1].task == (
        "Run impl\n\n[上一轮反馈]\nReviewer blocking feedback:\n- tests: 3 failing"
    )
    assert [event["payload"]["verdict"] for event in _events(events, "review_finished")] == ["fail", "pass"]
    assert [event["attempt_id"] for event in _events(events, "review_finished")] == [1, 2]
    retry_events = _events(events, "retry_scheduled")
    assert [(event["atom_id"], event["attempt_id"]) for event in retry_events] == [("impl", 2), ("review", 2)]


def test_verdict_file_missing_treated_as_reviewer_node_failure(tmp_path: Path) -> None:
    skeleton = _skeleton_with_downstream(max_repair_attempts=1)
    adapter = VerdictRunner([_pass_review("tests")], missing_verdict=True)

    states, events, _lock_base = _run_scheduler(skeleton, adapter, tmp_path, run_id="run-review-missing")

    assert states["impl"].status == "succeeded"
    assert states["review"].status == "failed"
    assert states["downstream"].status == "blocked"
    assert states["review"].result is not None
    assert states["review"].result.error is not None
    assert "declared output_file not produced: verdict.json" in states["review"].result.error
    assert _events(events, "review_finished") == []
    assert _single_event(events, "run_stopped")["payload"]["reason"] == "atom_failed"


def test_verdict_file_malformed_treated_as_fail(tmp_path: Path) -> None:
    skeleton = _review_skeleton(max_repair_attempts=2)
    adapter = VerdictRunner([_pass_review("tests")], malformed_verdict=True)

    states, events, lock_base = _run_scheduler(skeleton, adapter, tmp_path, run_id="run-review-malformed")

    assert states["impl"].status == "succeeded"
    assert states["review"].status == "failed"
    assert adapter.call_counts == {"impl": 2, "review": 2}
    reviews = _events(events, "review_finished")
    assert [event["payload"]["verdict"] for event in reviews] == ["fail", "fail"]
    assert all(event["payload"]["blocking_findings"] == ["verdict_file"] for event in reviews)
    retry_events = _events(events, "retry_scheduled")
    assert [(event["atom_id"], event["attempt_id"]) for event in retry_events] == [("impl", 2), ("review", 2)]
    stopped = _single_event(events, "run_stopped")
    assert stopped["payload"]["reason"] == "repair_failed"
    stop_report = _read_stop_report(lock_base, "run-review-malformed")
    assert stop_report["failed_atom"] == "review"
    assert stop_report["reviewer_evidence"][0]["criterion"] == "verdict_file"


class VerdictRunner:
    adapter_version = "verdict-runner-1"

    def __init__(
        self,
        verdicts: list[ReviewResult],
        *,
        missing_verdict: bool = False,
        malformed_verdict: bool = False,
    ) -> None:
        self.verdicts = list(verdicts)
        self.missing_verdict = missing_verdict
        self.malformed_verdict = malformed_verdict
        self.call_counts: dict[str, int] = {}
        self.calls_by_id: dict[str, list[AtomContract]] = {}

    def invoke(self, contract: AtomContract, timeout_sec: int) -> AtomResult:
        del timeout_sec
        call_index = self.call_counts.get(contract.atom_id, 0)
        self.call_counts[contract.atom_id] = call_index + 1
        self.calls_by_id.setdefault(contract.atom_id, []).append(contract)

        if contract.logical_role.strip().lower() == "reviewer":
            if not self.missing_verdict and contract.output_file:
                output_path = _workspace_path(contract)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                if self.malformed_verdict:
                    output_path.write_text("{not-json", encoding="utf-8")
                else:
                    verdict = self.verdicts[min(call_index, len(self.verdicts) - 1)]
                    output_path.write_text(json.dumps(verdict.to_dict()), encoding="utf-8")
        elif contract.output_file:
            output_path = _workspace_path(contract)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(f"product:{contract.atom_id}:{call_index + 1}", encoding="utf-8")

        return AtomResult(
            status="success",
            result=f"ok:{contract.atom_id}",
            artifacts=[],
            session_id=None,
            cost=0.1,
            duration_sec=0.01,
            raw_events_path=None,
            error=None,
            output_file="",
            output_sha256=None,
        )


def _workspace_path(contract: AtomContract) -> Path:
    output_path = Path(contract.output_file)
    if output_path.is_absolute():
        return output_path
    return Path(contract.workspace) / output_path


def _skeleton_with_downstream(*, max_repair_attempts: int) -> Skeleton:
    nodes = [
        _node("impl", "implementer", [], ["impl.txt"], False),
        _node(
            "review",
            "reviewer",
            ["impl"],
            [],
            True,
            inputs=[{"from": "impl", "field": "output_file"}],
            criteria=["tests"],
            output_file="verdict.json",
        ),
        _node("downstream", "publisher", ["review"], [], True),
    ]
    return _make_skeleton(nodes, max_repair_attempts=max_repair_attempts)


def _review_skeleton(*, max_repair_attempts: int) -> Skeleton:
    nodes = [
        _node("impl", "implementer", [], ["impl.txt"], False),
        _node(
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
    return _make_skeleton(nodes, max_repair_attempts=max_repair_attempts)


def _node(
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


def _make_skeleton(nodes: list[SkeletonNode], *, max_repair_attempts: int) -> Skeleton:
    return Skeleton(
        name="reviewer-node-test",
        version=1,
        nodes=nodes,
        edges=_edges_from_dependencies(nodes),
        run_limits={"max_repair_attempts": max_repair_attempts, "max_total_cost": 10.0},
        irreversible_ops=[],
    )


def _run_scheduler(
    skeleton: Skeleton,
    adapter: VerdictRunner,
    tmp_path: Path,
    *,
    run_id: str,
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
    )

    states = scheduler.run()
    return states, read_events(run_id, base_dir=lock_base), lock_base


def _pass_review(criterion: str) -> ReviewResult:
    return ReviewResult(
        passed=True,
        criteria=[_criterion(criterion, "pass", "ok")],
        blocking_findings=[],
        reviewer_session="rev-pass",
        feedback="",
    )


def _fail_review(criterion: str, evidence: str) -> ReviewResult:
    return ReviewResult(
        passed=False,
        criteria=[_criterion(criterion, "fail", evidence)],
        blocking_findings=[criterion],
        reviewer_session="rev-fail",
        feedback="",
    )


def _criterion(
    criterion: str,
    verdict: Literal["pass", "fail"],
    evidence: str,
    confidence: Literal["high", "medium", "low"] = "high",
) -> CriterionVerdict:
    return CriterionVerdict(
        criterion=criterion,
        verdict=verdict,
        evidence=evidence,
        confidence=confidence,
    )


def _edges_from_dependencies(nodes: list[SkeletonNode]) -> list[Edge]:
    return [{"from": dependency, "to": node.id} for node in nodes for dependency in node.depends_on]


def _events(events: list[dict[str, Any]], event_name: str) -> list[dict[str, Any]]:
    return [event for event in events if event["evt"] == event_name]


def _single_event(events: list[dict[str, Any]], event_name: str) -> dict[str, Any]:
    matching = _events(events, event_name)
    assert len(matching) == 1
    return matching[0]


def _path_taken(events: list[dict[str, Any]]) -> list[str]:
    return list(_single_event(events, "run_summary")["payload"]["path_taken"])


def _read_stop_report(lock_base: Path, run_id: str) -> dict[str, Any]:
    report_path = lock_base / run_id / "stop-report.json"
    with report_path.open("r", encoding="utf-8") as report_file:
        report = json.load(report_file)
    assert isinstance(report, dict)
    return report
