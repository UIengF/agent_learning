from __future__ import annotations

from typing import Any

from atomic_agents.mock_runner import MockRunner
from atomic_agents.models import AtomContract


def contract(atom_id: str = "impl", logical_role: str = "implementer", task: str = "Implement feature") -> AtomContract:
    return AtomContract(
        task=task,
        inputs=[],
        context_files=[],
        workspace=".",
        read_only=False,
        write_scope=["src/feature.py"],
        required_capabilities=["write_files"],
        status="success",
        result="",
        artifacts=[],
        output_file="src/feature.py",
        output_schema_ref=None,
        handoff={"completed": [], "pending": [], "decisions": [], "risks": []},
        consult=None,
        atom_id=atom_id,
        correlation_id="run-test",
        logical_role=logical_role,
        resolved_runner=None,
        session_id=None,
        hop_count=1,
        limits={"max_cost": 2.0, "timeout_sec": 1800, "max_internal_turns": 30},
        cost=0.0,
        duration_sec=0.0,
        timestamps={"started_at": "2026-06-27T00:00:00Z", "finished_at": "2026-06-27T00:00:00Z"},
    )


def test_mock_runner_default_echoes_task_and_tracks_calls() -> None:
    runner = MockRunner()
    atom_contract = contract(task="Plan the work")

    result = runner.invoke(atom_contract, timeout_sec=30)

    assert result.status == "success"
    assert result.result == "Plan the work"
    assert result.artifacts == []
    assert result.cost == 0.1
    assert result.duration_sec == 1.0
    assert result.raw_events_path is None
    assert result.error is None
    assert runner.adapter_version == "mock-1"
    assert runner.call_count == 1
    assert runner.calls == [atom_contract]
    assert runner.timeouts == [30]


def test_mock_runner_scripted_responses_are_deterministic_across_invokes() -> None:
    runner = MockRunner(
        {
            "impl": [
                {"status": "failed", "result": "first attempt", "error": "tests failed"},
                {"status": "success", "result": "second attempt"},
            ]
        }
    )
    atom_contract = contract(atom_id="impl")

    first = runner.invoke(atom_contract, timeout_sec=60)
    second = runner.invoke(atom_contract, timeout_sec=60)
    third = runner.invoke(atom_contract, timeout_sec=60)

    assert first.status == "failed"
    assert first.result == "first attempt"
    assert first.error == "tests failed"
    assert second.status == "success"
    assert second.result == "second attempt"
    assert third.status == "success"
    assert third.result == "second attempt"
    assert runner.call_count == 3
    assert runner.call_counts["impl"] == 3


def test_mock_runner_uses_logical_role_when_atom_id_is_not_scripted() -> None:
    runner = MockRunner({"reviewer": [{"result": "pass"}]})

    result = runner.invoke(contract(atom_id="review", logical_role="reviewer"), timeout_sec=30)

    assert result.result == "pass"
    assert runner.call_counts["reviewer"] == 1


def test_mock_runner_prefers_atom_id_over_logical_role_script() -> None:
    runner = MockRunner(
        {
            "impl": [{"result": "by atom"}],
            "implementer": [{"result": "by role"}],
        }
    )

    result = runner.invoke(contract(atom_id="impl", logical_role="implementer"), timeout_sec=30)

    assert result.result == "by atom"


def test_mock_runner_injects_failure_timeout_cost_and_artifacts() -> None:
    artifact: dict[str, Any] = {"path": "src/feature.py", "type": "code", "sha256": "abc123"}
    runner = MockRunner(
        {
            "implementer": [
                {
                    "status": "timeout",
                    "result": "partial",
                    "artifacts": [artifact],
                    "session_id": "sess-1",
                    "cost": 2.5,
                    "duration_sec": 61.0,
                    "raw_events_path": None,
                    "error": "timed out",
                }
            ]
        }
    )

    result = runner.invoke(contract(atom_id="other", logical_role="implementer"), timeout_sec=60)

    assert result.status == "timeout"
    assert result.result == "partial"
    assert result.artifacts == [artifact]
    assert result.session_id == "sess-1"
    assert result.cost == 2.5
    assert result.duration_sec == 61.0
    assert result.raw_events_path is None
    assert result.error == "timed out"
