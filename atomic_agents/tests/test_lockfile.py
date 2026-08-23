from __future__ import annotations

from pathlib import Path
from typing import Any

from atomic_agents.lockfile import LockWriter, read_events
from atomic_agents.validation import validate_lockfile_event


ENVELOPE_FIELDS = {
    "schema_version",
    "event_id",
    "run_id",
    "atom_id",
    "attempt_id",
    "adapter_version",
    "ts",
    "evt",
    "payload",
}


def test_lock_writer_emits_enveloped_incrementing_events(tmp_path: Path) -> None:
    writer = LockWriter("run-test", base_dir=tmp_path)

    run_events = [
        writer.run_started({"launcher": "mock"}),
        writer.skeleton_hashed({"sha256": "abc", "skeleton_ref": "skeleton.json"}),
        writer.binding_locked({"bindings": {"planner": {"runner": "mock"}}}),
        writer.user_approved({"gate": "plan", "summary_ref": "approval-summary.json"}),
        writer.run_stopped({"reason": "done", "stop_report_ref": None}),
        writer.run_summary({"total_cost": 0.0, "path_taken": [], "env": {"runner": "mock-1"}}),
    ]
    atom_events = [
        writer.atom_input({"snapshot": {}}, atom_id="plan", adapter_version="mock-1"),
        writer.atom_started({"runner": "mock"}, atom_id="plan", adapter_version="mock-1"),
        writer.atom_finished({"status": "success"}, atom_id="plan", adapter_version="mock-1"),
        writer.scope_drift({"declared": [], "actual": [], "drift": []}, atom_id="plan", adapter_version="mock-1"),
        writer.review_finished({"verdict": "pass"}, atom_id="review", adapter_version="mock-1"),
        writer.retry_scheduled({}, atom_id="impl", adapter_version="mock-1"),
    ]
    events = run_events + atom_events

    assert [event["event_id"] for event in events] == [f"evt-{index:04d}" for index in range(1, 13)]
    assert [event["evt"] for event in run_events] == [
        "run_started",
        "skeleton_hashed",
        "binding_locked",
        "user_approved",
        "run_stopped",
        "run_summary",
    ]
    assert [event["evt"] for event in atom_events] == [
        "atom_input",
        "atom_started",
        "atom_finished",
        "scope_drift",
        "review_finished",
        "retry_scheduled",
    ]

    for event in events:
        assert ENVELOPE_FIELDS <= set(event)
        assert event["run_id"] == "run-test"
        assert event["attempt_id"] == 1
        validate_lockfile_event(event)

    assert all(event["atom_id"] is None for event in run_events)
    assert all(isinstance(event["atom_id"], str) for event in atom_events)


def test_read_events_uses_tolerant_reader_and_skips_empty_lines(tmp_path: Path) -> None:
    writer = LockWriter("run-test", base_dir=tmp_path)
    written = [
        writer.run_started({"launcher": "mock"}),
        writer.atom_finished({"status": "success"}, atom_id="plan", adapter_version="mock-1"),
    ]
    with writer.path.open("a", encoding="utf-8") as lockfile:
        lockfile.write("\n")

    events = read_events("run-test", base_dir=tmp_path)

    assert events == written
    for event in events:
        validate_lockfile_event(event)


def test_lock_writer_path_is_under_run_directory(tmp_path: Path) -> None:
    writer = LockWriter("run-test", base_dir=tmp_path)

    assert writer.run_dir == tmp_path / "run-test"
    assert writer.path == tmp_path / "run-test" / "run.lock.jsonl"
