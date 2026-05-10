from __future__ import annotations

import json
from pathlib import Path

from aicoding_app.agent import TaskResult
from aicoding_app.cli import main
from aicoding_app.config import build_app_config
import aicoding_app.eval_harness as eval_harness
from aicoding_app.eval_harness import (
    _build_agent_task,
    format_eval_result,
    load_eval_suite,
    run_eval_suite,
)


class FakeValidation:
    def __init__(self, command: str, ok: bool = True):
        self.command = command
        self.ok = ok

    def format(self) -> str:
        status = "passed" if self.ok else "failed"
        return f"Validation command: {self.command}\nStatus: {status}"


def _write_suite(tmp_path: Path, tasks: list[dict[str, object]]) -> Path:
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"tasks": tasks}), encoding="utf-8")
    return suite


def test_load_eval_suite_maps_legacy_expected_validation(tmp_path: Path) -> None:
    suite = _write_suite(
        tmp_path,
        [
            {
                "name": "legacy",
                "task": "do work",
                "suggested_mode": "invalid",
                "expected_validation": "python -m pytest tests",
            }
        ],
    )

    task = load_eval_suite(suite)[0]

    assert task.name == "legacy"
    assert task.suggested_mode == "agent"
    assert task.visible_validation == ()
    assert task.hidden_validation == ("python -m pytest tests",)


def test_load_eval_suite_accepts_utf8_bom(tmp_path: Path) -> None:
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps({"tasks": [{"task": "inspect"}]}),
        encoding="utf-8-sig",
    )

    task = load_eval_suite(suite)[0]

    assert task.task == "inspect"


def test_load_eval_suite_supports_string_and_list_validations(tmp_path: Path) -> None:
    suite = _write_suite(
        tmp_path,
        [
            {
                "task": "do work",
                "visible_validation": "python -m pytest tests",
                "hidden_validation": [
                    "python -m pytest tests",
                    "python -m ruff check .",
                ],
            },
            {
                "task": "no validation",
            },
        ],
    )

    first, second = load_eval_suite(suite)

    assert first.visible_validation == ("python -m pytest tests",)
    assert first.hidden_validation == (
        "python -m pytest tests",
        "python -m ruff check .",
    )
    assert second.hidden_validation == ()
    assert "not specified" not in second.hidden_validation


def test_load_eval_suite_supports_group_and_continue_session(tmp_path: Path) -> None:
    suite = _write_suite(
        tmp_path,
        [
            {
                "task": "round one",
                "group_id": "worklog",
            },
            {
                "task": "round two",
                "group_id": "worklog",
                "continue_session": True,
            },
        ],
    )

    first, second = load_eval_suite(suite)

    assert first.group_id == "worklog"
    assert not first.continue_session
    assert second.group_id == "worklog"
    assert second.continue_session


def test_load_eval_suite_supports_setup_files(tmp_path: Path) -> None:
    suite = _write_suite(
        tmp_path,
        [
            {
                "task": "fix",
                "setup_files": {
                    "app.py": "value = 1\n",
                    "tests/test_app.py": "def test_ok():\n    assert True\n",
                },
            }
        ],
    )

    task = load_eval_suite(suite)[0]

    assert task.setup_files == (
        ("app.py", "value = 1\n"),
        ("tests/test_app.py", "def test_ok():\n    assert True\n"),
    )


def test_boundary_eval_suite_defines_first_twelve_cases() -> None:
    suite = Path("tests/fixtures/eval_cases/boundary_suite.json")

    tasks = load_eval_suite(suite)

    assert len(tasks) == 12
    assert {task.name for task in tasks} == {
        "case01 harness hidden pass",
        "case02 hidden regression edge",
        "case03 permission note",
        "case04 worklog round one",
        "case05 worklog round two",
        "case06 csv bom cli",
        "case07 json store export",
        "case08 note indexer round one",
        "case09 note indexer round two",
        "case10 repair public tests hidden edge",
        "case11 policy denial recovery",
        "case12 ambiguous output format",
    }
    assert tasks[4].group_id == "worklog"
    assert tasks[4].continue_session
    assert tasks[8].group_id == "note-indexer"
    assert tasks[8].continue_session
    assert all(task.hidden_validation for task in tasks)
    assert all("boundary_checks.py" not in _build_agent_task(task) for task in tasks)
    assert all("{{suite_dir}}" not in command for task in tasks for command in task.hidden_validation)
    assert all(Path(command.split()[3]).is_absolute() for task in tasks for command in task.hidden_validation)


def test_build_agent_task_includes_visible_and_hides_hidden_validation() -> None:
    task = eval_harness.EvalTask(
        name="demo",
        task="fix the app",
        suggested_mode="agent",
        visible_validation=("python -m pytest visible",),
        hidden_validation=("python -m pytest hidden",),
    )

    prompt = _build_agent_task(task)

    assert "fix the app" in prompt
    assert "Suggested validation:" in prompt
    assert "python -m pytest visible" in prompt
    assert "python -m pytest hidden" not in prompt


def test_run_eval_suite_runs_agent_then_hidden_validation(
    monkeypatch, tmp_path: Path
) -> None:
    suite = _write_suite(
        tmp_path,
        [
            {
                "name": "demo",
                "task": "fix the app",
                "visible_validation": "python -m pytest visible",
                "hidden_validation": "python -m pytest hidden",
            }
        ],
    )
    seen_prompts: list[str] = []
    seen_validations: list[str] = []

    class FakeAgent:
        def __init__(self, **kwargs):
            pass

        def run_mode_task(self, mode: str, task: str) -> TaskResult:
            seen_prompts.append(task)
            return TaskResult(session_id="fake-session", task_id="task-1", response="done")

    def fake_run_validation(policy, command: str, *, timeout_seconds: int) -> FakeValidation:
        seen_validations.append(command)
        return FakeValidation(command)

    monkeypatch.setattr(eval_harness, "CodingAgent", FakeAgent)
    monkeypatch.setattr(eval_harness, "run_validation", fake_run_validation)
    config = build_app_config(
        env_file=tmp_path / "missing.env",
        env={"AICODING_RUNTIME_DIR": str(tmp_path / "runtime")},
    )

    result = run_eval_suite(config, tmp_path, suite)

    assert result.task_results[0].status == "passed"
    assert "python -m pytest visible" in seen_prompts[0]
    assert "python -m pytest hidden" not in seen_prompts[0]
    assert seen_validations == ["python -m pytest hidden"]


def test_run_eval_suite_reuses_session_for_continue_session_group(
    monkeypatch, tmp_path: Path
) -> None:
    suite = _write_suite(
        tmp_path,
        [
            {"name": "round one", "task": "create", "group_id": "worklog"},
            {
                "name": "round two",
                "task": "extend",
                "group_id": "worklog",
                "continue_session": True,
            },
            {"name": "independent", "task": "inspect"},
        ],
    )
    seen_sessions: list[str] = []

    class FakeAgent:
        def __init__(self, **kwargs):
            seen_sessions.append(str(kwargs["session_id"]))

        def run_mode_task(self, mode: str, task: str) -> TaskResult:
            return TaskResult(session_id=seen_sessions[-1], task_id="task-1", response="done")

    monkeypatch.setattr(eval_harness, "CodingAgent", FakeAgent)
    config = build_app_config(
        env_file=tmp_path / "missing.env",
        env={"AICODING_RUNTIME_DIR": str(tmp_path / "runtime")},
    )

    result = run_eval_suite(config, tmp_path, suite)

    assert seen_sessions[0] == seen_sessions[1]
    assert seen_sessions[2] != seen_sessions[0]
    assert result.task_results[0].session_id == result.task_results[1].session_id


def test_run_eval_suite_uses_distinct_artifacts_for_continue_session_group(
    monkeypatch, tmp_path: Path
) -> None:
    suite = _write_suite(
        tmp_path,
        [
            {
                "name": "round one",
                "task": "create",
                "group_id": "worklog",
                "hidden_validation": "python -m pytest tests",
            },
            {
                "name": "round two",
                "task": "extend",
                "group_id": "worklog",
                "continue_session": True,
                "hidden_validation": "python -m pytest tests",
            },
        ],
    )
    calls = 0

    class FakeAgent:
        def __init__(self, **kwargs):
            pass

        def run_mode_task(self, mode: str, task: str) -> TaskResult:
            nonlocal calls
            calls += 1
            return TaskResult(
                session_id="shared",
                task_id=f"task-{calls}",
                response=f"response {calls}",
            )

    def fake_run_validation(policy, command: str, *, timeout_seconds: int) -> FakeValidation:
        return FakeValidation(command)

    monkeypatch.setattr(eval_harness, "CodingAgent", FakeAgent)
    monkeypatch.setattr(eval_harness, "run_validation", fake_run_validation)
    config = build_app_config(
        env_file=tmp_path / "missing.env",
        env={"AICODING_RUNTIME_DIR": str(tmp_path / "runtime")},
    )

    result = run_eval_suite(config, tmp_path, suite)
    first, second = result.task_results

    assert first.session_id == second.session_id
    assert first.response_path != second.response_path
    assert first.validations[0].output_path != second.validations[0].output_path
    assert Path(first.response_path).read_text(encoding="utf-8") == "response 1"
    assert Path(second.response_path).read_text(encoding="utf-8") == "response 2"


def test_run_eval_suite_writes_setup_files_before_agent(monkeypatch, tmp_path: Path) -> None:
    suite = _write_suite(
        tmp_path,
        [
            {
                "name": "setup",
                "task": "inspect seed",
                "setup_files": {"seed/app.py": "value = 1\n"},
            }
        ],
    )

    class FakeAgent:
        def __init__(self, **kwargs):
            pass

        def run_mode_task(self, mode: str, task: str) -> TaskResult:
            assert (tmp_path / "seed" / "app.py").read_text(encoding="utf-8") == "value = 1\n"
            return TaskResult(session_id="fake-session", task_id="task-1", response="done")

    monkeypatch.setattr(eval_harness, "CodingAgent", FakeAgent)
    config = build_app_config(
        env_file=tmp_path / "missing.env",
        env={"AICODING_RUNTIME_DIR": str(tmp_path / "runtime")},
    )

    result = run_eval_suite(config, tmp_path, suite)

    assert result.task_results[0].status == "passed_unvalidated"
    assert result.task_results[0].setup_errors == ()


def test_run_eval_suite_reports_setup_file_path_errors(monkeypatch, tmp_path: Path) -> None:
    suite = _write_suite(
        tmp_path,
        [{"name": "bad setup", "task": "inspect", "setup_files": {"../outside.py": "x = 1\n"}}],
    )

    class FakeAgent:
        def __init__(self, **kwargs):
            raise AssertionError("agent should not run when setup fails")

    monkeypatch.setattr(eval_harness, "CodingAgent", FakeAgent)
    config = build_app_config(
        env_file=tmp_path / "missing.env",
        env={"AICODING_RUNTIME_DIR": str(tmp_path / "runtime")},
    )

    result = run_eval_suite(config, tmp_path, suite)

    assert result.task_results[0].status == "error"
    assert result.task_results[0].setup_errors
    assert Path(result.task_results[0].response_path).read_text(encoding="utf-8")


def test_run_eval_suite_failed_hidden_validation_sets_failed(
    monkeypatch, tmp_path: Path
) -> None:
    suite = _write_suite(
        tmp_path,
        [{"name": "demo", "task": "fix", "hidden_validation": "python -m pytest tests"}],
    )

    class FakeAgent:
        def __init__(self, **kwargs):
            pass

        def run_mode_task(self, mode: str, task: str) -> TaskResult:
            return TaskResult(session_id="fake-session", task_id="task-1", response="done")

    def fake_run_validation(policy, command: str, *, timeout_seconds: int) -> FakeValidation:
        return FakeValidation(command, ok=False)

    monkeypatch.setattr(eval_harness, "CodingAgent", FakeAgent)
    monkeypatch.setattr(eval_harness, "run_validation", fake_run_validation)
    config = build_app_config(
        env_file=tmp_path / "missing.env",
        env={"AICODING_RUNTIME_DIR": str(tmp_path / "runtime")},
    )

    result = run_eval_suite(config, tmp_path, suite)

    assert result.task_results[0].status == "failed"
    assert result.has_failures


def test_run_eval_suite_denied_hidden_validation_sets_denied(
    monkeypatch, tmp_path: Path
) -> None:
    suite = _write_suite(
        tmp_path,
        [{"name": "demo", "task": "fix", "hidden_validation": "python -c \"print(1)\""}],
    )

    class FakeAgent:
        def __init__(self, **kwargs):
            pass

        def run_mode_task(self, mode: str, task: str) -> TaskResult:
            return TaskResult(session_id="fake-session", task_id="task-1", response="done")

    monkeypatch.setattr(eval_harness, "CodingAgent", FakeAgent)
    config = build_app_config(
        env_file=tmp_path / "missing.env",
        env={"AICODING_RUNTIME_DIR": str(tmp_path / "runtime")},
    )

    result = run_eval_suite(config, tmp_path, suite)

    assert result.task_results[0].status == "denied"
    assert result.task_results[0].validations[0].status == "denied"


def test_run_eval_suite_validation_error_sets_error(monkeypatch, tmp_path: Path) -> None:
    suite = _write_suite(
        tmp_path,
        [{"name": "demo", "task": "fix", "hidden_validation": "python -m pytest tests"}],
    )

    class FakeAgent:
        def __init__(self, **kwargs):
            pass

        def run_mode_task(self, mode: str, task: str) -> TaskResult:
            return TaskResult(session_id="fake-session", task_id="task-1", response="done")

    def fake_run_validation(policy, command: str, *, timeout_seconds: int) -> FakeValidation:
        raise RuntimeError("validation broke")

    monkeypatch.setattr(eval_harness, "CodingAgent", FakeAgent)
    monkeypatch.setattr(eval_harness, "run_validation", fake_run_validation)
    config = build_app_config(
        env_file=tmp_path / "missing.env",
        env={"AICODING_RUNTIME_DIR": str(tmp_path / "runtime")},
    )

    result = run_eval_suite(config, tmp_path, suite)

    assert result.task_results[0].status == "error"
    assert result.task_results[0].validations[0].status == "error"


def test_run_eval_suite_still_runs_hidden_validation_after_agent_error(
    monkeypatch, tmp_path: Path
) -> None:
    suite = _write_suite(
        tmp_path,
        [{"name": "demo", "task": "fix", "hidden_validation": "python -m pytest tests"}],
    )
    seen_validations: list[str] = []

    class FakeAgent:
        def __init__(self, **kwargs):
            pass

        def run_mode_task(self, mode: str, task: str) -> TaskResult:
            raise RuntimeError("agent broke after partial work")

    def fake_run_validation(policy, command: str, *, timeout_seconds: int) -> FakeValidation:
        seen_validations.append(command)
        return FakeValidation(command)

    monkeypatch.setattr(eval_harness, "CodingAgent", FakeAgent)
    monkeypatch.setattr(eval_harness, "run_validation", fake_run_validation)
    config = build_app_config(
        env_file=tmp_path / "missing.env",
        env={"AICODING_RUNTIME_DIR": str(tmp_path / "runtime")},
    )

    result = run_eval_suite(config, tmp_path, suite)

    assert result.task_results[0].status == "error"
    assert result.task_results[0].validations[0].status == "passed"
    assert seen_validations == ["python -m pytest tests"]


def test_run_eval_suite_flags_hidden_validation_command_leak(
    monkeypatch, tmp_path: Path
) -> None:
    suite = _write_suite(
        tmp_path,
        [{"name": "demo", "task": "fix", "hidden_validation": "python -m pytest hidden.py"}],
    )

    class FakeAgent:
        def __init__(self, **kwargs):
            pass

        def run_mode_task(self, mode: str, task: str) -> TaskResult:
            return TaskResult(
                session_id="fake-session",
                task_id="task-1",
                response="I ran python -m pytest hidden.py and it passed.",
            )

    def fake_run_validation(policy, command: str, *, timeout_seconds: int) -> FakeValidation:
        return FakeValidation(command)

    monkeypatch.setattr(eval_harness, "CodingAgent", FakeAgent)
    monkeypatch.setattr(eval_harness, "run_validation", fake_run_validation)
    config = build_app_config(
        env_file=tmp_path / "missing.env",
        env={"AICODING_RUNTIME_DIR": str(tmp_path / "runtime")},
    )

    result = run_eval_suite(config, tmp_path, suite)

    assert result.task_results[0].flags == ("hidden_leak",)


def test_run_eval_suite_flags_hidden_validation_path_leak(
    monkeypatch, tmp_path: Path
) -> None:
    hidden_path = "D:/hidden/test_secret.py"
    suite = _write_suite(
        tmp_path,
        [{"name": "demo", "task": "fix", "hidden_validation": f"python -m pytest {hidden_path}"}],
    )

    class FakeAgent:
        def __init__(self, **kwargs):
            pass

        def run_mode_task(self, mode: str, task: str) -> TaskResult:
            return TaskResult(
                session_id="fake-session",
                task_id="task-1",
                response=f"I inspected {hidden_path}.",
            )

    def fake_run_validation(policy, command: str, *, timeout_seconds: int) -> FakeValidation:
        return FakeValidation(command)

    monkeypatch.setattr(eval_harness, "CodingAgent", FakeAgent)
    monkeypatch.setattr(eval_harness, "run_validation", fake_run_validation)
    config = build_app_config(
        env_file=tmp_path / "missing.env",
        env={"AICODING_RUNTIME_DIR": str(tmp_path / "runtime")},
    )

    result = run_eval_suite(config, tmp_path, suite)

    assert result.task_results[0].flags == ("hidden_leak",)


def test_run_eval_suite_flags_hidden_validation_trace_leak(
    monkeypatch, tmp_path: Path
) -> None:
    suite = _write_suite(
        tmp_path,
        [{"name": "demo", "task": "fix", "hidden_validation": "python -m pytest hidden.py"}],
    )

    class FakeAgent:
        def __init__(self, **kwargs):
            self.runtime_dir = kwargs["config"].harness.runtime_dir
            self.session_id = kwargs["session_id"]

        def run_mode_task(self, mode: str, task: str) -> TaskResult:
            trace_dir = self.runtime_dir / "traces"
            trace_dir.mkdir(parents=True, exist_ok=True)
            (trace_dir / f"{self.session_id}.jsonl").write_text(
                '{"event_type":"tool_call","input_summary":"python -m pytest hidden.py"}\n',
                encoding="utf-8",
            )
            return TaskResult(session_id=self.session_id, task_id="task-1", response="Done.")

    def fake_run_validation(policy, command: str, *, timeout_seconds: int) -> FakeValidation:
        return FakeValidation(command)

    monkeypatch.setattr(eval_harness, "CodingAgent", FakeAgent)
    monkeypatch.setattr(eval_harness, "run_validation", fake_run_validation)
    config = build_app_config(
        env_file=tmp_path / "missing.env",
        env={"AICODING_RUNTIME_DIR": str(tmp_path / "runtime")},
    )

    result = run_eval_suite(config, tmp_path, suite)

    assert result.task_results[0].flags == ("hidden_leak",)
    assert result.task_results[0].status == "leak"
    assert result.has_failures


def test_run_eval_suite_leak_scan_uses_current_trace_segment(
    monkeypatch, tmp_path: Path
) -> None:
    suite = _write_suite(
        tmp_path,
        [
            {
                "name": "round one",
                "task": "fix",
                "group_id": "demo",
                "hidden_validation": "python -m pytest hidden.py",
            },
            {
                "name": "round two",
                "task": "continue",
                "group_id": "demo",
                "continue_session": True,
                "hidden_validation": "python -m pytest hidden.py",
            },
        ],
    )
    calls = 0

    class FakeAgent:
        def __init__(self, **kwargs):
            self.runtime_dir = kwargs["config"].harness.runtime_dir
            self.session_id = kwargs["session_id"]

        def run_mode_task(self, mode: str, task: str) -> TaskResult:
            nonlocal calls
            calls += 1
            trace_dir = self.runtime_dir / "traces"
            trace_dir.mkdir(parents=True, exist_ok=True)
            if calls == 1:
                with (trace_dir / f"{self.session_id}.jsonl").open(
                    "a",
                    encoding="utf-8",
                ) as trace:
                    trace.write(
                        '{"event_type":"tool_call","input_summary":"python -m pytest hidden.py"}\n'
                    )
            return TaskResult(session_id=self.session_id, task_id=f"task-{calls}", response="Done.")

    def fake_run_validation(policy, command: str, *, timeout_seconds: int) -> FakeValidation:
        return FakeValidation(command)

    monkeypatch.setattr(eval_harness, "CodingAgent", FakeAgent)
    monkeypatch.setattr(eval_harness, "run_validation", fake_run_validation)
    config = build_app_config(
        env_file=tmp_path / "missing.env",
        env={"AICODING_RUNTIME_DIR": str(tmp_path / "runtime")},
    )

    result = run_eval_suite(config, tmp_path, suite)

    assert result.task_results[0].status == "leak"
    assert result.task_results[1].status == "passed"
    assert result.task_results[1].flags == ()


def test_run_eval_suite_detects_hidden_path_with_mixed_separators(
    monkeypatch, tmp_path: Path
) -> None:
    suite = _write_suite(
        tmp_path,
        [
            {
                "name": "demo",
                "task": "fix",
                "hidden_validation": "python -m pytest hidden/tests/test_secret.py",
            }
        ],
    )

    class FakeAgent:
        def __init__(self, **kwargs):
            pass

        def run_mode_task(self, mode: str, task: str) -> TaskResult:
            return TaskResult(
                session_id="fake-session",
                task_id="task-1",
                response="I found hidden\\tests\\test_secret.py.",
            )

    def fake_run_validation(policy, command: str, *, timeout_seconds: int) -> FakeValidation:
        return FakeValidation(command)

    monkeypatch.setattr(eval_harness, "CodingAgent", FakeAgent)
    monkeypatch.setattr(eval_harness, "run_validation", fake_run_validation)
    config = build_app_config(
        env_file=tmp_path / "missing.env",
        env={"AICODING_RUNTIME_DIR": str(tmp_path / "runtime")},
    )

    result = run_eval_suite(config, tmp_path, suite)

    assert result.task_results[0].flags == ("hidden_leak",)


def test_run_eval_suite_avoids_generic_directory_leak_false_positive(
    monkeypatch, tmp_path: Path
) -> None:
    suite = _write_suite(
        tmp_path,
        [{"name": "demo", "task": "fix", "hidden_validation": "python -m pytest tests/"}],
    )

    class FakeAgent:
        def __init__(self, **kwargs):
            pass

        def run_mode_task(self, mode: str, task: str) -> TaskResult:
            return TaskResult(
                session_id="fake-session",
                task_id="task-1",
                response="I ran the visible tests/ directory.",
            )

    def fake_run_validation(policy, command: str, *, timeout_seconds: int) -> FakeValidation:
        return FakeValidation(command)

    monkeypatch.setattr(eval_harness, "CodingAgent", FakeAgent)
    monkeypatch.setattr(eval_harness, "run_validation", fake_run_validation)
    config = build_app_config(
        env_file=tmp_path / "missing.env",
        env={"AICODING_RUNTIME_DIR": str(tmp_path / "runtime")},
    )

    result = run_eval_suite(config, tmp_path, suite)

    assert result.task_results[0].flags == ()


def test_run_eval_suite_without_hidden_validation_is_unvalidated(
    monkeypatch, tmp_path: Path
) -> None:
    suite = _write_suite(tmp_path, [{"name": "demo", "task": "inspect"}])

    class FakeAgent:
        def __init__(self, **kwargs):
            pass

        def run_mode_task(self, mode: str, task: str) -> TaskResult:
            return TaskResult(session_id="fake-session", task_id="task-1", response="done")

    monkeypatch.setattr(eval_harness, "CodingAgent", FakeAgent)
    config = build_app_config(
        env_file=tmp_path / "missing.env",
        env={"AICODING_RUNTIME_DIR": str(tmp_path / "runtime")},
    )

    result = run_eval_suite(config, tmp_path, suite)

    assert result.task_results[0].status == "passed_unvalidated"
    assert not result.has_failures


def test_run_eval_suite_saves_response_and_validation_artifacts(
    monkeypatch, tmp_path: Path
) -> None:
    suite = _write_suite(
        tmp_path,
        [{"name": "demo", "task": "fix", "hidden_validation": "python -m pytest tests"}],
    )
    response = "full response text that should be persisted"

    class FakeAgent:
        def __init__(self, **kwargs):
            pass

        def run_mode_task(self, mode: str, task: str) -> TaskResult:
            return TaskResult(session_id="fake-session", task_id="task-1", response=response)

    def fake_run_validation(policy, command: str, *, timeout_seconds: int) -> FakeValidation:
        return FakeValidation(command)

    monkeypatch.setattr(eval_harness, "CodingAgent", FakeAgent)
    monkeypatch.setattr(eval_harness, "run_validation", fake_run_validation)
    config = build_app_config(
        env_file=tmp_path / "missing.env",
        env={"AICODING_RUNTIME_DIR": str(tmp_path / "runtime")},
    )

    result = run_eval_suite(config, tmp_path, suite)
    task = result.task_results[0]
    validation = task.validations[0]

    assert Path(task.response_path).read_text(encoding="utf-8") == response
    assert task.trace_path.endswith("eval-suite-001-demo.jsonl")
    assert Path(validation.output_path).read_text(encoding="utf-8") == (
        "Validation command: python -m pytest tests\nStatus: passed"
    )


def test_format_eval_result_includes_artifact_paths(monkeypatch, tmp_path: Path) -> None:
    suite = _write_suite(
        tmp_path,
        [{"name": "demo", "task": "fix", "hidden_validation": "python -m pytest tests"}],
    )

    class FakeAgent:
        def __init__(self, **kwargs):
            pass

        def run_mode_task(self, mode: str, task: str) -> TaskResult:
            return TaskResult(session_id="fake-session", task_id="task-1", response="done")

    def fake_run_validation(policy, command: str, *, timeout_seconds: int) -> FakeValidation:
        return FakeValidation(command)

    monkeypatch.setattr(eval_harness, "CodingAgent", FakeAgent)
    monkeypatch.setattr(eval_harness, "run_validation", fake_run_validation)
    config = build_app_config(
        env_file=tmp_path / "missing.env",
        env={"AICODING_RUNTIME_DIR": str(tmp_path / "runtime")},
    )

    formatted = format_eval_result(run_eval_suite(config, tmp_path, suite))

    assert "trace_path:" in formatted
    assert "response_path:" in formatted
    assert "output_path:" in formatted


def test_eval_cli_returns_zero_when_hidden_validation_passes(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    suite = _write_suite(
        tmp_path,
        [{"name": "demo", "task": "fix", "hidden_validation": "python -m pytest tests"}],
    )

    class FakeAgent:
        def __init__(self, **kwargs):
            pass

        def run_mode_task(self, mode: str, task: str) -> TaskResult:
            return TaskResult(session_id="fake-session", task_id="task-1", response="done")

    def fake_run_validation(policy, command: str, *, timeout_seconds: int) -> FakeValidation:
        return FakeValidation(command)

    monkeypatch.setattr(eval_harness, "CodingAgent", FakeAgent)
    monkeypatch.setattr(eval_harness, "run_validation", fake_run_validation)

    code = main(
        [
            "--env-file",
            str(tmp_path / "missing.env"),
            "dev",
            "eval",
            "run",
            "--workspace",
            str(tmp_path),
            "--suite",
            str(suite),
        ]
    )

    assert code == 0
    assert "passed: 1" in capsys.readouterr().out


def test_eval_cli_returns_one_when_hidden_validation_fails(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    suite = _write_suite(
        tmp_path,
        [{"name": "demo", "task": "fix", "hidden_validation": "python -m pytest tests"}],
    )

    class FakeAgent:
        def __init__(self, **kwargs):
            pass

        def run_mode_task(self, mode: str, task: str) -> TaskResult:
            return TaskResult(session_id="fake-session", task_id="task-1", response="done")

    def fake_run_validation(policy, command: str, *, timeout_seconds: int) -> FakeValidation:
        return FakeValidation(command, ok=False)

    monkeypatch.setattr(eval_harness, "CodingAgent", FakeAgent)
    monkeypatch.setattr(eval_harness, "run_validation", fake_run_validation)

    code = main(
        [
            "--env-file",
            str(tmp_path / "missing.env"),
            "dev",
            "eval",
            "run",
            "--workspace",
            str(tmp_path),
            "--suite",
            str(suite),
        ]
    )

    assert code == 1
    assert "failed: 1" in capsys.readouterr().out
