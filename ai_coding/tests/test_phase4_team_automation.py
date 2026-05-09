from __future__ import annotations

import json
from pathlib import Path

from aicoding_app.cli import main
from aicoding_app.memory import MemoryStore, SensitiveMemoryError


def test_memory_add_inspect_and_forget(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setenv("AICODING_RUNTIME_DIR", str(tmp_path / "runtime"))

    assert (
        main(
            [
                "memory",
                "add",
                "--kind",
                "command",
                "--text",
                "Run python -m pytest tests before PR",
            ]
        )
        == 0
    )
    added = capsys.readouterr().out.strip()
    memory_id = added.split(": ", 1)[1]

    assert main(["memory", "inspect"]) == 0
    inspected = capsys.readouterr().out
    assert memory_id in inspected
    assert "Run python -m pytest tests before PR" in inspected

    assert main(["memory", "forget", "--id", memory_id]) == 0
    assert "memory_forgotten" in capsys.readouterr().out
    assert main(["memory", "inspect"]) == 0
    assert "Memory: none" in capsys.readouterr().out


def test_memory_rejects_likely_secret(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "runtime")

    try:
        store.add("note", "api_key=sk-secret")
    except SensitiveMemoryError:
        pass
    else:  # pragma: no cover
        raise AssertionError("expected sensitive memory rejection")


def test_ask_and_plan_output_include_memory(monkeypatch, tmp_path: Path, capsys) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / "app.py").write_text("value = 1\n", encoding="utf-8")
    runtime_dir = tmp_path / "runtime"
    monkeypatch.setenv("AICODING_RUNTIME_DIR", str(runtime_dir))
    MemoryStore(runtime_dir).add("command", "Run python -m pytest tests before review")

    assert (
        main(
            [
                "--env-file",
                str(tmp_path / "missing.env"),
                "ask",
                "--workspace",
                str(workspace),
                "--task",
                "explain app.py",
            ]
        )
        == 0
    )
    ask_output = capsys.readouterr().out
    assert "Memory summary:" in ask_output
    assert "Run python -m pytest tests before review" in ask_output

    assert (
        main(
            [
                "--env-file",
                str(tmp_path / "missing.env"),
                "plan",
                "--workspace",
                str(workspace),
                "--task",
                "change app.py",
            ]
        )
        == 0
    )
    plan_output = capsys.readouterr().out
    assert "Memory summary:" in plan_output
    assert "Run python -m pytest tests before review" in plan_output


def test_schedule_plan_is_dry_run(tmp_path: Path, capsys) -> None:
    assert (
        main(
            [
                "schedule",
                "plan",
                "--workspace",
                str(tmp_path),
                "--task",
                "run pytest weekly",
                "--cadence",
                "weekly",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "Schedule plan:" in output
    assert "weekly" in output
    assert "no scheduler was created" in output


def test_eval_run_reads_fixture_and_skips_execution(tmp_path: Path, capsys) -> None:
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "tasks": [
                    {
                        "name": "demo",
                        "task": "inspect repo",
                        "suggested_mode": "ask",
                        "expected_validation": "python -m pytest tests",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )

    assert main(["eval", "run", "--workspace", str(tmp_path), "--suite", str(suite)]) == 0

    output = capsys.readouterr().out
    assert "Eval dry-run summary:" in output
    assert "task count: 1" in output
    assert "execution: skipped" in output
    assert "demo" in output


def test_connectors_list_outputs_disabled_placeholders(capsys) -> None:
    assert main(["connectors", "list"]) == 0

    output = capsys.readouterr().out
    assert "github: disabled/read-only placeholder" in output
    assert "mcp: disabled/read-only placeholder" in output
