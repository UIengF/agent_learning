from __future__ import annotations

import json
from pathlib import Path

from aicoding_app.cli import main
from aicoding_app.config import build_app_config


def test_config_inspect_masks_secret(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setenv("AICODING_MODEL_API_KEY", "secret-token-value")
    monkeypatch.setenv("AICODING_RUNTIME_DIR", str(tmp_path / "runtime"))

    assert main(["config", "inspect"]) == 0

    output = capsys.readouterr().out
    assert "secret-token-value" not in output
    assert "secr...alue" in output


def test_model_config_accepts_deepseek_aliases(tmp_path: Path) -> None:
    config = build_app_config(
        env_file=tmp_path / "missing.env",
        env={
            "DEEPSEEK_API_KEY": "<sk-test1234>",
            "DEEPSEEK_BASE_URL": "<https://api.deepseek.com>",
            "DEEPSEEK_MODEL": "<deepseek-chat>",
            "AICODING_RUNTIME_DIR": str(tmp_path / "runtime"),
        }
    )

    assert config.model.configured is True
    assert config.model.api_key == "sk-test1234"
    assert config.model.api_base == "https://api.deepseek.com"
    assert config.model.model_name == "deepseek-chat"


def test_run_resume_and_trace_for_simple_replace(monkeypatch, tmp_path: Path, capsys) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / "app.py").write_text("old\n", encoding="utf-8")
    runtime_dir = tmp_path / "runtime"
    monkeypatch.delenv("AICODING_MODEL_API_KEY", raising=False)
    monkeypatch.setenv("AICODING_RUNTIME_DIR", str(runtime_dir))
    monkeypatch.setenv("AICODING_ALLOWED_COMMANDS", "git status;git diff")

    assert (
        main(
            [
                "--env-file",
                str(tmp_path / "missing.env"),
                "run",
                "--workspace",
                str(workspace),
                "--session-id",
                "demo",
                "--task",
                "replace 'old' with 'new' in app.py",
            ]
        )
        == 0
    )
    run_output = capsys.readouterr().out

    assert "session_id: demo" in run_output
    assert "changed files: app.py" in run_output
    assert (workspace / "app.py").read_text(encoding="utf-8") == "new\n"

    assert main(["resume", "--session-id", "demo"]) == 0
    resume_payload = json.loads(capsys.readouterr().out)
    assert resume_payload["session_id"] == "demo"
    assert resume_payload["history"]

    assert main(["trace", "show", "--session-id", "demo"]) == 0
    trace_output = capsys.readouterr().out
    assert "task_started" in trace_output
    assert "apply_patch" in trace_output


def test_edit_patch_failure_reports_status_without_modifying_file(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / "app.py").write_text("stable\n", encoding="utf-8")
    runtime_dir = tmp_path / "runtime"
    monkeypatch.delenv("AICODING_MODEL_API_KEY", raising=False)
    monkeypatch.setenv("AICODING_RUNTIME_DIR", str(runtime_dir))
    monkeypatch.setenv("AICODING_ALLOWED_COMMANDS", "git status;git diff")

    assert (
        main(
            [
                "--env-file",
                str(tmp_path / "missing.env"),
                "edit",
                "--workspace",
                str(workspace),
                "--session-id",
                "mismatch",
                "--task",
                'replace "missing" with "new" in app.py',
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "patch_failed: patch hunk did not match file" in output
    assert "- git status --short: $ git status --short -- ." in output
    assert (workspace / "app.py").read_text(encoding="utf-8") == "stable\n"
