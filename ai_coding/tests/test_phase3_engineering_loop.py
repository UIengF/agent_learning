from __future__ import annotations

from pathlib import Path
import subprocess

from aicoding_app.cli import main
from aicoding_app.hooks import preview_hook
from aicoding_app.permissions import WorkspacePolicy
from aicoding_app.pr_summary import build_pr_summary


def test_verify_cli_runs_allowed_pytest_command(monkeypatch, tmp_path: Path, capsys) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_ok.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    monkeypatch.setenv("AICODING_RUNTIME_DIR", str(tmp_path / "runtime"))

    assert main(["verify", "--workspace", str(tmp_path), "--command", "python -m pytest tests"]) == 0

    output = capsys.readouterr().out
    assert "Validation command: python -m pytest tests" in output
    assert "Status: passed (0)" in output


def test_verify_cli_includes_pytest_failure_context(monkeypatch, tmp_path: Path, capsys) -> None:
    (tmp_path / "app.py").write_text("def answer():\n    return 1\n", encoding="utf-8")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_app.py").write_text(
        "from app import answer\n\n\ndef test_answer():\n    assert answer() == 2\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AICODING_RUNTIME_DIR", str(tmp_path / "runtime"))

    assert main(["verify", "--workspace", str(tmp_path), "--command", "python -m pytest tests"]) == 0

    output = capsys.readouterr().out
    assert "Status: failed" in output
    assert "Failure context:" in output
    assert "Pytest failure context:" in output
    assert "tests/test_app.py" in output


def test_git_summary_cli_outputs_branch_changed_files_and_commit_preview(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    subprocess.run(["git", "init"], cwd=tmp_path, check=True, capture_output=True, text=True)
    (tmp_path / "app.py").write_text("old\n", encoding="utf-8")
    subprocess.run(["git", "add", "app.py"], cwd=tmp_path, check=True, capture_output=True, text=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.email=test@example.com",
            "-c",
            "user.name=Test",
            "commit",
            "-m",
            "initial",
        ],
        cwd=tmp_path,
        check=True,
        capture_output=True,
        text=True,
    )
    (tmp_path / "app.py").write_text("new\n", encoding="utf-8")
    monkeypatch.setenv("AICODING_RUNTIME_DIR", str(tmp_path / "runtime"))

    assert main(["git", "summary", "--workspace", str(tmp_path)]) == 0

    output = capsys.readouterr().out
    assert "Git summary:" in output
    assert "Branch:" in output
    assert "app.py" in output
    assert "Commit preview:" in output


def test_pr_summary_contains_change_validation_and_risks() -> None:
    summary = build_pr_summary(
        change_summary="changed parser",
        changed_files=["aicoding_app/parser.py"],
        validation_result="python -m pytest tests passed",
        risks=["manual review required"],
    )

    assert "PR-ready summary:" in summary
    assert "changed parser" in summary
    assert "aicoding_app/parser.py" in summary
    assert "python -m pytest tests passed" in summary
    assert "manual review required" in summary


def test_after_verify_hook_preview_does_not_execute_denied_command(tmp_path: Path) -> None:
    (tmp_path / ".aicoding").mkdir()
    (tmp_path / ".aicoding" / "config.toml").write_text(
        "[hooks]\nafter_verify = ['python -m pytest tests', 'python scripts/deploy.py']\n",
        encoding="utf-8",
    )
    policy = WorkspacePolicy(workspace=tmp_path, allowed_commands=("python -m pytest",))

    preview = preview_hook(policy, "after_verify")
    text = preview.format()

    assert "preview only" in text
    assert "python -m pytest tests" in text
    assert "python scripts/deploy.py" in text
    assert preview.allowed == ("python -m pytest tests",)
    assert len(preview.denied) == 1
