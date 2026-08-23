from __future__ import annotations

import shutil
import subprocess
import time
from pathlib import Path

import pytest

from atomic_agents.drift import detect_drift, snapshot


def test_mtime_snapshot_detects_only_changes_outside_write_scope(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "src").mkdir()
    (workspace / "docs").mkdir()
    _write(workspace / "src" / "allowed.py", "before")
    _write(workspace / "docs" / "outside.md", "before")

    pre = snapshot(str(workspace))
    _advance_mtime_granularity()
    _write(workspace / "src" / "allowed.py", "after")
    _write(workspace / "docs" / "outside.md", "after")
    post = snapshot(str(workspace))

    assert detect_drift(str(workspace), pre, post, ["src/allowed.py"]) == ["docs/outside.md"]


def test_directory_write_scope_with_trailing_slash_covers_children(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "src").mkdir()
    _write(workspace / "src" / "a.py", "before")
    _write(workspace / "config.yaml", "before")

    pre = snapshot(str(workspace))
    _advance_mtime_granularity()
    _write(workspace / "src" / "a.py", "after")
    _write(workspace / "config.yaml", "after")
    post = snapshot(str(workspace))

    assert detect_drift(str(workspace), pre, post, ["src/"]) == ["config.yaml"]


def test_exact_file_write_scope_does_not_cover_sibling_files(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write(workspace / "a.py", "before")
    _write(workspace / "b.py", "before")

    pre = snapshot(str(workspace))
    _advance_mtime_granularity()
    _write(workspace / "a.py", "after")
    _write(workspace / "b.py", "after")
    post = snapshot(str(workspace))

    assert detect_drift(str(workspace), pre, post, ["a.py"]) == ["b.py"]


def test_no_changes_returns_no_drift(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write(workspace / "a.py", "same")

    pre = snapshot(str(workspace))
    post = snapshot(str(workspace))

    assert detect_drift(str(workspace), pre, post, ["a.py"]) == []


def test_bare_existing_directory_write_scope_covers_children(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "utils").mkdir()
    _write(workspace / "utils" / "log.py", "before")
    _write(workspace / "stray.py", "before")

    pre = snapshot(str(workspace))
    _advance_mtime_granularity()
    _write(workspace / "utils" / "log.py", "after")
    _write(workspace / "stray.py", "after")
    post = snapshot(str(workspace))

    assert detect_drift(str(workspace), pre, post, ["utils"]) == ["stray.py"]


def test_bare_non_directory_write_scope_is_exact_file(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    _write(workspace / "a.py", "before")
    _write(workspace / "b.py", "before")

    pre = snapshot(str(workspace))
    _advance_mtime_granularity()
    _write(workspace / "a.py", "after")
    _write(workspace / "b.py", "after")
    post = snapshot(str(workspace))

    assert detect_drift(str(workspace), pre, post, ["a.py"]) == ["b.py"]


@pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")
def test_git_snapshot_uses_porcelain_changed_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    _run_git(workspace, "init")
    _run_git(workspace, "config", "user.email", "test@example.com")
    _run_git(workspace, "config", "user.name", "Test User")
    _write(workspace / "allowed.py", "before")
    _write(workspace / "outside.py", "before")
    _run_git(workspace, "add", ".")
    _run_git(workspace, "commit", "-m", "initial")

    pre = snapshot(str(workspace))
    _write(workspace / "allowed.py", "after")
    _write(workspace / "outside.py", "after")
    post = snapshot(str(workspace))

    assert detect_drift(str(workspace), pre, post, ["allowed.py"]) == ["outside.py"]


def _write(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def _advance_mtime_granularity() -> None:
    time.sleep(0.01)


def _run_git(workspace: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=workspace,
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )
