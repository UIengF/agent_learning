from __future__ import annotations

from pathlib import Path

import pytest

from aicoding_app.permissions import CommandDenied, PermissionDenied, WorkspacePolicy


def test_workspace_policy_blocks_path_escape(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    policy = WorkspacePolicy(workspace=workspace, allowed_commands=("git status",))

    assert policy.resolve_path("src/app.py") == workspace / "src" / "app.py"

    with pytest.raises(PermissionDenied):
        policy.resolve_path("../outside.py")


def test_command_policy_allows_configured_prefix_and_blocks_destructive(tmp_path: Path) -> None:
    policy = WorkspacePolicy(
        workspace=tmp_path,
        allowed_commands=("python -m pytest", "git status", "git diff"),
    )

    assert policy.validate_command("python -m pytest tests") == "python -m pytest tests"

    with pytest.raises(CommandDenied):
        policy.validate_command("git reset --hard")

    with pytest.raises(CommandDenied):
        policy.validate_command("python -m pytest tests && del important.txt")

    with pytest.raises(CommandDenied):
        policy.validate_command("python setup.py install")


def test_command_policy_ignores_control_operators_inside_quotes(tmp_path: Path) -> None:
    policy = WorkspacePolicy(workspace=tmp_path, allowed_commands=("python -c",))

    assert (
        policy.validate_command('python -c "print(1); print(2)"')
        == 'python -c "print(1); print(2)"'
    )

    with pytest.raises(CommandDenied):
        policy.validate_command('python -c "print(1)" ; git status')

    with pytest.raises(CommandDenied):
        policy.validate_command('python -c "print(1)" | more')
