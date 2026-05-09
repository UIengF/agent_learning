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


def test_command_policy_allows_workspace_python_scripts(tmp_path: Path) -> None:
    (tmp_path / "task_manager.py").write_text("print('ok')\n", encoding="utf-8")
    policy = WorkspacePolicy(workspace=tmp_path, allowed_commands=("git status",))

    assert policy.validate_command("python task_manager.py") == "python task_manager.py"
    assert (
        policy.validate_command('python task_manager.py add "Buy groceries"')
        == 'python task_manager.py add "Buy groceries"'
    )


def test_command_policy_blocks_unsafe_python_invocations(tmp_path: Path) -> None:
    (tmp_path / "task_manager.py").write_text("print('ok')\n", encoding="utf-8")
    (tmp_path / "setup.py").write_text("print('setup')\n", encoding="utf-8")
    policy = WorkspacePolicy(workspace=tmp_path, allowed_commands=("git status",))

    with pytest.raises(CommandDenied):
        policy.validate_command('python -c "print(1)"')

    with pytest.raises(CommandDenied):
        policy.validate_command("python -m pip install pygments")

    with pytest.raises(CommandDenied):
        policy.validate_command("python ../outside.py")

    with pytest.raises(CommandDenied):
        policy.validate_command("python missing.py")

    with pytest.raises(CommandDenied):
        policy.validate_command("python setup.py install")
