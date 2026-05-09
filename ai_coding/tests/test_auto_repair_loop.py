from __future__ import annotations

from pathlib import Path

from aicoding_app.agent import CodingAgent
from aicoding_app.config import build_app_config
from aicoding_app.evidence_cache import EvidenceCache
from aicoding_app.permissions import WorkspacePolicy
from aicoding_app.plan import CodingPlan
from aicoding_app.skills import SkillRegistry
from aicoding_app.tools import CodingTools
from aicoding_app.trace import StructuredTraceWriter


def _write_simple_failing_repo(root: Path) -> None:
    (root / "app.py").write_text("def answer():\n    return 1\n", encoding="utf-8")
    (root / "tests").mkdir()
    (root / "tests" / "test_app.py").write_text(
        "from app import answer\n\n\ndef test_answer():\n    assert answer() == 2\n",
        encoding="utf-8",
    )


def test_auto_repair_loop_fixes_simple_pytest_return_value(tmp_path: Path) -> None:
    _write_simple_failing_repo(tmp_path)
    tools = CodingTools(
        policy=WorkspacePolicy(workspace=tmp_path, allowed_commands=("python -m pytest",)),
        plan=CodingPlan(goal="fix tests", edit_steps=["apply conservative pytest repair"]),
        evidence_cache=EvidenceCache(),
        skill_registry=SkillRegistry(tmp_path / "missing-skills"),
        trace_writer=StructuredTraceWriter(tmp_path / "runtime" / "traces", "repair"),
        command_timeout_seconds=30,
    )

    result = tools.auto_repair_loop("python -m pytest tests", max_attempts=2)

    assert "Repair candidate:" in result
    assert "changed files: app.py" in result
    assert "Auto repair result: validation passed." in result
    assert "return 2" in (tmp_path / "app.py").read_text(encoding="utf-8")


def test_agent_mode_runs_auto_repair_loop_for_test_tasks(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    _write_simple_failing_repo(workspace)
    config = build_app_config(
        env_file=tmp_path / "missing.env",
        env={
            "AICODING_RUNTIME_DIR": str(tmp_path / "runtime"),
            "AICODING_ALLOWED_COMMANDS": "python -m pytest;git status;git diff",
            "AICODING_TRACE_ENABLED": "true",
        }
    )
    agent = CodingAgent(config=config, workspace=workspace, session_id="repair-agent")

    result = agent.run_mode_task("agent", "fix failing tests")

    assert "Auto repair loop: python -m pytest tests" in result.response
    assert "Auto repair result: validation passed." in result.response
    assert "return 2" in (workspace / "app.py").read_text(encoding="utf-8")
