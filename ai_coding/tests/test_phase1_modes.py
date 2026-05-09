from __future__ import annotations

from pathlib import Path

from aicoding_app.cli import main
from aicoding_app.evidence_cache import EvidenceCache
from aicoding_app.permissions import WorkspacePolicy
from aicoding_app.plan import CodingPlan
from aicoding_app.project_instructions import load_project_instructions
from aicoding_app.repo_map import build_repo_map
from aicoding_app.skills import SkillRegistry
from aicoding_app.tools import CodingTools
from aicoding_app.trace import StructuredTraceWriter


def test_project_instructions_loads_agents_files(tmp_path: Path) -> None:
    (tmp_path / "AGENTS.md").write_text("Use pytest.", encoding="utf-8")
    (tmp_path / ".aicoding").mkdir()
    (tmp_path / ".aicoding" / "AGENTS.md").write_text("Prefer small patches.", encoding="utf-8")

    instructions = load_project_instructions(tmp_path)

    assert instructions.files == ("AGENTS.md", ".aicoding/AGENTS.md")
    assert "Use pytest." in instructions.content
    assert "Prefer small patches." in instructions.content


def test_repo_map_reports_python_symbols_tests_and_dependencies(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]\nname='demo'\n", encoding="utf-8")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text(
        "import os\n\nclass App:\n    def run(self):\n        return os.name\n",
        encoding="utf-8",
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_app.py").write_text("def test_app():\n    assert True\n", encoding="utf-8")

    repo_map = build_repo_map(tmp_path)
    formatted = repo_map.format()

    assert "pyproject.toml" in repo_map.dependency_files
    assert "tests" in repo_map.test_roots
    assert "class App" in formatted
    assert "function run" in formatted
    assert "src/app.py: os" in formatted


def test_preview_patch_does_not_modify_file(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("old\n", encoding="utf-8")
    tools = CodingTools(
        policy=WorkspacePolicy(workspace=tmp_path, allowed_commands=("git diff",)),
        plan=CodingPlan(goal="preview", edit_steps=["preview only"]),
        evidence_cache=EvidenceCache(),
        skill_registry=SkillRegistry(tmp_path / "missing-skills"),
        trace_writer=StructuredTraceWriter(tmp_path / "runtime" / "traces", "preview"),
    )

    result = tools.preview_patch(
        "*** Begin Patch\n*** Update File: app.py\n@@\n-old\n+new\n*** End Patch"
    )

    assert result == "preview changed files: app.py"
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "old\n"


def test_ask_mode_is_read_only_and_does_not_run_commands(monkeypatch, tmp_path: Path, capsys) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / "app.py").write_text("value = 1\n", encoding="utf-8")
    runtime_dir = tmp_path / "runtime"
    monkeypatch.delenv("AICODING_MODEL_API_KEY", raising=False)
    monkeypatch.setenv("AICODING_RUNTIME_DIR", str(runtime_dir))
    monkeypatch.setenv("AICODING_ALLOWED_COMMANDS", "git status;git diff")

    assert (
        main(
            [
                "--env-file",
                str(tmp_path / "missing.env"),
                "ask",
                "--workspace",
                str(workspace),
                "--session-id",
                "ask-demo",
                "--task",
                "explain this repository",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    trace = (runtime_dir / "traces" / "ask-demo.jsonl").read_text(encoding="utf-8")
    assert "Ask mode: read-only analysis" in output
    assert "No files were modified and no commands were run." in output
    assert "run_command" not in trace
    assert (workspace / "app.py").read_text(encoding="utf-8") == "value = 1\n"


def test_ask_mode_includes_overview_files_for_project_intro(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / "README.md").write_text(
        "# Demo Project\n\nThis project builds a local RAG service.\n",
        encoding="utf-8",
    )
    (workspace / "pyproject.toml").write_text(
        "[project]\nname = 'demo-rag'\n",
        encoding="utf-8",
    )
    (workspace / "app.py").write_text("def main():\n    return 1\n", encoding="utf-8")
    runtime_dir = tmp_path / "runtime"
    monkeypatch.delenv("AICODING_MODEL_API_KEY", raising=False)
    monkeypatch.setenv("AICODING_RUNTIME_DIR", str(runtime_dir))

    assert (
        main(
            [
                "--env-file",
                str(tmp_path / "missing.env"),
                "ask",
                "--workspace",
                str(workspace),
                "--session-id",
                "ask-overview",
                "--task",
                "介绍这个项目的结构和主要能力",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    trace = (runtime_dir / "traces" / "ask-overview.jsonl").read_text(encoding="utf-8")
    assert "Project overview summary:" in output
    assert "- Project: Demo Project" in output
    assert "Entry points: app.py" in output
    assert "Project overview context:" in output
    assert "Demo Project" in output
    assert "demo-rag" in output
    assert "No files were modified and no commands were run." in output
    assert "run_command" not in trace


def test_plan_mode_records_plan_without_modifying_files(monkeypatch, tmp_path: Path, capsys) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / "app.py").write_text("old\n", encoding="utf-8")
    runtime_dir = tmp_path / "runtime"
    monkeypatch.delenv("AICODING_MODEL_API_KEY", raising=False)
    monkeypatch.setenv("AICODING_RUNTIME_DIR", str(runtime_dir))

    assert (
        main(
            [
                "--env-file",
                str(tmp_path / "missing.env"),
                "plan",
                "--workspace",
                str(workspace),
                "--session-id",
                "plan-demo",
                "--task",
                "replace old with new",
            ]
        )
        == 0
    )

    output = capsys.readouterr().out
    assert "Plan mode: no repository files modified" in output
    assert "Goal: replace old with new" in output
    assert (workspace / "app.py").read_text(encoding="utf-8") == "old\n"


def test_repo_map_cli_prints_repository_summary(tmp_path: Path, capsys) -> None:
    (tmp_path / "app.py").write_text("def main():\n    return 1\n", encoding="utf-8")

    assert main(["repo", "map", "--workspace", str(tmp_path)]) == 0

    output = capsys.readouterr().out
    assert "Files:" in output
    assert "app.py" in output
    assert "function main" in output
