from __future__ import annotations

from pathlib import Path

import aicoding_app.agent as agent_module
from aicoding_app.agent import CodingAgent
from aicoding_app.config import build_app_config


def test_agent_fallback_create_and_session_state(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    config = build_app_config(
        env_file=tmp_path / "missing.env",
        env={
            "AICODING_RUNTIME_DIR": str(tmp_path / "runtime"),
            "AICODING_ALLOWED_COMMANDS": "git status;git diff",
            "AICODING_TRACE_ENABLED": "true",
        }
    )
    agent = CodingAgent(config=config, workspace=workspace, session_id="create-demo")

    result = agent.run_task('create hello.txt with "hello world"')

    assert "changed files: hello.txt" in result.response
    assert (workspace / "hello.txt").read_text(encoding="utf-8") == "hello world\n"
    assert (tmp_path / "runtime" / "sessions" / "create-demo.json").exists()
    assert (tmp_path / "runtime" / "traces" / "create-demo.jsonl").exists()


def test_edit_uses_direct_patch_when_langgraph_unavailable(
    monkeypatch, tmp_path: Path
) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    config = build_app_config(
        env_file=tmp_path / "missing.env",
        env={
            "AICODING_MODEL_API_KEY": "sk-test",
            "AICODING_MODEL_API_BASE": "https://example.invalid",
            "AICODING_MODEL_NAME": "test-model",
            "AICODING_RUNTIME_DIR": str(tmp_path / "runtime"),
            "AICODING_ALLOWED_COMMANDS": "git diff",
            "AICODING_TRACE_ENABLED": "true",
        },
    )
    monkeypatch.setattr(agent_module, "LANGGRAPH_AVAILABLE", False)
    monkeypatch.setattr(agent_module, "LANGGRAPH_IMPORT_ERROR", "No module named 'langchain_core'")
    monkeypatch.setattr(
        CodingAgent,
        "_call_direct_patch_model",
        lambda self, task, task_id, context: (
            "*** Begin Patch\n"
            "*** Add File: solution.py\n"
            "+def answer() -> int:\n"
            "+    return 42\n"
            "*** End Patch"
        ),
    )
    agent = CodingAgent(config=config, workspace=workspace, session_id="direct-edit")

    result = agent.run_mode_task("edit", "add a solution file")

    assert "direct patch edit was used" in result.response
    assert "changed files: solution.py" in result.response
    assert "return 42" in (workspace / "solution.py").read_text(encoding="utf-8")


def test_agent_falls_back_when_direct_patch_preview_fails(
    monkeypatch, tmp_path: Path
) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / "app.py").write_text("def value():\n    return 1\n", encoding="utf-8")
    config = build_app_config(
        env_file=tmp_path / "missing.env",
        env={
            "AICODING_MODEL_API_KEY": "sk-test",
            "AICODING_MODEL_API_BASE": "https://example.invalid",
            "AICODING_MODEL_NAME": "test-model",
            "AICODING_RUNTIME_DIR": str(tmp_path / "runtime"),
            "AICODING_ALLOWED_COMMANDS": "git status;git diff",
            "AICODING_TRACE_ENABLED": "true",
        },
    )
    monkeypatch.setattr(agent_module, "LANGGRAPH_AVAILABLE", False)
    monkeypatch.setattr(agent_module, "LANGGRAPH_IMPORT_ERROR", "No module named 'langchain_core'")
    monkeypatch.setattr(
        CodingAgent,
        "_call_direct_patch_model",
        lambda self, task, task_id, context: (
            "*** Begin Patch\n"
            "*** Read File: app.py\n"
            "def value():\n"
            "    return 2\n"
            "*** End Patch"
        ),
    )
    agent = CodingAgent(config=config, workspace=workspace, session_id="direct-preview-fail")

    result = agent.run_mode_task("agent", 'replace "return 1" with "return 2" in app.py')

    assert "deterministic fallback was used" in result.response
    assert "changed files: app.py" in result.response
    assert (workspace / "app.py").read_text(encoding="utf-8") == "def value():\n    return 2\n"


def test_extract_patch_block_normalizes_single_line_add_file() -> None:
    content = (
        "*** Begin Patch Add File: solution.py ```python\n"
        "def answer() -> int:\n"
        "    return 42\n"
        "``` *** End Patch"
    )

    patch = CodingAgent._extract_patch_block(content)

    assert patch == (
        "*** Begin Patch\n"
        "*** Add File: solution.py\n"
        "def answer() -> int:\n"
        "    return 42\n"
        "*** End Patch"
    )
