from __future__ import annotations

import json
from pathlib import Path

import aicoding_app.agent as agent_module
from aicoding_app.agent import CodingAgent, HarnessStop, PROMPT
from aicoding_app.config import build_app_config
from aicoding_app.harness_state import ToolRunState


def test_prompt_guides_policy_denial_recovery() -> None:
    assert "Commands already run with the workspace as current directory" in PROMPT
    assert "Do not prefix" in PROMPT
    assert "cd <path>" in PROMPT
    assert "cd /workspace" not in PROMPT
    assert "simplify" in PROMPT


def test_prompt_guides_documentation_target_selection() -> None:
    assert "target documentation" in PROMPT
    assert "verify it covers" in PROMPT
    assert "same topic" in PROMPT
    assert "dedicated nearby document" in PROMPT


def test_documentation_context_lists_readme_topics(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / "README.md").write_text("# Note Indexer CLI\n\nExisting docs.\n", encoding="utf-8")
    (workspace / "docs").mkdir()
    (workspace / "docs" / "README.md").write_text("# API Docs\n", encoding="utf-8")
    config = build_app_config(
        env_file=tmp_path / "missing.env",
        env={"AICODING_RUNTIME_DIR": str(tmp_path / "runtime")},
    )
    agent = CodingAgent(config=config, workspace=workspace, session_id="docs-context")

    context = agent._documentation_context()

    assert "- README.md: Note Indexer CLI" in context
    assert "- docs/README.md: API Docs" in context


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


def test_completion_check_exits_before_fallback_when_smoke_passes(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / "test_ready.py").write_text("print('ready')\n", encoding="utf-8")
    (workspace / "README.md").write_text("# Notes\n", encoding="utf-8")
    config = build_app_config(
        env_file=tmp_path / "missing.env",
        env={
            "AICODING_RUNTIME_DIR": str(tmp_path / "runtime"),
            "AICODING_ALLOWED_COMMANDS": "python test_ready.py;git status;git diff",
        },
    )
    agent = CodingAgent(config=config, workspace=workspace, session_id="completion")

    result = agent.run_mode_task(
        "agent",
        "\n".join(
            [
                "Ensure test_ready.py and README.md are ready.",
                "- python test_ready.py",
            ]
        ),
    )

    assert "Task status:" in result.response
    assert "completed" in result.response
    assert "Smoke command passed: python test_ready.py" in result.response
    assert "Deterministic fallback was used" not in result.response


def test_completion_check_does_not_skip_project_specific_mutating_command(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / "note_indexer.py").write_text(
        "import sys\n"
        "if len(sys.argv) > 1 and sys.argv[1] == 'add':\n"
        "    raise SystemExit(99)\n"
        "print('ok')\n",
        encoding="utf-8",
    )
    (workspace / "test_note_indexer.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    (workspace / "README.md").write_text("# Notes\n", encoding="utf-8")
    notes = [{"id": 1, "title": "First Note", "content": "Agent testing note", "tags": ["ai"]}]
    (workspace / "notes.json").write_text(json.dumps(notes), encoding="utf-8")
    config = build_app_config(
        env_file=tmp_path / "missing.env",
        env={
            "AICODING_RUNTIME_DIR": str(tmp_path / "runtime"),
            "AICODING_ALLOWED_COMMANDS": "python note_indexer.py;git status;git diff",
        },
    )
    agent = CodingAgent(config=config, workspace=workspace, session_id="completion-idempotent")

    completion = agent._check_task_completion(
        "\n".join(
            [
                "Ensure note_indexer.py, test_note_indexer.py, README.md, and notes.json are ready.",
                '- python note_indexer.py add "First Note" --tags work,ai --content "Agent testing note"',
                "- python note_indexer.py list",
            ]
        ),
        "task-id",
    )

    saved = json.loads((workspace / "notes.json").read_text(encoding="utf-8"))
    assert saved == notes
    assert completion.status == "unknown"
    assert completion.skipped_commands == [
        'Skipped unsafe or mutating smoke command during completion check: python note_indexer.py add "First Note" --tags work,ai --content "Agent testing note"'
    ]


def test_completion_check_treats_project_specific_command_with_equals_content_as_unsafe(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / "note_indexer.py").write_text(
        "import sys\n"
        "if len(sys.argv) > 1 and sys.argv[1] == 'add':\n"
        "    raise SystemExit(99)\n"
        "print('ok')\n",
        encoding="utf-8",
    )
    (workspace / "test_note_indexer.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    (workspace / "README.md").write_text("# Notes\n", encoding="utf-8")
    notes = [{"title": "First Note", "content": "Agent testing note"}]
    (workspace / "notes.json").write_text(json.dumps(notes), encoding="utf-8")
    config = build_app_config(
        env_file=tmp_path / "missing.env",
        env={
            "AICODING_RUNTIME_DIR": str(tmp_path / "runtime"),
            "AICODING_ALLOWED_COMMANDS": "python note_indexer.py;git status;git diff",
        },
    )
    agent = CodingAgent(config=config, workspace=workspace, session_id="completion-idempotent-equals")

    completion = agent._check_task_completion(
        "\n".join(
            [
                "Ensure note_indexer.py, test_note_indexer.py, README.md, and notes.json are ready.",
                '- python note_indexer.py add "First Note" --content="Agent testing note"',
                "- python note_indexer.py list",
            ]
        ),
        "task-id",
    )

    saved = json.loads((workspace / "notes.json").read_text(encoding="utf-8"))
    assert saved == notes
    assert completion.status == "unknown"
    assert completion.skipped_commands == [
        'Skipped unsafe or mutating smoke command during completion check: python note_indexer.py add "First Note" --content="Agent testing note"'
    ]


def test_completion_check_does_not_short_circuit_explicit_validation(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / "test_app.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    (workspace / "README.md").write_text("# App\n", encoding="utf-8")
    config = build_app_config(
        env_file=tmp_path / "missing.env",
        env={
            "AICODING_RUNTIME_DIR": str(tmp_path / "runtime"),
            "AICODING_ALLOWED_COMMANDS": "python -m pytest;git status;git diff",
        },
    )
    agent = CodingAgent(config=config, workspace=workspace, session_id="completion-explicit-validation")

    completion = agent._check_task_completion(
        "\n".join(
            [
                "Ensure test_app.py and README.md are ready.",
                "- python -m pytest test_app.py",
            ]
        ),
        "task-id",
    )

    assert completion.status == "unknown"
    assert "Explicit validation command present" in completion.skipped_commands[0]


def test_completion_check_does_not_execute_unknown_mutating_smoke_command(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / "danger.py").write_text(
        "from pathlib import Path\n"
        "Path('mutated.txt').write_text('mutated', encoding='utf-8')\n",
        encoding="utf-8",
    )
    (workspace / "README.md").write_text("# Boundary\n", encoding="utf-8")
    config = build_app_config(
        env_file=tmp_path / "missing.env",
        env={
            "AICODING_MODEL_API_KEY": "",
            "AICODING_RUNTIME_DIR": str(tmp_path / "runtime"),
            "AICODING_ALLOWED_COMMANDS": "python danger.py;git status;git diff",
        },
    )
    agent = CodingAgent(config=config, workspace=workspace, session_id="completion-unsafe")

    completion = agent._check_task_completion(
        "\n".join(
            [
                "Ensure README.md is ready.",
                "- python danger.py mutate",
            ]
        ),
        "task-id",
    )

    assert completion.status == "unknown"
    assert completion.skipped_commands == [
        "Skipped unsafe or mutating smoke command during completion check: python danger.py mutate"
    ]
    assert not (workspace / "mutated.txt").exists()


def test_completion_check_unknown_unsafe_smoke_does_not_overwrite_session_plan(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / "danger.py").write_text("print('unsafe')\n", encoding="utf-8")
    (workspace / "README.md").write_text("# Boundary\n", encoding="utf-8")
    config = build_app_config(
        env_file=tmp_path / "missing.env",
        env={
            "AICODING_RUNTIME_DIR": str(tmp_path / "runtime"),
            "AICODING_ALLOWED_COMMANDS": "python danger.py;git status;git diff",
        },
    )
    agent = CodingAgent(config=config, workspace=workspace, session_id="completion-plan-preserve")
    original_plan = {"goal": "existing plan"}
    agent.session.plan = original_plan.copy()

    completion = agent._check_task_completion(
        "Ensure README.md is ready.\n- python danger.py mutate",
        "task-id",
    )

    assert completion.status == "unknown"
    assert agent.session.plan == original_plan


def test_required_files_reject_paths_outside_workspace(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / "subdir").mkdir()
    (workspace / "subdir" / "valid.json").write_text("{}", encoding="utf-8")
    config = build_app_config(
        env_file=tmp_path / "missing.env",
        env={"AICODING_RUNTIME_DIR": str(tmp_path / "runtime")},
    )
    agent = CodingAgent(config=config, workspace=workspace, session_id="required-files-boundary")

    files = agent._required_files_for_task(
        "Check ../../outside.json and subdir/valid.json are ready."
    )

    assert files == ["subdir/valid.json"]


def test_completion_check_missing_required_file_does_not_exit(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    config = build_app_config(
        env_file=tmp_path / "missing.env",
        env={
            "AICODING_RUNTIME_DIR": str(tmp_path / "runtime"),
            "AICODING_ALLOWED_COMMANDS": "git status;git diff",
        },
    )
    agent = CodingAgent(config=config, workspace=workspace, session_id="completion-missing")

    completion = agent._check_task_completion(
        "Ensure note_indexer.py and README.md are ready.",
        "task-id",
    )

    assert completion.status == "incomplete"
    assert "Required file missing: note_indexer.py" in completion.missing_items


def test_prompt_treats_validation_environment_failures_as_blockers() -> None:
    assert "environment/tooling blocker" in PROMPT
    assert "Do not retry the same failing validation command." in PROMPT
    assert "Call check_environment once" in PROMPT
    assert "Do not automatically use install_python_package" in PROMPT
    assert "missing Python environment dependency, use" not in PROMPT


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


def test_final_response_correction_lists_denied_and_failed_validations(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    config = build_app_config(
        env_file=tmp_path / "missing.env",
        env={"AICODING_RUNTIME_DIR": str(tmp_path / "runtime")},
    )
    agent = CodingAgent(config=config, workspace=workspace, session_id="correction")
    run_state = ToolRunState()
    run_state.record_denied_command(
        "python -c 'print(1)'",
        "command is outside whitelist: python -c 'print(1)'",
    )
    run_state.record_policy_denial(
        "python -c 'print(1)'",
        "command is outside whitelist: python -c 'print(1)'",
    )
    run_state.record_validation(
        command="python -m pytest tests",
        status="failed",
        stdout="1 failed",
    )
    run_state.record_validation(
        command="python -m ruff check .",
        status="passed",
        stdout="All checks passed!",
    )

    response = agent._append_harness_correction("All validations pass.", run_state)

    assert "All validations pass." in response
    assert "Harness correction:" in response
    assert "Policy denials:" in response
    assert "python -c 'print(1)': command is outside whitelist" in response
    assert "Denied commands:" not in response
    assert "Failed validation commands:" in response
    assert "python -m pytest tests" in response
    assert "Passed validation commands:" in response
    assert "python -m ruff check ." in response


def test_final_response_correction_lists_problem_categories(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    config = build_app_config(
        env_file=tmp_path / "missing.env",
        env={"AICODING_RUNTIME_DIR": str(tmp_path / "runtime")},
    )
    agent = CodingAgent(config=config, workspace=workspace, session_id="classified-correction")
    run_state = ToolRunState()
    run_state.record_environment_failure(
        "python -m pytest tests",
        "Validation environment failure detected.",
    )
    run_state.record_policy_denial('python -c "print(1)"', "command is outside whitelist")
    run_state.record_tool_misuse("bad_tool", "unknown_tool")
    run_state.record_code_failure("python -m pytest tests/test_app.py", stdout="1 failed")

    response = agent._append_harness_correction("Done.", run_state)

    assert "Environment blockers:" in response
    assert "Policy denials:" in response
    assert "Tool misuses:" in response
    assert "Code validation failures:" in response


def test_harness_stop_returns_partial_summary_without_fallback(monkeypatch, tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    config = build_app_config(
        env_file=tmp_path / "missing.env",
        env={
            "AICODING_MODEL_API_KEY": "sk-test",
            "AICODING_MODEL_API_BASE": "https://example.invalid",
            "AICODING_MODEL_NAME": "test-model",
            "AICODING_RUNTIME_DIR": str(tmp_path / "runtime"),
        },
    )
    monkeypatch.setattr(agent_module, "LANGGRAPH_AVAILABLE", True)

    def fake_langgraph(self, task, task_id, *, run_state=None):
        state = run_state or ToolRunState()
        state.record_environment_failure(
            "python -m pytest test_note_indexer.py",
            "Validation environment failure detected.",
        )
        state.mark_environment_diagnosed()
        state.should_stop()
        raise HarnessStop("dependency_stop: environment_blocker_detected", state)

    monkeypatch.setattr(CodingAgent, "_run_langgraph_task", fake_langgraph)
    agent = CodingAgent(config=config, workspace=workspace, session_id="harness-stop")

    result = agent.run_mode_task(
        "agent",
        "Run validation:\n- python -m pytest test_note_indexer.py",
    )

    assert "Task status:" in result.response
    assert "completed_with_environment_blocker" in result.response
    assert "Blocked:" in result.response
    assert "Next action:" in result.response
    assert "python -m pytest test_note_indexer.py" in result.response
    assert "Harness correction:" not in result.response
    assert "deterministic fallback was used" not in result.response
    assert "direct patch edit was used" not in result.response


def test_partial_summary_environment_blocker_uses_explicit_rerun_command(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    config = build_app_config(
        env_file=tmp_path / "missing.env",
        env={"AICODING_RUNTIME_DIR": str(tmp_path / "runtime")},
    )
    agent = CodingAgent(config=config, workspace=workspace, session_id="partial-env")
    run_state = ToolRunState()
    run_state.mark_file_written("note_indexer.py")
    run_state.record_validation(command="python note_indexer.py --help", status="passed")
    run_state.record_environment_failure(
        "python -m pytest test_note_indexer.py",
        "Validation environment failure detected.",
    )
    run_state.mark_environment_diagnosed()
    run_state.should_stop()

    response = agent._build_partial_summary(
        "Validate with:\n- python -m pytest test_note_indexer.py",
        "task-id",
        run_state,
        failure_reason="dependency_stop",
    )

    assert "Task status:" in response
    assert "completed_with_environment_blocker" in response
    assert "Modified files:" in response
    assert "note_indexer.py" in response
    assert "Environment blocker:" in response
    assert "Repair local validation environment, then rerun python -m pytest test_note_indexer.py." in response
    assert "python -m pytest tests" not in response


def test_partial_summary_policy_denial_status(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    config = build_app_config(
        env_file=tmp_path / "missing.env",
        env={"AICODING_RUNTIME_DIR": str(tmp_path / "runtime")},
    )
    agent = CodingAgent(config=config, workspace=workspace, session_id="partial-policy")
    run_state = ToolRunState()
    run_state.record_policy_denial('python -c "print(1)"', "command is outside whitelist")
    run_state.record_policy_denial("python -m pip list", "command is outside whitelist")
    run_state.record_policy_denial("diff output.json expected.json", "command is outside whitelist")
    run_state.record_policy_denial("rm tmp.json", "destructive command is not allowed")
    run_state.should_stop()

    response = agent._build_partial_summary("run tests", "task-id", run_state)

    assert "stopped_by_policy" in response
    assert "Use an allowed validation command or extend the whitelist deliberately." in response


def test_policy_denial_loop_allows_recovery_after_two_denials() -> None:
    run_state = ToolRunState()

    run_state.record_policy_denial('python -c "print(1)"', "command is outside whitelist")
    run_state.record_policy_denial("diff output.json expected.json", "command is outside whitelist")

    assert not run_state.should_stop()


def test_validation_commands_extract_explicit_commands(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    config = build_app_config(
        env_file=tmp_path / "missing.env",
        env={"AICODING_RUNTIME_DIR": str(tmp_path / "runtime")},
    )
    agent = CodingAgent(config=config, workspace=workspace, session_id="validation-explicit")

    commands = agent._validation_commands_for_task(
        "\n".join(
            [
                "- python -m pytest test_note_indexer.py",
                "- python -m ruff check .",
                "- python -m pyright",
            ]
        )
    )

    assert commands == [
        "python -m pytest test_note_indexer.py",
        "python -m ruff check .",
        "python -m pyright",
    ]


def test_validation_command_prefers_explicit_over_workspace_default(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / "test_note_indexer.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    config = build_app_config(
        env_file=tmp_path / "missing.env",
        env={"AICODING_RUNTIME_DIR": str(tmp_path / "runtime")},
    )
    agent = CodingAgent(config=config, workspace=workspace, session_id="validation-explicit-first")

    assert (
        agent._validation_command_for_task("Run python -m pytest test_note_indexer.py")
        == "python -m pytest test_note_indexer.py"
    )


def test_validation_command_defaults_to_root_tests(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / "test_note_indexer.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    config = build_app_config(
        env_file=tmp_path / "missing.env",
        env={"AICODING_RUNTIME_DIR": str(tmp_path / "runtime")},
    )
    agent = CodingAgent(config=config, workspace=workspace, session_id="validation-root-tests")

    assert agent._validation_command_for_task("run tests") == "python -m pytest"


def test_validation_command_defaults_to_tests_dir(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    (workspace / "tests").mkdir(parents=True)
    config = build_app_config(
        env_file=tmp_path / "missing.env",
        env={"AICODING_RUNTIME_DIR": str(tmp_path / "runtime")},
    )
    agent = CodingAgent(config=config, workspace=workspace, session_id="validation-tests-dir")

    assert agent._validation_command_for_task("run tests") == "python -m pytest tests"


def test_validation_commands_skip_unsafe_commands(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    config = build_app_config(
        env_file=tmp_path / "missing.env",
        env={"AICODING_RUNTIME_DIR": str(tmp_path / "runtime")},
    )
    agent = CodingAgent(config=config, workspace=workspace, session_id="validation-unsafe")

    commands = agent._validation_commands_for_task(
        "\n".join(
            [
                'python -c "print(1)"',
                "python -m pip list",
                "python -m pytest tests || python -c 'print(1)'",
            ]
        )
    )

    assert commands == []


def test_validation_commands_do_not_extract_prose_mentions(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    config = build_app_config(
        env_file=tmp_path / "missing.env",
        env={"AICODING_RUNTIME_DIR": str(tmp_path / "runtime")},
    )
    agent = CodingAgent(config=config, workspace=workspace, session_id="validation-prose")

    commands = agent._validation_commands_for_task("Read the pytest documentation.")

    assert commands == []


def test_command_inspection_preserves_windows_backslash_paths(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    config = build_app_config(
        env_file=tmp_path / "missing.env",
        env={"AICODING_RUNTIME_DIR": str(tmp_path / "runtime")},
    )
    agent = CodingAgent(config=config, workspace=workspace, session_id="windows-command-split")

    parts = agent._split_command_for_inspection(r"python subdir\note_indexer.py list")

    assert parts == ["python", "subdir/note_indexer.py", "list"]

    quoted_parts = agent._split_command_for_inspection(
        r'python note_indexer.py add "First Note" --content="C:\Users\data"'
    )
    assert quoted_parts == [
        "python",
        "note_indexer.py",
        "add",
        "First Note",
        r"--content=C:\Users\data",
    ]

    arg_parts = agent._split_command_for_inspection(
        r"python script.py --data=C:\Users\data\input.csv"
    )
    assert arg_parts == ["python", "script.py", r"--data=C:\Users\data\input.csv"]


def test_run_state_problem_counters_reset_before_fallback(tmp_path: Path) -> None:
    run_state = ToolRunState(max_dependency_violations=2)
    run_state.record_dependency_violation("apply_patch", "preview_required")
    run_state.record_dependency_violation("apply_patch", "preview_required")
    assert run_state.should_stop()

    run_state.reset_problem_counters_for_fallback()

    assert not run_state.should_stop()
    assert run_state.dependency_violations == []
    assert run_state.consecutive_problem_count == 0


def test_final_response_correction_lists_dependency_violations(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    config = build_app_config(
        env_file=tmp_path / "missing.env",
        env={"AICODING_RUNTIME_DIR": str(tmp_path / "runtime")},
    )
    agent = CodingAgent(config=config, workspace=workspace, session_id="dependency-correction")
    run_state = ToolRunState(max_dependency_violations=1)
    run_state.record_dependency_violation(
        "apply_patch",
        "preview_required: call preview_patch successfully before apply_patch",
    )

    response = agent._append_harness_correction("Model says done.", run_state)

    assert "Harness correction:" in response
    assert "Dependency violations:" in response
    assert "apply_patch: preview_required" in response
    assert "Harness stop reason: too_many_dependency_violations" in response


def test_final_response_trace_summary_preserves_harness_correction(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    config = build_app_config(
        env_file=tmp_path / "missing.env",
        env={"AICODING_RUNTIME_DIR": str(tmp_path / "runtime")},
    )
    agent = CodingAgent(config=config, workspace=workspace, session_id="trace-summary")
    response = "\n".join(
        [
            "A" * 3000,
            "",
            "Harness correction:",
            "- Failed validation commands:",
            "  - python -m pytest tests",
        ]
    )

    summary = agent._final_response_trace_summary(response)

    assert len(summary) < len(response)
    assert "Harness correction:" in summary
    assert "python -m pytest tests" in summary


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


def test_normalize_patch_block_keeps_existing_add_file_header() -> None:
    patch = CodingAgent._normalize_patch_block(
        "*** Begin Patch\n"
        "*** Add File: app.py\n"
        "+print('ok')\n"
        "*** End Patch"
    )

    assert patch == (
        "*** Begin Patch\n"
        "*** Add File: app.py\n"
        "+print('ok')\n"
        "*** End Patch"
    )


def test_normalize_patch_block_does_not_rewrite_file_content_headers() -> None:
    patch = CodingAgent._normalize_patch_block(
        "*** Begin Patch\n"
        "*** Add File: README.md\n"
        "+Add File: should remain documentation text\n"
        "+Update File: should also remain documentation text\n"
        "*** End Patch"
    )

    assert "+Add File: should remain documentation text" in patch
    assert "+Update File: should also remain documentation text" in patch
    assert "+*** Add File:" not in patch


def test_normalize_patch_block_does_not_rewrite_unprefixed_content_headers() -> None:
    patch = CodingAgent._normalize_patch_block(
        "*** Begin Patch\n"
        "*** Add File: README.md\n"
        "Add File: should remain documentation text\n"
        "Update File: should also remain documentation text\n"
        "*** End Patch"
    )

    assert "Add File: should remain documentation text" in patch
    assert "Update File: should also remain documentation text" in patch
    assert "*** Add File: should remain documentation text" not in patch
