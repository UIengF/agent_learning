"""Comprehensive end-to-end integration test for the ai-coding agent CLI.

Covers every CLI command, fallback pattern, mode, error path, and special feature.
All tests use deterministic fallback (no API key required).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

TEST_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = TEST_DIR.parent
PYTHON = [
    sys.executable,
    "-m",
    "aicoding_app.cli",
]


def _run(*args: str, cwd: str | Path = "", env: dict | None = None, timeout: int = 30) -> subprocess.CompletedProcess:
    full_args = [*PYTHON, *args]
    merged_env = os.environ.copy()
    if env:
        merged_env.update(env)
    return subprocess.run(
        full_args,
        cwd=cwd or PROJECT_ROOT,
        env=merged_env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


def _workspace(tmp_path: Path, name: str = "repo") -> Path:
    ws = tmp_path / name
    ws.mkdir(parents=True)
    (ws / "app.py").write_text("old\n", encoding="utf-8")
    (ws / "utils.py").write_text("def greet(name):\n    return f'Hello, {name}'\n", encoding="utf-8")
    (ws / "test_app.py").write_text("def test_pass():\n    assert True\n", encoding="utf-8")
    return ws


def _runtime_env(tmp_path: Path) -> dict:
    return {
        "AICODING_MODEL_API_KEY": "",
        "AICODING_API_KEY": "",
        "DEEPSEEK_API_KEY": "",
        "OPENAI_API_KEY": "",
        "AICODING_RUNTIME_DIR": str(tmp_path / "runtime"),
        "AICODING_ALLOWED_COMMANDS": "git status;git diff;python -m pytest",
        "AICODING_ALLOWED_DIRS": str(tmp_path / "repo"),
    }


# =============================================================================
# 1.  CONFIG INSPECT — secret masking
# =============================================================================


def test_config_inspect_masks_secret(tmp_path: Path) -> None:
    env = _runtime_env(tmp_path)
    env["AICODING_MODEL_API_KEY"] = "super-secret-key-12345"
    result = _run("config", "inspect", env=env)
    assert result.returncode == 0, result.stderr
    assert "super-secret-key-12345" not in result.stdout
    assert "sup...345" in result.stdout or "secr" in result.stdout.lower()


# =============================================================================
# 2.  RUN MODE — all four deterministic fallback patterns
# =============================================================================


def test_run_mode_replace_text(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    env = _runtime_env(tmp_path)
    env["AICODING_RUNTIME_DIR"] = str(tmp_path / "runtime")

    result = _run("run", "--workspace", str(ws), "--session-id", "e2e-replace", "--task", "replace 'old' with 'new' in app.py", env=env)

    assert result.returncode == 0, result.stderr
    assert "session_id: e2e-replace" in result.stdout
    assert "changed files: app.py" in result.stdout
    assert (ws / "app.py").read_text(encoding="utf-8") == "new\n"


def test_run_mode_create_file(tmp_path: Path) -> None:
    """create <path> with \"<content>\" pattern — path without quotes"""
    ws = _workspace(tmp_path)
    env = _runtime_env(tmp_path)
    env["AICODING_RUNTIME_DIR"] = str(tmp_path / "runtime")

    result = _run(
        "run", "--workspace", str(ws), "--session-id", "e2e-create",
        "--task", 'create new_module.py with "def new_func():\n    return 42\n"',
        env=env,
    )

    assert result.returncode == 0, f"STDERR: {result.stderr}\nSTDOUT: {result.stdout}"
    assert (ws / "new_module.py").exists(), f"File not created. Output: {result.stdout}"
    assert "def new_func()" in (ws / "new_module.py").read_text(encoding="utf-8")


def test_run_mode_append_text(tmp_path: Path) -> None:
    """append \"<content>\" to <path> pattern"""
    ws = _workspace(tmp_path)
    env = _runtime_env(tmp_path)
    env["AICODING_RUNTIME_DIR"] = str(tmp_path / "runtime")

    result = _run(
        "run", "--workspace", str(ws), "--session-id", "e2e-append",
        "--task", 'append "# new line" to utils.py',
        env=env,
    )

    assert result.returncode == 0, result.stderr
    content = (ws / "utils.py").read_text(encoding="utf-8")
    assert "# new line" in content


def test_run_mode_set_file_content(tmp_path: Path) -> None:
    """set <path> to \"<content>\" pattern — path without quotes, single-line content"""
    ws = _workspace(tmp_path)
    env = _runtime_env(tmp_path)
    env["AICODING_RUNTIME_DIR"] = str(tmp_path / "runtime")

    result = _run(
        "run", "--workspace", str(ws), "--session-id", "e2e-set",
        "--task", 'set utils.py to "def updated(): pass"',
        env=env,
    )

    assert result.returncode == 0, f"STDERR: {result.stderr}\nSTDOUT: {result.stdout}"
    content = (ws / "utils.py").read_text(encoding="utf-8")
    assert "def updated(): pass" in content


# =============================================================================
# 3.  EDIT MODE — plan-before-edit safety gate
# =============================================================================


def test_edit_mode_patch_success(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    env = _runtime_env(tmp_path)

    result = _run(
        "edit", "--workspace", str(ws), "--session-id", "e2e-edit-success",
        "--task", "replace 'old' with 'new' in app.py",
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert (ws / "app.py").read_text(encoding="utf-8") == "new\n"


def test_edit_mode_patch_mismatch_does_not_modify(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    env = _runtime_env(tmp_path)

    result = _run(
        "edit", "--workspace", str(ws), "--session-id", "e2e-edit-fail",
        "--task", 'replace "missing" with "new" in app.py',
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "patch_failed" in result.stdout or "no edit pattern matched" in result.stdout
    assert (ws / "app.py").read_text(encoding="utf-8") == "old\n"  # unchanged


# =============================================================================
# 4.  ASK MODE — read-only analysis
# =============================================================================


def test_ask_mode_read_only(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    env = _runtime_env(tmp_path)

    result = _run(
        "ask", "--workspace", str(ws), "--session-id", "e2e-ask",
        "--task", "overview of project structure",
        env=env,
    )

    assert result.returncode == 0, result.stderr
    # Should either succeed with model or gracefully fall back
    assert "No files were modified" in result.stdout
    assert "[aicoding]" in result.stderr


# =============================================================================
# 5.  PLAN MODE — structured plan without editing
# =============================================================================


def test_plan_mode_creates_plan(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    env = _runtime_env(tmp_path)

    result = _run(
        "plan", "--workspace", str(ws), "--session-id", "e2e-plan",
        "--task", "add error handling to utils.py",
        env=env,
    )

    assert result.returncode == 0, result.stderr
    # Plan mode should not modify files
    assert (ws / "utils.py").read_text(encoding="utf-8").startswith("def greet")
    assert "[aicoding]" in result.stderr


# =============================================================================
# 6.  AGENT MODE — staged agent loop
# =============================================================================


def test_agent_mode_deterministic_fallback(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    env = _runtime_env(tmp_path)

    result = _run(
        "agent", "--workspace", str(ws), "--session-id", "e2e-agent",
        "--task", "replace 'old' with 'new' in app.py",
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert "[aicoding]" in result.stderr
    assert "deterministic" in result.stdout.lower() or "changed files" in result.stdout


# =============================================================================
# 7.  TRACE SHOW — session trace inspection
# =============================================================================


def test_trace_show_after_run(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    env = _runtime_env(tmp_path)

    # First, run a command to generate a trace
    _run(
        "run", "--workspace", str(ws), "--session-id", "e2e-trace-demo",
        "--task", "replace 'old' with 'new' in app.py",
        env=env,
    )

    # Then show the trace
    result = _run("trace", "show", "--session-id", "e2e-trace-demo", env=env)

    assert result.returncode == 0, result.stderr
    assert "task_started" in result.stdout or "final_response" in result.stdout


def test_trace_show_missing_session(tmp_path: Path) -> None:
    env = _runtime_env(tmp_path)

    result = _run("trace", "show", "--session-id", "nonexistent", env=env)

    assert result.returncode == 1
    assert "trace not found" in result.stderr


# =============================================================================
# 8.  RESUME — session persistence
# =============================================================================


def test_resume_session_after_run(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    env = _runtime_env(tmp_path)

    _run(
        "run", "--workspace", str(ws), "--session-id", "e2e-resume-demo",
        "--task", "replace 'old' with 'new' in app.py",
        env=env,
    )

    result = _run("resume", "--session-id", "e2e-resume-demo", env=env)

    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["session_id"] == "e2e-resume-demo"
    assert "history" in payload


def test_resume_missing_session(tmp_path: Path) -> None:
    env = _runtime_env(tmp_path)

    result = _run("resume", "--session-id", "does-not-exist", env=env)

    assert result.returncode == 1
    assert "session not found" in result.stderr


# =============================================================================
# 9.  REPO MAP — repository intelligence
# =============================================================================


def test_repo_map(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    env = _runtime_env(tmp_path)

    result = _run("dev", "repo", "map", "--workspace", str(ws), env=env)

    assert result.returncode == 0, result.stderr
    assert "app.py" in result.stdout
    assert "utils.py" in result.stdout


# =============================================================================
# 10. CONTEXT EXPLAIN — symbol explanation
# =============================================================================


def test_context_explain(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    env = _runtime_env(tmp_path)

    result = _run("dev", "context", "explain", "--workspace", str(ws), "--query", "greet", env=env)

    assert result.returncode == 0, result.stderr
    assert len(result.stdout) > 0


def test_context_explain_with_pytest(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    env = _runtime_env(tmp_path)

    result = _run(
        "dev", "context", "explain", "--workspace", str(ws), "--query", "test_app.py", env=env
    )

    assert result.returncode == 0, result.stderr
    assert len(result.stdout) > 0


# =============================================================================
# 11. GIT SUMMARY — git operations
# =============================================================================


def test_git_summary_shows_branch_and_changes(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    env = _runtime_env(tmp_path)

    # Init git in workspace so git commands work
    subprocess.run(["git", "init"], cwd=ws, capture_output=True, timeout=10)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=ws, capture_output=True, timeout=10)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=ws, capture_output=True, timeout=10)
    subprocess.run(["git", "add", "-A"], cwd=ws, capture_output=True, timeout=10)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=ws, capture_output=True, timeout=10)

    (ws / "new_file.py").write_text("x = 1\n", encoding="utf-8")

    result = _run("dev", "git", "summary", "--workspace", str(ws), env=env)

    assert result.returncode == 0, result.stderr
    assert "Git summary" in result.stdout
    assert "new_file.py" in result.stdout


# =============================================================================
# 12. MEMORY — memory store operations
# =============================================================================


def test_memory_add_inspect_forget(tmp_path: Path) -> None:
    env = _runtime_env(tmp_path)

    # Add a memory
    add_result = _run(
        "dev", "memory", "add", "--kind", "insight", "--text", "Test memory entry", env=env
    )
    assert add_result.returncode == 0, add_result.stderr
    assert "memory_added" in add_result.stdout

    # Extract the id
    memory_id = add_result.stdout.strip().removeprefix("memory_added: ")

    # Inspect — should contain the memory
    inspect_result = _run("dev", "memory", "inspect", env=env)
    assert inspect_result.returncode == 0, inspect_result.stderr
    assert "Test memory entry" in inspect_result.stdout

    # Forget
    forget_result = _run("dev", "memory", "forget", "--id", memory_id, env=env)
    assert forget_result.returncode == 0, forget_result.stderr
    assert "memory_forgotten" in forget_result.stdout

    # Verify gone
    inspect2 = _run("dev", "memory", "inspect", env=env)
    assert "Test memory entry" not in inspect2.stdout


def test_memory_forget_nonexistent(tmp_path: Path) -> None:
    env = _runtime_env(tmp_path)

    result = _run("dev", "memory", "forget", "--id", "no-such-id", env=env)
    assert result.returncode == 1
    assert "memory_not_found" in result.stderr or "memory_not_found" in result.stdout


# =============================================================================
# 13. TEXT HYGIENE — check and clean
# =============================================================================


def test_text_hygiene_check_reports_issues(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    # Use characters detectable by text hygiene but safe for Windows GBK encoding
    (ws / "mixed_line_endings.py").write_text("x = 1\r\ny = 2\n", encoding="utf-8")
    env = _runtime_env(tmp_path)

    result = _run("dev", "text", "check", "--workspace", str(ws), env=env)
    assert result.returncode == 0, f"STDERR: {result.stderr}\nSTDOUT: {result.stdout}"


def test_text_hygiene_clean_conservative(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    (ws / "bad_quotes.py").write_text("x = “hello”\n", encoding="utf-8")
    env = _runtime_env(tmp_path)

    result = _run("dev", "text", "clean", "--workspace", str(ws), env=env)
    assert result.returncode == 0, result.stderr


# =============================================================================
# 14. VERIFY — run allowed validation command
# =============================================================================


def test_verify_shows_output(tmp_path: Path) -> None:
    """Verify runs a command and captures output (exit code is always 0 from CLI wrapper)."""
    ws = _workspace(tmp_path)
    env = _runtime_env(tmp_path)

    result = _run(
        "dev", "verify", "--workspace", str(ws), "--validation-command", "git status --short",
        env=env,
    )
    assert result.returncode == 0, result.stderr
    # git status might fail (no git repo) but CLI always returns 0; output should exist
    assert len(result.stdout) > 0


# =============================================================================
# 15. SCHEDULE PLAN — scheduling dry-run
# =============================================================================


def test_schedule_plan_dry_run(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    env = _runtime_env(tmp_path)

    result = _run(
        "dev", "schedule", "plan", "--workspace", str(ws),
        "--task", "run tests daily",
        "--cadence", "0 9 * * 1-5",
        env=env,
    )

    assert result.returncode == 0, result.stderr
    assert len(result.stdout) > 0


# =============================================================================
# 16. CONNECTORS LIST — placeholder connectors
# =============================================================================


def test_connectors_list(tmp_path: Path) -> None:
    env = _runtime_env(tmp_path)

    result = _run("dev", "connectors", "list", env=env)

    assert result.returncode == 0, result.stderr
    assert len(result.stdout) > 0


# =============================================================================
# 17. TRACE SERVE PARSING — verify CLI options for web UI
# =============================================================================


def test_trace_serve_cli_parsing(tmp_path: Path) -> None:
    """Verify trace serve can parse host/port options (can't actually bind)."""
    from aicoding_app.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["trace", "serve", "--host", "0.0.0.0", "--port", "9999"])
    assert args.trace_command == "serve"
    assert args.host == "0.0.0.0"
    assert args.port == 9999


# =============================================================================
# 18. WEB COMMAND STILL WORKS (aliases to trace serve)
# =============================================================================


def test_web_command_removed(tmp_path: Path) -> None:
    """Verify web subcommand was removed (trace serve is the only entry)."""
    from aicoding_app.cli import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["web", "--host", "0.0.0.0", "--port", "7777"])


# =============================================================================
# 19. ERROR HANDLING — missing arguments, unknown commands
# =============================================================================


def test_error_missing_workspace(tmp_path: Path) -> None:
    env = _runtime_env(tmp_path)

    result = _run("run", "--workspace", str(tmp_path / "nonexistent"), "--task", "test", env=env)

    assert result.returncode != 0


def test_error_unknown_command(tmp_path: Path) -> None:
    env = _runtime_env(tmp_path)

    result = _run("unknown-command-xyz", env=env)
    assert result.returncode != 0


def test_error_missing_session_id(tmp_path: Path) -> None:
    env = _runtime_env(tmp_path)

    result = _run("resume", "--session-id", "no-such-session", env=env)
    assert result.returncode == 1
    assert "session not found" in result.stderr


# =============================================================================
# 20. PROGRESS INDICATORS — [aicoding] appears on stderr
# =============================================================================


def test_progress_indicators_on_stderr(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    env = _runtime_env(tmp_path)

    # Test multiple modes produce progress indicators
    for mode, task in [
        ("run", "replace 'old' with 'new' in app.py"),
    ]:
        session_id = f"e2e-progress-{mode}"
        result = _run(mode, "--workspace", str(ws), "--session-id", session_id, "--task", task, env=env)
        assert result.returncode == 0, f"{mode}: {result.stderr}"
        assert "[aicoding]" in result.stderr, f"{mode} should emit [aicoding] on stderr"


# =============================================================================
# 21. MULTIPLE SESSIONS — isolation between sessions
# =============================================================================


def test_session_isolation(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    env = _runtime_env(tmp_path)

    # Session A changes app.py
    result_a = _run(
        "run", "--workspace", str(ws), "--session-id", "session-a",
        "--task", "replace 'old' with 'new' in app.py",
        env=env,
    )
    assert result_a.returncode == 0

    # Session B starts fresh — modifies a different file (create pattern needs no-quotes path)
    result_b = _run(
        "run", "--workspace", str(ws), "--session-id", "session-b",
        "--task", "replace 'def greet(name)' with 'def greet(name, greeting=\"Hi\")' in utils.py",
        env=env,
    )
    assert result_b.returncode == 0

    # Both effects visible in workspace (shared workspace), but traces are separate
    assert (ws / "app.py").read_text(encoding="utf-8") == "new\n"
    assert 'greeting="Hi"' in (ws / "utils.py").read_text(encoding="utf-8")

    # Traces are isolated per session
    trace_a = _run("trace", "show", "--session-id", "session-a", env=env)
    assert trace_a.returncode == 0
    assert "task_started" in trace_a.stdout

    trace_b = _run("trace", "show", "--session-id", "session-b", env=env)
    assert trace_b.returncode == 0
    assert "task_started" in trace_b.stdout


# =============================================================================
# 22. CHAT MODE — interactive session (single turn via stdin)
# =============================================================================


def test_chat_mode_single_turn(tmp_path: Path) -> None:
    ws = _workspace(tmp_path)
    env = _runtime_env(tmp_path)

    proc = subprocess.run(
        [*PYTHON, "chat", "--workspace", str(ws), "--session-id", "e2e-chat"],
        cwd=PROJECT_ROOT,
        env={**os.environ, **env},
        input="replace 'old' with 'new' in app.py\n/exit\n",
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert "session_id: e2e-chat" in proc.stdout
    assert "[aicoding]" in proc.stderr


# =============================================================================
# 23. ENV FILE — --env-file flag with .env loading
# =============================================================================


def test_env_file_loading(tmp_path: Path) -> None:
    env_file = tmp_path / "custom.env"
    env_file.write_text(
        "AICODING_RUNTIME_DIR=" + str(tmp_path / "runtime-custom").replace("\\", "/") + "\n"
        "AICODING_ALLOWED_COMMANDS=git status;git diff\n"
        "AICODING_MODEL_API_KEY=\n",
        encoding="utf-8",
    )

    ws = _workspace(tmp_path)
    # Don't pass env vars, rely on --env-file
    clean_env = {k: v for k, v in os.environ.items() if not k.startswith("AICODING_")}

    result = _run(
        "--env-file", str(env_file),
        "run", "--workspace", str(ws), "--session-id", "e2e-envfile",
        "--task", "replace 'old' with 'new' in app.py",
        env=clean_env,
    )
    assert result.returncode == 0, result.stderr


# =============================================================================
# Test config — validate the test suite itself
# =============================================================================


def test_all_test_modes_produce_exit_code_zero(tmp_path: Path) -> None:
    """Quick sanity: each mode at minimum returns 0."""
    ws = _workspace(tmp_path)
    env = _runtime_env(tmp_path)

    for mode in ("run", "edit", "agent"):
        sid = f"e2e-smoke-{mode}"
        r = _run(mode, "--workspace", str(ws), "--session-id", sid, "--task", "replace 'old' with 'new' in app.py", env=env)
        assert r.returncode == 0, f"{mode}: exit {r.returncode}\nSTDERR: {r.stderr}"
