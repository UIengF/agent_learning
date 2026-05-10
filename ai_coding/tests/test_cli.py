from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

from aicoding_app.cli import build_parser, main
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


def test_trace_serve_command_parses_server_options() -> None:
    args = build_parser().parse_args(["trace", "serve", "--host", "127.0.0.1", "--port", "8766"])

    assert args.command == "trace"
    assert args.trace_command == "serve"
    assert args.host == "127.0.0.1"
    assert args.port == 8766


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


def test_module_cli_agent_detects_completed_workspace_and_skips_duplicate_note_add(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    (workspace / "note_indexer.py").write_text(
        "\n".join(
            [
                "from __future__ import annotations",
                "import json",
                "import sys",
                "from pathlib import Path",
                "NOTES = Path('notes.json')",
                "def load_notes():",
                "    return json.loads(NOTES.read_text(encoding='utf-8')) if NOTES.exists() else []",
                "def save_notes(notes):",
                "    NOTES.write_text(json.dumps(notes, indent=2), encoding='utf-8')",
                "if len(sys.argv) > 1 and sys.argv[1] == 'add':",
                "    notes = load_notes()",
                "    notes.append({'title': sys.argv[2], 'content': sys.argv[-1]})",
                "    save_notes(notes)",
                "elif len(sys.argv) > 1 and sys.argv[1] == 'list':",
                "    print('\\n'.join(note['title'] for note in load_notes()))",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (workspace / "test_note_indexer.py").write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    (workspace / "README.md").write_text("# Notes\n", encoding="utf-8")
    notes = [{"title": "Boundary_Note", "content": "Boundary_Content"}]
    (workspace / "notes.json").write_text(json.dumps(notes), encoding="utf-8")
    runtime_dir = tmp_path / "runtime"
    task = "\n".join(
        [
            "Ensure note_indexer.py, test_note_indexer.py, README.md, and notes.json are ready.",
            "- python note_indexer.py add Boundary_Note --content Boundary_Content",
            "- python note_indexer.py list",
        ]
    )
    env = os.environ.copy()
    env.update(
        {
            "AICODING_MODEL_API_KEY": "",
            "AICODING_API_KEY": "",
            "DEEPSEEK_API_KEY": "",
            "OPENAI_API_KEY": "",
            "AICODING_RUNTIME_DIR": str(runtime_dir),
            "AICODING_ALLOWED_COMMANDS": "python note_indexer.py;git status;git diff",
        }
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "aicoding_app.cli",
            "--env-file",
            str(tmp_path / "missing.env"),
            "agent",
            "--workspace",
            str(workspace),
            "--session-id",
            "module-boundary",
            "--task",
            task,
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "Task status:" in result.stdout
    assert "- completed" in result.stdout
    assert "Skipped idempotent side-effect command" in result.stdout
    assert "Smoke command passed: python note_indexer.py list" in result.stdout
    assert "deterministic fallback was used" not in result.stdout.lower()
    assert json.loads((workspace / "notes.json").read_text(encoding="utf-8")) == notes
    trace_text = (runtime_dir / "traces" / "module-boundary.jsonl").read_text(encoding="utf-8")
    assert "task_started" in trace_text
    assert "final_response" in trace_text
