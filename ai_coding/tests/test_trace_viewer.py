from __future__ import annotations

import json
from pathlib import Path
from functools import partial
from http.server import ThreadingHTTPServer
import threading
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from aicoding_app.config import build_app_config
from aicoding_app.trace_viewer import (
    _resolve_web_workspace,
    TraceViewerRequestHandler,
    api_response,
    chat_session_payload,
    list_chat_sessions,
    list_trace_sessions,
    load_session_events,
    parse_trace_file,
    session_summary,
    web_run_response,
)
from aicoding_app.session_store import SessionStore


def _write_trace(trace_dir: Path, session_id: str) -> Path:
    trace_dir.mkdir(parents=True)
    path = trace_dir / f"{session_id}.jsonl"
    events = [
        {
            "timestamp": "2026-01-01T00:00:00+00:00",
            "session_id": session_id,
            "task_id": "task-a",
            "event_type": "model_call",
            "tool_name": None,
            "input_summary": "messages",
            "output_summary": "calling model",
            "status": "ok",
            "payload": {},
        },
        {
            "timestamp": "2026-01-01T00:00:01+00:00",
            "session_id": session_id,
            "task_id": "task-a",
            "event_type": "tool_call",
            "tool_name": "apply_patch",
            "input_summary": "patch",
            "output_summary": "changed files: app.py",
            "status": "ok",
            "payload": {},
        },
        {
            "timestamp": "2026-01-01T00:00:02+00:00",
            "session_id": session_id,
            "task_id": "task-a",
            "event_type": "tool_call",
            "tool_name": "run_command",
            "input_summary": "python -m pytest tests",
            "output_summary": "returncode: 0",
            "status": "ok",
            "payload": {},
        },
        {
            "timestamp": "2026-01-01T00:00:03+00:00",
            "session_id": session_id,
            "task_id": "task-a",
            "event_type": "tool_call",
            "tool_name": "run_command",
            "input_summary": "python -c print(1)",
            "output_summary": "command is outside whitelist",
            "status": "denied",
            "payload": {},
        },
        {
            "timestamp": "2026-01-01T00:00:04+00:00",
            "session_id": session_id,
            "task_id": "task-a",
            "event_type": "final_response",
            "tool_name": None,
            "input_summary": "response",
            "output_summary": "done",
            "status": "ok",
            "payload": {},
        },
    ]
    path.write_text(
        "\n".join(json.dumps(event) for event in events) + "\n{bad json\n",
        encoding="utf-8",
    )
    return path


def test_parse_trace_file_ignores_malformed_lines(tmp_path: Path) -> None:
    path = _write_trace(tmp_path / "traces", "demo")

    parsed = parse_trace_file(path)

    assert parsed.session_id == "demo"
    assert len(parsed.events) == 5
    assert parsed.malformed_count == 1


def test_list_trace_sessions(tmp_path: Path) -> None:
    _write_trace(tmp_path / "runtime" / "traces", "demo")

    sessions = list_trace_sessions(tmp_path / "runtime")

    assert sessions[0]["session_id"] == "demo"
    assert sessions[0]["event_count"] == 5
    assert sessions[0]["malformed_count"] == 1


def test_session_summary_extracts_counts_and_validation(tmp_path: Path) -> None:
    _write_trace(tmp_path / "runtime" / "traces", "demo")

    summary = session_summary(tmp_path / "runtime", "demo")

    assert summary["model_call_count"] == 1
    assert summary["tool_call_count"] == 3
    assert summary["changed_files"] == ["app.py"]
    assert summary["validation_commands"][0]["command"] == "python -m pytest tests"
    assert summary["failed_or_denied_tool_calls"][0]["status"] == "denied"


def test_load_session_events_rejects_missing_session(tmp_path: Path) -> None:
    try:
        load_session_events(tmp_path / "runtime", "missing")
    except FileNotFoundError:
        pass
    else:  # pragma: no cover
        raise AssertionError("missing session should raise FileNotFoundError")


def test_api_response_sessions_and_missing_session(tmp_path: Path) -> None:
    _write_trace(tmp_path / "runtime" / "traces", "demo")

    status, sessions_payload = api_response(tmp_path / "runtime", "/api/sessions")
    summary_status, summary_payload = api_response(
        tmp_path / "runtime", "/api/sessions/demo/summary"
    )
    missing_status, missing_payload = api_response(tmp_path / "runtime", "/api/sessions/missing")

    assert status == 200
    assert sessions_payload["sessions"][0]["session_id"] == "demo"
    assert summary_status == 200
    assert summary_payload["session_id"] == "demo"
    assert missing_status == 404
    assert missing_payload["error"] == "session_not_found"


def test_chat_session_api_lists_and_loads_saved_conversation(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"
    store = SessionStore(runtime_dir)
    session = store.get_or_create("web-demo", str(tmp_path / "workspace"))
    session.add_message("user", "first request")
    session.add_message("assistant", "first response")
    store.save(session)

    listed = list_chat_sessions(runtime_dir)
    payload = chat_session_payload(runtime_dir, "web-demo")
    status, api_payload = api_response(runtime_dir, "/api/chat/sessions/web-demo")

    assert listed[0]["session_id"] == "web-demo"
    assert listed[0]["message_count"] == 2
    assert payload["history"][0]["content"] == "first request"
    assert status == 200
    assert api_payload["history"][1]["role"] == "assistant"


def test_web_run_rejects_workspace_outside_project(tmp_path: Path) -> None:
    config = build_app_config(
        env_file=tmp_path / "missing.env",
        env={"AICODING_RUNTIME_DIR": str(tmp_path / "runtime")},
        project_root=tmp_path / "project",
    )

    status, payload = web_run_response(
        config,
        {
            "mode": "ask",
            "workspace": str(tmp_path / "outside"),
            "task": "explain",
            "session_id": "demo",
        },
    )

    assert status == 400
    assert payload["error"] == "invalid_workspace"


def test_web_run_allows_runtime_workspace(monkeypatch, tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    runtime_dir = project_root / "runtime"
    config = build_app_config(
        env_file=tmp_path / "missing.env",
        env={"AICODING_RUNTIME_DIR": str(runtime_dir)},
        project_root=project_root,
    )

    class FakeAgent:
        def __init__(self, *, config, workspace, session_id):  # noqa: ANN001
            self.workspace = workspace
            self.session_id = session_id

        def run_mode_task(self, mode: str, task: str):  # noqa: ANN201
            class Result:
                session_id = "web-demo"
                task_id = "task-web"
                response = f"{mode}: {task}"

            return Result()

    monkeypatch.setattr("aicoding_app.trace_viewer.CodingAgent", FakeAgent)

    status, payload = web_run_response(
        config,
        {
            "mode": "ask",
            "workspace": "runtime/web-demo",
            "task": "explain project",
            "session_id": "web-demo",
        },
    )

    assert status == 200
    assert payload["session_id"] == "web-demo"
    assert payload["trace_url"] == "/?session=web-demo"
    assert (runtime_dir / "web-demo").exists()


def test_resolve_web_workspace_has_no_creation_side_effect(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    runtime_dir = project_root / "runtime"
    config = build_app_config(
        env_file=tmp_path / "missing.env",
        env={"AICODING_RUNTIME_DIR": str(runtime_dir)},
        project_root=project_root,
    )

    workspace = _resolve_web_workspace(config, "runtime/not-yet-created")

    assert workspace == (runtime_dir / "not-yet-created").resolve()
    assert not workspace.exists()


def test_web_run_rejects_non_json_content_type(tmp_path: Path) -> None:
    project_root = tmp_path / "project"
    project_root.mkdir()
    config = build_app_config(
        env_file=tmp_path / "missing.env",
        env={"AICODING_RUNTIME_DIR": str(project_root / "runtime")},
        project_root=project_root,
    )
    handler = partial(
        TraceViewerRequestHandler,
        runtime_dir=config.harness.runtime_dir,
        app_config=config,
    )
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        request = Request(
            f"http://127.0.0.1:{server.server_port}/api/run",
            data=b"{}",
            headers={"Content-Type": "text/plain"},
            method="POST",
        )
        try:
            urlopen(request, timeout=5)
        except HTTPError as exc:
            status = exc.code
            payload = json.loads(exc.read().decode("utf-8"))
        else:  # pragma: no cover
            raise AssertionError("non-JSON request should be rejected")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert status == 415
    assert payload["error"] == "unsupported_media_type"
