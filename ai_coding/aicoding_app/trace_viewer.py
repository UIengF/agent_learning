from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse
import webbrowser

from .agent import CodingAgent, generate_session_id
from .config import AppConfig
from .evidence_cache import compact_text
from .session_store import SessionStore


STATIC_DIR = Path(__file__).resolve().parent / "static" / "trace_viewer"
WEB_RUN_MODES = {"ask", "plan", "edit", "agent"}
MAX_RUN_BODY_BYTES = 64 * 1024


@dataclass(frozen=True)
class ParsedTrace:
    session_id: str
    path: Path
    events: list[dict[str, Any]]
    malformed_count: int = 0


def _trace_dir(runtime_dir: str | Path) -> Path:
    return Path(runtime_dir).resolve() / "traces"


def _trace_path(runtime_dir: str | Path, session_id: str) -> Path:
    trace_dir = _trace_dir(runtime_dir)
    path = (trace_dir / f"{session_id}.jsonl").resolve()
    if path.parent != trace_dir:
        raise ValueError("invalid session id")
    return path


def parse_trace_file(path: str | Path) -> ParsedTrace:
    trace_path = Path(path)
    events: list[dict[str, Any]] = []
    malformed_count = 0
    if trace_path.exists():
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                malformed_count += 1
                continue
            if isinstance(payload, dict):
                events.append(payload)
            else:
                malformed_count += 1
    session_id = trace_path.stem
    if events:
        session_id = str(events[0].get("session_id") or session_id)
    return ParsedTrace(
        session_id=session_id,
        path=trace_path,
        events=events,
        malformed_count=malformed_count,
    )


def list_trace_sessions(runtime_dir: str | Path) -> list[dict[str, Any]]:
    trace_dir = _trace_dir(runtime_dir)
    if not trace_dir.exists():
        return []
    sessions: list[dict[str, Any]] = []
    for path in sorted(trace_dir.glob("*.jsonl")):
        parsed = parse_trace_file(path)
        latest_timestamp = ""
        if parsed.events:
            latest_timestamp = str(parsed.events[-1].get("timestamp") or "")
        sessions.append(
            {
                "session_id": parsed.session_id,
                "trace_file": str(path),
                "event_count": len(parsed.events),
                "latest_timestamp": latest_timestamp,
                "malformed_count": parsed.malformed_count,
            }
        )
    sessions.sort(key=lambda item: str(item.get("latest_timestamp") or ""), reverse=True)
    return sessions


def list_chat_sessions(runtime_dir: str | Path) -> list[dict[str, Any]]:
    store = SessionStore(runtime_dir)
    sessions_dir = store.sessions_dir
    if not sessions_dir.exists():
        return []
    sessions: list[dict[str, Any]] = []
    for path in sorted(sessions_dir.glob("*.json")):
        try:
            session = store.load(path.stem)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            continue
        if session is None:
            continue
        sessions.append(
            {
                "session_id": session.session_id,
                "workspace": session.workspace,
                "message_count": len(session.history),
                "updated_at": session.updated_at,
                "created_at": session.created_at,
                "latest_message": compact_text(
                    session.history[-1]["content"] if session.history else "",
                    160,
                ),
            }
        )
    sessions.sort(key=lambda item: str(item.get("updated_at") or ""), reverse=True)
    return sessions


def chat_session_payload(runtime_dir: str | Path, session_id: str) -> dict[str, Any]:
    store = SessionStore(runtime_dir)
    session = store.load(session_id)
    if session is None:
        raise FileNotFoundError(session_id)
    return session.to_jsonable()


def load_session_events(runtime_dir: str | Path, session_id: str) -> ParsedTrace:
    path = _trace_path(runtime_dir, session_id)
    if not path.exists():
        raise FileNotFoundError(session_id)
    return parse_trace_file(path)


def summarize_trace(parsed: ParsedTrace) -> dict[str, Any]:
    task_ids = sorted(
        {
            str(event.get("task_id"))
            for event in parsed.events
            if event.get("task_id") not in {None, ""}
        }
    )
    model_calls = [event for event in parsed.events if event.get("event_type") == "model_call"]
    tool_calls = [event for event in parsed.events if event.get("event_type") == "tool_call"]
    final_responses = [
        event for event in parsed.events if event.get("event_type") == "final_response"
    ]
    changed_files = _extract_changed_files(tool_calls)
    validation_commands = _extract_validation_commands(tool_calls)
    failed_or_denied = [
        _event_summary(event)
        for event in parsed.events
        if event.get("status") in {"failed", "denied"}
    ]
    final_response_summary = ""
    if final_responses:
        final_response_summary = compact_text(
            str(final_responses[-1].get("output_summary") or ""),
            1200,
        )
    return {
        "session_id": parsed.session_id,
        "trace_file": str(parsed.path),
        "event_count": len(parsed.events),
        "malformed_count": parsed.malformed_count,
        "task_ids": task_ids,
        "model_call_count": len(model_calls),
        "tool_call_count": len(tool_calls),
        "final_response_summary": final_response_summary,
        "changed_files": changed_files,
        "validation_commands": validation_commands,
        "failed_or_denied_tool_calls": failed_or_denied,
    }


def session_summary(runtime_dir: str | Path, session_id: str) -> dict[str, Any]:
    return summarize_trace(load_session_events(runtime_dir, session_id))


def _event_summary(event: dict[str, Any]) -> dict[str, str]:
    return {
        "event_type": str(event.get("event_type") or ""),
        "tool_name": str(event.get("tool_name") or ""),
        "input_summary": str(event.get("input_summary") or ""),
        "output_summary": str(event.get("output_summary") or ""),
        "status": str(event.get("status") or ""),
    }


def _extract_changed_files(tool_calls: list[dict[str, Any]]) -> list[str]:
    files: list[str] = []
    for event in tool_calls:
        if event.get("tool_name") != "apply_patch":
            continue
        output = str(event.get("output_summary") or "")
        prefix = "changed files: "
        if output.startswith(prefix):
            files.extend(item.strip() for item in output.removeprefix(prefix).split(","))
    return sorted({item for item in files if item})


def _extract_validation_commands(tool_calls: list[dict[str, Any]]) -> list[dict[str, str]]:
    commands: list[dict[str, str]] = []
    for event in tool_calls:
        tool_name = event.get("tool_name")
        input_summary = str(event.get("input_summary") or "")
        if tool_name == "run_validation" or (
            tool_name == "run_command" and _looks_like_validation(input_summary)
        ):
            commands.append(
                {
                    "command": input_summary,
                    "status": str(event.get("status") or ""),
                    "summary": compact_text(str(event.get("output_summary") or ""), 600),
                }
            )
    return commands


def _looks_like_validation(command: str) -> bool:
    lowered = command.lower()
    return "pytest" in lowered or "ruff" in lowered or "pyright" in lowered


def api_response(runtime_dir: str | Path, path: str) -> tuple[int, dict[str, Any]]:
    parsed = urlparse(path)
    route = parsed.path.rstrip("/") or "/"
    if route == "/api/chat/sessions":
        return HTTPStatus.OK, {"sessions": list_chat_sessions(runtime_dir)}
    if route.startswith("/api/chat/sessions/"):
        session_id = unquote(route.removeprefix("/api/chat/sessions/"))
        try:
            return HTTPStatus.OK, chat_session_payload(runtime_dir, session_id)
        except FileNotFoundError:
            return HTTPStatus.NOT_FOUND, {"error": "session_not_found", "session_id": session_id}
    if route == "/api/sessions":
        return HTTPStatus.OK, {"sessions": list_trace_sessions(runtime_dir)}
    if route.startswith("/api/sessions/"):
        suffix = route.removeprefix("/api/sessions/")
        is_summary = suffix.endswith("/summary")
        session_id = suffix.removesuffix("/summary") if is_summary else suffix
        session_id = unquote(session_id)
        try:
            parsed_trace = load_session_events(runtime_dir, session_id)
        except (FileNotFoundError, ValueError):
            return HTTPStatus.NOT_FOUND, {"error": "session_not_found", "session_id": session_id}
        if is_summary:
            return HTTPStatus.OK, summarize_trace(parsed_trace)
        return HTTPStatus.OK, {
            "session_id": parsed_trace.session_id,
            "trace_file": str(parsed_trace.path),
            "events": parsed_trace.events,
            "malformed_count": parsed_trace.malformed_count,
        }
    return HTTPStatus.NOT_FOUND, {"error": "not_found"}


def web_run_response(config: AppConfig, payload: dict[str, Any]) -> tuple[int, dict[str, Any]]:
    mode = str(payload.get("mode") or "agent").strip().lower()
    task = str(payload.get("task") or "").strip()
    session_id = str(payload.get("session_id") or "").strip() or generate_session_id()
    if mode not in WEB_RUN_MODES:
        return HTTPStatus.BAD_REQUEST, {
            "error": "invalid_mode",
            "allowed_modes": sorted(WEB_RUN_MODES),
        }
    if not task:
        return HTTPStatus.BAD_REQUEST, {"error": "missing_task"}
    try:
        workspace = _resolve_web_workspace(config, str(payload.get("workspace") or ""))
    except ValueError as exc:
        return HTTPStatus.BAD_REQUEST, {"error": "invalid_workspace", "detail": str(exc)}
    if workspace != config.project_root.resolve():
        workspace.mkdir(parents=True, exist_ok=True)

    agent = CodingAgent(config=config, workspace=workspace, session_id=session_id)
    result = agent.run_mode_task(mode, task)
    return HTTPStatus.OK, {
        "session_id": result.session_id,
        "task_id": result.task_id,
        "mode": mode,
        "workspace": str(workspace),
        "response": result.response,
        "trace_url": f"/?session={result.session_id}",
    }


def _resolve_web_workspace(config: AppConfig, workspace: str) -> Path:
    project_root = config.project_root.resolve()
    runtime_dir = config.harness.runtime_dir.resolve()
    raw = workspace.strip()
    if not raw:
        raise ValueError("workspace is required")
    candidate = Path(raw)
    if not candidate.is_absolute():
        candidate = project_root / candidate
    resolved = candidate.resolve()

    if resolved == project_root:
        return resolved
    if resolved == runtime_dir or runtime_dir in resolved.parents:
        return resolved
    raise ValueError(
        f"workspace must be the project root or inside runtime: {project_root}, {runtime_dir}"
    )


class TraceViewerRequestHandler(SimpleHTTPRequestHandler):
    def __init__(
        self,
        *args: Any,
        runtime_dir: str | Path,
        app_config: AppConfig | None = None,
        **kwargs: Any,
    ) -> None:
        self.runtime_dir = Path(runtime_dir).resolve()
        self.app_config = app_config
        super().__init__(*args, directory=str(STATIC_DIR), **kwargs)

    def do_GET(self) -> None:
        if self.path.startswith("/api/"):
            status, payload = api_response(self.runtime_dir, self.path)
            self._send_json(status, payload)
            return
        parsed = urlparse(self.path)
        if parsed.path == "/favicon.ico":
            self.send_response(HTTPStatus.NO_CONTENT)
            self.end_headers()
            return
        if parsed.path in {"", "/"}:
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self) -> None:
        if self.path != "/api/run":
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return
        if self.app_config is None:
            self._send_json(HTTPStatus.SERVICE_UNAVAILABLE, {"error": "run_unavailable"})
            return
        content_type = self.headers.get("Content-Type", "")
        if not content_type.lower().startswith("application/json"):
            self._send_json(
                HTTPStatus.UNSUPPORTED_MEDIA_TYPE,
                {"error": "unsupported_media_type"},
            )
            return
        try:
            payload = self._read_json_body()
        except ValueError as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid_json", "detail": str(exc)})
            return
        try:
            status, response = web_run_response(self.app_config, payload)
        except Exception as exc:
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {"error": "run_failed", "detail": compact_text(str(exc), 1000)},
            )
            return
        self._send_json(status, response)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length") or "0")
        if length <= 0:
            raise ValueError("empty request body")
        if length > MAX_RUN_BODY_BYTES:
            raise ValueError("request body is too large")
        raw = self.rfile.read(length).decode("utf-8")
        payload = json.loads(raw)
        if not isinstance(payload, dict):
            raise ValueError("request body must be a JSON object")
        return payload

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve_trace_viewer(
    runtime_dir: str | Path,
    *,
    app_config: AppConfig | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    open_browser: bool = False,
    session_id: str | None = None,
) -> None:
    handler = partial(TraceViewerRequestHandler, runtime_dir=runtime_dir, app_config=app_config)
    server = ThreadingHTTPServer((host, port), handler)
    server.socket.settimeout(1.0)
    query = f"?session={session_id}" if session_id else ""
    url = f"http://{host}:{server.server_port}/{query}"
    print(f"Trace viewer serving at {url}", flush=True)
    print("Press Ctrl+C to stop.", flush=True)
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nTrace viewer stopped.")
    finally:
        server.server_close()
