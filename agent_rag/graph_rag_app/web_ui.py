from __future__ import annotations

from dataclasses import dataclass
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path
import time
from typing import Any

from .config import DEFAULT_CHECKPOINT_DB, DEFAULT_SESSION_ID
from .runtime import build_sqlite_checkpointer, run_or_resume


STATIC_DIR = Path(__file__).resolve().parent / "static"


@dataclass(frozen=True)
class AskPayload:
    question: str
    index_dir: str
    session_id: str = DEFAULT_SESSION_ID
    resume: bool = False
    history: tuple[dict[str, str], ...] = ()


def _parse_history(value: Any) -> tuple[dict[str, str], ...]:
    if not isinstance(value, list):
        return ()
    history: list[dict[str, str]] = []
    for item in value[-4:]:
        if not isinstance(item, dict):
            continue
        question = str(item.get("question", "")).strip()
        answer = str(item.get("answer", "")).strip()
        if question and answer:
            history.append({"question": question, "answer": answer})
    return tuple(history)


def _build_question_with_history(payload: AskPayload) -> str:
    if not payload.resume or not payload.history:
        return payload.question
    lines = ["Previous conversation context:"]
    for item in payload.history:
        lines.append(f"User: {item['question']}")
        lines.append(f"Assistant: {item['answer']}")
    lines.append(f"Current question: {payload.question}")
    return "\n".join(lines)


def parse_ask_payload(data: dict[str, Any]) -> AskPayload:
    question = str(data.get("question", "")).strip()
    index_dir = str(data.get("index_dir", "")).strip()
    session_id = str(data.get("session_id", "")).strip() or DEFAULT_SESSION_ID
    resume = bool(data.get("resume", False))
    if not question:
        raise ValueError("Question is required.")
    if not index_dir:
        raise ValueError("Index directory is required.")
    return AskPayload(
        question=question,
        index_dir=index_dir,
        session_id=session_id,
        resume=resume,
        history=_parse_history(data.get("history")),
    )


def build_ask_response(data: dict[str, Any]) -> dict[str, Any]:
    payload = parse_ask_payload(data)
    started = time.perf_counter()
    question = _build_question_with_history(payload)
    answer = run_or_resume(
        question=question,
        index_dir=payload.index_dir,
        session_id=payload.session_id,
        checkpointer=build_sqlite_checkpointer(DEFAULT_CHECKPOINT_DB),
        resume=False,
        interrupt_after=None,
    )
    return {
        "answer": answer,
        "question": payload.question,
        "index_dir": payload.index_dir,
        "session_id": payload.session_id,
        "resume": False,
        "context_included": bool(payload.resume and payload.history),
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }


class GraphRAGUIHandler(SimpleHTTPRequestHandler):
    default_index_dir = "agent"

    def log_message(self, format: str, *args: Any) -> None:
        return

    def do_GET(self) -> None:
        if self.path == "/api/config":
            self._send_json({"index_dir": self.default_index_dir, "session_id": DEFAULT_SESSION_ID})
            return
        if self.path in {"/", "/index.html"}:
            self._send_static_file(STATIC_DIR / "index.html")
            return
        if self.path.startswith("/static/"):
            relative = self.path.removeprefix("/static/").split("?", 1)[0]
            self._send_static_file(STATIC_DIR / relative)
            return
        self.send_error(HTTPStatus.NOT_FOUND, "Not found")

    def do_POST(self) -> None:
        if self.path != "/api/ask":
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        try:
            data = self._read_json_body()
            response = build_ask_response(data)
        except ValueError as exc:
            self._send_json({"error": str(exc)}, status=HTTPStatus.BAD_REQUEST)
        except Exception as exc:  # pragma: no cover - exact runtime errors depend on local setup
            self._send_json({"error": str(exc)}, status=HTTPStatus.INTERNAL_SERVER_ERROR)
        else:
            self._send_json(response)

    def _read_json_body(self) -> dict[str, Any]:
        length = int(self.headers.get("Content-Length", "0") or "0")
        raw_body = self.rfile.read(length).decode("utf-8")
        try:
            data = json.loads(raw_body or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError("Request body must be valid JSON.") from exc
        if not isinstance(data, dict):
            raise ValueError("Request body must be a JSON object.")
        return data

    def _send_json(self, data: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_static_file(self, path: Path) -> None:
        resolved_static = STATIC_DIR.resolve()
        resolved_path = path.resolve()
        if resolved_static != resolved_path and resolved_static not in resolved_path.parents:
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return
        if not resolved_path.is_file():
            self.send_error(HTTPStatus.NOT_FOUND, "Not found")
            return

        content_type = mimetypes.guess_type(str(resolved_path))[0] or "application/octet-stream"
        body = resolved_path.read_bytes()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def serve_ui(*, index_dir: str = "agent", host: str = "127.0.0.1", port: int = 8765) -> int:
    handler_cls = type(
        "ConfiguredGraphRAGUIHandler",
        (GraphRAGUIHandler,),
        {"default_index_dir": index_dir},
    )
    server = ThreadingHTTPServer((host, port), handler_cls)
    url = f"http://{host}:{port}"
    print(f"Graph RAG UI running at {url}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nGraph RAG UI stopped.")
    finally:
        server.server_close()
    return 0
