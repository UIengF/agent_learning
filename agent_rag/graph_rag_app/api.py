from __future__ import annotations

from dataclasses import asdict, is_dataclass
import logging
from pathlib import Path
import sys
import time
from typing import Any
from uuid import uuid4

from fastapi import FastAPI, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .config import DEFAULT_CHECKPOINT_DB, DEFAULT_SESSION_ID, build_app_config
from .indexing import inspect_index, load_index
from .runtime import build_sqlite_checkpointer, run_or_resume_with_trace
from .scholar_search import run_scholar_search
from .schemas import (
    AskRequest,
    AskResponse,
    ConfigResponse,
    HealthResponse,
    RetrievalResultItem,
    RetrieveRequest,
    RetrieveResponse,
    ScholarPaper,
    ScholarSearchRequest,
    ScholarSearchResponse,
    StatusResponse,
    WebFetchRequest,
    WebSearchHit,
    WebSearchRequest,
    WebSearchResponse,
)
from .sources import extract_sources_from_messages
from .web_fetch import fetch_url
from .web_runtime import build_configured_web_search_backend


STATIC_DIR = Path(__file__).resolve().parent / "static"
LOGGER = logging.getLogger(__name__)
REQUEST_ID_HEADER = "X-Request-ID"


def _build_question_with_history(payload: AskRequest) -> tuple[str, bool]:
    history = payload.history[-4:]
    if not payload.resume or not history:
        return payload.question, False
    lines = ["Previous conversation context:"]
    for item in history:
        lines.append(f"User: {item.question}")
        lines.append(f"Assistant: {item.answer}")
    lines.append(f"Current question: {payload.question}")
    return "\n".join(lines), True


def _result_to_item(result: Any) -> RetrievalResultItem:
    if is_dataclass(result):
        data = asdict(result)
    elif isinstance(result, dict):
        data = result
    else:
        data = {
            "chunk_id": getattr(result, "chunk_id"),
            "score": getattr(result, "score"),
            "text": getattr(result, "text"),
            "document_id": getattr(result, "document_id"),
            "source_path": getattr(result, "source_path"),
            "section_title": getattr(result, "section_title", ""),
            "strategy": getattr(result, "strategy"),
        }
    return RetrievalResultItem(**data)


def _object_to_dict(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return value
    data: dict[str, Any] = {}
    for name in dir(value):
        if name.startswith("_"):
            continue
        item = getattr(value, name)
        if not callable(item):
            data[name] = item
    return data


def _build_web_search_backend():
    web_config = build_app_config(".").web
    return build_configured_web_search_backend(web_config)


def _request_id(request: Request) -> str:
    value = getattr(request.state, "request_id", "")
    return str(value) if value else ""


def _error_payload(
    code: str,
    message: str,
    details: dict[str, Any] | list[Any] | None = None,
    *,
    request_id: str = "",
) -> dict[str, Any]:
    return {
        "error": {
            "code": code,
            "message": message,
            "details": details or {},
            "request_id": request_id,
        }
    }


def create_app(*, default_index_dir: str = "agent") -> FastAPI:
    app = FastAPI(title="Graph RAG API", version="1.0.0")

    @app.middleware("http")
    async def request_context_middleware(request: Request, call_next):
        request_id = request.headers.get(REQUEST_ID_HEADER) or uuid4().hex
        request.state.request_id = request_id
        started = time.perf_counter()
        response: Response = await call_next(request)
        response.headers[REQUEST_ID_HEADER] = request_id
        elapsed_ms = int((time.perf_counter() - started) * 1000)
        LOGGER.info(
            "request_id=%s method=%s path=%s status_code=%s elapsed_ms=%s",
            request_id,
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
        )
        return response

    @app.exception_handler(RequestValidationError)
    async def validation_exception_handler(
        request: Request,
        exc: RequestValidationError,
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content=_error_payload(
                "validation_error",
                "Request validation failed.",
                {"errors": exc.errors()},
                request_id=_request_id(request),
            ),
            headers={REQUEST_ID_HEADER: _request_id(request)},
        )

    @app.exception_handler(ImportError)
    async def import_error_handler(request: Request, exc: ImportError) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content=_error_payload(
                "runtime_dependency_error",
                str(exc),
                request_id=_request_id(request),
            ),
            headers={REQUEST_ID_HEADER: _request_id(request)},
        )

    @app.exception_handler(Exception)
    async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=500,
            content=_error_payload("internal_error", str(exc), request_id=_request_id(request)),
            headers={REQUEST_ID_HEADER: _request_id(request)},
        )

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return (STATIC_DIR / "index.html").read_text(encoding="utf-8")

    @app.get("/healthz", response_model=HealthResponse)
    def healthz() -> HealthResponse:
        return HealthResponse(status="ok", service="graph-rag-api", index_dir=default_index_dir)

    @app.get("/api/status", response_model=StatusResponse)
    def status() -> StatusResponse:
        app_config = build_app_config(default_index_dir)
        from .agent import LANGGRAPH_AVAILABLE

        return StatusResponse(
            service="graph-rag-api",
            index_dir=default_index_dir,
            python_executable=sys.executable,
            langgraph_available=bool(LANGGRAPH_AVAILABLE),
            api_key_configured=bool(app_config.model.api_key),
            model_name=app_config.model.model_name,
            web_enabled=app_config.web.enabled,
            web_search_provider=app_config.web.search_provider,
            scholar_enabled=app_config.scholar.enabled,
        )

    @app.get("/api/config", response_model=ConfigResponse)
    def config() -> ConfigResponse:
        return ConfigResponse(index_dir=default_index_dir, session_id=DEFAULT_SESSION_ID)

    @app.post("/api/ask", response_model=AskResponse)
    def ask(payload: AskRequest) -> AskResponse:
        started = time.perf_counter()
        question, context_included = _build_question_with_history(payload)
        checkpointer = build_sqlite_checkpointer(DEFAULT_CHECKPOINT_DB) if payload.resume else None
        trace = run_or_resume_with_trace(
            question=question,
            index_dir=payload.index_dir,
            session_id=payload.session_id,
            checkpointer=checkpointer,
            resume=False,
            interrupt_after=None,
        )
        source_messages = trace.source_messages if trace.source_messages is not None else trace.messages
        sources = extract_sources_from_messages(source_messages)
        return AskResponse(
            answer=trace.answer,
            question=payload.question,
            index_dir=payload.index_dir,
            session_id=payload.session_id,
            resume=False,
            context_included=context_included,
            elapsed_seconds=round(time.perf_counter() - started, 2),
            sources=sources,
            retrieval_debug={
                "source_count": len(sources),
                "index_dir": payload.index_dir,
            },
        )

    @app.post("/api/retrieve", response_model=RetrieveResponse)
    def retrieve(payload: RetrieveRequest) -> RetrieveResponse:
        results = load_index(payload.index_dir).retrieve(
            payload.query,
            top_k=payload.top_k,
            strategy=payload.strategy,
        )
        items = [_result_to_item(result) for result in results]
        return RetrieveResponse(
            query=payload.query,
            index_dir=payload.index_dir,
            top_k=payload.top_k,
            strategy=payload.strategy,
            result_count=len(items),
            results=items,
        )

    @app.get("/api/index/inspect")
    def index_inspect(index_dir: str) -> dict[str, Any]:
        return inspect_index(index_dir)

    @app.post("/api/web/search", response_model=WebSearchResponse)
    def web_search(payload: WebSearchRequest) -> WebSearchResponse:
        backend = _build_web_search_backend()
        hits = backend.search(payload.query, top_k=payload.top_k)
        results = [WebSearchHit(**_object_to_dict(hit)) for hit in hits]
        return WebSearchResponse(
            query=payload.query,
            result_count=len(results),
            results=results,
            debug=getattr(backend, "last_debug", {}) or {},
        )

    @app.post("/api/web/fetch")
    def web_fetch(payload: WebFetchRequest) -> dict[str, Any]:
        return _object_to_dict(fetch_url(payload.url))

    @app.post("/api/scholar/search", response_model=ScholarSearchResponse)
    def scholar_search(payload: ScholarSearchRequest) -> ScholarSearchResponse:
        result = run_scholar_search(
            topic=payload.topic,
            count=payload.count,
            app_config=build_app_config(payload.index_dir),
        )
        data = _object_to_dict(result)
        papers = [ScholarPaper(**_object_to_dict(paper)) for paper in data.get("papers", [])]
        return ScholarSearchResponse(
            topic=str(data.get("topic", payload.topic)),
            count_requested=int(data.get("count_requested", payload.count)),
            query_count=int(data.get("query_count", 0)),
            paper_count=len(papers),
            papers=papers,
        )

    return app
