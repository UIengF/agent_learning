from __future__ import annotations

import argparse
from dataclasses import replace
from dataclasses import asdict
from datetime import datetime
import json
import os
from pathlib import Path

from .config import (
    DEFAULT_CHECKPOINT_DB,
    DEFAULT_SESSION_ID,
    IndexBuildConfig,
    RetrievalRuntimeConfig,
    build_app_config,
    parse_bool_env,
)
from .agent_evaluation import run_evaluation_dataset
from .eval_datasets import load_evaluation_dataset
from .eval_reporting import load_evaluation_report, save_evaluation_report
from .indexing import (
    build_index,
    inspect_index,
    load_index,
    resolve_existing_index_for_kb,
)
from .jobs import BackgroundJobManager, JobNotFound, job_to_dict
from .runtime import build_sqlite_checkpointer, run_or_resume
from .scholar_export import save_scholar_search_markdown
from .scholar_search import run_scholar_search
from .web_fetch import fetch_url
from .web_runtime import build_configured_web_search_backend
from .server import serve_fastapi

DEFAULT_QUESTION = "What does the knowledge base say about Anthropic agent technology?"


def _bounded_int(minimum: int, maximum: int):
    def parser(value: str) -> int:
        parsed = int(value)
        if parsed < minimum or parsed > maximum:
            raise argparse.ArgumentTypeError(
                f"Expected an integer between {minimum} and {maximum}."
            )
        return parsed

    return parser


def _add_runtime_args(parser: argparse.ArgumentParser, *, include_index_dir: bool) -> None:
    if include_index_dir:
        parser.add_argument("--index-dir", required=True)
    parser.add_argument("--question", default=os.getenv("RAG_QUESTION", ""))
    parser.add_argument("--session-id", default=os.getenv("RAG_SESSION_ID", DEFAULT_SESSION_ID))
    parser.add_argument(
        "--checkpoint-db",
        default=os.getenv("RAG_CHECKPOINT_DB", DEFAULT_CHECKPOINT_DB),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=parse_bool_env("RAG_RESUME", False),
        help="Resume from an existing checkpoint for the same session id.",
    )
    parser.add_argument(
        "--interrupt-after",
        action="append",
        default=None,
        help="Interrupt after the given node name. Repeatable.",
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or resume the graph_rag LangGraph agent.")
    subparsers = parser.add_subparsers(dest="command")

    index_parser = subparsers.add_parser("index", help="Build or inspect a retrieval index.")
    index_subparsers = index_parser.add_subparsers(dest="index_command")

    index_build = index_subparsers.add_parser("build", help="Build a retrieval index.")
    index_build.add_argument("--kb-path", required=True)
    index_build.add_argument("--output-dir", required=True)
    index_build.add_argument("--chunk-size", type=int, default=None)
    index_build.add_argument("--chunk-overlap", type=int, default=None)
    index_build.add_argument("--keyword-weight", type=float, default=None)

    index_inspect = index_subparsers.add_parser(
        "inspect", help="Inspect an existing retrieval index."
    )
    index_inspect.add_argument("--index-dir", required=True)

    query_parser = subparsers.add_parser("query", help="Run retrieval against an existing index.")
    query_subparsers = query_parser.add_subparsers(dest="query_command")
    query_run = query_subparsers.add_parser("run", help="Run a retrieval query.")
    query_run.add_argument("--index-dir", required=True)
    query_run.add_argument("--question", required=True)
    query_run.add_argument("--top-k", type=int, default=None)
    query_run.add_argument("--strategy", choices=["sparse", "dense", "hybrid"], default=None)

    web_parser = subparsers.add_parser("web", help="Run web search and fetch debug commands.")
    web_subparsers = web_parser.add_subparsers(dest="web_command", required=True)

    web_search = web_subparsers.add_parser("search", help="Search the public web.")
    web_search.add_argument("--query", required=True)
    web_search.add_argument("--top-k", type=int, default=None)

    web_fetch = web_subparsers.add_parser("fetch", help="Fetch a public web page.")
    web_fetch.add_argument("--url", required=True)

    scholar_parser = subparsers.add_parser("scholar", help="Run Google Scholar search commands.")
    scholar_subparsers = scholar_parser.add_subparsers(dest="scholar_command", required=True)
    scholar_search = scholar_subparsers.add_parser(
        "search", help="Search Google Scholar from a topic."
    )
    scholar_search.add_argument("--topic", required=True)
    scholar_search.add_argument("--count", type=_bounded_int(1, 20), default=5)
    scholar_search.add_argument("--save-md", action="store_true", default=False)
    scholar_search.add_argument("--output-dir", default="agent/scholar")

    ask_parser = subparsers.add_parser(
        "ask",
        help="Run or resume the graph_rag LangGraph agent with an existing index.",
    )
    _add_runtime_args(ask_parser, include_index_dir=True)

    ui_parser = subparsers.add_parser("ui", help="Run the browser question-answering UI.")
    ui_parser.add_argument("--index-dir", default="agent")
    ui_parser.add_argument("--host", default="127.0.0.1")
    ui_parser.add_argument("--port", type=int, default=8765)

    serve_parser = subparsers.add_parser("serve", help="Run the FastAPI backend service.")
    serve_parser.add_argument("--index-dir", default="agent")
    serve_parser.add_argument("--host", default="127.0.0.1")
    serve_parser.add_argument("--port", type=int, default=8765)
    serve_parser.add_argument("--reload", action="store_true", default=False)

    eval_parser = subparsers.add_parser("eval", help="Run evaluation datasets and persist reports.")
    eval_subparsers = eval_parser.add_subparsers(dest="eval_command", required=True)
    eval_run = eval_subparsers.add_parser("run", help="Run a local evaluation dataset.")
    eval_run.add_argument("--dataset", required=True)
    eval_run.add_argument("--index-dir", default="agent")
    eval_run.add_argument("--output-dir", default=str(Path("runtime") / "evals"))
    eval_run.add_argument("--baseline-run", default=None)
    eval_run.add_argument("--tag", action="append", default=None)
    judge_toggle = eval_run.add_mutually_exclusive_group()
    judge_toggle.add_argument("--judge-enabled", action="store_true", default=False)
    judge_toggle.add_argument("--judge-disabled", action="store_true", default=False)
    eval_run.add_argument("--judge-model", default=None)
    eval_run.add_argument("--judge-api-base", default=None)
    eval_run.add_argument("--judge-api-key", default=None)

    job_parser = subparsers.add_parser("job", help="Inspect background job status and logs.")
    job_subparsers = job_parser.add_subparsers(dest="job_command", required=True)
    job_status = job_subparsers.add_parser("status", help="Read a background job record.")
    job_status.add_argument("--job-id", required=True)
    job_status.add_argument("--runtime-dir", default="runtime/jobs")
    job_log = job_subparsers.add_parser("log", help="Read a background job log.")
    job_log.add_argument("--job-id", required=True)
    job_log.add_argument("--runtime-dir", default="runtime/jobs")
    job_log.add_argument("--max-chars", type=int, default=12000)

    _add_runtime_args(parser, include_index_dir=False)
    parser.add_argument(
        "--kb-path",
        default=os.getenv("RAG_KB_PATH", os.getenv("RAG_DOCX_PATH", "")),
    )
    parser.add_argument("--docx-path", default="", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def _print_json(data: object) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2))


def _handle_index_build(args: argparse.Namespace) -> int:
    config = IndexBuildConfig(
        chunk_size=args.chunk_size or IndexBuildConfig().chunk_size,
        chunk_overlap=args.chunk_overlap or IndexBuildConfig().chunk_overlap,
        keyword_weight=(
            args.keyword_weight
            if args.keyword_weight is not None
            else IndexBuildConfig().keyword_weight
        ),
    )
    report = build_index(kb_path=args.kb_path, output_dir=args.output_dir, config=config)
    _print_json(
        {
            "index_dir": report.index_dir,
            "chunk_count": report.chunk_count,
            "document_count": report.document_count,
            "built_at": report.built_at,
            "index_db_path": report.index_db_path,
            "manifest_path": report.manifest_path,
        }
    )
    return 0


def _handle_index_inspect(args: argparse.Namespace) -> int:
    _print_json(inspect_index(args.index_dir))
    return 0


def _handle_query_run(args: argparse.Namespace) -> int:
    retrieval_config = RetrievalRuntimeConfig(
        top_k=args.top_k or RetrievalRuntimeConfig().top_k,
        strategy=args.strategy or RetrievalRuntimeConfig().strategy,
    )
    results = load_index(args.index_dir).retrieve(
        args.question,
        top_k=retrieval_config.top_k,
        strategy=retrieval_config.strategy,
    )
    _print_json([result.__dict__ for result in results])
    return 0


def _build_web_config(args: argparse.Namespace):
    kb_path = args.kb_path or args.docx_path or "."
    return build_app_config(kb_path).web


def _handle_web_search(args: argparse.Namespace) -> int:
    web_config = _build_web_config(args)
    backend = build_configured_web_search_backend(web_config)
    top_k = args.top_k if args.top_k is not None else web_config.search_top_k
    results = backend.search(args.query, top_k=top_k)
    _print_json(
        {
            "query": args.query,
            "result_count": len(results),
            "results": [asdict(result) for result in results],
        }
    )
    return 0


def _handle_web_fetch(args: argparse.Namespace) -> int:
    web_config = _build_web_config(args)
    result = fetch_url(
        args.url,
        timeout_seconds=web_config.fetch_timeout_seconds,
        max_bytes=web_config.fetch_max_bytes,
        max_chars=web_config.fetch_max_chars,
        user_agent=web_config.user_agent,
    )
    _print_json(asdict(result))
    return 0


def _handle_scholar_search(args: argparse.Namespace) -> int:
    app_config = build_app_config(args.kb_path or args.docx_path or ".")
    result = run_scholar_search(
        topic=args.topic,
        count=args.count,
        app_config=app_config,
    )
    _print_json(asdict(result))
    if args.save_md:
        output_path = save_scholar_search_markdown(
            result,
            output_dir=args.output_dir,
            count_requested=args.count,
            now=datetime.now().astimezone(),
        )
        print(f"Saved Markdown to {output_path}")
    return 0


def _handle_ask(args: argparse.Namespace) -> int:
    checkpointer = build_sqlite_checkpointer(args.checkpoint_db)
    question = args.question or DEFAULT_QUESTION
    answer = run_or_resume(
        question=question,
        index_dir=args.index_dir,
        session_id=args.session_id,
        checkpointer=checkpointer,
        resume=args.resume,
        interrupt_after=args.interrupt_after,
    )
    print(answer)
    return 0


def _handle_eval_run(args: argparse.Namespace) -> int:
    include_tags = set(args.tag) if args.tag else None
    dataset = load_evaluation_dataset(Path(args.dataset), include_tags=include_tags)
    app_config = override_eval_judge_config(build_app_config(args.index_dir), args)
    report = run_evaluation_dataset(dataset, index_dir=args.index_dir, app_config=app_config)
    baseline_report = load_evaluation_report(Path(args.baseline_run)) if args.baseline_run else None
    artifact = save_evaluation_report(
        report,
        output_root=Path(args.output_dir),
        baseline_report=baseline_report,
    )
    print(f"Saved evaluation report to {artifact.run_dir}")
    return 0


def _handle_job_status(args: argparse.Namespace) -> int:
    manager = BackgroundJobManager(args.runtime_dir)
    try:
        _print_json(job_to_dict(manager.get(args.job_id)))
    except JobNotFound:
        _print_json({"error": "job_not_found", "job_id": args.job_id})
        return 1
    return 0


def _handle_job_log(args: argparse.Namespace) -> int:
    manager = BackgroundJobManager(args.runtime_dir)
    try:
        print(manager.read_log(args.job_id, max_chars=args.max_chars))
    except JobNotFound:
        _print_json({"error": "job_not_found", "job_id": args.job_id})
        return 1
    return 0


def override_eval_judge_config(app_config, args: argparse.Namespace):
    enabled = app_config.eval_judge.enabled
    if getattr(args, "judge_enabled", False):
        enabled = True
    if getattr(args, "judge_disabled", False):
        enabled = False
    eval_judge = replace(
        app_config.eval_judge,
        enabled=enabled,
        model_name=args.judge_model or app_config.eval_judge.model_name,
        api_base=args.judge_api_base or app_config.eval_judge.api_base,
        api_key=args.judge_api_key or app_config.eval_judge.api_key,
    )
    return replace(app_config, eval_judge=eval_judge)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "index" and args.index_command == "build":
        return _handle_index_build(args)
    if args.command == "index" and args.index_command == "inspect":
        return _handle_index_inspect(args)
    if args.command == "query" and args.query_command == "run":
        return _handle_query_run(args)
    if args.command == "web" and args.web_command == "search":
        return _handle_web_search(args)
    if args.command == "web" and args.web_command == "fetch":
        return _handle_web_fetch(args)
    if args.command == "scholar" and args.scholar_command == "search":
        return _handle_scholar_search(args)
    if args.command == "ask":
        return _handle_ask(args)
    if args.command == "eval" and args.eval_command == "run":
        return _handle_eval_run(args)
    if args.command == "job" and args.job_command == "status":
        return _handle_job_status(args)
    if args.command == "job" and args.job_command == "log":
        return _handle_job_log(args)
    if args.command == "ui":
        return serve_fastapi(index_dir=args.index_dir, host=args.host, port=args.port, reload=False)
    if args.command == "serve":
        return serve_fastapi(
            index_dir=args.index_dir,
            host=args.host,
            port=args.port,
            reload=args.reload,
        )

    kb_path = args.kb_path or args.docx_path
    if not kb_path:
        raise EnvironmentError("Missing RAG_KB_PATH environment variable.")

    args.index_dir = str(resolve_existing_index_for_kb(kb_path))
    return _handle_ask(args)
