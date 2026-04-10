from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os

from .config import (
    DEFAULT_CHECKPOINT_DB,
    DEFAULT_SESSION_ID,
    IndexBuildConfig,
    RetrievalRuntimeConfig,
    build_app_config,
    parse_bool_env,
)
from .indexing import (
    build_index,
    inspect_index,
    load_index,
    resolve_existing_index_for_kb,
)
from .runtime import build_sqlite_checkpointer, run_or_resume
from .web_fetch import fetch_url
from .web_search import DuckDuckGoHtmlSearchBackend, MultiQuerySearchBackend

DEFAULT_QUESTION = "What does the knowledge base say about Anthropic agent technology?"


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

    index_inspect = index_subparsers.add_parser("inspect", help="Inspect an existing retrieval index.")
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

    ask_parser = subparsers.add_parser(
        "ask",
        help="Run or resume the graph_rag LangGraph agent with an existing index.",
    )
    _add_runtime_args(ask_parser, include_index_dir=True)

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
    if web_config.search_provider != "duckduckgo_html":
        raise ValueError(f"Unsupported web search provider: {web_config.search_provider}")
    backend = MultiQuerySearchBackend(
        DuckDuckGoHtmlSearchBackend(
            timeout_seconds=web_config.fetch_timeout_seconds,
            user_agent=web_config.user_agent,
        )
    )
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
    if args.command == "ask":
        return _handle_ask(args)

    kb_path = args.kb_path or args.docx_path
    if not kb_path:
        raise EnvironmentError("Missing RAG_KB_PATH environment variable.")

    args.index_dir = str(resolve_existing_index_for_kb(kb_path))
    return _handle_ask(args)
