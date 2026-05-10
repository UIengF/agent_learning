from __future__ import annotations

import argparse
from pathlib import Path

from .agent import build_agent
from .config import DEFAULT_CHECKPOINT_DB, DEFAULT_SESSION_ID, build_app_config
from .report_generator import generate_report, save_report
from .runtime import (
    build_sqlite_checkpointer,
    extract_messages_from_state,
    get_graph_state,
    get_thread_config,
)


def add_report_parser(subparsers) -> None:
    """Add the report subcommand to argparse."""

    parser = subparsers.add_parser(
        "report",
        help="Generate a Markdown academic report from a completed agent session.",
    )
    parser.add_argument("--session", "--session-id", dest="session_id", default=DEFAULT_SESSION_ID)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--checkpoint-db", default=DEFAULT_CHECKPOINT_DB)
    parser.add_argument("--output-dir", default="reports")
    parser.add_argument("--slug", default="")
    parser.add_argument("--model", default="")


def _load_session_messages(args: argparse.Namespace) -> list:
    checkpointer = build_sqlite_checkpointer(args.checkpoint_db)
    if checkpointer is None:
        raise RuntimeError("SQLite checkpoint support is not available in this environment.")
    app_config = build_app_config(
        args.index_dir,
        session_id=args.session_id,
        checkpoint_db=args.checkpoint_db,
    )
    agent = build_agent(args.index_dir, checkpointer=checkpointer, app_config=app_config)
    state = get_graph_state(agent.graph, get_thread_config(args.session_id))
    messages = extract_messages_from_state(state)
    if not messages:
        raise RuntimeError(f"No checkpoint messages found for session '{args.session_id}'.")
    return messages


def handle_report(args) -> int:
    """Load a session from checkpoint, generate a report, and save it to file."""

    messages = _load_session_messages(args)
    markdown = generate_report(
        messages=messages,
        source_messages=messages,
        session_id=args.session_id,
        index_dir=args.index_dir,
        model_name=args.model,
    )
    path = save_report(markdown, Path(args.output_dir), slug=args.slug or args.session_id)
    print(f"Saved report to {path}")
    return 0
