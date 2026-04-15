from __future__ import annotations

import sys
from pathlib import Path


CURRENT_DIR = Path(__file__).resolve().parent
MAIN_DIR = CURRENT_DIR / "main"
if str(MAIN_DIR) not in sys.path:
    sys.path.insert(0, str(MAIN_DIR))

from graph_rag_app import (  # noqa: E402
    DashScopeEmbeddingClient,
    build_index,
    LocalRAGStore,
    LocalRAGRetrieveTool,
    build_agent,
    build_sqlite_checkpointer,
    load_index,
    load_corpus_text,
    parse_args,
    run_demo,
    run_or_resume,
)
from graph_rag_app.cli import main  # noqa: E402


__all__ = [
    "DashScopeEmbeddingClient",
    "build_index",
    "LocalRAGStore",
    "LocalRAGRetrieveTool",
    "build_agent",
    "build_sqlite_checkpointer",
    "load_index",
    "load_corpus_text",
    "main",
    "parse_args",
    "run_demo",
    "run_or_resume",
]


if __name__ == "__main__":
    raise SystemExit(main())
