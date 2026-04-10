from __future__ import annotations

from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from graph_rag_app.cli import main, parse_args


class CliStandardModeTests(TestCase):
    def test_parse_args_supports_ask_command(self) -> None:
        args = parse_args(["ask", "--index-dir", "index-dir", "--question", "What is this?"])

        self.assertEqual(args.command, "ask")
        self.assertEqual(args.index_dir, "index-dir")
        self.assertEqual(args.question, "What is this?")

    def test_main_ask_uses_run_or_resume_with_index_dir(self) -> None:
        with patch("graph_rag_app.cli.build_sqlite_checkpointer", return_value="checkpointer"):
            with patch("graph_rag_app.cli.run_or_resume", return_value="final answer") as run_or_resume:
                exit_code = main(
                    [
                        "ask",
                        "--index-dir",
                        "index-dir",
                        "--question",
                        "What is this?",
                        "--session-id",
                        "session-1",
                    ]
                )

        self.assertEqual(exit_code, 0)
        run_or_resume.assert_called_once_with(
            question="What is this?",
            index_dir="index-dir",
            session_id="session-1",
            checkpointer="checkpointer",
            resume=False,
            interrupt_after=None,
        )

    def test_legacy_default_path_resolves_existing_index_before_ask(self) -> None:
        with patch(
            "graph_rag_app.cli.resolve_existing_index_for_kb",
            return_value=Path("existing-index"),
        ) as resolve_existing_index:
            with patch("graph_rag_app.cli.build_sqlite_checkpointer", return_value="checkpointer"):
                with patch("graph_rag_app.cli.run_or_resume", return_value="final answer") as run_or_resume:
                    exit_code = main(["--kb-path", "kb-path", "--question", "What is this?"])

        self.assertEqual(exit_code, 0)
        resolve_existing_index.assert_called_once_with("kb-path")
        run_or_resume.assert_called_once_with(
            question="What is this?",
            index_dir=str(Path("existing-index")),
            session_id="graph_rag_default",
            checkpointer="checkpointer",
            resume=False,
            interrupt_after=None,
        )
