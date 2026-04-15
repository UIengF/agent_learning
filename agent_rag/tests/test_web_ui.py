from __future__ import annotations

from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from graph_rag_app.cli import main, parse_args
from graph_rag_app.web_ui import build_ask_response, parse_ask_payload


class WebUiTests(TestCase):
    def test_parse_args_supports_ui_command(self) -> None:
        args = parse_args(["ui", "--index-dir", ".\\agent", "--host", "127.0.0.1", "--port", "8765"])

        self.assertEqual(args.command, "ui")
        self.assertEqual(args.index_dir, ".\\agent")
        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.port, 8765)

    def test_main_dispatches_ui_command(self) -> None:
        with patch("graph_rag_app.cli.serve_ui", return_value=0) as serve_ui:
            exit_code = main(["ui", "--index-dir", ".\\agent", "--host", "127.0.0.1", "--port", "8765"])

        self.assertEqual(exit_code, 0)
        serve_ui.assert_called_once_with(index_dir=".\\agent", host="127.0.0.1", port=8765)

    def test_parse_ask_payload_requires_question(self) -> None:
        with self.assertRaisesRegex(ValueError, "Question is required"):
            parse_ask_payload({"index_dir": ".\\agent"})

    def test_parse_ask_payload_requires_index_dir(self) -> None:
        with self.assertRaisesRegex(ValueError, "Index directory is required"):
            parse_ask_payload({"question": "What is this?"})

    def test_parse_ask_payload_uses_defaults(self) -> None:
        payload = parse_ask_payload({"question": "What is this?", "index_dir": ".\\agent"})

        self.assertEqual(payload.question, "What is this?")
        self.assertEqual(payload.index_dir, ".\\agent")
        self.assertEqual(payload.session_id, "graph_rag_default")
        self.assertFalse(payload.resume)

    def test_build_ask_response_calls_runtime(self) -> None:
        with patch("graph_rag_app.web_ui.build_sqlite_checkpointer", return_value="checkpointer"):
            with patch("graph_rag_app.web_ui.run_or_resume", return_value="final answer") as run_or_resume:
                response = build_ask_response(
                    {
                        "question": "What is this?",
                        "index_dir": ".\\agent",
                        "session_id": "session-1",
                        "resume": True,
                    }
                )

        self.assertEqual(response["answer"], "final answer")
        self.assertEqual(response["question"], "What is this?")
        self.assertEqual(response["index_dir"], ".\\agent")
        self.assertEqual(response["session_id"], "session-1")
        self.assertFalse(response["resume"])
        self.assertGreaterEqual(response["elapsed_seconds"], 0)
        run_or_resume.assert_called_once_with(
            question="What is this?",
            index_dir=".\\agent",
            session_id="session-1",
            checkpointer="checkpointer",
            resume=False,
            interrupt_after=None,
        )

    def test_build_ask_response_adds_history_without_checkpoint_resume(self) -> None:
        with patch("graph_rag_app.web_ui.build_sqlite_checkpointer", return_value="checkpointer"):
            with patch("graph_rag_app.web_ui.run_or_resume", return_value="final answer") as run_or_resume:
                response = build_ask_response(
                    {
                        "question": "What changed?",
                        "index_dir": ".\\agent",
                        "session_id": "session-1",
                        "resume": True,
                        "history": [
                            {"question": "What is Agent?", "answer": "An agent uses tools."},
                        ],
                    }
                )

        self.assertTrue(response["context_included"])
        call_kwargs = run_or_resume.call_args.kwargs
        self.assertFalse(call_kwargs["resume"])
        self.assertIn("Previous conversation context:", call_kwargs["question"])
        self.assertIn("User: What is Agent?", call_kwargs["question"])
        self.assertIn("Assistant: An agent uses tools.", call_kwargs["question"])
        self.assertIn("Current question: What changed?", call_kwargs["question"])

    def test_static_app_renders_answers_as_markdown(self) -> None:
        app_js = Path("graph_rag_app/static/app.js").read_text(encoding="utf-8")

        self.assertIn("function renderMarkdown", app_js)
        self.assertIn('class="markdown-answer"', app_js)
        self.assertNotIn("<pre>${escapeHtml(data.answer)}</pre>", app_js)

    def test_static_page_uses_project_friendly_copy(self) -> None:
        index_html = Path("graph_rag_app/static/index.html").read_text(encoding="utf-8")

        self.assertIn("<title>Agent 资料助手</title>", index_html)
        self.assertIn('<h1 id="app-title">Agent 资料助手</h1>', index_html)
        self.assertIn("围绕本地 Agent 资料提问，必要时自动补充网页和论文线索。", index_html)
        self.assertNotIn("知识库问答", index_html)
        self.assertNotIn("Graph RAG Agent", index_html)

    def test_static_app_sends_history_for_context_without_resume(self) -> None:
        app_js = Path("graph_rag_app/static/app.js").read_text(encoding="utf-8")

        self.assertIn("const conversationHistory = [];", app_js)
        self.assertIn("history: resumeInput.checked ? conversationHistory.slice(-4) : []", app_js)
        self.assertIn("conversationHistory.push", app_js)

