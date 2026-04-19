from __future__ import annotations

import json
import os
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from fastapi.testclient import TestClient

from graph_rag_app.api import create_app
from graph_rag_app.config import build_app_config
from graph_rag_app.cli import main, parse_args
from graph_rag_app.runtime import AgentRunTrace, get_graph_state


class FakeGraphWithoutCheckpointer:
    def get_state(self, config):
        raise ValueError("No checkpointer set")


class RuntimeStateTests(TestCase):
    def test_get_graph_state_treats_missing_checkpointer_as_no_state(self) -> None:
        state = get_graph_state(FakeGraphWithoutCheckpointer(), {"configurable": {"thread_id": "fresh"}})

        self.assertIsNone(state)


class FastApiAppTests(TestCase):
    def test_healthz_returns_service_status(self) -> None:
        client = TestClient(create_app(default_index_dir="agent"))

        response = client.get("/healthz")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["x-request-id"])
        data = response.json()
        self.assertEqual(data["status"], "ok")
        self.assertEqual(data["service"], "graph-rag-api")
        self.assertEqual(data["index_dir"], "agent")

    def test_status_returns_backend_runtime_metadata_without_secret_values(self) -> None:
        client = TestClient(create_app(default_index_dir="agent"))

        response = client.get("/api/status")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["service"], "graph-rag-api")
        self.assertEqual(data["index_dir"], "agent")
        self.assertIn("python_executable", data)
        self.assertIn("langgraph_available", data)
        self.assertIn("api_key_configured", data)
        self.assertIn("web_search_provider", data)
        self.assertNotIn("api_key", data)
        self.assertNotIn("sk-", str(data))

    def test_config_returns_default_index_and_session(self) -> None:
        client = TestClient(create_app(default_index_dir="agent"))

        response = client.get("/api/config")

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["index_dir"], "agent")
        self.assertEqual(data["session_id"], "graph_rag_default")

    def test_ask_requires_question(self) -> None:
        client = TestClient(create_app(default_index_dir="agent"))

        response = client.post("/api/ask", json={"index_dir": "agent"})

        self.assertEqual(response.status_code, 422)
        data = response.json()
        self.assertEqual(data["error"]["code"], "validation_error")
        self.assertIn("question", str(data["error"]["details"]))

    def test_ask_runtime_error_uses_unified_error_response(self) -> None:
        client = TestClient(create_app(default_index_dir="agent"))

        with patch(
            "graph_rag_app.api.run_or_resume_with_trace",
            side_effect=ImportError("missing dependency"),
        ):
            response = client.post(
                "/api/ask",
                json={"question": "What is this?", "index_dir": "agent"},
            )

        self.assertEqual(response.status_code, 500)
        data = response.json()
        self.assertTrue(response.headers["x-request-id"])
        self.assertEqual(data["error"]["request_id"], response.headers["x-request-id"])
        self.assertEqual(data["error"]["code"], "runtime_dependency_error")
        self.assertEqual(data["error"]["message"], "missing dependency")
        self.assertEqual(data["error"]["details"], {})

    def test_ask_calls_runtime_with_history_context(self) -> None:
        client = TestClient(create_app(default_index_dir="agent"))

        with patch("graph_rag_app.api.build_sqlite_checkpointer", return_value="checkpointer"):
            with patch(
                "graph_rag_app.api.run_or_resume_with_trace",
                return_value=AgentRunTrace(answer="final answer", messages=[]),
            ) as run_or_resume:
                response = client.post(
                    "/api/ask",
                    json={
                        "question": "What changed?",
                        "index_dir": "agent",
                        "session_id": "session-1",
                        "resume": True,
                        "history": [
                            {"question": "What is Agent?", "answer": "An agent uses tools."},
                        ],
                    },
                )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["answer"], "final answer")
        self.assertEqual(data["sources"], [])
        self.assertEqual(data["retrieval_debug"]["source_count"], 0)
        self.assertTrue(data["context_included"])
        call_kwargs = run_or_resume.call_args.kwargs
        self.assertEqual(call_kwargs["index_dir"], "agent")
        self.assertEqual(call_kwargs["session_id"], "session-1")
        self.assertFalse(call_kwargs["resume"])
        self.assertIn("Previous conversation context:", call_kwargs["question"])
        self.assertIn("Current question: What changed?", call_kwargs["question"])

    def test_ask_without_resume_does_not_use_checkpoint_history(self) -> None:
        client = TestClient(create_app(default_index_dir="agent"))

        with patch("graph_rag_app.api.build_sqlite_checkpointer", return_value="checkpointer") as build_checkpointer:
            with patch(
                "graph_rag_app.api.run_or_resume_with_trace",
                return_value=AgentRunTrace(answer="fresh answer", messages=[]),
            ) as run_or_resume:
                response = client.post(
                    "/api/ask",
                    json={
                        "question": "Fresh question",
                        "index_dir": "agent",
                        "session_id": "graph_rag_default",
                        "resume": False,
                    },
                )

        self.assertEqual(response.status_code, 200)
        build_checkpointer.assert_not_called()
        call_kwargs = run_or_resume.call_args.kwargs
        self.assertIsNone(call_kwargs["checkpointer"])
        self.assertEqual(call_kwargs["session_id"], "graph_rag_default")
        self.assertFalse(call_kwargs["resume"])

    def test_ask_returns_local_and_web_sources_from_trace_messages(self) -> None:
        client = TestClient(create_app(default_index_dir="agent"))
        local_payload = {
            "results": [
                {
                    "chunk_id": 1,
                    "score": 0.8,
                    "text": "Local evidence.",
                    "document_id": "doc-1",
                    "source_path": "Agent/local.md",
                    "section_title": "Overview",
                    "strategy": "hybrid",
                }
            ]
        }
        web_payload = {
            "results": [
                {
                    "title": "Web evidence",
                    "url": "https://example.com/agent",
                    "snippet": "External web evidence.",
                    "source": "duckduckgo",
                    "rank": 1,
                }
            ]
        }
        fetched_payload = {
            "url": "https://example.com/agent",
            "final_url": "https://example.com/agent",
            "title": "Web evidence",
            "text": "Fetched page body.",
            "status_code": 200,
        }
        trace = AgentRunTrace(
            answer="answer with sources",
            messages=[
                {"role": "tool", "name": "local_rag_retrieve", "content": json.dumps(local_payload)},
                {"role": "tool", "name": "web_search", "content": json.dumps(web_payload)},
                {"role": "tool", "name": "web_fetch", "content": json.dumps(fetched_payload)},
            ],
        )

        with patch("graph_rag_app.api.build_sqlite_checkpointer", return_value="checkpointer"):
            with patch("graph_rag_app.api.run_or_resume_with_trace", return_value=trace):
                response = client.post(
                    "/api/ask",
                    json={"question": "What is this?", "index_dir": "agent"},
                )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["answer"], "answer with sources")
        self.assertEqual(data["retrieval_debug"]["source_count"], 2)
        self.assertEqual(data["sources"][0]["source_type"], "local")
        self.assertEqual(data["sources"][0]["source_path"], "Agent/local.md")
        self.assertEqual(data["sources"][1]["source_type"], "web")
        self.assertEqual(data["sources"][1]["url"], "https://example.com/agent")
        self.assertEqual(data["sources"][1]["text"], "Fetched page body.")

    def test_ask_does_not_return_unfetched_web_search_candidates(self) -> None:
        client = TestClient(create_app(default_index_dir="agent"))
        web_payload = {
            "results": [
                {
                    "title": "Candidate A",
                    "url": "https://example.com/a",
                    "snippet": "Candidate snippet A.",
                    "source": "duckduckgo",
                    "rank": 1,
                },
                {
                    "title": "Candidate B",
                    "url": "https://example.com/b",
                    "snippet": "Candidate snippet B.",
                    "source": "duckduckgo",
                    "rank": 2,
                },
            ]
        }
        fetched_payload = {
            "url": "https://example.com/b",
            "final_url": "https://example.com/b",
            "title": "Candidate B",
            "text": "Chosen page body.",
            "status_code": 200,
        }
        trace = AgentRunTrace(
            answer="answer with selected web source",
            messages=[
                {"role": "tool", "name": "web_search", "content": json.dumps(web_payload)},
                {"role": "tool", "name": "web_fetch", "content": json.dumps(fetched_payload)},
            ],
        )

        with patch("graph_rag_app.api.run_or_resume_with_trace", return_value=trace):
            response = client.post(
                "/api/ask",
                json={"question": "What is this?", "index_dir": "agent"},
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["retrieval_debug"]["source_count"], 1)
        self.assertEqual(len(data["sources"]), 1)
        self.assertEqual(data["sources"][0]["url"], "https://example.com/b")

    def test_ask_sources_ignore_prior_session_messages(self) -> None:
        client = TestClient(create_app(default_index_dir="agent"))
        old_payload = {
            "results": [
                {
                    "chunk_id": 1,
                    "score": 0.9,
                    "text": "Old unrelated evidence.",
                    "document_id": "doc-old",
                    "source_path": "Anthropic/old.md",
                    "section_title": "Old",
                    "strategy": "hybrid",
                }
            ]
        }
        current_payload = {
            "results": [
                {
                    "chunk_id": 2,
                    "score": 0.8,
                    "text": "Current OpenAI evidence.",
                    "document_id": "doc-current",
                    "source_path": "OpenAI/current.md",
                    "section_title": "Current",
                    "strategy": "hybrid",
                }
            ]
        }
        trace = AgentRunTrace(
            answer="answer with current source",
            messages=[
                {"role": "tool", "name": "local_rag_retrieve", "content": json.dumps(old_payload)},
                {"role": "tool", "name": "local_rag_retrieve", "content": json.dumps(current_payload)},
            ],
            source_messages=[
                {"role": "tool", "name": "local_rag_retrieve", "content": json.dumps(current_payload)},
            ],
        )

        with patch("graph_rag_app.api.run_or_resume_with_trace", return_value=trace):
            response = client.post(
                "/api/ask",
                json={"question": "What is this?", "index_dir": "agent"},
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["retrieval_debug"]["source_count"], 1)
        self.assertEqual(data["sources"][0]["source_path"], "OpenAI/current.md")

    def test_retrieve_returns_ranked_results(self) -> None:
        class FakeResult:
            chunk_id = 7
            score = 0.9
            text = "Agent systems use tools."
            document_id = "doc-1"
            source_path = "agent.md"
            section_title = "Tools"
            strategy = "hybrid"

        class FakeIndex:
            def retrieve(self, query: str, top_k: int, strategy: str):
                self.query = query
                self.top_k = top_k
                self.strategy = strategy
                return [FakeResult()]

        client = TestClient(create_app(default_index_dir="agent"))

        with patch("graph_rag_app.api.load_index", return_value=FakeIndex()) as load_index:
            response = client.post(
                "/api/retrieve",
                json={"query": "agent tools", "index_dir": "agent", "top_k": 1, "strategy": "hybrid"},
            )

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["query"], "agent tools")
        self.assertEqual(data["result_count"], 1)
        self.assertEqual(data["results"][0]["source_path"], "agent.md")
        load_index.assert_called_once_with("agent")

    def test_index_inspect_returns_index_metadata(self) -> None:
        client = TestClient(create_app(default_index_dir="agent"))

        with patch("graph_rag_app.api.inspect_index", return_value={"chunk_count": 3}):
            response = client.get("/api/index/inspect", params={"index_dir": "agent"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"chunk_count": 3})

    def test_web_search_returns_ranked_hits(self) -> None:
        class FakeHit:
            title = "Agent docs"
            url = "https://example.com/agent"
            snippet = "Agent documentation"

        class FakeBackend:
            last_debug = {"selected_provider": "fake"}

            def search(self, query: str, top_k: int):
                return [FakeHit()]

        client = TestClient(create_app(default_index_dir="agent"))

        with patch("graph_rag_app.api._build_web_search_backend", return_value=FakeBackend()):
            response = client.post("/api/web/search", json={"query": "agent", "top_k": 1})

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["query"], "agent")
        self.assertEqual(data["result_count"], 1)
        self.assertEqual(data["results"][0]["url"], "https://example.com/agent")
        self.assertEqual(data["debug"], {"selected_provider": "fake"})

    def test_web_fetch_returns_page_payload(self) -> None:
        client = TestClient(create_app(default_index_dir="agent"))

        with patch(
            "graph_rag_app.api.fetch_url",
            return_value={
                "url": "https://example.com",
                "status_code": 200,
                "title": "Example",
                "text": "Example body",
                "truncated": False,
                "error": "",
            },
        ):
            response = client.post("/api/web/fetch", json={"url": "https://example.com"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["title"], "Example")

    def test_scholar_search_returns_paper_metadata(self) -> None:
        class FakePaper:
            title = "Agent Paper"
            link = "https://example.com/paper"
            snippet = "Paper abstract"
            publication_info = "2026"
            cited_by = 10

        class FakeResult:
            topic = "agent"
            count_requested = 1
            query_count = 1
            papers = [FakePaper()]

        client = TestClient(create_app(default_index_dir="agent"))

        with patch("graph_rag_app.api.build_app_config", return_value="config"):
            with patch("graph_rag_app.api.run_scholar_search", return_value=FakeResult()):
                response = client.post("/api/scholar/search", json={"topic": "agent", "count": 1})

        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["topic"], "agent")
        self.assertEqual(data["paper_count"], 1)
        self.assertEqual(data["papers"][0]["title"], "Agent Paper")

    def test_static_root_serves_index_html(self) -> None:
        client = TestClient(create_app(default_index_dir="agent"))

        response = client.get("/")

        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("<title>Agent 资料助手</title>", response.text)

    def test_request_logging_records_method_path_status_and_request_id(self) -> None:
        client = TestClient(create_app(default_index_dir="agent"))

        with self.assertLogs("graph_rag_app.api", level="INFO") as logs:
            response = client.get("/healthz")

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["x-request-id"])
        joined_logs = "\n".join(logs.output)
        self.assertIn("method=GET", joined_logs)
        self.assertIn("path=/healthz", joined_logs)
        self.assertIn("status_code=200", joined_logs)
        self.assertIn(f"request_id={response.headers['x-request-id']}", joined_logs)


class FastApiCliTests(TestCase):
    def test_parse_args_supports_serve_command(self) -> None:
        args = parse_args(["serve", "--index-dir", ".\\agent", "--host", "127.0.0.1", "--port", "8765"])

        self.assertEqual(args.command, "serve")
        self.assertEqual(args.index_dir, ".\\agent")
        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.port, 8765)

    def test_main_dispatches_serve_command(self) -> None:
        with patch("graph_rag_app.cli.serve_fastapi", return_value=0) as serve_fastapi:
            exit_code = main(["serve", "--index-dir", ".\\agent", "--host", "127.0.0.1", "--port", "8765"])

        self.assertEqual(exit_code, 0)
        serve_fastapi.assert_called_once_with(
            index_dir=".\\agent",
            host="127.0.0.1",
            port=8765,
            reload=False,
        )

    def test_main_dispatches_ui_command_to_fastapi(self) -> None:
        with patch("graph_rag_app.cli.serve_fastapi", return_value=0) as serve_fastapi:
            exit_code = main(["ui", "--index-dir", ".\\agent", "--host", "127.0.0.1", "--port", "8765"])

        self.assertEqual(exit_code, 0)
        serve_fastapi.assert_called_once_with(
            index_dir=".\\agent",
            host="127.0.0.1",
            port=8765,
            reload=False,
        )
