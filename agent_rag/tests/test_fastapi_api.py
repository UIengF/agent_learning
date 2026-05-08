from __future__ import annotations

import json
from pathlib import Path
import time
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
        state = get_graph_state(
            FakeGraphWithoutCheckpointer(), {"configurable": {"thread_id": "fresh"}}
        )

        self.assertIsNone(state)

    def test_run_or_resume_with_trace_forwards_langsmith_metadata(self) -> None:
        from graph_rag_app.runtime import run_or_resume_with_trace

        class FakeAgent:
            def __init__(self) -> None:
                self.graph = object()
                self.log_path = Path("runtime/fake.log")
                self.system = "system"
                self.user_memory = object()

            @staticmethod
            def _message_content(message):
                if isinstance(message, dict):
                    return str(message.get("content", ""))
                return str(getattr(message, "content", ""))

            @staticmethod
            def _is_human_message(message):
                if isinstance(message, dict):
                    return message.get("role") in {"human", "user"}
                return False

            @staticmethod
            def _shorten(text: str, _max_len: int) -> str:
                return text

        class FakeState:
            def __init__(self, messages, next_nodes=()):
                self.values = {"messages": messages}
                self.next = next_nodes

        config = build_app_config(".")
        langsmith_calls: list[dict[str, object]] = []

        from contextlib import contextmanager

        class FakeRun:
            id = "run-123"

        @contextmanager
        def fake_langsmith_context(*, config, run_name, metadata, inputs, tags=None):
            langsmith_calls.append(
                {
                    "config": config,
                    "run_name": run_name,
                    "metadata": metadata,
                    "inputs": inputs,
                    "tags": tags,
                }
            )
            yield FakeRun()

        with patch("graph_rag_app.runtime.build_app_config", return_value=config):
            with patch("graph_rag_app.runtime.build_agent", return_value=FakeAgent()):
                with patch(
                    "graph_rag_app.runtime.get_graph_state",
                    side_effect=[
                        None,
                        FakeState([{"role": "assistant", "content": "done"}]),
                    ],
                ):
                    with patch(
                        "graph_rag_app.runtime.run_graph_invoke",
                        return_value={"messages": [{"role": "assistant", "content": "done"}]},
                    ):
                        with patch(
                            "graph_rag_app.runtime.build_user_memory", return_value={"memory": []}
                        ):
                            with patch(
                                "graph_rag_app.runtime.merge_user_memory",
                                return_value={"memory": []},
                            ):
                                with patch("graph_rag_app.runtime.save_user_memory"):
                                    with patch("graph_rag_app.runtime.ensure_log_file"):
                                        with patch("graph_rag_app.runtime.append_log"):
                                            with patch(
                                                "graph_rag_app.runtime.langsmith_run_context",
                                                side_effect=fake_langsmith_context,
                                            ):
                                                trace = run_or_resume_with_trace(
                                                    question="What changed?",
                                                    index_dir="agent",
                                                    run_metadata={
                                                        "mode": "eval",
                                                        "dataset_name": "agent-smoke",
                                                        "case_id": "case-1",
                                                        "tags": ["smoke", "local"],
                                                    },
                                                )

        self.assertEqual(trace.answer, "done")
        self.assertEqual(trace.langsmith_run_id, "run-123")
        self.assertEqual(len(langsmith_calls), 1)
        self.assertEqual(langsmith_calls[0]["run_name"], "graph_rag.eval.case:case-1")
        self.assertEqual(langsmith_calls[0]["metadata"]["case_id"], "case-1")
        self.assertEqual(langsmith_calls[0]["metadata"]["dataset_name"], "agent-smoke")
        self.assertEqual(
            langsmith_calls[0]["tags"],
            ["smoke", "local", "mode:eval", "dataset:agent-smoke", "case:case-1"],
        )


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

        with patch(
            "graph_rag_app.api.build_sqlite_checkpointer", return_value="checkpointer"
        ) as build_checkpointer:
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
                    "score": 0.95,
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
                {
                    "role": "tool",
                    "name": "local_rag_retrieve",
                    "content": json.dumps(local_payload),
                },
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
        sources_by_type = {source["source_type"]: source for source in data["sources"]}
        self.assertEqual(sources_by_type["local"]["source_path"], "Agent/local.md")
        self.assertEqual(sources_by_type["web"]["url"], "https://example.com/agent")
        self.assertEqual(sources_by_type["web"]["text"], "Fetched page body.")

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
                    "score": 0.95,
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
                {
                    "role": "tool",
                    "name": "local_rag_retrieve",
                    "content": json.dumps(current_payload),
                },
            ],
            source_messages=[
                {
                    "role": "tool",
                    "name": "local_rag_retrieve",
                    "content": json.dumps(current_payload),
                },
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
                json={
                    "query": "agent tools",
                    "index_dir": "agent",
                    "top_k": 1,
                    "strategy": "hybrid",
                },
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
        ) as fetch_url:
            response = client.post("/api/web/fetch", json={"url": "https://example.com"})

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["title"], "Example")
        fetch_url.assert_called_once()
        self.assertIn("timeout_seconds", fetch_url.call_args.kwargs)
        self.assertIn("user_agent", fetch_url.call_args.kwargs)

    def test_web_fetch_returns_permission_error_when_policy_blocks_url(self) -> None:
        client = TestClient(create_app(default_index_dir="agent"))
        client.app.state.permission_policy = build_app_config(".").permissions

        from graph_rag_app.permissions import ToolPermissionPolicy
        from graph_rag_app.config import PermissionConfig

        client.app.state.permission_policy = ToolPermissionPolicy(
            PermissionConfig(allow_local_web_fetch=False)
        )

        response = client.post("/api/web/fetch", json={"url": "http://localhost:8000"})

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["error"]["code"], "permission_denied")

    def test_retrieve_returns_permission_error_for_disallowed_index_dir(self) -> None:
        from graph_rag_app.permissions import ToolPermissionPolicy
        from graph_rag_app.config import PermissionConfig

        client = TestClient(create_app(default_index_dir="agent"))
        client.app.state.permission_policy = ToolPermissionPolicy(
            PermissionConfig(allowed_index_roots="agent")
        )

        response = client.post(
            "/api/retrieve",
            json={"query": "agent", "index_dir": "..\\outside"},
        )

        self.assertEqual(response.status_code, 403)
        self.assertEqual(response.json()["error"]["code"], "permission_denied")

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

    def test_job_endpoints_submit_status_and_log(self) -> None:
        client = TestClient(create_app(default_index_dir="agent"))

        with patch("graph_rag_app.api.submit_index_build_job") as submit_job:
            submit_job.return_value = client.app.state.job_manager.submit(
                "index_build",
                lambda _log_path: {"index_dir": "agent", "chunk_count": 1},
            )
            response = client.post(
                "/api/jobs/index-build",
                json={"kb_path": "docs", "output_dir": "agent/job-index"},
            )

        self.assertEqual(response.status_code, 200)
        job_id = response.json()["job_id"]
        for _ in range(100):
            status_response = client.get(f"/api/jobs/{job_id}")
            if status_response.json()["status"] == "completed":
                break
            time.sleep(0.01)
        status_response = client.get(f"/api/jobs/{job_id}")
        log_response = client.get(f"/api/jobs/{job_id}/log")

        self.assertEqual(status_response.status_code, 200)
        self.assertEqual(status_response.json()["job_id"], job_id)
        self.assertEqual(log_response.status_code, 200)
        self.assertIn("job_id=", log_response.json()["log"])

    def test_missing_job_returns_404(self) -> None:
        client = TestClient(create_app(default_index_dir="agent"))

        response = client.get("/api/jobs/missing")

        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.json()["error"]["code"], "job_not_found")

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
        args = parse_args(
            ["serve", "--index-dir", ".\\agent", "--host", "127.0.0.1", "--port", "8765"]
        )

        self.assertEqual(args.command, "serve")
        self.assertEqual(args.index_dir, ".\\agent")
        self.assertEqual(args.host, "127.0.0.1")
        self.assertEqual(args.port, 8765)

    def test_main_dispatches_serve_command(self) -> None:
        with patch("graph_rag_app.cli.serve_fastapi", return_value=0) as serve_fastapi:
            exit_code = main(
                ["serve", "--index-dir", ".\\agent", "--host", "127.0.0.1", "--port", "8765"]
            )

        self.assertEqual(exit_code, 0)
        serve_fastapi.assert_called_once_with(
            index_dir=".\\agent",
            host="127.0.0.1",
            port=8765,
            reload=False,
        )

    def test_main_dispatches_ui_command_to_fastapi(self) -> None:
        with patch("graph_rag_app.cli.serve_fastapi", return_value=0) as serve_fastapi:
            exit_code = main(
                ["ui", "--index-dir", ".\\agent", "--host", "127.0.0.1", "--port", "8765"]
            )

        self.assertEqual(exit_code, 0)
        serve_fastapi.assert_called_once_with(
            index_dir=".\\agent",
            host="127.0.0.1",
            port=8765,
            reload=False,
        )

    def test_parse_args_supports_job_status_and_log_commands(self) -> None:
        status_args = parse_args(["job", "status", "--job-id", "abc", "--runtime-dir", "jobs"])
        log_args = parse_args(
            ["job", "log", "--job-id", "abc", "--runtime-dir", "jobs", "--max-chars", "100"]
        )

        self.assertEqual(status_args.command, "job")
        self.assertEqual(status_args.job_command, "status")
        self.assertEqual(status_args.job_id, "abc")
        self.assertEqual(log_args.job_command, "log")
        self.assertEqual(log_args.max_chars, 100)

    def test_main_job_status_returns_stable_error_for_missing_job(self) -> None:
        with patch("graph_rag_app.cli._print_json") as print_json:
            exit_code = main(
                ["job", "status", "--job-id", "missing", "--runtime-dir", "runtime/jobs"]
            )

        self.assertEqual(exit_code, 1)
        print_json.assert_called_once_with({"error": "job_not_found", "job_id": "missing"})
