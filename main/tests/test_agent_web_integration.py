from __future__ import annotations

import json
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from graph_rag_app.agent import Agent, build_agent
from graph_rag_app.config import (
    AppConfig,
    EmbeddingConfig,
    GenerationConfig,
    ModelConfig,
    RetrievalConfig,
    RuntimeConfig,
    WebConfig,
)
from graph_rag_app.web_search import MultiQuerySearchBackend
from graph_rag_app.web_types import FetchResult


class _ToolCallCarrier:
    def __init__(self, tool_calls):
        self.tool_calls = tool_calls


class AgentWebIntegrationTests(TestCase):
    def test_build_agent_registers_web_tools_when_enabled(self) -> None:
        app_config = AppConfig(
            kb_path=Path("unused-kb"),
            model=ModelConfig(api_key="test-key", api_base="https://example.invalid", model_name="model"),
            embedding=EmbeddingConfig(model="embed-model", max_batch_size=8),
            retrieval=RetrievalConfig(keyword_weight=0.25),
            web=WebConfig(
                enabled=True,
                search_provider="duckduckgo_html",
                search_top_k=7,
                fetch_timeout_seconds=9,
                fetch_max_bytes=2048,
                fetch_max_chars=512,
                user_agent="agent-test/1.0",
            ),
            generation=GenerationConfig(min_evidence_score=0.15, max_rounds=4),
            runtime=RuntimeConfig(),
        )

        captured_fetcher = None

        def build_web_fetch_tool(*, fetcher):
            nonlocal captured_fetcher
            captured_fetcher = fetcher
            return "web-fetch-tool"

        with patch("graph_rag_app.agent.ChatOpenAI", return_value="model-instance"):
            with patch("graph_rag_app.agent.load_index", return_value="retriever") as load_index:
                with patch("graph_rag_app.agent.LocalRAGRetrieveTool", return_value="local-tool") as rag_tool:
                    with patch(
                        "graph_rag_app.agent.DuckDuckGoHtmlSearchBackend",
                        return_value="search-backend",
                    ) as backend_cls:
                        with patch(
                            "graph_rag_app.agent.WebSearchTool",
                            return_value="web-search-tool",
                        ) as web_search_tool:
                            with patch(
                                "graph_rag_app.agent.WebFetchTool",
                                side_effect=build_web_fetch_tool,
                            ) as web_fetch_tool:
                                with patch(
                                    "graph_rag_app.agent.fetch_url",
                                    return_value=FetchResult(
                                        url="https://example.com/page",
                                        final_url="https://example.com/page",
                                        title="Fetched page",
                                        text="Expanded page content.",
                                        status_code=200,
                                        content_type="text/html",
                                        truncated=False,
                                    ),
                                ) as fetch_url:
                                    with patch(
                                        "graph_rag_app.agent.Agent",
                                        return_value="agent",
                                    ) as agent_cls:
                                        result = build_agent(
                                            index_dir="existing-index",
                                            app_config=app_config,
                                        )
                                        self.assertIsNotNone(captured_fetcher)
                                        self.assertEqual(
                                            captured_fetcher("https://example.com/page").title,
                                            "Fetched page",
                                        )

        self.assertEqual(result, "agent")
        load_index.assert_called_once_with("existing-index", keyword_weight=0.25)
        rag_tool.assert_called_once_with(store="retriever", min_evidence_score=0.15)
        backend_cls.assert_called_once_with(timeout_seconds=9, user_agent="agent-test/1.0")
        web_search_tool.assert_called_once()
        wrapped_backend = web_search_tool.call_args.kwargs["backend"]
        self.assertIsInstance(wrapped_backend, MultiQuerySearchBackend)
        self.assertEqual(wrapped_backend.backend, "search-backend")
        self.assertEqual(web_search_tool.call_args.kwargs["default_top_k"], 7)
        web_fetch_tool.assert_called_once()
        fetch_url.assert_called_once_with(
            "https://example.com/page",
            timeout_seconds=9,
            max_bytes=2048,
            max_chars=512,
            user_agent="agent-test/1.0",
        )
        agent_cls.assert_called_once_with(
            "model-instance",
            ["local-tool", "web-search-tool", "web-fetch-tool"],
            checkpointer=None,
            system=agent_cls.call_args.kwargs["system"],
            ensure_log_file=None,
            append_log=None,
            shorten_text=None,
            max_rounds=4,
        )

    def test_take_action_converts_tool_failures_into_tool_messages(self) -> None:
        class FailingTool:
            name = "web_fetch"

            def invoke(self, args):
                raise TimeoutError(f"timed out for {args['url']}")

        runner = Agent(tools=[FailingTool()])
        state = {
            "messages": [
                _ToolCallCarrier(
                    [
                        {
                            "id": "call-1",
                            "name": "web_fetch",
                            "args": {"url": "https://example.com/slow"},
                        }
                    ]
                )
            ]
        }

        result = runner.take_action(state)

        self.assertEqual(len(result["messages"]), 1)
        tool_message = result["messages"][0]
        payload = json.loads(tool_message.content)
        self.assertEqual(tool_message.name, "web_fetch")
        self.assertEqual(payload["error"], "tool_execution_failed")
        self.assertEqual(payload["tool_name"], "web_fetch")
        self.assertEqual(payload["tool_args"], {"url": "https://example.com/slow"})
        self.assertEqual(payload["error_type"], "TimeoutError")
        self.assertIn("timed out", payload["message"])
