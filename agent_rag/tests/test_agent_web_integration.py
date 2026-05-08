from __future__ import annotations

import json
from pathlib import Path
from unittest import TestCase
from unittest.mock import ANY, patch

from graph_rag_app.agent import Agent, build_agent
from graph_rag_app.config import (
    AppConfig,
    EmbeddingConfig,
    GenerationConfig,
    ModelConfig,
    RetrievalConfig,
    RuntimeConfig,
    ScholarConfig,
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
            model=ModelConfig(
                api_key="test-key", api_base="https://example.invalid", model_name="model"
            ),
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
            scholar=ScholarConfig(
                enabled=True, api_key="serp-api-key", default_count=4, max_count=20
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
                with patch(
                    "graph_rag_app.agent.LocalRAGRetrieveTool", return_value="local-tool"
                ) as rag_tool:
                    with patch(
                        "graph_rag_app.agent.build_configured_web_search_backend",
                        return_value=MultiQuerySearchBackend("search-backend"),
                    ) as backend_builder:
                        with patch(
                            "graph_rag_app.agent.WebSearchTool",
                            return_value="web-search-tool",
                        ) as web_search_tool:
                            with patch(
                                "graph_rag_app.agent.WebFetchTool",
                                side_effect=build_web_fetch_tool,
                            ) as web_fetch_tool:
                                with patch(
                                    "graph_rag_app.agent.build_scholar_search_service",
                                    return_value="scholar-service",
                                ) as build_scholar_service:
                                    with patch(
                                        "graph_rag_app.agent.ScholarSearchTool",
                                        return_value="scholar-search-tool",
                                    ) as scholar_tool:
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
                                                    captured_fetcher(
                                                        "https://example.com/page"
                                                    ).title,
                                                    "Fetched page",
                                                )

        self.assertEqual(result, "agent")
        load_index.assert_called_once_with("existing-index", keyword_weight=0.25)
        rag_tool.assert_called_once_with(store="retriever", min_evidence_score=0.15)
        backend_builder.assert_called_once_with(app_config.web)
        web_search_tool.assert_called_once()
        wrapped_backend = web_search_tool.call_args.kwargs["backend"]
        self.assertIsInstance(wrapped_backend, MultiQuerySearchBackend)
        self.assertEqual(wrapped_backend.backend, "search-backend")
        self.assertEqual(web_search_tool.call_args.kwargs["default_top_k"], 7)
        build_scholar_service.assert_called_once_with(app_config)
        scholar_tool.assert_called_once_with(searcher="scholar-service", default_count=4)
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
            [
                "local-tool",
                ANY,
                ANY,
                "web-search-tool",
                "scholar-search-tool",
                "web-fetch-tool",
            ],
            checkpointer=None,
            system=agent_cls.call_args.kwargs["system"],
            ensure_log_file=None,
            append_log=None,
            shorten_text=None,
            max_rounds=4,
            max_recent_messages=8,
            recent_full_turns=3,
            max_context_chars=app_config.context.max_context_chars,
            max_context_tokens=100000,
            live_messages_compression_enabled=True,
            live_messages_keep_turns=1,
            live_messages_max_fetch_chars=180,
            live_messages_max_search_results=3,
            user_memory=ANY,
            token_estimator=ANY,
            skill_registry=ANY,
            trace_writer=None,
        )

    def test_build_agent_accepts_searxng_provider_when_web_enabled(self) -> None:
        app_config = AppConfig(
            kb_path=Path("unused-kb"),
            model=ModelConfig(
                api_key="test-key", api_base="https://example.invalid", model_name="model"
            ),
            embedding=EmbeddingConfig(model="embed-model", max_batch_size=8),
            retrieval=RetrievalConfig(keyword_weight=0.25),
            web=WebConfig(enabled=True, search_provider="searxng"),
            scholar=ScholarConfig(enabled=False),
            generation=GenerationConfig(min_evidence_score=0.15, max_rounds=4),
            runtime=RuntimeConfig(),
        )

        with patch("graph_rag_app.agent.ChatOpenAI", return_value="model-instance"):
            with patch("graph_rag_app.agent.load_index", return_value="retriever"):
                with patch("graph_rag_app.agent.LocalRAGRetrieveTool", return_value="local-tool"):
                    with patch(
                        "graph_rag_app.agent.build_configured_web_search_backend",
                        return_value="search-backend",
                    ) as backend_builder:
                        with patch(
                            "graph_rag_app.agent.WebSearchTool", return_value="web-search-tool"
                        ):
                            with patch(
                                "graph_rag_app.agent.WebFetchTool", return_value="web-fetch-tool"
                            ):
                                with patch(
                                    "graph_rag_app.agent.Agent", return_value="agent"
                                ) as agent_cls:
                                    result = build_agent(
                                        index_dir="existing-index", app_config=app_config
                                    )

        self.assertEqual(result, "agent")
        backend_builder.assert_called_once_with(app_config.web)
        agent_cls.assert_called_once()

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

    def test_take_action_falls_back_to_next_search_result_after_web_fetch_failure(self) -> None:
        class FallbackFetchTool:
            name = "web_fetch"

            def __init__(self) -> None:
                self.urls: list[str] = []

            def invoke(self, args):
                url = args["url"]
                self.urls.append(url)
                if url == "https://docs.example.com/broken":
                    raise TimeoutError("fetch timed out")
                return json.dumps(
                    {
                        "url": url,
                        "final_url": url,
                        "title": "Fallback page",
                        "text": "Fetched fallback page body.",
                    }
                )

        tool = FallbackFetchTool()
        agent = Agent(tools=[tool])
        state = {
            "messages": [
                {
                    "role": "human",
                    "content": "How do the official docs describe tool use?",
                },
                {
                    "role": "tool",
                    "name": "web_search",
                    "content": (
                        '{"query":"official docs tool use","result_count":3,"results":['
                        '{"title":"Broken official docs","url":"https://docs.example.com/broken",'
                        '"snippet":"Official tool use docs","source":"search","rank":1,"is_official":true},'
                        '{"title":"Working official docs","url":"https://docs.example.com/working",'
                        '"snippet":"Official tool use fallback","source":"search","rank":2,"is_official":true},'
                        '{"title":"Third party summary","url":"https://blog.example.net/tool-use",'
                        '"snippet":"Summary","source":"search","rank":3,"is_official":false}]}'
                    ),
                },
                _ToolCallCarrier(
                    [
                        {
                            "id": "fetch-1",
                            "name": "web_fetch",
                            "args": {"url": "https://docs.example.com/broken"},
                        }
                    ]
                ),
            ]
        }

        result = agent.take_action(state)

        self.assertEqual(
            tool.urls,
            ["https://docs.example.com/broken", "https://docs.example.com/working"],
        )
        payload = json.loads(result["messages"][0].content)
        self.assertEqual(payload["url"], "https://docs.example.com/working")
        self.assertEqual(payload["title"], "Fallback page")

    def test_take_action_fallback_considers_all_current_web_search_results(self) -> None:
        class FallbackFetchTool:
            name = "web_fetch"

            def __init__(self) -> None:
                self.urls: list[str] = []

            def invoke(self, args):
                url = args["url"]
                self.urls.append(url)
                if url != "https://cloud.example.com/working":
                    raise TimeoutError("fetch failed")
                return json.dumps(
                    {
                        "url": url,
                        "final_url": url,
                        "title": "Working page",
                        "text": "Fetched from an earlier current-turn search result.",
                    }
                )

        tool = FallbackFetchTool()
        agent = Agent(tools=[tool])
        state = {
            "messages": [
                {"role": "human", "content": "How do Gemini docs describe tool use?"},
                {
                    "role": "tool",
                    "name": "web_search",
                    "content": (
                        '{"query":"Gemini function calling cloud docs","result_count":1,"results":['
                        '{"title":"Cloud function calling","url":"https://cloud.example.com/working",'
                        '"snippet":"Tool use docs","source":"search","rank":1,"is_official":true}]}'
                    ),
                },
                {
                    "role": "tool",
                    "name": "web_search",
                    "content": (
                        '{"query":"Gemini function calling ai docs","result_count":1,"results":['
                        '{"title":"AI docs","url":"https://ai.example.com/broken",'
                        '"snippet":"Tool use docs","source":"search","rank":1,"is_official":true}]}'
                    ),
                },
                _ToolCallCarrier(
                    [
                        {
                            "id": "fetch-1",
                            "name": "web_fetch",
                            "args": {"url": "https://ai.example.com/broken"},
                        }
                    ]
                ),
            ]
        }

        result = agent.take_action(state)

        self.assertEqual(
            tool.urls,
            ["https://ai.example.com/broken", "https://cloud.example.com/working"],
        )
        payload = json.loads(result["messages"][0].content)
        self.assertEqual(payload["url"], "https://cloud.example.com/working")
