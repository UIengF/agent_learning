from __future__ import annotations

import json
from unittest import TestCase

from pydantic import ValidationError

from graph_rag_app.web_tools import WebFetchTool, WebSearchTool
from graph_rag_app.web_types import FetchResult, SearchHit


class _FakeSearchBackend:
    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, *, top_k: int) -> list[SearchHit]:
        self.calls.append((query, top_k))
        return [
            SearchHit(
                title="Recent agent update",
                url="https://example.com/agent-update",
                snippet="Latest public update for the agent.",
                source="duckduckgo_html",
                rank=1,
            )
        ]

    @property
    def last_debug(self) -> dict[str, object]:
        query, top_k = self.calls[-1]
        return {
            "expanded_queries": [query],
            "per_query_counts": [{"query": query, "result_count": 1}],
            "raw_result_count": 1,
            "merged_result_count": 1,
            "selected_urls": ["https://example.com/agent-update"],
        }


class WebToolTests(TestCase):
    def test_web_search_tool_returns_expected_json_payload(self) -> None:
        backend = _FakeSearchBackend()
        tool = WebSearchTool(backend=backend)

        payload = json.loads(tool.invoke({"query": "agent updates", "top_k": 2}))

        self.assertEqual(backend.calls, [("agent updates", 2)])
        self.assertEqual(payload["query"], "agent updates")
        self.assertEqual(payload["result_count"], 1)
        self.assertEqual(
            payload["debug"],
            {
                "expanded_queries": ["agent updates"],
                "per_query_counts": [{"query": "agent updates", "result_count": 1}],
                "raw_result_count": 1,
                "merged_result_count": 1,
                "selected_urls": ["https://example.com/agent-update"],
            },
        )
        self.assertEqual(
            payload["results"],
            [
                {
                    "title": "Recent agent update",
                    "url": "https://example.com/agent-update",
                    "snippet": "Latest public update for the agent.",
                    "source": "duckduckgo_html",
                    "rank": 1,
                }
            ],
        )

    def test_web_search_tool_rejects_invalid_top_k_via_schema_validation(self) -> None:
        backend = _FakeSearchBackend()
        tool = WebSearchTool(backend=backend)

        with self.assertRaises(ValidationError):
            tool.invoke({"query": "agent updates", "top_k": 0})

        self.assertEqual(backend.calls, [])

    def test_web_fetch_tool_returns_fetch_result_as_json(self) -> None:
        calls: list[str] = []

        def fetcher(url: str) -> FetchResult:
            calls.append(url)
            return FetchResult(
                url=url,
                final_url=f"{url}/final",
                title="Fetched page",
                text="Expanded page content.",
                status_code=200,
                content_type="text/html",
                truncated=False,
            )

        tool = WebFetchTool(fetcher=fetcher)

        payload = json.loads(tool.invoke({"url": "https://example.com/page"}))

        self.assertEqual(calls, ["https://example.com/page"])
        self.assertEqual(
            payload,
            {
                "url": "https://example.com/page",
                "final_url": "https://example.com/page/final",
                "title": "Fetched page",
                "text": "Expanded page content.",
                "status_code": 200,
                "content_type": "text/html",
                "truncated": False,
            },
        )

    def test_web_fetch_tool_schema_allows_local_urls(self) -> None:
        calls: list[str] = []

        def fetcher(url: str) -> FetchResult:
            calls.append(url)
            return FetchResult(
                url=url,
                final_url=url,
                title="Local page",
                text="Local content.",
                status_code=200,
                content_type="text/html",
                truncated=False,
            )

        tool = WebFetchTool(fetcher=fetcher)

        payload = json.loads(tool.invoke({"url": "http://127.0.0.1/admin"}))

        self.assertEqual(calls, ["http://127.0.0.1/admin"])
        self.assertEqual(payload["url"], "http://127.0.0.1/admin")

    def test_web_fetch_tool_runtime_allows_local_urls(self) -> None:
        calls: list[str] = []

        def fetcher(url: str) -> FetchResult:
            calls.append(url)
            return FetchResult(
                url=url,
                final_url=url,
                title="Local page",
                text="Local content.",
                status_code=200,
                content_type="text/html",
                truncated=False,
            )

        tool = WebFetchTool(fetcher=fetcher)

        payload = json.loads(tool._run("http://127.0.0.1/admin"))

        self.assertEqual(calls, ["http://127.0.0.1/admin"])
        self.assertEqual(payload["url"], "http://127.0.0.1/admin")
