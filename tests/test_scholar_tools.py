from __future__ import annotations

import json
from unittest import TestCase

from pydantic import ValidationError

from graph_rag_app.scholar_search import ScholarSearchResponse
from graph_rag_app.scholar_tools import ScholarSearchTool
from graph_rag_app.web_types import ScholarHit


class ScholarToolTests(TestCase):
    def test_scholar_search_tool_returns_expected_json_payload(self) -> None:
        class FakeSearcher:
            def __init__(self) -> None:
                self.calls: list[tuple[str, int]] = []

            def search(self, topic: str, count: int) -> ScholarSearchResponse:
                self.calls.append((topic, count))
                return ScholarSearchResponse(
                    topic=topic,
                    planned_queries=["query one", "query two"],
                    result_count=1,
                    results=[
                        ScholarHit(
                            title="Paper A",
                            url="https://example.com/a",
                            snippet="snippet",
                            publication_summary="2024",
                            year=2024,
                            cited_by_count=12,
                            resources=(),
                            source_query="query one",
                            rank=1,
                        )
                    ],
                )

        searcher = FakeSearcher()
        tool = ScholarSearchTool(searcher=searcher, default_count=5)

        payload = json.loads(tool.invoke({"topic": "graph rag", "count": 2}))

        self.assertEqual(searcher.calls, [("graph rag", 2)])
        self.assertEqual(payload["topic"], "graph rag")
        self.assertEqual(payload["planned_queries"], ["query one", "query two"])
        self.assertEqual(payload["result_count"], 1)
        self.assertEqual(payload["results"][0]["title"], "Paper A")

    def test_scholar_search_tool_rejects_invalid_count(self) -> None:
        class FakeSearcher:
            def search(self, topic: str, count: int) -> ScholarSearchResponse:
                raise AssertionError("should not be called")

        tool = ScholarSearchTool(searcher=FakeSearcher())

        with self.assertRaises(ValidationError):
            tool.invoke({"topic": "graph rag", "count": 0})
