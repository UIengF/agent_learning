from __future__ import annotations

from pathlib import Path
from unittest import TestCase

from graph_rag_app.web_search import (
    MultiQuerySearchBackend,
    SearchHit,
    merge_ranked_hits,
    parse_duckduckgo_html,
)


class WebSearchTests(TestCase):
    def test_parse_duckduckgo_html_returns_ranked_hits(self) -> None:
        fixture_path = Path(__file__).parent / "fixtures" / "web" / "search_results.html"
        html = fixture_path.read_text(encoding="utf-8")

        hits = parse_duckduckgo_html(html, top_k=2)

        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0].title, "OpenAI agents overview")
        self.assertEqual(hits[0].url, "https://example.com/openai-agents")
        self.assertEqual(hits[0].rank, 1)
        self.assertEqual(hits[0].source, "duckduckgo_html")
        self.assertEqual(hits[1].title, "Agent web search patterns")
        self.assertEqual(hits[1].url, "https://example.com/agent-search")
        self.assertEqual(hits[1].rank, 2)

    def test_merge_ranked_hits_dedupes_and_prefers_official_domains(self) -> None:
        merged = merge_ranked_hits(
            [
                SearchHit(
                    title="Third-party comparison",
                    url="https://example.com/openai-vs-adk",
                    snippet="blog summary",
                    source="duckduckgo_html",
                    rank=1,
                ),
                SearchHit(
                    title="OpenAI Agents SDK docs",
                    url="https://openai.github.io/openai-agents-python/",
                    snippet="official docs",
                    source="duckduckgo_html",
                    rank=3,
                ),
                SearchHit(
                    title="Third-party comparison duplicate",
                    url="https://example.com/openai-vs-adk/",
                    snippet="duplicate result",
                    source="duckduckgo_html",
                    rank=2,
                ),
            ],
            top_k=3,
        )

        self.assertEqual([hit.url for hit in merged], [
            "https://openai.github.io/openai-agents-python",
            "https://example.com/openai-vs-adk",
        ])
        self.assertEqual(merged[0].rank, 1)
        self.assertEqual(merged[1].rank, 2)

    def test_multi_query_backend_expands_and_merges_openai_gemini_queries(self) -> None:
        class FakeBackend:
            def __init__(self) -> None:
                self.calls: list[tuple[str, int]] = []

            def search(self, query: str, *, top_k: int) -> list[SearchHit]:
                self.calls.append((query, top_k))
                if query == "OpenAI agent implementation vs Google Gemini":
                    return [
                        SearchHit(
                            title="General comparison",
                            url="https://example.com/general-comparison",
                            snippet="general blog",
                            source="duckduckgo_html",
                            rank=1,
                        )
                    ]
                if query == "OpenAI Agents SDK vs Google ADK":
                    return [
                        SearchHit(
                            title="OpenAI Agents SDK docs",
                            url="https://openai.github.io/openai-agents-python/",
                            snippet="official docs",
                            source="duckduckgo_html",
                            rank=1,
                        ),
                        SearchHit(
                            title="Google ADK docs",
                            url="https://google.github.io/adk-docs/",
                            snippet="official docs",
                            source="duckduckgo_html",
                            rank=2,
                        ),
                    ]
                return []

        backend = FakeBackend()
        multi = MultiQuerySearchBackend(backend=backend)

        hits = multi.search("OpenAI agent implementation vs Google Gemini", top_k=3)

        self.assertEqual(
            backend.calls,
            [
                ("OpenAI agent implementation vs Google Gemini", 3),
                ("OpenAI Agents SDK vs Google ADK", 3),
            ],
        )
        self.assertEqual(
            [hit.url for hit in hits],
            [
                "https://openai.github.io/openai-agents-python",
                "https://google.github.io/adk-docs",
                "https://example.com/general-comparison",
            ],
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
