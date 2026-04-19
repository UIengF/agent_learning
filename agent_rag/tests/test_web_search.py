from __future__ import annotations

from pathlib import Path
from unittest import TestCase

from graph_rag_app.web_search import (
    DuckDuckGoHtmlSearchBackend,
    FallbackSearchBackend,
    MultiQuerySearchBackend,
    SearchBackendError,
    SearchHit,
    SearXngSearchBackend,
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

    def test_duckduckgo_backend_raises_when_challenge_page_is_returned(self) -> None:
        class FakeResponse:
            headers = type("Headers", (), {"get_content_charset": lambda self: "utf-8"})()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback) -> None:
                return None

            def read(self) -> bytes:
                return b"<html><body>anomaly challenge</body></html>"

        def fake_urlopen(request, timeout):  # noqa: ANN001
            return FakeResponse()

        backend = DuckDuckGoHtmlSearchBackend(
            timeout_seconds=1,
            user_agent="test-agent",
            opener=fake_urlopen,
        )

        with self.assertRaisesRegex(SearchBackendError, "search_backend_blocked"):
            backend.search("OpenAI Agents SDK", top_k=3)

    def test_searxng_backend_maps_json_results_to_search_hits(self) -> None:
        class FakeResponse:
            headers = type("Headers", (), {"get_content_charset": lambda self: "utf-8"})()

            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, traceback) -> None:
                return None

            def read(self) -> bytes:
                return (
                    b'{"results":[{"title":"Google ADK docs","url":"https://google.github.io/adk-docs/",'
                    b'"content":"Build agents with ADK.","engine":"google"}]}'
                )

        calls = []

        def fake_urlopen(request, timeout):  # noqa: ANN001
            calls.append(request.full_url)
            return FakeResponse()

        backend = SearXngSearchBackend(
            base_url="http://127.0.0.1:8080",
            timeout_seconds=1,
            user_agent="test-agent",
            engines="google,bing",
            categories="general",
            language="zh-CN",
            opener=fake_urlopen,
        )

        hits = backend.search("Google ADK", top_k=3)

        self.assertIn("/search?", calls[0])
        self.assertIn("format=json", calls[0])
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].title, "Google ADK docs")
        self.assertEqual(hits[0].url, "https://google.github.io/adk-docs/")
        self.assertEqual(hits[0].snippet, "Build agents with ADK.")
        self.assertEqual(hits[0].source, "searxng:google")

    def test_fallback_search_backend_uses_next_backend_after_failure(self) -> None:
        class BrokenBackend:
            source = "broken"

            def search(self, query: str, *, top_k: int) -> list[SearchHit]:
                raise SearchBackendError("search_backend_blocked", provider="broken")

        class WorkingBackend:
            source = "working"

            def search(self, query: str, *, top_k: int) -> list[SearchHit]:
                return [
                    SearchHit(
                        title="OpenAI Agents SDK",
                        url="https://openai.github.io/openai-agents-python/",
                        snippet="Official docs",
                        source="working",
                        rank=1,
                    )
                ]

        backend = FallbackSearchBackend([BrokenBackend(), WorkingBackend()])

        hits = backend.search("OpenAI Agents SDK", top_k=3)

        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].source, "working")
        self.assertEqual(
            backend.last_debug["provider_errors"],
            [{"provider": "broken", "error": "search_backend_blocked"}],
        )

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
        self.assertTrue(merged[0].is_official)
        self.assertFalse(merged[1].is_official)
        self.assertEqual(merged[0].rank, 1)
        self.assertEqual(merged[1].rank, 2)

    def test_merge_ranked_hits_prefers_query_specific_official_domain(self) -> None:
        merged = merge_ranked_hits(
            [
                SearchHit(
                    title="OpenAI News",
                    url="https://openai.com/news",
                    snippet="official OpenAI news",
                    source="duckduckgo_html",
                    rank=2,
                ),
                SearchHit(
                    title="Computerworld OpenAI roundup",
                    url="https://www.computerworld.com/article/4015023/openai-latest-news-and-insights.html",
                    snippet="third-party roundup",
                    source="duckduckgo_html",
                    rank=1,
                ),
            ],
            top_k=2,
            query="recent changes OpenAI",
        )

        self.assertEqual(
            [hit.url for hit in merged],
            [
                "https://openai.com/news",
                "https://www.computerworld.com/article/4015023/openai-latest-news-and-insights.html",
            ],
        )
        self.assertTrue(merged[0].is_official)
        self.assertFalse(merged[1].is_official)

    def test_merge_ranked_hits_demotes_low_quality_domains_but_keeps_balanced_third_party(self) -> None:
        merged = merge_ranked_hits(
            [
                SearchHit(
                    title="个人怎么才能使用OpenAI? - 知乎",
                    url="https://www.zhihu.com/question/572946129",
                    snippet="问答站内容",
                    source="searxng:duckduckgo",
                    rank=1,
                ),
                SearchHit(
                    title="Claude Agents SDK vs. OpenAI Agents SDK vs. Google ADK",
                    url="https://composio.dev/content/claude-agents-sdk-vs-openai-agents-sdk-vs-google-adk",
                    snippet="框架能力对比",
                    source="searxng:duckduckgo",
                    rank=2,
                ),
                SearchHit(
                    title="OpenAI Agents SDK",
                    url="https://openai.github.io/openai-agents-python/",
                    snippet="官方 SDK 文档",
                    source="searxng:startpage",
                    rank=3,
                ),
                SearchHit(
                    title="Overview of Agent Development Kit",
                    url="https://docs.cloud.google.com/agent-builder/agent-development-kit/overview",
                    snippet="Google ADK 官方文档",
                    source="searxng:duckduckgo",
                    rank=4,
                ),
            ],
            top_k=4,
            query="openai和gemini在agent实现中有哪些异同点",
        )

        self.assertEqual(
            [hit.url for hit in merged],
            [
                "https://openai.github.io/openai-agents-python",
                "https://docs.cloud.google.com/agent-builder/agent-development-kit/overview",
                "https://composio.dev/content/claude-agents-sdk-vs-openai-agents-sdk-vs-google-adk",
                "https://www.zhihu.com/question/572946129",
            ],
        )
        self.assertTrue(merged[0].is_official)
        self.assertTrue(merged[1].is_official)
        self.assertFalse(merged[2].is_official)
        self.assertFalse(merged[3].is_official)

    def test_merge_ranked_hits_prefers_official_github_org_repo_over_marketing_pages(self) -> None:
        merged = merge_ranked_hits(
            [
                SearchHit(
                    title="Top OpenAI agent frameworks in 2026",
                    url="https://www.eesel.ai/blog/openai-api-vs-gemini-api",
                    snippet="build vs buy comparison",
                    source="searxng:duckduckgo",
                    rank=1,
                ),
                SearchHit(
                    title="google/adk-python",
                    url="https://github.com/google/adk-python",
                    snippet="official Google ADK repository",
                    source="searxng:google",
                    rank=3,
                ),
            ],
            top_k=2,
            query="Google ADK Gemini agent framework",
        )

        self.assertEqual(
            [hit.url for hit in merged],
            [
                "https://github.com/google/adk-python",
                "https://www.eesel.ai/blog/openai-api-vs-gemini-api",
            ],
        )
        self.assertTrue(merged[0].is_official)
        self.assertFalse(merged[1].is_official)

    def test_merge_ranked_hits_demotes_generic_home_and_login_pages(self) -> None:
        merged = merge_ranked_hits(
            [
                SearchHit(
                    title="Google",
                    url="https://www.google.com",
                    snippet="搜索首页",
                    source="searxng:google",
                    rank=1,
                ),
                SearchHit(
                    title="登录 - Google 账号",
                    url="https://accounts.google.com/login",
                    snippet="登录页面",
                    source="searxng:google",
                    rank=2,
                ),
                SearchHit(
                    title="OpenAI compatibility | Gemini API - Google AI for Developers",
                    url="https://ai.google.dev/gemini-api/docs/openai",
                    snippet="官方兼容性文档",
                    source="searxng:google",
                    rank=4,
                ),
                SearchHit(
                    title="Claude Agents SDK vs. OpenAI Agents SDK vs. Google ADK",
                    url="https://composio.dev/content/claude-agents-sdk-vs-openai-agents-sdk-vs-google-adk",
                    snippet="对比分析",
                    source="searxng:duckduckgo",
                    rank=3,
                ),
            ],
            top_k=4,
            query="openai和gemini在agent实现中有哪些异同点",
        )

        self.assertEqual(
            [hit.url for hit in merged],
            [
                "https://ai.google.dev/gemini-api/docs/openai",
                "https://composio.dev/content/claude-agents-sdk-vs-openai-agents-sdk-vs-google-adk",
                "https://www.google.com",
                "https://accounts.google.com/login",
            ],
        )

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
                ("OpenAI agent differences vs Gemini agent", 3),
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
        self.assertTrue(hits[0].is_official)
        self.assertTrue(hits[1].is_official)
        self.assertFalse(hits[2].is_official)

    def test_multi_query_backend_adds_more_specific_compare_query_for_openai_gemini(self) -> None:
        class FakeBackend:
            def __init__(self) -> None:
                self.calls: list[tuple[str, int]] = []

            def search(self, query: str, *, top_k: int) -> list[SearchHit]:
                self.calls.append((query, top_k))
                return []

        backend = FakeBackend()
        multi = MultiQuerySearchBackend(backend=backend)

        multi.search("openai和gemini在agent实现中有哪些异同点", top_k=3)

        self.assertEqual(
            backend.calls,
            [
                ("openai和gemini在agent实现中有哪些异同点", 3),
                ("OpenAI Agents SDK vs Google ADK", 3),
                ("OpenAI agent differences vs Gemini agent", 3),
            ],
        )


if __name__ == "__main__":
    import unittest

    unittest.main()
