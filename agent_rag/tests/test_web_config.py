from __future__ import annotations

import os
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

from graph_rag_app.config import build_app_config


class WebConfigTests(TestCase):
    def test_build_app_config_includes_web_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with patch("graph_rag_app.config.PROJECT_ENV_PATH", Path("__missing__.env")):
                config = build_app_config("index-dir")

        self.assertTrue(config.web.enabled)
        self.assertEqual(config.web.search_provider, "duckduckgo_html")
        self.assertEqual(config.web.searxng_url, "http://127.0.0.1:8080")
        self.assertEqual(config.web.searxng_engines, "")
        self.assertEqual(config.web.searxng_categories, "general")
        self.assertEqual(config.web.searxng_language, "zh-CN")
        self.assertEqual(config.web.search_top_k, 5)
        self.assertEqual(config.web.fetch_timeout_seconds, 15)
        self.assertEqual(config.web.fetch_max_bytes, 1_500_000)
        self.assertEqual(config.web.fetch_max_chars, 6000)
        self.assertIn("Mozilla/5.0", config.web.user_agent)
        self.assertIn("graph-rag-agent/1.0", config.web.user_agent)
        self.assertTrue(config.scholar.enabled)
        self.assertEqual(config.scholar.api_key, "")
        self.assertEqual(config.scholar.default_count, 5)
        self.assertEqual(config.scholar.max_count, 20)
        self.assertEqual(config.scholar.engine, "google_scholar")
        self.assertEqual(config.generation.max_rounds, 8)

    def test_build_app_config_reads_searxng_env(self) -> None:
        with patch.dict(
            os.environ,
            {
                "RAG_WEB_SEARCH_PROVIDER": "searxng",
                "RAG_SEARXNG_URL": "http://searxng.local:8080",
                "RAG_SEARXNG_ENGINES": "google,bing",
                "RAG_SEARXNG_CATEGORIES": "general,science",
                "RAG_SEARXNG_LANGUAGE": "en-US",
            },
            clear=True,
        ):
            config = build_app_config("index-dir")

        self.assertEqual(config.web.search_provider, "searxng")
        self.assertEqual(config.web.searxng_url, "http://searxng.local:8080")
        self.assertEqual(config.web.searxng_engines, "google,bing")
        self.assertEqual(config.web.searxng_categories, "general,science")
        self.assertEqual(config.web.searxng_language, "en-US")

    def test_build_app_config_allows_max_rounds_env_override(self) -> None:
        with patch.dict(os.environ, {"RAG_MAX_ROUNDS": "6"}, clear=True):
            config = build_app_config("index-dir")

        self.assertEqual(config.generation.max_rounds, 6)

    def test_build_app_config_reads_scholar_env(self) -> None:
        with patch.dict(
            os.environ,
            {
                "RAG_SCHOLAR_ENABLED": "false",
                "SERPAPI_API_KEY": "serp-key",
                "RAG_SCHOLAR_DEFAULT_COUNT": "7",
                "RAG_SCHOLAR_MAX_COUNT": "12",
                "RAG_SCHOLAR_ENGINE": "google_scholar",
            },
            clear=True,
        ):
            config = build_app_config("index-dir")

        self.assertFalse(config.scholar.enabled)
        self.assertEqual(config.scholar.api_key, "serp-key")
        self.assertEqual(config.scholar.default_count, 7)
        self.assertEqual(config.scholar.max_count, 12)
        self.assertEqual(config.scholar.engine, "google_scholar")

    def test_build_app_config_includes_context_defaults_and_env(self) -> None:
        with patch.dict(
            os.environ,
            {
                "RAG_MAX_RECENT_MESSAGES": "5",
                "RAG_RECENT_FULL_TURNS": "3",
                "RAG_MAX_CONTEXT_CHARS": "12000",
                "RAG_MAX_CONTEXT_TOKENS": "100000",
                "RAG_LIVE_MESSAGES_COMPRESSION_ENABLED": "false",
                "RAG_LIVE_MESSAGES_KEEP_TURNS": "2",
                "RAG_LIVE_MESSAGES_MAX_FETCH_CHARS": "240",
                "RAG_LIVE_MESSAGES_MAX_SEARCH_RESULTS": "4",
            },
            clear=True,
        ):
            config = build_app_config("index-dir")

        self.assertEqual(config.context.max_recent_messages, 5)
        self.assertEqual(config.context.recent_full_turns, 3)
        self.assertEqual(config.context.max_context_chars, 12000)
        self.assertEqual(config.context.max_context_tokens, 100000)
        self.assertFalse(config.context.live_messages_compression_enabled)
        self.assertEqual(config.context.live_messages_keep_turns, 2)
        self.assertEqual(config.context.live_messages_max_fetch_chars, 240)
        self.assertEqual(config.context.live_messages_max_search_results, 4)

