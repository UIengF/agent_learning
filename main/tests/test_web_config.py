from __future__ import annotations

import os
from unittest import TestCase
from unittest.mock import patch

from graph_rag_app.config import build_app_config


class WebConfigTests(TestCase):
    def test_build_app_config_includes_web_defaults(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            config = build_app_config("index-dir")

        self.assertTrue(config.web.enabled)
        self.assertEqual(config.web.search_provider, "duckduckgo_html")
        self.assertEqual(config.web.search_top_k, 5)
        self.assertEqual(config.web.fetch_timeout_seconds, 15)
        self.assertEqual(config.web.fetch_max_bytes, 1_500_000)
        self.assertEqual(config.web.fetch_max_chars, 6000)
        self.assertEqual(config.web.user_agent, "graph-rag-agent/1.0")
