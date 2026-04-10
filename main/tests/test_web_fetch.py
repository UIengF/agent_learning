from __future__ import annotations

import socket
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch
from urllib.parse import urlsplit
from urllib.request import Request

from graph_rag_app.web_fetch import (
    extract_main_text,
    fetch_url,
    truncate_text,
)


class _FakeHeaders:
    def __init__(self, content_type: str, charset: str | None = "utf-8") -> None:
        self._content_type = content_type
        self._charset = charset

    def get_content_type(self) -> str:
        return self._content_type

    def get_content_charset(self) -> str | None:
        return self._charset


class _FakeResponse:
    def __init__(
        self,
        *,
        final_url: str,
        body: str,
        content_type: str = "text/html",
        charset: str | None = "utf-8",
        status: int = 200,
    ) -> None:
        self._final_url = final_url
        self._body = body.encode(charset or "utf-8")
        self.headers = _FakeHeaders(content_type=content_type, charset=charset)
        self.status = status

    def __enter__(self) -> _FakeResponse:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        return None

    def geturl(self) -> str:
        return self._final_url

    def read(self, limit: int | None = None) -> bytes:
        if limit is None:
            return self._body
        return self._body[:limit]


class _FakeOpener:
    def __init__(self, response: _FakeResponse) -> None:
        self.response = response
        self.calls: list[tuple[str, float | None]] = []

    def open(self, request: Request, timeout: float | None = None) -> _FakeResponse:
        self.calls.append((request.full_url, timeout))
        return self.response


class WebFetchTests(TestCase):
    def _fetch_with_response(
        self,
        *,
        body: str,
        final_url: str = "https://example.com/article",
        content_type: str = "text/html",
        charset: str | None = "utf-8",
        status: int = 200,
        url: str = "https://example.com/article",
        max_bytes: int = 10_000,
        max_chars: int = 16,
        opener: _FakeOpener | None = None,
    ):
        opener = opener or _FakeOpener(
            _FakeResponse(
                final_url=final_url,
                body=body,
                content_type=content_type,
                charset=charset,
                status=status,
            )
        )
        with patch("graph_rag_app.web_fetch.build_opener", return_value=opener):
            result = fetch_url(
                url,
                timeout_seconds=2,
                max_bytes=max_bytes,
                max_chars=max_chars,
                user_agent="graph-rag-agent/1.0",
            )
        return result, opener

    def test_extract_main_text_prefers_article_and_excludes_noise(self) -> None:
        fixture_path = Path(__file__).parent / "fixtures" / "web" / "article.html"
        html = fixture_path.read_text(encoding="utf-8")

        result = extract_main_text(html)

        self.assertEqual(result["title"], "Example Article Title")
        self.assertIn("Lead paragraph from the article.", result["text"])
        self.assertIn("Second paragraph with the actual story.", result["text"])
        self.assertNotIn("Subscribe to our newsletter", result["text"])
        self.assertNotIn("Footer navigation", result["text"])

    def test_extract_main_text_ignores_article_like_strings_in_scripts(self) -> None:
        html = """
            <html>
              <head><title>Script Trap</title></head>
              <body>
                <script>const snippet = "<article>not real</article>";</script>
                <p>Visible body text.</p>
              </body>
            </html>
        """

        result = extract_main_text(html)

        self.assertEqual(result["title"], "Script Trap")
        self.assertEqual(result["text"], "Visible body text.")

    def test_truncate_text_preserves_word_boundaries_and_marks_truncation(self) -> None:
        text = "Alpha   beta\n gamma delta epsilon"

        truncated, did_truncate = truncate_text(text, max_chars=16)

        self.assertEqual(truncated, "Alpha beta gamma")
        self.assertTrue(did_truncate)

    def test_fetch_url_extracts_html_fixture(self) -> None:
        fixture_path = Path(__file__).parent / "fixtures" / "web" / "article.html"
        html = fixture_path.read_text(encoding="utf-8")
        result, opener = self._fetch_with_response(body=html)

        self.assertEqual(opener.calls, [("https://example.com/article", 2)])
        self.assertEqual(result.final_url, "https://example.com/article")
        self.assertEqual(result.status_code, 200)
        self.assertEqual(result.content_type, "text/html")
        self.assertEqual(result.title, "Example Article Title")
        self.assertEqual(result.text, "Example Article")
        self.assertTrue(result.truncated)

    def test_fetch_url_rejects_unsupported_content_type(self) -> None:
        fixture_path = Path(__file__).parent / "fixtures" / "web" / "article.html"
        html = fixture_path.read_text(encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "unsupported content type: application/json"):
            self._fetch_with_response(body=html, content_type="application/json")

    def test_fetch_url_with_zero_byte_limit_marks_truncation(self) -> None:
        fixture_path = Path(__file__).parent / "fixtures" / "web" / "article.html"
        html = fixture_path.read_text(encoding="utf-8")
        result, _ = self._fetch_with_response(body=html, max_bytes=0)

        self.assertEqual(result.text, "")
        self.assertTrue(result.truncated)
        self.assertEqual(result.content_type, "text/html")

    def test_fetch_url_allows_localhost_and_private_urls(self) -> None:
        fixture_path = Path(__file__).parent / "fixtures" / "web" / "article.html"
        html = fixture_path.read_text(encoding="utf-8")
        result, opener = self._fetch_with_response(
            body=html,
            url="http://127.0.0.1/article",
            final_url="http://127.0.0.1/article",
        )

        self.assertEqual(opener.calls, [("http://127.0.0.1/article", 2)])
        self.assertEqual(result.url, "http://127.0.0.1/article")
        self.assertEqual(result.final_url, "http://127.0.0.1/article")

    def test_fetch_url_allows_private_redirect_targets(self) -> None:
        fixture_path = Path(__file__).parent / "fixtures" / "web" / "article.html"
        html = fixture_path.read_text(encoding="utf-8")
        result, opener = self._fetch_with_response(
            body=html,
            url="https://example.com/article",
            final_url="http://127.0.0.1/internal",
        )

        self.assertEqual(opener.calls, [("https://example.com/article", 2)])
        self.assertEqual(result.final_url, "http://127.0.0.1/internal")


if __name__ == "__main__":
    import unittest

    unittest.main()
