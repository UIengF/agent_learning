from __future__ import annotations

from html.parser import HTMLParser
import socket
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, build_opener

from .web_types import FetchResult

_NOISE_TAGS = {"script", "style", "noscript", "template", "header", "footer", "nav", "aside"}
_DEFAULT_FETCH_ATTEMPTS = 2
_RETRYABLE_HTTP_STATUSES = {408, 425, 429, 500, 502, 503, 504}


def _build_fetch_headers(user_agent: str) -> dict[str, str]:
    return {
        "User-Agent": user_agent,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Upgrade-Insecure-Requests": "1",
    }


def _is_retryable_fetch_error(exc: Exception) -> bool:
    if isinstance(exc, HTTPError):
        return exc.code in _RETRYABLE_HTTP_STATUSES
    if isinstance(exc, (TimeoutError, socket.timeout, URLError)):
        return True
    return False


def truncate_text(text: str, max_chars: int) -> tuple[str, bool]:
    compacted = " ".join(text.split())
    if max_chars < 0:
        raise ValueError("max_chars must be non-negative")
    if len(compacted) <= max_chars:
        return compacted, False
    if max_chars == 0:
        return "", True

    cut = max_chars
    if cut < len(compacted) and not compacted[cut].isspace():
        whitespace_cut = compacted.rfind(" ", 0, max_chars + 1)
        if whitespace_cut > 0:
            cut = whitespace_cut

    truncated = compacted[:cut].rstrip()
    if not truncated:
        truncated = compacted[:max_chars].rstrip()
    return truncated, True


def extract_main_text(html: str) -> dict[str, str]:
    title = _extract_title(html)
    preferred_tag = _detect_preferred_content_tag(html)
    text = _extract_text_from_tag(html, preferred_tag) if preferred_tag else _extract_visible_text(html)
    return {"title": title, "text": text}


def fetch_url(
    url: str,
    *,
    timeout_seconds: float,
    max_bytes: int,
    max_chars: int,
    user_agent: str,
) -> FetchResult:
    if max_bytes < 0:
        raise ValueError("max_bytes must be non-negative")

    opener = build_opener()
    last_error: Exception | None = None
    for attempt in range(_DEFAULT_FETCH_ATTEMPTS):
        request = Request(url, headers=_build_fetch_headers(user_agent))
        try:
            with opener.open(request, timeout=timeout_seconds) as response:
                final_url = response.geturl()
                status_code = response.status
                content_type = response.headers.get_content_type()
                if content_type != "text/html":
                    raise ValueError(f"unsupported content type: {content_type}")

                raw = response.read(max_bytes + 1 if max_bytes else 1)
                bytes_truncated = len(raw) > max_bytes if max_bytes else bool(raw)
                html_bytes = raw[:max_bytes] if max_bytes else b""
                charset = response.headers.get_content_charset() or "utf-8"
                html = html_bytes.decode(charset, errors="replace")
            break
        except Exception as exc:
            last_error = exc
            if attempt + 1 >= _DEFAULT_FETCH_ATTEMPTS or not _is_retryable_fetch_error(exc):
                raise
            time.sleep(min(0.25 * (2**attempt), 1.0))
    else:  # pragma: no cover - loop always breaks or raises
        if last_error is not None:
            raise last_error
        raise RuntimeError("fetch_url exhausted attempts without a result")

    extracted = extract_main_text(html)
    text, chars_truncated = truncate_text(extracted["text"], max_chars)
    return FetchResult(
        url=url,
        final_url=final_url,
        title=extracted["title"],
        text=text,
        status_code=status_code,
        content_type=content_type,
        truncated=bytes_truncated or chars_truncated,
    )


class _HTMLTextExtractor(HTMLParser):
    def __init__(self, *, target_tag: str | None = None) -> None:
        super().__init__(convert_charrefs=True)
        self.target_tag = target_tag
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self._capture_depth = 0
        self._skip_depth = 0
        self._in_title = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "title":
            self._in_title = True
        if self.target_tag and tag == self.target_tag and self._capture_depth == 0:
            self._capture_depth = 1
            return
        if self._capture_depth:
            if tag in _NOISE_TAGS:
                self._skip_depth += 1
            else:
                self._capture_depth += 1
            return
        if self.target_tag is None and tag in _NOISE_TAGS:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        if self.target_tag and self._capture_depth:
            if tag in _NOISE_TAGS and self._skip_depth:
                self._skip_depth -= 1
            else:
                self._capture_depth -= 1
            return
        if self.target_tag is None and tag in _NOISE_TAGS and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        stripped = data.strip()
        if not stripped:
            return
        if self._in_title:
            self.title_parts.append(stripped)
            return
        if self.target_tag:
            if self._capture_depth and not self._skip_depth:
                self.text_parts.append(stripped)
        elif not self._skip_depth:
            self.text_parts.append(stripped)


def _extract_title(html: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(html)
    return " ".join(parser.title_parts).strip()


def _extract_visible_text(html: str) -> str:
    parser = _HTMLTextExtractor()
    parser.feed(html)
    return " ".join(parser.text_parts).strip()


def _extract_text_from_tag(html: str, tag: str | None) -> str:
    if tag is None:
        return _extract_visible_text(html)
    parser = _HTMLTextExtractor(target_tag=tag)
    parser.feed(html)
    return " ".join(parser.text_parts).strip()


def _detect_preferred_content_tag(html: str) -> str | None:
    parser = _TagPresenceDetector()
    parser.feed(html)
    if parser.has_article:
        return "article"
    if parser.has_main:
        return "main"
    return None


class _TagPresenceDetector(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.has_article = False
        self.has_main = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "article":
            self.has_article = True
        elif tag == "main":
            self.has_main = True
