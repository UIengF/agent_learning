from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
from urllib.parse import parse_qs, quote_plus, unquote, urljoin, urlparse, urlsplit, urlunsplit
from urllib.request import Request, urlopen

from .web_types import SearchHit

_DDG_SEARCH_URL = "https://html.duckduckgo.com/html/?q={query}"
_OFFICIAL_DOMAINS = (
    "openai.com",
    "openai.github.io",
    "platform.openai.com",
    "developers.openai.com",
    "google.com",
    "ai.google.dev",
    "cloud.google.com",
    "developers.googleblog.com",
    "google.github.io",
    "deepmind.google",
)
_DOC_HINTS = ("/docs", "/doc", "/guide", "/guides", "/api", "/reference", "/sdk")
_QUERY_ENTITY_DOMAIN_HINTS = {
    "openai": ("openai.com", "openai.github.io", "platform.openai.com", "developers.openai.com"),
    "anthropic": ("anthropic.com", "docs.anthropic.com"),
    "gemini": ("google.com", "ai.google.dev", "google.github.io", "deepmind.google"),
    "google": ("google.com", "ai.google.dev", "cloud.google.com", "google.github.io", "deepmind.google"),
}


def parse_duckduckgo_html(html: str, top_k: int) -> list[SearchHit]:
    if top_k <= 0:
        return []

    parser = _DuckDuckGoHtmlParser()
    parser.feed(html)
    return parser.hits[:top_k]


class DuckDuckGoHtmlSearchBackend:
    source = "duckduckgo_html"

    def __init__(self, timeout_seconds: float, user_agent: str) -> None:
        self.timeout_seconds = timeout_seconds
        self.user_agent = user_agent

    def search(self, query: str, *, top_k: int) -> list[SearchHit]:
        url = _DDG_SEARCH_URL.format(query=quote_plus(query))
        request = Request(
            url,
            headers={
                "User-Agent": self.user_agent,
                "Accept": "text/html,application/xhtml+xml",
            },
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            html = response.read().decode(charset, errors="replace")
        return parse_duckduckgo_html(html, top_k=top_k)


class MultiQuerySearchBackend:
    def __init__(self, backend) -> None:  # noqa: ANN001
        self.backend = backend
        self.source = getattr(backend, "source", "merged_search")
        self.last_debug: dict[str, object] = {}

    def search(self, query: str, *, top_k: int) -> list[SearchHit]:
        expanded_queries = expand_search_queries(query)
        all_hits: list[SearchHit] = []
        per_query_counts: list[dict[str, object]] = []
        for expanded_query in expanded_queries:
            hits = self.backend.search(expanded_query, top_k=top_k)
            all_hits.extend(hits)
            per_query_counts.append({"query": expanded_query, "result_count": len(hits)})

        merged = merge_ranked_hits(all_hits, top_k=top_k, query=query)
        self.last_debug = {
            "expanded_queries": expanded_queries,
            "per_query_counts": per_query_counts,
            "raw_result_count": len(all_hits),
            "merged_result_count": len(merged),
            "selected_urls": [hit.url for hit in merged],
            "official_urls": [hit.url for hit in merged if hit.is_official],
        }
        return merged


def expand_search_queries(query: str) -> list[str]:
    normalized = " ".join(query.split())
    expanded = [normalized]
    lowered = normalized.lower()
    if "openai" in lowered and ("gemini" in lowered or "google" in lowered):
        _append_unique(expanded, "OpenAI Agents SDK vs Google ADK")
    if "openai" in lowered and "adk" in lowered and "google" not in lowered:
        _append_unique(expanded, "OpenAI Agents SDK vs Google ADK")
    return expanded


def merge_ranked_hits(hits: list[SearchHit], *, top_k: int, query: str = "") -> list[SearchHit]:
    if top_k <= 0:
        return []

    best_by_url: dict[str, tuple[tuple[int, int, int], SearchHit]] = {}
    for hit in hits:
        normalized_url = _normalize_url(hit.url)
        candidate_key = _rank_key(hit, query=query)
        existing = best_by_url.get(normalized_url)
        if existing is None or candidate_key > existing[0]:
            best_by_url[normalized_url] = (candidate_key, hit)

    merged = [item[1] for item in sorted(best_by_url.values(), key=lambda item: item[0], reverse=True)]
    return [
        SearchHit(
            title=hit.title,
            url=_normalize_url(hit.url),
            snippet=hit.snippet,
            source=hit.source,
            rank=index + 1,
            is_official=_is_official_url(hit.url, query=query),
        )
        for index, hit in enumerate(merged[:top_k])
    ]


@dataclass
class _ResultCard:
    title: str = ""
    url: str = ""
    snippet: str = ""


class _DuckDuckGoHtmlParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.hits: list[SearchHit] = []
        self._current_card: _ResultCard | None = None
        self._result_depth = 0
        self._capture_title = False
        self._capture_snippet = False
        self._title_parts: list[str] = []
        self._snippet_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attr_map = {name: value for name, value in attrs}
        classes = _split_classes(attr_map.get("class"))
        if self._current_card is not None and tag == "div":
            self._result_depth += 1
            return

        if tag == "div" and "result" in classes:
            self._start_result_card()
            return

        if self._current_card is None:
            return

        if tag == "a" and "result__a" in classes:
            self._capture_title = True
            self._title_parts = []
            href = attr_map.get("href") or ""
            self._current_card.url = _resolve_duckduckgo_url(href)
            return

        if tag == "a" and "result__snippet" in classes:
            self._capture_snippet = True
            self._snippet_parts = []

    def handle_endtag(self, tag: str) -> None:
        if self._current_card is None:
            return

        if tag == "a" and self._capture_title:
            self._capture_title = False
            self._current_card.title = " ".join(self._title_parts).strip()
            return

        if tag == "a" and self._capture_snippet:
            self._capture_snippet = False
            self._current_card.snippet = " ".join(self._snippet_parts).strip()
            return

        if tag == "div":
            if self._result_depth > 0:
                self._result_depth -= 1
            if self._result_depth == 0:
                self._finalize_result_card()

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if not text or self._current_card is None:
            return
        if self._capture_title:
            self._title_parts.append(text)
        elif self._capture_snippet:
            self._snippet_parts.append(text)

    def _start_result_card(self) -> None:
        self._current_card = _ResultCard()
        self._result_depth = 1

    def _finalize_result_card(self) -> None:
        card = self._current_card
        self._current_card = None
        self._result_depth = 0
        self._capture_title = False
        self._capture_snippet = False
        self._title_parts = []
        self._snippet_parts = []
        if card is None or not card.title or not card.url:
            return
        self.hits.append(
            SearchHit(
                title=card.title,
                url=card.url,
                snippet=card.snippet,
                source="duckduckgo_html",
                rank=len(self.hits) + 1,
            )
        )


def _split_classes(class_attr: str | None) -> set[str]:
    if not class_attr:
        return set()
    return {part for part in class_attr.split() if part}


def _resolve_duckduckgo_url(href: str) -> str:
    parsed = urlparse(href)
    if parsed.path == "/l/":
        query = parse_qs(parsed.query)
        redirected = query.get("uddg", [""])[0]
        if redirected:
            return unquote(redirected)
    return urljoin("https://html.duckduckgo.com", href)


def _append_unique(items: list[str], value: str) -> None:
    if value not in items:
        items.append(value)


def _normalize_url(url: str) -> str:
    parts = urlsplit(url)
    normalized_path = parts.path.rstrip("/") or "/"
    if normalized_path == "/":
        normalized_path = ""
    return urlunsplit((parts.scheme, parts.netloc.lower(), normalized_path, "", ""))


def _rank_key(hit: SearchHit, *, query: str) -> tuple[int, int, int]:
    return (
        _official_domain_boost(hit.url, query=query),
        _doc_path_boost(hit.url),
        -int(hit.rank),
    )


def _official_domain_boost(url: str, *, query: str) -> int:
    hostname = urlparse(url).hostname or ""
    hostname = hostname.lower()
    preferred_domains = _preferred_domains_for_query(query)
    for domain in preferred_domains:
        if hostname == domain or hostname.endswith(f".{domain}"):
            return 2
    for domain in _OFFICIAL_DOMAINS:
        if hostname == domain or hostname.endswith(f".{domain}"):
            return 1
    return 0


def _preferred_domains_for_query(query: str) -> tuple[str, ...]:
    lowered = query.lower()
    matched_domains: list[str] = []
    for token, domains in _QUERY_ENTITY_DOMAIN_HINTS.items():
        if token in lowered:
            for domain in domains:
                if domain not in matched_domains:
                    matched_domains.append(domain)
    return tuple(matched_domains)


def _is_official_url(url: str, *, query: str) -> bool:
    return _official_domain_boost(url, query=query) > 0


def _doc_path_boost(url: str) -> int:
    path = (urlparse(url).path or "").lower()
    return 1 if any(hint in path for hint in _DOC_HINTS) else 0
