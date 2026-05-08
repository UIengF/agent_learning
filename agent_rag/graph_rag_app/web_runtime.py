from __future__ import annotations

from .config import WebConfig
from .web_search import (
    DuckDuckGoHtmlSearchBackend,
    FallbackSearchBackend,
    MultiQuerySearchBackend,
    SearXngSearchBackend,
)


def build_configured_web_search_backend(web_config: WebConfig) -> MultiQuerySearchBackend:
    duckduckgo = DuckDuckGoHtmlSearchBackend(
        timeout_seconds=web_config.fetch_timeout_seconds,
        user_agent=web_config.user_agent,
    )
    searxng = SearXngSearchBackend(
        base_url=web_config.searxng_url,
        timeout_seconds=web_config.fetch_timeout_seconds,
        user_agent=web_config.user_agent,
        engines=web_config.searxng_engines,
        categories=web_config.searxng_categories,
        language=web_config.searxng_language,
    )
    if web_config.search_provider == "searxng":
        backend = FallbackSearchBackend([searxng, duckduckgo])
    elif web_config.search_provider == "duckduckgo_html":
        backend = FallbackSearchBackend([duckduckgo, searxng])
    else:
        raise ValueError(f"Unsupported web search provider: {web_config.search_provider}")
    return MultiQuerySearchBackend(backend)
