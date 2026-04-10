from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass(frozen=True)
class EvidenceCache:
    local_results_by_query: dict[str, str] = field(default_factory=dict)
    web_results_by_query: dict[str, str] = field(default_factory=dict)
    fetched_pages_by_url: dict[str, str] = field(default_factory=dict)


def _parse_tool_payload(content: str) -> dict[str, Any]:
    try:
        payload = json.loads(content)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def build_evidence_cache(
    messages: list[Any],
    *,
    message_content: Callable[[Any], str],
    is_tool_message: Callable[[Any], bool],
    tool_name: Callable[[Any], str],
) -> EvidenceCache:
    local_results_by_query: dict[str, str] = {}
    web_results_by_query: dict[str, str] = {}
    fetched_pages_by_url: dict[str, str] = {}

    for message in messages:
        if not is_tool_message(message):
            continue
        content = message_content(message)
        payload = _parse_tool_payload(content)
        name = tool_name(message)
        if name == "local_rag_retrieve":
            query = str(payload.get("query", "")).strip()
            if query:
                local_results_by_query[query] = content
        elif name == "web_search":
            query = str(payload.get("query", "")).strip()
            if query:
                web_results_by_query[query] = content
        elif name == "web_fetch":
            for key in ("url", "final_url"):
                url = str(payload.get(key, "")).strip()
                if url:
                    fetched_pages_by_url[url] = content

    return EvidenceCache(
        local_results_by_query=local_results_by_query,
        web_results_by_query=web_results_by_query,
        fetched_pages_by_url=fetched_pages_by_url,
    )


def lookup_cached_tool_result(
    evidence_cache: EvidenceCache,
    *,
    tool_name: str,
    tool_args: dict[str, Any],
) -> str | None:
    if tool_name == "local_rag_retrieve":
        query = str(tool_args.get("query", "")).strip()
        return evidence_cache.local_results_by_query.get(query)
    if tool_name == "web_search":
        query = str(tool_args.get("query", "")).strip()
        return evidence_cache.web_results_by_query.get(query)
    if tool_name == "web_fetch":
        url = str(tool_args.get("url", "")).strip()
        return evidence_cache.fetched_pages_by_url.get(url)
    return None


def format_evidence_cache(evidence_cache: EvidenceCache) -> str | None:
    if not (
        evidence_cache.local_results_by_query
        or evidence_cache.web_results_by_query
        or evidence_cache.fetched_pages_by_url
    ):
        return None

    lines = ["Evidence cache:"]
    if evidence_cache.local_results_by_query:
        lines.append(
            "cached_local_queries: " + ", ".join(sorted(evidence_cache.local_results_by_query.keys()))
        )
    if evidence_cache.web_results_by_query:
        lines.append(
            "cached_web_queries: " + ", ".join(sorted(evidence_cache.web_results_by_query.keys()))
        )
    if evidence_cache.fetched_pages_by_url:
        lines.append(
            "cached_fetched_urls: " + ", ".join(sorted(evidence_cache.fetched_pages_by_url.keys()))
        )
    return "\n".join(lines)
