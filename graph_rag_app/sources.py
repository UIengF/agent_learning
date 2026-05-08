from __future__ import annotations

import json
from typing import Any

DEFAULT_LOCAL_SOURCE_CONFIDENCE_THRESHOLD = 0.9
DEFAULT_MAX_WEB_SOURCES = 3


def _message_name(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("name", ""))
    return str(getattr(message, "name", ""))


def _message_content(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("content", ""))
    return str(getattr(message, "content", ""))


def _parse_json_payload(message: Any) -> dict[str, Any]:
    try:
        payload = json.loads(_message_content(message))
    except (TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _short_text(value: Any, limit: int = 700) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _source_key(source: dict[str, Any]) -> tuple[str, str, str]:
    if source.get("source_type") == "web":
        return ("web", str(source.get("url", "")), "")
    if source.get("source_type") == "scholar":
        return ("scholar", str(source.get("url", "")), str(source.get("title", "")))
    return (
        "local",
        str(source.get("source_path", "")),
        str(source.get("section_title", "")),
    )


def _merge_source(existing: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    merged = dict(existing)
    for key, value in incoming.items():
        if value in {"", None} or value == []:
            continue
        if key == "text" and merged.get("text"):
            continue
        merged[key] = value
    return merged


def _local_sources(payload: dict[str, Any]) -> list[dict[str, Any]]:
    results = payload.get("results", [])
    if not isinstance(results, list):
        return []
    sources: list[dict[str, Any]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        source_path = str(item.get("source_path", "")).strip()
        text = _short_text(item.get("text", ""))
        if not source_path and not text:
            continue
        sources.append(
            {
                "source_type": "local",
                "source_path": source_path,
                "section_title": str(item.get("section_title", "") or ""),
                "score": float(item.get("score", 0.0) or 0.0),
                "strategy": str(item.get("strategy", "") or ""),
                "text": text,
                "document_id": str(item.get("document_id", "") or ""),
                "chunk_id": int(item.get("chunk_id", -1) or -1),
            }
        )
    return sources


def _web_fetch_source(payload: dict[str, Any]) -> list[dict[str, Any]]:
    url = str(payload.get("final_url") or payload.get("url") or "").strip()
    if not url:
        return []
    return [
        {
            "source_type": "web",
            "url": url,
            "title": str(payload.get("title", "") or ""),
            "snippet": "",
            "text": _short_text(payload.get("text", "")),
            "rank": 0,
            "source": "web_fetch",
        }
    ]


def _scholar_sources(payload: dict[str, Any]) -> list[dict[str, Any]]:
    results = payload.get("results", [])
    if not isinstance(results, list):
        return []
    sources: list[dict[str, Any]] = []
    for item in results:
        if not isinstance(item, dict):
            continue
        url = str(item.get("url", "") or "").strip()
        title = str(item.get("title", "") or "").strip()
        snippet = _short_text(item.get("snippet", ""))
        publication_summary = _short_text(item.get("publication_summary", ""))
        if not url and not title and not snippet:
            continue
        sources.append(
            {
                "source_type": "scholar",
                "url": url,
                "title": title,
                "snippet": snippet,
                "text": publication_summary,
                "year": item.get("year"),
                "cited_by_count": int(item.get("cited_by_count", 0) or 0),
                "source_query": str(item.get("source_query", "") or ""),
                "rank": int(item.get("rank", 0) or 0),
                "source": str(item.get("source", "") or "scholar_search"),
                "resources": item.get("resources", []),
            }
        )
    return sources


def _sorted_local_sources(
    sources: list[dict[str, Any]],
    *,
    confidence_threshold: float,
) -> list[dict[str, Any]]:
    filtered = [
        source
        for source in sources
        if float(source.get("score", 0.0) or 0.0) >= confidence_threshold
    ]
    return sorted(filtered, key=lambda item: float(item.get("score", 0.0) or 0.0), reverse=True)


def _sorted_web_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(sources, key=lambda item: int(item.get("rank", 0) or 0))


def _sorted_scholar_sources(sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        sources,
        key=lambda item: (
            int(item.get("rank", 0) or 0),
            -int(item.get("cited_by_count", 0) or 0),
            -(int(item.get("year", 0) or 0)),
        ),
    )


def _select_sources(
    local_sources: list[dict[str, Any]],
    web_sources: list[dict[str, Any]],
    scholar_sources: list[dict[str, Any]],
    *,
    limit: int,
    local_confidence_threshold: float,
    max_web_sources: int,
) -> list[dict[str, Any]]:
    selected_local = _sorted_local_sources(
        local_sources,
        confidence_threshold=local_confidence_threshold,
    )
    selected_web = _sorted_web_sources(web_sources)[:max_web_sources]
    selected_scholar = _sorted_scholar_sources(scholar_sources)

    if selected_scholar:
        remaining = max(0, limit - len(selected_local) - len(selected_web))
        return (selected_local + selected_web + selected_scholar[:remaining])[:limit]

    remaining = max(0, limit - len(selected_web))
    return (selected_web + selected_local[:remaining])[:limit]


def extract_sources_from_messages(
    messages: list[Any],
    *,
    limit: int = 8,
    local_confidence_threshold: float = DEFAULT_LOCAL_SOURCE_CONFIDENCE_THRESHOLD,
    max_web_sources: int = DEFAULT_MAX_WEB_SOURCES,
) -> list[dict[str, Any]]:
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for message in messages:
        name = _message_name(message)
        payload = _parse_json_payload(message)
        if not payload:
            continue
        if name == "web_search":
            continue
        if name == "local_rag_retrieve":
            candidates = _local_sources(payload)
        elif name == "web_fetch":
            candidates = _web_fetch_source(payload)
        elif name == "scholar_search":
            candidates = _scholar_sources(payload)
        else:
            candidates = []
        for candidate in candidates:
            key = _source_key(candidate)
            if key in by_key:
                by_key[key] = _merge_source(by_key[key], candidate)
            else:
                by_key[key] = candidate

    local_sources: list[dict[str, Any]] = []
    web_sources: list[dict[str, Any]] = []
    scholar_sources: list[dict[str, Any]] = []
    for source in by_key.values():
        source_type = str(source.get("source_type", "") or "")
        if source_type == "local":
            local_sources.append(source)
        elif source_type == "web":
            web_sources.append(source)
        elif source_type == "scholar":
            scholar_sources.append(source)

    return _select_sources(
        local_sources,
        web_sources,
        scholar_sources,
        limit=limit,
        local_confidence_threshold=local_confidence_threshold,
        max_web_sources=max_web_sources,
    )
