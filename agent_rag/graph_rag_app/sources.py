from __future__ import annotations

import json
from typing import Any


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


def extract_sources_from_messages(
    messages: list[Any],
    *,
    limit: int = 8,
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
        else:
            candidates = []
        for candidate in candidates:
            key = _source_key(candidate)
            if key in by_key:
                by_key[key] = _merge_source(by_key[key], candidate)
            else:
                by_key[key] = candidate
    return list(by_key.values())[:limit]
