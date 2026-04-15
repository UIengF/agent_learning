from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .config import AppConfig, ModelConfig
from .web_types import ScholarHit, ScholarResource

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover - runtime dependency is optional in unit tests
    ChatOpenAI = None


_SERPAPI_SEARCH_URL = "https://serpapi.com/search.json"
_YEAR_PATTERN = re.compile(r"\b(19|20)\d{2}\b")
_BULLET_PREFIX = "-*0123456789. )\t"


@dataclass(frozen=True)
class ScholarSearchResponse:
    topic: str
    planned_queries: list[str]
    result_count: int
    results: list[ScholarHit]


class SerpApiGoogleScholarBackend:
    source = "google_scholar_serpapi"

    def __init__(
        self,
        *,
        api_key: str,
        engine: str = "google_scholar",
        timeout_seconds: float = 15,
        user_agent: str = "",
    ) -> None:
        self.api_key = api_key
        self.engine = engine
        self.timeout_seconds = timeout_seconds
        self.user_agent = user_agent

    def search(self, query: str, *, count: int) -> list[ScholarHit]:
        if not self.api_key:
            raise EnvironmentError("Missing SERPAPI_API_KEY environment variable.")

        params = {
            "engine": self.engine,
            "q": query,
            "num": max(1, min(20, int(count))),
            "api_key": self.api_key,
        }
        request = Request(
            f"{_SERPAPI_SEARCH_URL}?{urlencode(params)}",
            headers={
                "User-Agent": self.user_agent,
                "Accept": "application/json",
            },
        )
        with urlopen(request, timeout=self.timeout_seconds) as response:
            charset = response.headers.get_content_charset() or "utf-8"
            payload = json.loads(response.read().decode(charset, errors="replace"))

        if isinstance(payload, dict) and payload.get("error"):
            raise RuntimeError(str(payload["error"]))
        return parse_scholar_organic_results(payload, source_query=query)


class ScholarSearchService:
    def __init__(self, *, planner: Any | None, backend: Any, max_queries: int = 4) -> None:
        self.planner = planner
        self.backend = backend
        self.max_queries = max(1, max_queries)

    def search(self, topic: str, count: int) -> ScholarSearchResponse:
        resolved_count = max(1, min(20, int(count)))
        planned_queries = plan_scholar_queries(
            topic,
            planner=self.planner,
            max_queries=self.max_queries,
        )
        all_hits: list[ScholarHit] = []
        for planned_query in planned_queries:
            all_hits.extend(self.backend.search(planned_query, count=resolved_count))

        merged_hits = merge_scholar_hits(all_hits, count=resolved_count)
        return ScholarSearchResponse(
            topic=topic,
            planned_queries=planned_queries,
            result_count=len(merged_hits),
            results=merged_hits,
        )


def build_scholar_query_planner(model_config: ModelConfig) -> Any | None:
    if ChatOpenAI is None or not model_config.api_key:
        return None
    return ChatOpenAI(
        model=model_config.model_name,
        openai_api_key=model_config.api_key,
        openai_api_base=model_config.api_base,
    )


def build_scholar_search_service(app_config: AppConfig) -> ScholarSearchService:
    return ScholarSearchService(
        planner=build_scholar_query_planner(app_config.model),
        backend=SerpApiGoogleScholarBackend(
            api_key=app_config.scholar.api_key,
            engine=app_config.scholar.engine,
            timeout_seconds=app_config.web.fetch_timeout_seconds,
            user_agent=app_config.web.user_agent,
        ),
    )


def run_scholar_search(*, topic: str, count: int, app_config: AppConfig) -> ScholarSearchResponse:
    if not app_config.scholar.enabled:
        raise EnvironmentError("Scholar search is disabled by RAG_SCHOLAR_ENABLED.")
    if not app_config.scholar.api_key:
        raise EnvironmentError("Missing SERPAPI_API_KEY environment variable.")
    service = build_scholar_search_service(app_config)
    return service.search(topic, count)


def plan_scholar_queries(topic: str, *, planner: Any | None, max_queries: int = 4) -> list[str]:
    normalized_topic = " ".join(topic.split())
    if not normalized_topic:
        return []
    if planner is None:
        return [normalized_topic]

    prompt = (
        "You generate Google Scholar search queries.\n"
        "Return a JSON array with 2 to 4 concise English search queries only.\n"
        "Favor academic terminology, paper-title phrasing, survey terms, and common research keywords.\n"
        "Do not include explanations.\n"
        f"Topic: {normalized_topic}"
    )

    try:
        response = planner.invoke(prompt)
    except Exception:
        return [normalized_topic]

    content = _response_text(response)
    parsed_queries = _parse_query_list(content)
    if not parsed_queries:
        return [normalized_topic]

    deduped: list[str] = []
    for item in parsed_queries:
        value = " ".join(item.split())
        if value and value not in deduped:
            deduped.append(value)
        if len(deduped) >= max_queries:
            break
    return deduped or [normalized_topic]


def parse_scholar_organic_results(payload: dict[str, Any], *, source_query: str) -> list[ScholarHit]:
    organic_results = payload.get("organic_results", [])
    if not isinstance(organic_results, list):
        return []

    hits: list[ScholarHit] = []
    for index, item in enumerate(organic_results, start=1):
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        url = str(item.get("link", "")).strip()
        if not title:
            continue
        publication_info = item.get("publication_info", {})
        publication_summary = ""
        if isinstance(publication_info, dict):
            publication_summary = str(publication_info.get("summary", "")).strip()
        inline_links = item.get("inline_links", {})
        cited_by_count = 0
        if isinstance(inline_links, dict):
            cited_by = inline_links.get("cited_by", {})
            if isinstance(cited_by, dict):
                cited_by_count = _safe_int(cited_by.get("total"))
        resources = _parse_resources(item.get("resources", []))
        hits.append(
            ScholarHit(
                title=title,
                url=url,
                snippet=str(item.get("snippet", "")).strip(),
                publication_summary=publication_summary,
                year=_extract_year(publication_summary),
                cited_by_count=cited_by_count,
                resources=resources,
                source_query=source_query,
                rank=index,
            )
        )
    return hits


def merge_scholar_hits(hits: list[ScholarHit], *, count: int) -> list[ScholarHit]:
    if count <= 0:
        return []

    merged: dict[tuple[str, str], dict[str, Any]] = {}
    for hit in hits:
        key = (_normalize_title(hit.title), _normalize_url(hit.url))
        bucket = merged.setdefault(
            key,
            {
                "representative": hit,
                "queries": {hit.source_query},
                "best_rank": hit.rank,
                "best_cited_by": hit.cited_by_count,
                "best_year": hit.year or 0,
            },
        )
        bucket["queries"].add(hit.source_query)
        bucket["best_rank"] = min(bucket["best_rank"], hit.rank)
        bucket["best_cited_by"] = max(bucket["best_cited_by"], hit.cited_by_count)
        bucket["best_year"] = max(bucket["best_year"], hit.year or 0)
        representative = bucket["representative"]
        if hit.rank < representative.rank:
            bucket["representative"] = hit
        elif not representative.snippet and hit.snippet:
            bucket["representative"] = hit

    ranked_entries = sorted(
        merged.values(),
        key=lambda item: (
            -len(item["queries"]),
            item["best_rank"],
            -_citation_bucket(item["best_cited_by"]),
            -(item["best_year"] or 0),
            item["representative"].title.lower(),
        ),
    )

    merged_hits: list[ScholarHit] = []
    for index, item in enumerate(ranked_entries[:count], start=1):
        representative = item["representative"]
        merged_hits.append(
            ScholarHit(
                title=representative.title,
                url=representative.url,
                snippet=representative.snippet,
                publication_summary=representative.publication_summary,
                year=representative.year,
                cited_by_count=representative.cited_by_count,
                resources=representative.resources,
                source_query=representative.source_query,
                rank=index,
                source=representative.source,
            )
        )
    return merged_hits


def _parse_resources(resources: Any) -> tuple[ScholarResource, ...]:
    if not isinstance(resources, list):
        return ()
    parsed: list[ScholarResource] = []
    for resource in resources:
        if not isinstance(resource, dict):
            continue
        link = str(resource.get("link", "")).strip()
        title = str(resource.get("title", "")).strip()
        if not link and not title:
            continue
        parsed.append(
            ScholarResource(
                title=title,
                link=link,
                file_format=str(resource.get("file_format", "")).strip() or None,
            )
        )
    return tuple(parsed)


def _extract_year(text: str) -> int | None:
    match = _YEAR_PATTERN.search(text or "")
    if not match:
        return None
    return int(match.group(0))


def _citation_bucket(value: int) -> int:
    return min(max(0, int(value)) // 25, 20)


def _normalize_title(value: str) -> str:
    return " ".join((value or "").lower().split())


def _normalize_url(value: str) -> str:
    return (value or "").strip().rstrip("/").lower()


def _safe_int(value: Any) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(str(text))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return str(content)


def _parse_query_list(content: str) -> list[str]:
    text = content.strip()
    if not text:
        return []
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = None

    if isinstance(payload, list):
        return [str(item).strip() for item in payload if str(item).strip()]

    lines = []
    for raw_line in text.splitlines():
        line = raw_line.strip().strip(",")
        line = line.lstrip(_BULLET_PREFIX).strip()
        if line.startswith('"') and line.endswith('"'):
            line = line[1:-1]
        if line:
            lines.append(line)
    return lines
