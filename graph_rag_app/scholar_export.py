from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re

from .scholar_search import ScholarSearchResponse
from .web_types import ScholarHit, ScholarResource


_OUTPUT_VERSION = 1
_DEFAULT_SOURCE = "google_scholar_serpapi"


def build_scholar_filename(
    response: ScholarSearchResponse,
    now: datetime | None = None,
) -> str:
    timestamp = (now or datetime.now().astimezone()).strftime("%Y-%m-%d-%H%M%S")
    slug = _slugify_topic(response.topic)
    return f"{timestamp}-{slug}.md"


def render_scholar_search_markdown(
    response: ScholarSearchResponse,
    *,
    generated_at: datetime | None = None,
    count_requested: int | None = None,
) -> str:
    created_at = (generated_at or datetime.now().astimezone()).isoformat()
    source = _infer_source(response)

    lines = [
        "---",
        f'topic: "{_escape_yaml(response.topic)}"',
        f"planned_queries: {_render_frontmatter_queries(response.planned_queries)}",
        f"result_count: {response.result_count}",
        f'generated_at: "{created_at}"',
        f'source: "{_escape_yaml(source)}"',
        f"count_requested: {0 if count_requested is None else count_requested}",
        f"output_version: {_OUTPUT_VERSION}",
        "---",
        "",
        f"# Scholar Search: {response.topic}",
        "",
        "## Query Summary",
        f"- Topic: `{response.topic}`",
        f"- Generated at: `{created_at}`",
        f"- Planned queries: {_render_summary_queries(response.planned_queries)}",
        f"- Result count: `{response.result_count}`",
    ]

    for item in response.results:
        lines.extend(_render_scholar_hit(item))

    return "\n".join(lines).rstrip() + "\n"


def save_scholar_search_markdown(
    response: ScholarSearchResponse,
    output_dir: str | Path,
    *,
    now: datetime | None = None,
    generated_at: datetime | None = None,
    count_requested: int | None = None,
) -> Path:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    created_at = generated_at or now or datetime.now().astimezone()
    file_path = output_path / build_scholar_filename(response, now=created_at)
    file_path.write_text(
        render_scholar_search_markdown(
            response,
            generated_at=created_at,
            count_requested=count_requested,
        ),
        encoding="utf-8",
    )
    return file_path


def _render_scholar_hit(item: ScholarHit) -> list[str]:
    return [
        "",
        f"## {item.rank}. {item.title}",
        f"- Title: {item.title}",
        f"- URL: {item.url}",
        f"- Year: {item.year if item.year is not None else 'Unknown'}",
        f"- Cited by: {item.cited_by_count}",
        f"- Publication summary: {_value_or_none(item.publication_summary)}",
        f"- Source query: {item.source_query}",
        f"- Source: {item.source}",
        "",
        "### Snippet",
        _value_or_none(item.snippet),
        "",
        "### Resources",
        *_render_resources(item.resources),
    ]


def _render_resources(resources: tuple[ScholarResource, ...]) -> list[str]:
    if not resources:
        return ["None"]
    return [
        f"- {_value_or_none(resource.title)} | {_value_or_none(resource.file_format)} | {resource.link}"
        for resource in resources
    ]


def _render_frontmatter_queries(queries: list[str]) -> str:
    if not queries:
        return "[]"
    rendered = ", ".join(f'"{_escape_yaml(query)}"' for query in queries)
    return f"[{rendered}]"


def _render_summary_queries(queries: list[str]) -> str:
    if not queries:
        return "None"
    return "; ".join(f"`{query}`" for query in queries)


def _slugify_topic(topic: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")
    return normalized or "scholar-search"


def _escape_yaml(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _value_or_none(value: str | None) -> str:
    text = str(value or "").strip()
    return text or "None"


def _infer_source(response: ScholarSearchResponse) -> str:
    if response.results:
        return response.results[0].source
    return _DEFAULT_SOURCE
