from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchHit:
    title: str
    url: str
    snippet: str
    source: str
    rank: int
    is_official: bool = False


@dataclass(frozen=True)
class ScholarResource:
    title: str
    link: str
    file_format: str | None = None


@dataclass(frozen=True)
class ScholarHit:
    title: str
    url: str
    snippet: str
    publication_summary: str
    year: int | None
    cited_by_count: int
    resources: tuple[ScholarResource, ...]
    source_query: str
    rank: int
    source: str = "google_scholar_serpapi"


@dataclass(frozen=True)
class FetchResult:
    url: str
    final_url: str
    title: str
    text: str
    status_code: int
    content_type: str
    truncated: bool
