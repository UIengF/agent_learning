from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SearchHit:
    title: str
    url: str
    snippet: str
    source: str
    rank: int


@dataclass(frozen=True)
class FetchResult:
    url: str
    final_url: str
    title: str
    text: str
    status_code: int
    content_type: str
    truncated: bool
