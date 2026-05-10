from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ReportSource:
    id: str
    source_type: str
    title: str
    url_or_path: str
    snippet: str = ""
    year: str = ""
    accessed_at: str = ""


@dataclass
class ReportSection:
    heading: str
    content: str
    source_ids: list[str]


@dataclass
class CitationAudit:
    unused_sources: list[str]
    missing_refs: list[str]
    unsupported_claims: list[str]
    passed: bool


@dataclass
class ReportBrief:
    question: str
    research_plan: dict | None
    evidence_summary: str
    sources: list[ReportSource]
    gaps: list[str]
    generated_at: str
    session_id: str
    model: str
