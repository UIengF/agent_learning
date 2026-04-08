from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from local_research_swarm.models import CritiqueOutcome, PlanOutcome, PlannedTask, ResearchOutcome, SynthesisOutcome


def _normalize_tasks(items: Iterable[dict[str, Any] | PlannedTask]) -> list[PlannedTask]:
    tasks: list[PlannedTask] = []
    for item in items:
        if isinstance(item, PlannedTask):
            tasks.append(item)
            continue
        tasks.append(
            PlannedTask(
                title=item["title"],
                query=item["query"],
                description=item.get("description", ""),
            )
        )
    return tasks


def _normalize_summary(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(str(item).strip() for item in value if str(item).strip())
    if value is None:
        return ""
    return str(value)


def _normalize_citations(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list):
        return []
    citations: list[dict[str, str]] = []
    for item in value:
        if isinstance(item, dict):
            citations.append(
                {
                    "title": str(item.get("title", "")).strip(),
                    "url": str(item.get("url", "")).strip(),
                    "snippet": str(item.get("snippet", "")).strip(),
                }
            )
    return citations


class PlannerAgent:
    def __init__(self, *, llm_backend: Any) -> None:
        self.llm_backend = llm_backend

    async def plan(self, goal: str, round_index: int, artifacts: list[Any]) -> PlanOutcome:
        raw = await self.llm_backend.plan(goal, round_index, artifacts)
        if isinstance(raw, PlanOutcome):
            return raw
        return PlanOutcome(
            tasks=_normalize_tasks(raw.get("tasks", [])),
            notes=raw.get("notes", ""),
            token_usage=int(raw.get("token_usage", 0)),
            cost_usd=float(raw.get("cost_usd", 0.0)),
        )


class ResearcherAgent:
    def __init__(self, *, llm_backend: Any, search_tool: Any, web_reader: Any, search_limit: int = 2) -> None:
        self.llm_backend = llm_backend
        self.search_tool = search_tool
        self.web_reader = web_reader
        self.search_limit = search_limit

    async def research(self, goal: str, task: Any, artifacts: list[Any]) -> ResearchOutcome:
        del artifacts
        query = task.payload.get("query", task.title) if hasattr(task, "payload") else task.query
        results = await self.search_tool.search(query, limit=self.search_limit)
        pages = []
        for result in results:
            try:
                pages.append(await self.web_reader.read(result.url))
            except Exception:
                continue
        raw = await self.llm_backend.summarize_research(goal, task, pages)
        if isinstance(raw, ResearchOutcome):
            return raw
        return ResearchOutcome(
            summary=_normalize_summary(raw.get("summary", "")),
            citations=_normalize_citations(raw.get("citations", [])),
            token_usage=int(raw.get("token_usage", 0)),
            cost_usd=float(raw.get("cost_usd", 0.0)),
        )


class SynthesizerAgent:
    def __init__(self, *, llm_backend: Any) -> None:
        self.llm_backend = llm_backend

    async def synthesize(self, goal: str, artifacts: list[Any]) -> SynthesisOutcome:
        raw = await self.llm_backend.synthesize(goal, artifacts)
        if isinstance(raw, SynthesisOutcome):
            return raw
        return SynthesisOutcome(
            markdown=raw.get("markdown", ""),
            token_usage=int(raw.get("token_usage", 0)),
            cost_usd=float(raw.get("cost_usd", 0.0)),
        )


class CriticAgent:
    def __init__(self, *, llm_backend: Any) -> None:
        self.llm_backend = llm_backend

    async def critique(self, goal: str, draft_markdown: str, artifacts: list[Any], round_index: int) -> CritiqueOutcome:
        raw = await self.llm_backend.critique(goal, draft_markdown, artifacts, round_index)
        if isinstance(raw, CritiqueOutcome):
            return raw
        return CritiqueOutcome(
            approved=bool(raw.get("approved", False)),
            feedback=raw.get("feedback", ""),
            follow_up_tasks=_normalize_tasks(raw.get("follow_up_tasks", [])),
            token_usage=int(raw.get("token_usage", 0)),
            cost_usd=float(raw.get("cost_usd", 0.0)),
        )
