from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


def utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid4().hex[:12]}"


class RunStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_REVIEW = "waiting_review"
    COMPLETED = "completed"
    FAILED = "failed"
    ABORTED = "aborted"


class TaskStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    DONE = "done"
    DROPPED = "dropped"


class AgentRole(str, Enum):
    PLANNER = "planner"
    RESEARCHER = "researcher"
    SYNTHESIZER = "synthesizer"
    CRITIC = "critic"
    SINGLE_AGENT = "single_agent"


class AgentAction(str, Enum):
    PLAN = "plan"
    RESEARCH = "research"
    SYNTHESIZE = "synthesize"
    CRITIQUE = "critique"
    STOP = "stop"


@dataclass(slots=True)
class RunRecord:
    id: str
    goal: str
    runtime_mode: str
    status: RunStatus
    current_agent: str
    created_at: str
    updated_at: str
    round_index: int
    max_rounds: int
    max_parallel_agents: int
    budget_tokens: int
    budget_cost_usd: float
    spent_tokens: int = 0
    spent_cost_usd: float = 0.0
    auto_execute: bool = True
    result_artifact_id: str = ""
    result_path: str = ""
    last_error: str = ""


@dataclass(slots=True)
class TaskRecord:
    id: str
    run_id: str
    role: AgentRole
    title: str
    description: str
    status: TaskStatus
    round_index: int
    payload: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)
    updated_at: str = field(default_factory=utc_now)
    started_at: str = ""
    completed_at: str = ""
    last_error: str = ""

    @classmethod
    def new(
        cls,
        *,
        run_id: str,
        role: AgentRole,
        title: str,
        description: str,
        round_index: int,
        payload: dict[str, Any] | None = None,
    ) -> "TaskRecord":
        return cls(
            id=new_id("task"),
            run_id=run_id,
            role=role,
            title=title,
            description=description,
            status=TaskStatus.PENDING,
            round_index=round_index,
            payload=payload or {},
        )


@dataclass(slots=True)
class ArtifactRecord:
    id: str
    run_id: str
    task_id: str
    kind: str
    title: str
    content: str
    source_url: str = ""
    citation_label: str = ""
    citations: list[dict[str, str]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utc_now)
    file_path: str = ""

    @classmethod
    def new(
        cls,
        *,
        run_id: str,
        task_id: str,
        kind: str,
        title: str,
        content: str,
        source_url: str = "",
        citation_label: str = "",
        citations: list[dict[str, str]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> "ArtifactRecord":
        return cls(
            id=new_id("artifact"),
            run_id=run_id,
            task_id=task_id,
            kind=kind,
            title=title,
            content=content,
            source_url=source_url,
            citation_label=citation_label,
            citations=citations or [],
            metadata=metadata or {},
        )


@dataclass(slots=True)
class TraceEvent:
    id: str
    run_id: str
    task_id: str
    agent_role: AgentRole
    action: AgentAction
    message: str
    token_usage: int = 0
    cost_usd: float = 0.0
    duration_ms: int = 0
    created_at: str = field(default_factory=utc_now)

    @classmethod
    def new(
        cls,
        *,
        run_id: str,
        task_id: str,
        agent_role: AgentRole,
        action: AgentAction,
        message: str,
        token_usage: int = 0,
        cost_usd: float = 0.0,
        duration_ms: int = 0,
    ) -> "TraceEvent":
        return cls(
            id=new_id("trace"),
            run_id=run_id,
            task_id=task_id,
            agent_role=agent_role,
            action=action,
            message=message,
            token_usage=token_usage,
            cost_usd=cost_usd,
            duration_ms=duration_ms,
        )


@dataclass(slots=True)
class CheckpointRecord:
    id: str
    run_id: str
    phase: str
    payload: dict[str, Any]
    created_at: str = field(default_factory=utc_now)
    file_path: str = ""


@dataclass(slots=True)
class PlannedTask:
    title: str
    query: str
    description: str = ""


@dataclass(slots=True)
class PlanOutcome:
    tasks: list[PlannedTask]
    notes: str = ""
    token_usage: int = 0
    cost_usd: float = 0.0


@dataclass(slots=True)
class ResearchOutcome:
    summary: str
    citations: list[dict[str, str]]
    token_usage: int = 0
    cost_usd: float = 0.0


@dataclass(slots=True)
class SynthesisOutcome:
    markdown: str
    token_usage: int = 0
    cost_usd: float = 0.0


@dataclass(slots=True)
class CritiqueOutcome:
    approved: bool
    feedback: str
    follow_up_tasks: list[PlannedTask] = field(default_factory=list)
    token_usage: int = 0
    cost_usd: float = 0.0


@dataclass(slots=True)
class AgentRegistry:
    planner: Any
    researcher: Any
    synthesizer: Any
    critic: Any
