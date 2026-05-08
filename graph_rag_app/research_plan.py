from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from typing import Any, Type

from pydantic import BaseModel, Field

try:
    from langchain_core.tools import BaseTool
except ImportError:  # pragma: no cover
    BaseTool = object  # type: ignore[assignment]


@dataclass(frozen=True)
class ResearchPlan:
    question: str = ""
    status: str = "in_progress"
    steps: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    gaps: tuple[str, ...] = ()
    next_action: str = ""


class ResearchPlanInput(BaseModel):
    question: str = Field("", description="The current research question or task.")
    status: str = Field(
        "in_progress",
        description="Current plan status: in_progress, blocked, or completed.",
    )
    steps: list[str] = Field(default_factory=list, description="Planned or completed steps.")
    evidence: list[str] = Field(
        default_factory=list,
        description="Evidence already gathered and why it matters.",
    )
    gaps: list[str] = Field(
        default_factory=list,
        description="Known missing evidence or unresolved uncertainty.",
    )
    next_action: str = Field("", description="The smallest useful next action.")


class ResearchPlanTool(BaseTool):
    name: str = "research_plan_update"
    description: str = (
        "Record or update the research plan, gathered evidence, evidence gaps, "
        "and the smallest next action for complex RAG questions."
    )
    args_schema: Type[BaseModel] = ResearchPlanInput

    def invoke(self, input: dict[str, Any], **_: Any) -> str:
        return self._run(**input)

    def _run(
        self,
        question: str = "",
        status: str = "in_progress",
        steps: list[str] | None = None,
        evidence: list[str] | None = None,
        gaps: list[str] | None = None,
        next_action: str = "",
    ) -> str:
        plan = ResearchPlan(
            question=question.strip(),
            status=status.strip() or "in_progress",
            steps=tuple(item.strip() for item in (steps or []) if item.strip()),
            evidence=tuple(item.strip() for item in (evidence or []) if item.strip()),
            gaps=tuple(item.strip() for item in (gaps or []) if item.strip()),
            next_action=next_action.strip(),
        )
        return json.dumps({"research_plan": asdict(plan)}, ensure_ascii=False)


def _message_name(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("name", "") or "")
    return str(getattr(message, "name", "") or "")


def _message_content(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("content", "") or "")
    return str(getattr(message, "content", "") or "")


def _parse_plan_payload(content: str) -> ResearchPlan | None:
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    plan = payload.get("research_plan")
    if not isinstance(plan, dict):
        return None
    return ResearchPlan(
        question=str(plan.get("question", "") or ""),
        status=str(plan.get("status", "") or "in_progress"),
        steps=tuple(str(item) for item in plan.get("steps", []) if str(item).strip()),
        evidence=tuple(str(item) for item in plan.get("evidence", []) if str(item).strip()),
        gaps=tuple(str(item) for item in plan.get("gaps", []) if str(item).strip()),
        next_action=str(plan.get("next_action", "") or ""),
    )


def latest_research_plan(messages: list[Any]) -> ResearchPlan | None:
    for message in reversed(messages):
        if _message_name(message) != "research_plan_update":
            continue
        plan = _parse_plan_payload(_message_content(message))
        if plan is not None:
            return plan
    return None


def format_research_plan(plan: ResearchPlan | None) -> str | None:
    if plan is None:
        return None

    lines = [
        "Research plan:",
        f"status: {plan.status}",
    ]
    if plan.question:
        lines.append(f"question: {plan.question}")
    if plan.steps:
        lines.append("steps:")
        lines.extend(f"- {item}" for item in plan.steps)
    if plan.evidence:
        lines.append("evidence:")
        lines.extend(f"- {item}" for item in plan.evidence)
    if plan.gaps:
        lines.append("gaps:")
        lines.extend(f"- {item}" for item in plan.gaps)
    if plan.next_action:
        lines.append(f"next_action: {plan.next_action}")
    return "\n".join(lines)
