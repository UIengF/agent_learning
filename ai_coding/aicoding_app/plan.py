from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass
class CodingPlan:
    goal: str = ""
    assumptions: list[str] = field(default_factory=list)
    files_to_check: list[str] = field(default_factory=list)
    edit_steps: list[str] = field(default_factory=list)
    validation_steps: list[str] = field(default_factory=list)
    risks: list[str] = field(default_factory=list)
    updated_at: str = ""

    def update(
        self,
        *,
        goal: str | None = None,
        assumptions: list[str] | None = None,
        files_to_check: list[str] | None = None,
        edit_steps: list[str] | None = None,
        validation_steps: list[str] | None = None,
        risks: list[str] | None = None,
    ) -> "CodingPlan":
        if goal is not None:
            self.goal = goal
        if assumptions is not None:
            self.assumptions = assumptions
        if files_to_check is not None:
            self.files_to_check = files_to_check
        if edit_steps is not None:
            self.edit_steps = edit_steps
        if validation_steps is not None:
            self.validation_steps = validation_steps
        if risks is not None:
            self.risks = risks
        self.updated_at = datetime.now(timezone.utc).isoformat()
        return self

    @property
    def ready_for_edit(self) -> bool:
        return bool(self.goal and self.edit_steps)

    def to_jsonable(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_jsonable(cls, value: dict[str, Any] | None) -> "CodingPlan":
        if not value:
            return cls()
        return cls(
            goal=str(value.get("goal") or ""),
            assumptions=list(value.get("assumptions") or []),
            files_to_check=list(value.get("files_to_check") or []),
            edit_steps=list(value.get("edit_steps") or []),
            validation_steps=list(value.get("validation_steps") or []),
            risks=list(value.get("risks") or []),
            updated_at=str(value.get("updated_at") or ""),
        )


def format_plan(plan: CodingPlan) -> str:
    if not plan.goal:
        return "No coding plan has been recorded yet."
    return "\n".join(
        [
            f"Goal: {plan.goal}",
            f"Assumptions: {', '.join(plan.assumptions) or 'none'}",
            f"Files to check: {', '.join(plan.files_to_check) or 'none'}",
            f"Edit steps: {', '.join(plan.edit_steps) or 'none'}",
            f"Validation steps: {', '.join(plan.validation_steps) or 'none'}",
            f"Risks: {', '.join(plan.risks) or 'none'}",
        ]
    )
