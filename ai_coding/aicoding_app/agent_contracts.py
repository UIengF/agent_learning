from __future__ import annotations

from dataclasses import dataclass

from .harness_state import ToolRunState


@dataclass(frozen=True)
class TaskResult:
    session_id: str
    task_id: str
    response: str


@dataclass(frozen=True)
class CompletionStatus:
    status: str
    completed_items: list[str]
    missing_items: list[str]
    smoke_commands: list[str]
    blocked_items: list[str]
    skipped_commands: list[str]


class HarnessStop(RuntimeError):
    def __init__(self, message: str, run_state: ToolRunState):
        super().__init__(message)
        self.run_state = run_state
