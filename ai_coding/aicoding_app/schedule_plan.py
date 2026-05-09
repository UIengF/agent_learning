from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SchedulePlan:
    task: str
    cadence: str
    workspace: str
    suggested_command: str

    def format(self) -> str:
        return "\n".join(
            [
                "Schedule plan:",
                f"- task: {self.task}",
                f"- cadence: {self.cadence}",
                f"- workspace: {self.workspace}",
                f"- suggested command: {self.suggested_command}",
                "- execution: dry-run plan only; no scheduler was created and nothing will run in the background.",
                "- risks: review command safety and runtime environment before creating any real automation.",
            ]
        )


def create_schedule_plan(workspace: str, task: str, cadence: str) -> SchedulePlan:
    suggested_command = f'python aicoding.py agent --workspace "{workspace}" --task "{task}"'
    return SchedulePlan(
        task=task,
        cadence=cadence,
        workspace=workspace,
        suggested_command=suggested_command,
    )
