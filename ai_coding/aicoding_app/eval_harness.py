from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class EvalTask:
    name: str
    task: str
    suggested_mode: str
    expected_validation: str


def load_eval_suite(path: str | Path) -> tuple[EvalTask, ...]:
    data: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    tasks = []
    for raw in data.get("tasks", []):
        if not isinstance(raw, dict):
            continue
        tasks.append(
            EvalTask(
                name=str(raw.get("name") or "unnamed"),
                task=str(raw.get("task") or ""),
                suggested_mode=str(raw.get("suggested_mode") or "agent"),
                expected_validation=str(raw.get("expected_validation") or "not specified"),
            )
        )
    return tuple(tasks)


def format_eval_dry_run(workspace: str, suite_path: str | Path) -> str:
    tasks = load_eval_suite(suite_path)
    lines = [
        "Eval dry-run summary:",
        f"- workspace: {workspace}",
        f"- suite: {suite_path}",
        f"- task count: {len(tasks)}",
        "- execution: skipped; no model calls, file edits, or validation commands were run.",
    ]
    for index, task in enumerate(tasks, start=1):
        lines.extend(
            [
                f"{index}. {task.name}",
                f"   task: {task.task}",
                f"   suggested mode: {task.suggested_mode}",
                f"   expected validation: {task.expected_validation}",
            ]
        )
    return "\n".join(lines)
