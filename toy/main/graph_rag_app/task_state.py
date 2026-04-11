from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class TaskState:
    question: str
    entities: tuple[str, ...]
    missing_information: tuple[str, ...]
    next_action: str
    evidence_sufficiency: str


_TOKEN_PATTERN = re.compile(r"\b[\w-]+\b")
_ENTITY_STOPWORDS = {"What", "Why", "How", "Which", "Who", "Where", "When", "Can", "Could"}


def _extract_entities(text: str) -> tuple[str, ...]:
    seen: list[str] = []
    for token in _TOKEN_PATTERN.findall(text):
        if token in _ENTITY_STOPWORDS:
            continue
        has_internal_upper = any(char.isupper() for char in token[1:])
        looks_like_named_entity = token[0].isupper() and len(token) > 2
        if not (has_internal_upper or looks_like_named_entity or token.isupper()):
            continue
        if token not in seen:
            seen.append(token)
    return tuple(seen)


def _parse_tool_payload(content: str) -> dict[str, Any]:
    try:
        payload = json.loads(content)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def build_task_state(
    messages: list[Any],
    *,
    message_content: Callable[[Any], str],
    is_tool_message: Callable[[Any], bool],
    is_human_message: Callable[[Any], bool],
    tool_name: Callable[[Any], str],
) -> TaskState | None:
    if not messages:
        return None

    question = ""
    latest_tool_name = ""
    latest_tool_payload: dict[str, Any] = {}

    for message in messages:
        if is_human_message(message):
            content = message_content(message).strip()
            if content:
                question = content
        if is_tool_message(message):
            latest_tool_name = tool_name(message)
            latest_tool_payload = _parse_tool_payload(message_content(message))

    if not question:
        return None

    entities = _extract_entities(question)
    missing_information: list[str] = []
    next_action = "answer"
    evidence_sufficiency = "medium"

    if latest_tool_name == "local_rag_retrieve":
        if latest_tool_payload.get("reason") == "insufficient_evidence" or latest_tool_payload.get("result_count", 0) == 0:
            missing_information.append("insufficient_evidence")
            next_action = "web_search"
            evidence_sufficiency = "low"
        else:
            next_action = "answer"
            evidence_sufficiency = "medium"
    elif latest_tool_name == "web_search":
        if latest_tool_payload.get("result_count", 0) > 0:
            next_action = "web_fetch"
            evidence_sufficiency = "medium"
        else:
            missing_information.append("no_search_results")
            next_action = "answer"
            evidence_sufficiency = "low"
    elif latest_tool_name == "web_fetch":
        if latest_tool_payload.get("text"):
            next_action = "answer"
            evidence_sufficiency = "high"
        else:
            missing_information.append("page_content_missing")
            next_action = "web_search"
            evidence_sufficiency = "low"
    elif latest_tool_payload.get("error"):
        missing_information.append(str(latest_tool_payload.get("error")))
        next_action = "answer"
        evidence_sufficiency = "low"

    return TaskState(
        question=question,
        entities=entities,
        missing_information=tuple(missing_information),
        next_action=next_action,
        evidence_sufficiency=evidence_sufficiency,
    )


def format_task_state(task_state: TaskState, *, shorten: Callable[[str, int], str], max_chars: int) -> str:
    lines = [
        "Task state:",
        f"question: {shorten(task_state.question, 240)}",
        f"entities: {', '.join(task_state.entities)}" if task_state.entities else "entities:",
        f"next_action: {task_state.next_action}",
        f"evidence_sufficiency: {task_state.evidence_sufficiency}",
    ]
    for item in task_state.missing_information:
        lines.append(f"missing_information: {shorten(item, 120)}")
    return shorten("\n".join(lines), max_chars)
