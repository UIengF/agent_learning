from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


_CONSTRAINT_MARKERS = (
    "must",
    "should",
    "only",
    "do not",
    "don't",
    "use ",
    "based on",
    "without",
)
_QUESTION_PREFIXES = ("what", "why", "how", "which", "who", "where", "when", "can", "could")


@dataclass(frozen=True)
class SessionSummary:
    previous_topics: tuple[str, ...]
    persistent_user_constraints: tuple[str, ...]
    confirmed_facts: tuple[str, ...]
    open_questions: tuple[str, ...]
    tool_history: tuple[str, ...]


def _looks_like_constraint(text: str) -> bool:
    lowered = text.strip().lower()
    return any(marker in lowered for marker in _CONSTRAINT_MARKERS)


def _looks_like_question(text: str) -> bool:
    stripped = text.strip().lower()
    if stripped.endswith("?"):
        return True
    return stripped.startswith(_QUESTION_PREFIXES)


def build_session_summary(
    messages: list[Any],
    *,
    message_content: Callable[[Any], str],
    message_role: Callable[[Any], str],
    is_tool_message: Callable[[Any], bool],
    is_human_message: Callable[[Any], bool],
    tool_name: Callable[[Any], str],
    shorten: Callable[[str, int], str],
) -> SessionSummary | None:
    if not messages:
        return None

    previous_topics: list[str] = []
    persistent_user_constraints: list[str] = []
    confirmed_facts: list[str] = []
    open_questions: list[str] = []
    tool_history: list[str] = []

    for message in messages:
        content = message_content(message).strip()
        if not content:
            continue

        shortened = shorten(content, 180)
        if is_tool_message(message):
            tool_history.append(f"{tool_name(message)}: {shortened}")
            continue

        if is_human_message(message):
            if _looks_like_question(content):
                open_questions.append(shortened)
            elif _looks_like_constraint(content):
                persistent_user_constraints.append(shortened)
            else:
                previous_topics.append(shortened)
            continue

        if message_role(message) == "assistant":
            confirmed_facts.append(shortened)

    if not (
        previous_topics
        or persistent_user_constraints
        or confirmed_facts
        or open_questions
        or tool_history
    ):
        return None

    return SessionSummary(
        previous_topics=tuple(previous_topics),
        persistent_user_constraints=tuple(persistent_user_constraints),
        confirmed_facts=tuple(confirmed_facts),
        open_questions=tuple(open_questions),
        tool_history=tuple(tool_history),
    )


def format_session_summary(summary: SessionSummary, *, shorten: Callable[[str, int], str], max_chars: int) -> str:
    lines = ["Session summary of earlier messages:"]
    for item in summary.previous_topics:
        lines.append(f"- previous_topic: {item}")
    for item in summary.persistent_user_constraints:
        lines.append(f"- persistent_user_constraint: {item}")
    for item in summary.confirmed_facts:
        lines.append(f"- confirmed_fact: {item}")
    for item in summary.open_questions:
        lines.append(f"- open_question: {item}")
    for item in summary.tool_history:
        lines.append(f"- tool_history: {item}")
    return shorten("\n".join(lines), max_chars)
