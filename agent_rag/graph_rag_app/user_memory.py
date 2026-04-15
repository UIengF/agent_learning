from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable


@dataclass(frozen=True)
class UserMemory:
    preferred_language: str = ""
    answer_style: str = ""
    preferred_index_dir: str = ""
    stable_constraints: tuple[str, ...] = ()
    recurring_topics: tuple[str, ...] = ()


def _contains_chinese(text: str) -> bool:
    return bool(re.search(r"[\u4e00-\u9fff]", text))


def _dedupe(items: list[str]) -> tuple[str, ...]:
    seen: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.append(item)
    return tuple(seen)


def load_user_memory(path: str | Path, user_id: str = "default_user") -> UserMemory:
    file_path = Path(path)
    if not file_path.exists():
        return UserMemory()
    try:
        payload = json.loads(file_path.read_text(encoding="utf-8"))
    except Exception:
        return UserMemory()
    data = payload.get(user_id, {}) if isinstance(payload, dict) else {}
    if not isinstance(data, dict):
        return UserMemory()
    return UserMemory(
        preferred_language=str(data.get("preferred_language", "")),
        answer_style=str(data.get("answer_style", "")),
        preferred_index_dir=str(data.get("preferred_index_dir", "")),
        stable_constraints=tuple(data.get("stable_constraints", ()) or ()),
        recurring_topics=tuple(data.get("recurring_topics", ()) or ()),
    )


def save_user_memory(path: str | Path, memory: UserMemory, user_id: str = "default_user") -> None:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {}
    if file_path.exists():
        try:
            loaded = json.loads(file_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                payload = loaded
        except Exception:
            payload = {}
    payload[user_id] = asdict(memory)
    file_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def build_user_memory(
    messages: list[Any],
    *,
    message_content: Callable[[Any], str],
    is_human_message: Callable[[Any], bool],
    shorten: Callable[[str, int], str],
) -> UserMemory:
    preferred_language = ""
    answer_style = ""
    preferred_index_dir = ""
    stable_constraints: list[str] = []
    recurring_topics: list[str] = []

    for message in messages:
        if not is_human_message(message):
            continue
        content = message_content(message).strip()
        lowered = content.lower()
        shortened = shorten(content, 180)

        if "中文" in content or "chinese" in lowered or _contains_chinese(content):
            preferred_language = "zh"
        elif "english" in lowered or "英文" in content:
            preferred_language = "en"

        if any(token in content for token in ("简洁", "简短")) or any(token in lowered for token in ("concise", "brief")):
            answer_style = "concise"
        elif "结构化" in content or "structured" in lowered:
            answer_style = "structured"
        elif any(token in content for token in ("详细", "展开")) or "detailed" in lowered:
            answer_style = "detailed"

        if "index-dir" in lowered or "index_dir" in lowered:
            preferred_index_dir = shortened

        if any(
            marker in lowered
            for marker in (
                "本地优先",
                "不要编造",
                "local evidence",
                "local knowledge base",
                "grounded evidence",
                "source",
                "sources",
            )
        ):
            stable_constraints.append(shortened)
        else:
            recurring_topics.append(shortened)

    return UserMemory(
        preferred_language=preferred_language,
        answer_style=answer_style,
        preferred_index_dir=preferred_index_dir,
        stable_constraints=_dedupe(stable_constraints),
        recurring_topics=_dedupe(recurring_topics[:3]),
    )


def merge_user_memory(existing: UserMemory, observed: UserMemory) -> UserMemory:
    return UserMemory(
        preferred_language=observed.preferred_language or existing.preferred_language,
        answer_style=observed.answer_style or existing.answer_style,
        preferred_index_dir=observed.preferred_index_dir or existing.preferred_index_dir,
        stable_constraints=_dedupe(list(existing.stable_constraints) + list(observed.stable_constraints)),
        recurring_topics=_dedupe(list(existing.recurring_topics) + list(observed.recurring_topics))[:5],
    )


def format_user_memory(memory: UserMemory, *, shorten: Callable[[str, int], str], max_chars: int) -> str | None:
    if not any(
        (
            memory.preferred_language,
            memory.answer_style,
            memory.preferred_index_dir,
            memory.stable_constraints,
            memory.recurring_topics,
        )
    ):
        return None

    lines = ["User memory:"]
    if memory.preferred_language:
        lines.append(f"preferred_language: {memory.preferred_language}")
    if memory.answer_style:
        lines.append(f"answer_style: {memory.answer_style}")
    if memory.preferred_index_dir:
        lines.append(f"preferred_index_dir: {memory.preferred_index_dir}")
    for item in memory.stable_constraints:
        lines.append(f"stable_constraint: {item}")
    for item in memory.recurring_topics:
        lines.append(f"recurring_topic: {item}")
    return shorten("\n".join(lines), max_chars)
