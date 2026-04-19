from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class EvidenceScope:
    prefer_local: bool
    prefer_official: bool
    allow_third_party: bool
    require_recency: bool


@dataclass(frozen=True)
class QuestionFrame:
    question: str
    target_entities: tuple[str, ...]
    task_intent: str
    focus_dimensions: tuple[str, ...]
    evidence_scope: EvidenceScope
    success_criteria: tuple[str, ...]


_ENTITY_ALIASES = {
    "openai": "OpenAI",
    "gemini": "Gemini",
    "google": "Google",
    "google deepmind": "Google DeepMind",
    "anthropic": "Anthropic",
    "claude": "Claude",
    "langgraph": "LangGraph",
    "adk": "ADK",
    "mcp": "MCP",
    "rag": "RAG",
    "codex": "Codex",
}
_CAPITALIZED_TOKEN_PATTERN = re.compile(r"\b[A-Z][A-Za-z0-9_.-]{2,}\b")
_ENTITY_TOKEN_STOPWORDS = {
    "What",
    "Why",
    "How",
    "Which",
    "Who",
    "Where",
    "When",
    "Can",
    "Could",
    "Should",
    "Would",
    "Recent",
    "Latest",
}
_INTENT_KEYWORDS = {
    "compare": ("对比", "比较", "区别", "差异", "异同", "vs", "versus"),
    "explain": ("是什么", "how", "why", "原理", "解释", "实现"),
    "debug": ("为什么", "报错", "失败", "问题", "原因", "debug", "error"),
    "recommend": ("推荐", "适合", "选择", "最好", "best", "recommend"),
    "plan": ("计划", "规划", "roadmap", "方案", "design"),
}
_FOCUS_KEYWORDS = {
    "implementation": ("实现", "implementation", "how it works", "原理", "mechanism"),
    "architecture": ("架构", "architecture", "设计", "design"),
    "tool_use": ("tool", "工具", "function calling", "tool use", "调用"),
    "reasoning": ("reasoning", "推理", "planning", "决策", "思维链"),
    "integration": ("integration", "集成", "protocol", "协议", "sdk", "api", "mcp"),
    "evaluation": ("benchmark", "评测", "效果", "性能", "quality"),
}
_RECENCY_KEYWORDS = ("最新", "最近", "当前", "today", "latest", "recent", "newest")


def build_question_frame(question: str) -> QuestionFrame:
    normalized_question = " ".join((question or "").split())
    lowered = normalized_question.lower()
    target_entities = _extract_target_entities(normalized_question, lowered)
    task_intent = _classify_task_intent(lowered)
    focus_dimensions = _extract_focus_dimensions(lowered)
    evidence_scope = EvidenceScope(
        prefer_local=True,
        prefer_official=True,
        allow_third_party=True,
        require_recency=any(keyword in lowered for keyword in _RECENCY_KEYWORDS),
    )
    success_criteria = _build_success_criteria(
        target_entities=target_entities,
        task_intent=task_intent,
        focus_dimensions=focus_dimensions,
        evidence_scope=evidence_scope,
    )
    return QuestionFrame(
        question=normalized_question,
        target_entities=target_entities,
        task_intent=task_intent,
        focus_dimensions=focus_dimensions,
        evidence_scope=evidence_scope,
        success_criteria=success_criteria,
    )


def format_question_frame(frame: QuestionFrame) -> str:
    lines = [
        "Question frame:",
        f"question: {frame.question}",
        "target_entities: " + ", ".join(frame.target_entities) if frame.target_entities else "target_entities:",
        f"task_intent: {frame.task_intent}",
        "focus_dimensions: " + ", ".join(frame.focus_dimensions)
        if frame.focus_dimensions
        else "focus_dimensions:",
        (
            "evidence_scope: "
            f"prefer_local={str(frame.evidence_scope.prefer_local).lower()}, "
            f"prefer_official={str(frame.evidence_scope.prefer_official).lower()}, "
            f"allow_third_party={str(frame.evidence_scope.allow_third_party).lower()}, "
            f"require_recency={str(frame.evidence_scope.require_recency).lower()}"
        ),
    ]
    for criterion in frame.success_criteria:
        lines.append(f"success_criteria: {criterion}")
    return "\n".join(lines)


def _extract_target_entities(question: str, lowered: str) -> tuple[str, ...]:
    seen: list[str] = []
    for alias, canonical in sorted(_ENTITY_ALIASES.items(), key=lambda item: len(item[0]), reverse=True):
        if alias in lowered and canonical not in seen:
            seen.append(canonical)
    for token in _CAPITALIZED_TOKEN_PATTERN.findall(question):
        if token in _ENTITY_TOKEN_STOPWORDS:
            continue
        if token not in seen:
            seen.append(token)
    return tuple(seen)


def _classify_task_intent(lowered: str) -> str:
    for intent, keywords in _INTENT_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            return intent
    return "explain"


def _extract_focus_dimensions(lowered: str) -> tuple[str, ...]:
    matched: list[str] = []
    for dimension, keywords in _FOCUS_KEYWORDS.items():
        if any(keyword in lowered for keyword in keywords):
            matched.append(dimension)
    if not matched:
        matched.append("core_facts")
    return tuple(matched)


def _build_success_criteria(
    *,
    target_entities: tuple[str, ...],
    task_intent: str,
    focus_dimensions: tuple[str, ...],
    evidence_scope: EvidenceScope,
) -> tuple[str, ...]:
    criteria: list[str] = []
    if len(target_entities) >= 2:
        criteria.append("cover all target entities")
    elif target_entities:
        criteria.append("ground claims about the target entity")
    if task_intent == "compare":
        criteria.append("surface meaningful differences and commonalities")
    if focus_dimensions and focus_dimensions != ("core_facts",):
        criteria.append("address the requested focus dimensions")
    if evidence_scope.prefer_official:
        criteria.append("prefer official or primary sources when available")
    if evidence_scope.require_recency:
        criteria.append("use up-to-date evidence")
    criteria.append("avoid generic sources that do not directly answer the question")
    return tuple(criteria)
