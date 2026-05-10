from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .corpus import tokenize

_METADATA_STOPWORDS = {
    "a",
    "about",
    "agent",
    "agents",
    "ai",
    "an",
    "and",
    "announcement",
    "announcements",
    "change",
    "changes",
    "compare",
    "comparison",
    "difference",
    "differences",
    "for",
    "how",
    "in",
    "latest",
    "news",
    "of",
    "on",
    "recent",
    "the",
    "to",
    "update",
    "updates",
    "vs",
    "what",
    "which",
    "source",
    "explains",
    "are",
}

_RULES_PATH = Path(__file__).with_name("query_rewrite_rules.json")


def _load_query_rewrite_rules() -> tuple[tuple[tuple[str, ...], str], ...]:
    raw_rules: Any = json.loads(_RULES_PATH.read_text(encoding="utf-8"))
    return tuple(
        (tuple(str(token) for token in rule["required_tokens"]), str(rule["rewrite"]))
        for rule in raw_rules
    )


_QUERY_REWRITE_RULES = _load_query_rewrite_rules()


def _query_focus_tokens(query: str) -> tuple[str, ...]:
    tokens: list[str] = []
    for token in tokenize(query):
        normalized = token.strip().lower()
        if not normalized or normalized.isdigit():
            continue
        if len(normalized) < 3:
            continue
        if normalized in _METADATA_STOPWORDS:
            continue
        tokens.append(normalized)
    return tuple(dict.fromkeys(tokens))


def expand_query_for_retrieval(query: str) -> str:
    """Add compact domain terms for paraphrased agent-research queries."""

    original_tokens = set(_query_focus_tokens(query))
    additions: list[str] = []
    for required_tokens, rewrite in _QUERY_REWRITE_RULES:
        if all(token in original_tokens for token in required_tokens):
            additions.append(rewrite)
    if not additions:
        return query
    terms = " ".join(dict.fromkeys(" ".join(additions).split()))
    return f"{query} {terms}"
