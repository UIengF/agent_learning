from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Callable


_ASCII_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_CJK_RE = re.compile(r"[\u3400-\u9fff]")
_SPACE_RE = re.compile(r"\s")


@dataclass(frozen=True)
class HeuristicTokenEstimator:
    chars_per_token: float = 4.0
    cjk_chars_per_token: float = 1.0
    punctuation_chars_per_token: float = 2.0
    message_overhead_tokens: int = 4
    tool_message_overhead_tokens: int = 8
    system_message_overhead_tokens: int = 5

    def estimate_text_tokens(self, text: str) -> int:
        if not text:
            return 0

        cjk_chars = len(_CJK_RE.findall(text))
        ascii_word_chars = sum(len(match.group(0)) for match in _ASCII_WORD_RE.finditer(text))
        whitespace_chars = len(_SPACE_RE.findall(text))
        punctuation_chars = max(0, len(text) - cjk_chars - ascii_word_chars - whitespace_chars)

        total = 0
        if ascii_word_chars:
            total += math.ceil(ascii_word_chars / self.chars_per_token)
        if cjk_chars:
            total += math.ceil(cjk_chars / self.cjk_chars_per_token)
        if punctuation_chars:
            total += math.ceil(punctuation_chars / self.punctuation_chars_per_token)
        return max(1, total)

    def estimate_message_tokens(
        self,
        message: Any,
        *,
        message_content: Callable[[Any], str],
        message_role: Callable[[Any], str],
    ) -> int:
        role = message_role(message).lower()
        content_tokens = self.estimate_text_tokens(message_content(message))
        if role == "tool":
            overhead = self.tool_message_overhead_tokens
        elif role == "system":
            overhead = self.system_message_overhead_tokens
        else:
            overhead = self.message_overhead_tokens
        return overhead + content_tokens


def select_token_estimator(model_name: str | None) -> HeuristicTokenEstimator:
    normalized = (model_name or "").strip().lower()
    if normalized.startswith("gpt"):
        return HeuristicTokenEstimator(
            chars_per_token=3.8,
            cjk_chars_per_token=1.0,
            punctuation_chars_per_token=1.8,
            message_overhead_tokens=4,
            tool_message_overhead_tokens=8,
            system_message_overhead_tokens=6,
        )
    if normalized.startswith("qwen"):
        return HeuristicTokenEstimator(
            chars_per_token=3.5,
            cjk_chars_per_token=0.9,
            punctuation_chars_per_token=1.8,
            message_overhead_tokens=3,
            tool_message_overhead_tokens=7,
            system_message_overhead_tokens=5,
        )
    return HeuristicTokenEstimator()
