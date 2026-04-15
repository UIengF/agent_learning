from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ContextBudget:
    max_chars: int
    max_tokens: int | None = None
    reserved_chars: int = 0
    reserved_tokens: int = 0
    layer_caps: dict[str, int] = field(default_factory=dict)
    layer_token_caps: dict[str, int] = field(default_factory=dict)
    protected_layers: tuple[str, ...] = (
        "system_prompt",
        "live_messages",
        "live_messages_recent",
        "reflection_prompt",
    )
    drop_order: tuple[str, ...] = (
        "live_messages_compressed",
        "evidence_cache",
        "session_summary",
        "user_memory",
        "task_state",
    )

    @property
    def available_chars(self) -> int:
        return max(0, self.max_chars - self.reserved_chars)

    @property
    def available_tokens(self) -> int | None:
        if self.max_tokens is None:
            return None
        return max(0, self.max_tokens - self.reserved_tokens)
