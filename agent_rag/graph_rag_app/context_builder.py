from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from .context_budget import ContextBudget
from .context_metrics import build_context_metrics
from .token_estimation import HeuristicTokenEstimator

if TYPE_CHECKING:
    from .context_metrics import ContextMetrics


@dataclass(frozen=True)
class ContextLayer:
    name: str
    role: str
    content: str
    status: str = "kept"
    estimated_chars: int = 0
    estimated_tokens: int = 0


@dataclass(frozen=True)
class ContextBuildResult:
    messages: list[Any]
    layers: tuple[ContextLayer, ...]
    dropped_layers: tuple[ContextLayer, ...] = ()
    estimated_total_chars: int = 0
    estimated_total_tokens: int = 0
    metrics: ContextMetrics | None = None


@dataclass(frozen=True)
class _LayerSpec:
    name: str
    role: str
    content: str
    estimated_chars: int
    estimated_tokens: int
    status: str = "kept"


def _build_message(
    role: str,
    content: str,
    *,
    system_message_factory: Callable[..., Any] | None,
    human_message_factory: Callable[..., Any] | None,
) -> Any:
    if role == "system" and system_message_factory is not None:
        return system_message_factory(content=content)
    if role == "human" and human_message_factory is not None:
        return human_message_factory(content=content)
    return {"role": role, "content": content}


def _estimate_message_chars(message: Any, *, message_content: Callable[[Any], str] | None) -> int:
    if message_content is not None:
        return len(message_content(message))
    if isinstance(message, dict):
        return len(str(message.get("content", "")))
    return len(str(getattr(message, "content", "")))


def _default_message_content(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("content", ""))
    return str(getattr(message, "content", ""))


def _default_message_role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role", ""))
    return str(getattr(message, "role", getattr(message, "type", "")))


def _default_tool_name(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("name", ""))
    return str(getattr(message, "name", ""))


def _message_tool_calls(message: Any) -> list[dict[str, Any]]:
    if isinstance(message, dict):
        tool_calls = message.get("tool_calls", [])
    else:
        tool_calls = getattr(message, "tool_calls", [])
    return list(tool_calls) if isinstance(tool_calls, list) else []


def _parse_tool_payload(content: str) -> dict[str, Any]:
    try:
        payload = json.loads(content)
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def _split_message_turns(
    messages: list[Any],
    *,
    message_role: Callable[[Any], str],
) -> list[list[Any]]:
    if not messages:
        return []

    turns: list[list[Any]] = []
    current_turn: list[Any] = []
    for message in messages:
        role = message_role(message).lower()
        if role in {"human", "user"}:
            if current_turn:
                turns.append(current_turn)
            current_turn = [message]
            continue
        if not current_turn:
            current_turn = [message]
        else:
            current_turn.append(message)
    if current_turn:
        turns.append(current_turn)
    return turns


def _shorten_value(text: str, limit: int, shorten: Callable[[str, int], str] | None) -> str:
    compact = " ".join(str(text).split())
    if len(compact) <= limit:
        return compact
    if shorten is not None:
        return shorten(compact, limit)
    return compact[:limit]


def _compress_tool_message(
    message: Any,
    *,
    message_content: Callable[[Any], str],
    shorten: Callable[[str, int], str] | None,
    max_fetch_chars: int,
    max_search_results: int,
) -> str:
    name = _default_tool_name(message)
    payload = _parse_tool_payload(message_content(message))
    if name == "web_search":
        query = str(payload.get("query", "")).strip()
        results = payload.get("results", [])
        urls: list[str] = []
        if isinstance(results, list):
            for result in results[:max_search_results]:
                if not isinstance(result, dict):
                    continue
                url = str(result.get("url", "")).strip()
                if not url:
                    continue
                if result.get("is_official"):
                    url = f"{url} [official]"
                urls.append(url)
        summary = ", ".join(urls) if urls else "no ranked urls"
        return f"compressed web_search: {query} -> {summary}"

    if name == "web_fetch":
        url = str(payload.get("final_url") or payload.get("url") or "").strip()
        title = str(payload.get("title", "")).strip()
        text = _shorten_value(str(payload.get("text", "")), max_fetch_chars, shorten)
        parts = [part for part in (title, url, text) if part]
        return "compressed web_fetch: " + " | ".join(parts)

    if name == "local_rag_retrieve":
        query = str(payload.get("query", "")).strip()
        count = payload.get("result_count", 0)
        return f"compressed local_rag_retrieve: {query} ({count} results)"

    return f"compressed {name}: {_shorten_value(message_content(message), 120, shorten)}"


def _compress_message(
    message: Any,
    *,
    message_content: Callable[[Any], str],
    message_role: Callable[[Any], str],
    shorten: Callable[[str, int], str] | None,
    max_fetch_chars: int,
    max_search_results: int,
) -> str:
    role = message_role(message).lower()
    if role == "tool":
        return _compress_tool_message(
            message,
            message_content=message_content,
            shorten=shorten,
            max_fetch_chars=max_fetch_chars,
            max_search_results=max_search_results,
        )

    tool_calls = _message_tool_calls(message)
    if tool_calls:
        call_summaries: list[str] = []
        for tool_call in tool_calls:
            name = str(tool_call.get("name", "tool")).strip()
            args = tool_call.get("args", {})
            if isinstance(args, dict):
                if "query" in args:
                    arg_summary = f"query={args['query']}"
                elif "url" in args:
                    arg_summary = f"url={args['url']}"
                else:
                    arg_summary = json.dumps(args, ensure_ascii=False, sort_keys=True)
            else:
                arg_summary = str(args)
            call_summaries.append(f"{name}({arg_summary})")
        return "assistant tool call: " + "; ".join(call_summaries)

    content = _shorten_value(message_content(message), 120, shorten)
    return f"{role}: {content}"


def _compress_live_messages(
    messages: list[Any],
    *,
    compression_enabled: bool,
    keep_turns: int,
    message_content: Callable[[Any], str],
    message_role: Callable[[Any], str],
    shorten: Callable[[str, int], str] | None,
    max_fetch_chars: int,
    max_search_results: int,
) -> tuple[list[Any], str | None]:
    if not messages or not compression_enabled:
        return messages, None

    turns = _split_message_turns(messages, message_role=message_role)
    if len(turns) <= keep_turns:
        return messages, None

    recent_turns = turns[-keep_turns:]
    older_turns = turns[:-keep_turns]
    recent_messages = [message for turn in recent_turns for message in turn]
    older_messages = [message for turn in older_turns for message in turn]
    if not older_messages:
        return recent_messages, None

    lines = ["Compressed live messages:"]
    for message in older_messages:
        lines.append(
            "- "
            + _compress_message(
                message,
                message_content=message_content,
                message_role=message_role,
                shorten=shorten,
                max_fetch_chars=max_fetch_chars,
                max_search_results=max_search_results,
            )
        )
    return recent_messages, "\n".join(lines)


def _truncate_layer(
    layer: _LayerSpec,
    *,
    cap: int,
    shorten: Callable[[str, int], str] | None,
) -> _LayerSpec:
    if cap <= 0 or layer.estimated_chars <= cap:
        return layer

    if shorten is not None:
        content = shorten(layer.content, cap)
    else:
        content = layer.content[:cap]

    return _LayerSpec(
        name=layer.name,
        role=layer.role,
        content=content,
        estimated_chars=len(content),
        estimated_tokens=layer.estimated_tokens,
        status="truncated",
    )


def _truncate_layer_tokens(
    layer: _LayerSpec,
    *,
    cap: int,
    shorten: Callable[[str, int], str] | None,
) -> _LayerSpec:
    if cap <= 0 or layer.estimated_tokens <= cap:
        return layer

    target_chars = max(1, min(len(layer.content), max(1, int(len(layer.content) * (cap / max(layer.estimated_tokens, 1))))))
    if shorten is not None:
        content = shorten(layer.content, target_chars)
    else:
        content = layer.content[:target_chars]

    return _LayerSpec(
        name=layer.name,
        role=layer.role,
        content=content,
        estimated_chars=len(content),
        estimated_tokens=cap,
        status="truncated",
    )


def _apply_budget(
    layers: list[_LayerSpec],
    *,
    budget: ContextBudget,
    shorten: Callable[[str, int], str] | None,
    using_tokens: bool,
) -> tuple[list[_LayerSpec], list[_LayerSpec], int, int]:
    capped_layers: list[_LayerSpec] = []
    for layer in layers:
        cap = budget.layer_caps.get(layer.name)
        token_cap = budget.layer_token_caps.get(layer.name)
        if layer.role != "conversation" and token_cap is not None:
            capped_layers.append(_truncate_layer_tokens(layer, cap=token_cap, shorten=shorten))
        elif layer.role != "conversation" and cap is not None:
            capped_layers.append(_truncate_layer(layer, cap=cap, shorten=shorten))
        else:
            capped_layers.append(layer)

    kept_layers = list(capped_layers)
    dropped_layers: list[_LayerSpec] = []

    def total_chars() -> int:
        return sum(layer.estimated_chars for layer in kept_layers)

    def total_tokens() -> int:
        return sum(layer.estimated_tokens for layer in kept_layers)

    def within_budget() -> bool:
        if using_tokens and budget.available_tokens is not None:
            return total_tokens() <= budget.available_tokens
        return total_chars() <= budget.available_chars

    if within_budget():
        return kept_layers, dropped_layers, total_chars(), total_tokens()

    for layer_name in budget.drop_order:
        if within_budget():
            break
        for index, layer in enumerate(list(kept_layers)):
            if layer.name != layer_name or layer.name in budget.protected_layers:
                continue
            dropped_layers.append(
                _LayerSpec(
                    name=layer.name,
                    role=layer.role,
                    content=layer.content,
                    estimated_chars=layer.estimated_chars,
                    estimated_tokens=layer.estimated_tokens,
                    status="dropped",
                )
            )
            del kept_layers[index]
            break

    return kept_layers, dropped_layers, total_chars(), total_tokens()


def build_context_messages(
    *,
    base_messages: list[Any],
    system_text: str | None,
    summary_text: str | None,
    user_memory_text: str | None,
    evidence_cache_text: str | None,
    task_state_text: str | None,
    reflection_text: str | None,
    budget: ContextBudget | None = None,
    shorten: Callable[[str, int], str] | None = None,
    message_content: Callable[[Any], str] | None = None,
    message_role: Callable[[Any], str] | None = None,
    token_estimator: HeuristicTokenEstimator | None = None,
    live_messages_compression_enabled: bool = True,
    live_messages_keep_turns: int = 1,
    live_messages_max_fetch_chars: int = 180,
    live_messages_max_search_results: int = 3,
    system_message_factory: Callable[..., Any] | None,
    human_message_factory: Callable[..., Any] | None,
) -> ContextBuildResult:
    planned_layers: list[_LayerSpec] = []
    estimator = token_estimator or HeuristicTokenEstimator()
    resolved_message_content = message_content or _default_message_content
    resolved_message_role = message_role or _default_message_role

    if system_text:
        planned_layers.append(
            _LayerSpec(
                name="system_prompt",
                role="system",
                content=system_text,
                estimated_chars=len(system_text),
                estimated_tokens=estimator.estimate_text_tokens(system_text),
            )
        )

    if user_memory_text:
        planned_layers.append(
            _LayerSpec(
                name="user_memory",
                role="system",
                content=user_memory_text,
                estimated_chars=len(user_memory_text),
                estimated_tokens=estimator.estimate_text_tokens(user_memory_text),
            )
        )

    if summary_text:
        planned_layers.append(
            _LayerSpec(
                name="session_summary",
                role="system",
                content=summary_text,
                estimated_chars=len(summary_text),
                estimated_tokens=estimator.estimate_text_tokens(summary_text),
            )
        )

    recent_messages, compressed_live_messages_text = _compress_live_messages(
        base_messages,
        compression_enabled=live_messages_compression_enabled,
        keep_turns=max(1, live_messages_keep_turns),
        message_content=resolved_message_content,
        message_role=resolved_message_role,
        shorten=shorten,
        max_fetch_chars=max(80, live_messages_max_fetch_chars),
        max_search_results=max(1, live_messages_max_search_results),
    )

    if compressed_live_messages_text:
        planned_layers.append(
            _LayerSpec(
                name="live_messages_compressed",
                role="system",
                content=compressed_live_messages_text,
                estimated_chars=len(compressed_live_messages_text),
                estimated_tokens=estimator.estimate_text_tokens(compressed_live_messages_text),
            )
        )

    if recent_messages:
        planned_layers.append(
            _LayerSpec(
                name="live_messages_recent" if compressed_live_messages_text else "live_messages",
                role="conversation",
                content=f"{len(recent_messages)} messages",
                estimated_chars=sum(
                    _estimate_message_chars(message, message_content=resolved_message_content)
                    for message in recent_messages
                ),
                estimated_tokens=sum(
                    estimator.estimate_message_tokens(
                        message,
                        message_content=resolved_message_content,
                        message_role=resolved_message_role,
                    )
                    for message in recent_messages
                ),
            )
        )

    if evidence_cache_text:
        planned_layers.append(
            _LayerSpec(
                name="evidence_cache",
                role="system",
                content=evidence_cache_text,
                estimated_chars=len(evidence_cache_text),
                estimated_tokens=estimator.estimate_text_tokens(evidence_cache_text),
            )
        )

    if task_state_text:
        planned_layers.append(
            _LayerSpec(
                name="task_state",
                role="system",
                content=task_state_text,
                estimated_chars=len(task_state_text),
                estimated_tokens=estimator.estimate_text_tokens(task_state_text),
            )
        )

    if reflection_text:
        planned_layers.append(
            _LayerSpec(
                name="reflection_prompt",
                role="human",
                content=reflection_text,
                estimated_chars=len(reflection_text),
                estimated_tokens=estimator.estimate_text_tokens(reflection_text),
            )
        )

    if budget is not None:
        kept_layers, dropped_layers, estimated_total_chars, estimated_total_tokens = _apply_budget(
            planned_layers,
            budget=budget,
            shorten=shorten,
            using_tokens=budget.max_tokens is not None,
        )
    else:
        kept_layers = planned_layers
        dropped_layers = []
        estimated_total_chars = sum(layer.estimated_chars for layer in kept_layers)
        estimated_total_tokens = sum(layer.estimated_tokens for layer in kept_layers)

    messages: list[Any] = []
    layers: list[ContextLayer] = []
    for layer in kept_layers:
        if layer.name in {"live_messages", "live_messages_recent"}:
            messages.extend(recent_messages)
        else:
            messages.append(
                _build_message(
                    layer.role,
                    layer.content,
                    system_message_factory=system_message_factory,
                    human_message_factory=human_message_factory,
                )
            )
        layers.append(
            ContextLayer(
                name=layer.name,
                role=layer.role,
                content=layer.content,
                status=layer.status,
                estimated_chars=layer.estimated_chars,
                estimated_tokens=layer.estimated_tokens,
            )
        )

    result = ContextBuildResult(
        messages=messages,
        layers=tuple(layers),
        dropped_layers=tuple(
            ContextLayer(
                name=layer.name,
                role=layer.role,
                content=layer.content,
                status=layer.status,
                estimated_chars=layer.estimated_chars,
                estimated_tokens=layer.estimated_tokens,
            )
            for layer in dropped_layers
        ),
        estimated_total_chars=estimated_total_chars,
        estimated_total_tokens=estimated_total_tokens,
    )
    return ContextBuildResult(
        messages=result.messages,
        layers=result.layers,
        dropped_layers=result.dropped_layers,
        estimated_total_chars=result.estimated_total_chars,
        estimated_total_tokens=result.estimated_total_tokens,
        metrics=build_context_metrics(result),
    )
