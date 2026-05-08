from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ContextMetrics:
    estimated_total_chars: int
    estimated_total_tokens: int
    kept_layer_names: tuple[str, ...]
    dropped_layer_names: tuple[str, ...]
    truncated_layer_names: tuple[str, ...]


def build_context_metrics(result: Any) -> ContextMetrics:
    kept_layers = tuple(getattr(result, "layers", ()) or ())
    dropped_layers = tuple(getattr(result, "dropped_layers", ()) or ())
    kept_layer_names = tuple(getattr(layer, "name", "") for layer in kept_layers)
    dropped_layer_names = tuple(getattr(layer, "name", "") for layer in dropped_layers)
    truncated_layer_names = tuple(
        getattr(layer, "name", "")
        for layer in kept_layers
        if getattr(layer, "status", "") == "truncated"
    )
    return ContextMetrics(
        estimated_total_chars=int(getattr(result, "estimated_total_chars", 0) or 0),
        estimated_total_tokens=int(getattr(result, "estimated_total_tokens", 0) or 0),
        kept_layer_names=kept_layer_names,
        dropped_layer_names=dropped_layer_names,
        truncated_layer_names=truncated_layer_names,
    )


def format_context_metrics(metrics: ContextMetrics) -> str:
    lines = [
        "Context metrics:",
        f"estimated_total_chars: {metrics.estimated_total_chars}",
        f"estimated_total_tokens: {metrics.estimated_total_tokens}",
        "kept_layers: " + ", ".join(metrics.kept_layer_names),
    ]
    if metrics.dropped_layer_names:
        lines.append("dropped_layers: " + ", ".join(metrics.dropped_layer_names))
    if metrics.truncated_layer_names:
        lines.append("truncated_layers: " + ", ".join(metrics.truncated_layer_names))
    return "\n".join(lines)
