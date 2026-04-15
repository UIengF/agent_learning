from __future__ import annotations

from unittest import TestCase

from graph_rag_app.context_builder import build_context_messages
from graph_rag_app.context_metrics import build_context_metrics, format_context_metrics


class ContextMetricsTests(TestCase):
    def test_build_context_metrics_summarizes_kept_dropped_and_truncated_layers(self) -> None:
        result = build_context_messages(
            base_messages=[{"role": "human", "content": "question"}],
            system_text="system prompt",
            summary_text="summary layer is very long",
            user_memory_text=None,
            evidence_cache_text="evidence cache is very long too",
            task_state_text=None,
            reflection_text=None,
            budget=None,
            shorten=lambda text, max_len: text[:max_len],
            message_content=lambda message: str(message.get("content", "")),
            system_message_factory=None,
            human_message_factory=None,
        )

        metrics = build_context_metrics(result)

        self.assertEqual(metrics.kept_layer_names, ("system_prompt", "session_summary", "live_messages", "evidence_cache"))
        self.assertEqual(metrics.dropped_layer_names, ())
        self.assertEqual(metrics.truncated_layer_names, ())
        self.assertGreater(metrics.estimated_total_chars, 0)
        self.assertGreater(metrics.estimated_total_tokens, 0)

    def test_format_context_metrics_includes_truncated_and_dropped_layers(self) -> None:
        result = build_context_messages(
            base_messages=[{"role": "human", "content": "current question"}],
            system_text="system prompt",
            summary_text="summary layer that becomes expendable",
            user_memory_text="user memory",
            evidence_cache_text="evidence cache",
            task_state_text="task state",
            reflection_text=None,
            budget=None,
            shorten=lambda text, max_len: text[:max_len],
            message_content=lambda message: str(message.get("content", "")),
            system_message_factory=None,
            human_message_factory=None,
        )
        metrics = build_context_metrics(result)
        rendered = format_context_metrics(metrics)

        self.assertIn("Context metrics:", rendered)
        self.assertIn("estimated_total_chars:", rendered)
        self.assertIn("estimated_total_tokens:", rendered)
        self.assertIn("kept_layers:", rendered)

