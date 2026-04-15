from __future__ import annotations

from unittest import TestCase

from graph_rag_app.context_budget import ContextBudget
from graph_rag_app.context_builder import ContextBuildResult, ContextLayer, build_context_messages
from graph_rag_app.token_estimation import HeuristicTokenEstimator


class ContextBuilderTests(TestCase):
    def test_build_context_messages_tracks_layer_order(self) -> None:
        result = build_context_messages(
            base_messages=[{"role": "human", "content": "current question"}],
            system_text="system prompt",
            summary_text="Session summary of earlier messages:\n- previous_topic: earlier",
            user_memory_text="User memory:\npreferred_language: zh",
            evidence_cache_text="Evidence cache:\ncached_web_queries: OpenAI",
            task_state_text="Task state:\nnext_action: web_search",
            reflection_text="You just received tool results.",
            system_message_factory=None,
            human_message_factory=None,
        )

        self.assertIsInstance(result, ContextBuildResult)
        self.assertEqual(
            tuple(layer.name for layer in result.layers),
            (
                "system_prompt",
                "user_memory",
                "session_summary",
                "live_messages",
                "evidence_cache",
                "task_state",
                "reflection_prompt",
            ),
        )
        self.assertEqual(result.messages[0]["content"], "system prompt")
        self.assertEqual(result.messages[1]["content"], "User memory:\npreferred_language: zh")
        self.assertTrue(result.messages[-1]["content"].startswith("You just received tool results."))

    def test_build_context_messages_omits_empty_layers(self) -> None:
        result = build_context_messages(
            base_messages=[{"role": "human", "content": "current question"}],
            system_text="system prompt",
            summary_text=None,
            user_memory_text=None,
            evidence_cache_text=None,
            task_state_text=None,
            reflection_text=None,
            system_message_factory=None,
            human_message_factory=None,
        )

        self.assertEqual(
            tuple(layer.name for layer in result.layers),
            ("system_prompt", "live_messages"),
        )
        self.assertEqual(
            result.messages,
            [
                {"role": "system", "content": "system prompt"},
                {"role": "human", "content": "current question"},
            ],
        )

    def test_layer_metadata_preserves_role_and_content(self) -> None:
        result = build_context_messages(
            base_messages=[{"role": "human", "content": "current question"}],
            system_text="system prompt",
            summary_text="summary",
            user_memory_text=None,
            evidence_cache_text=None,
            task_state_text="task",
            reflection_text=None,
            system_message_factory=None,
            human_message_factory=None,
        )

        self.assertEqual(tuple(layer.name for layer in result.layers), ("system_prompt", "session_summary", "live_messages", "task_state"))
        self.assertEqual(result.layers[0].content, "system prompt")
        self.assertEqual(result.layers[1].content, "summary")
        self.assertEqual(result.layers[2].content, "1 messages")
        self.assertEqual(result.layers[3].content, "task")
        self.assertEqual(result.layers[0].estimated_chars, len("system prompt"))
        self.assertEqual(result.layers[2].estimated_chars, len("current question"))
        self.assertGreater(result.layers[0].estimated_tokens, 0)
        self.assertGreater(result.layers[2].estimated_tokens, 0)

    def test_build_context_messages_drops_low_priority_layers_when_budget_is_tight(self) -> None:
        result = build_context_messages(
            base_messages=[{"role": "human", "content": "current question with enough length"}],
            system_text="system prompt",
            summary_text="summary layer that should be dropped first",
            user_memory_text="user memory that should still fit",
            evidence_cache_text="evidence cache that should be dropped first",
            task_state_text="task state that should still fit",
            reflection_text=None,
            budget=ContextBudget(max_chars=140),
            shorten=lambda text, max_len: text[:max_len],
            message_content=lambda message: str(message.get("content", "")),
            system_message_factory=None,
            human_message_factory=None,
        )

        self.assertEqual(
            tuple(layer.name for layer in result.layers),
            ("system_prompt", "user_memory", "live_messages", "task_state"),
        )
        self.assertEqual(
            tuple(layer.name for layer in result.dropped_layers),
            ("evidence_cache", "session_summary"),
        )
        self.assertLessEqual(result.estimated_total_chars, 140)

    def test_build_context_messages_applies_layer_caps_before_budget_pruning(self) -> None:
        result = build_context_messages(
            base_messages=[{"role": "human", "content": "question"}],
            system_text="system prompt",
            summary_text="summary layer is very long",
            user_memory_text=None,
            evidence_cache_text="evidence cache is very long too",
            task_state_text=None,
            reflection_text=None,
            budget=ContextBudget(
                max_chars=500,
                layer_caps={"session_summary": 12, "evidence_cache": 14},
            ),
            shorten=lambda text, max_len: text[:max_len],
            message_content=lambda message: str(message.get("content", "")),
            system_message_factory=None,
            human_message_factory=None,
        )

        self.assertEqual(result.messages[1]["content"], "summary laye")
        self.assertEqual(result.messages[3]["content"], "evidence cache")
        self.assertEqual(result.layers[1].status, "truncated")
        self.assertEqual(result.layers[3].status, "truncated")

    def test_build_context_messages_supports_token_budget(self) -> None:
        result = build_context_messages(
            base_messages=[{"role": "human", "content": "current question"}],
            system_text="system prompt",
            summary_text="summary layer that becomes expendable",
            user_memory_text="user memory",
            evidence_cache_text="evidence cache",
            task_state_text="task state",
            reflection_text=None,
            budget=ContextBudget(max_chars=500, max_tokens=20),
            shorten=lambda text, max_len: text[:max_len],
            message_content=lambda message: str(message.get("content", "")),
            message_role=lambda message: str(message.get("role", "")),
            token_estimator=HeuristicTokenEstimator(message_overhead_tokens=2),
            system_message_factory=None,
            human_message_factory=None,
        )

        self.assertLessEqual(result.estimated_total_tokens, 20)
        self.assertIn("session_summary", tuple(layer.name for layer in result.dropped_layers))
        self.assertGreater(result.layers[0].estimated_tokens, 0)

    def test_build_context_messages_splits_live_messages_into_recent_and_compressed_layers(self) -> None:
        result = build_context_messages(
            base_messages=[
                {"role": "human", "content": "older question"},
                {"role": "assistant", "content": "older reasoning"},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "search-1", "name": "web_search", "args": {"query": "recent openai news"}}],
                },
                {
                    "role": "tool",
                    "name": "web_search",
                    "content": (
                        '{"query":"recent openai news","result_count":2,"results":['
                        '{"title":"OpenAI News","url":"https://openai.com/news","snippet":"Official",'
                        '"source":"duckduckgo_html","rank":1,"is_official":true},'
                        '{"title":"BBC","url":"https://bbc.com/story","snippet":"Third-party",'
                        '"source":"duckduckgo_html","rank":2,"is_official":false}]}'
                    ),
                },
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"id": "fetch-1", "name": "web_fetch", "args": {"url": "https://openai.com/news"}}],
                },
                {
                    "role": "tool",
                    "name": "web_fetch",
                    "content": '{"url":"https://openai.com/news","title":"OpenAI News","text":"'
                    + ("A" * 240)
                    + '","truncated":false}',
                },
                {"role": "human", "content": "current question"},
            ],
            system_text="system prompt",
            summary_text=None,
            user_memory_text=None,
            evidence_cache_text="Evidence cache:\ncached_web_queries: recent openai news",
            task_state_text="Task state:\nnext_action: answer",
            reflection_text=None,
            budget=None,
            shorten=lambda text, max_len: text[:max_len],
            message_content=lambda message: str(message.get("content", "")),
            message_role=lambda message: str(message.get("role", "")),
            system_message_factory=None,
            human_message_factory=None,
        )

        self.assertEqual(
            tuple(layer.name for layer in result.layers),
            (
                "system_prompt",
                "live_messages_compressed",
                "live_messages_recent",
                "evidence_cache",
                "task_state",
            ),
        )
        self.assertIn("older question", result.messages[1]["content"])
        self.assertIn("recent openai news", result.messages[1]["content"])
        self.assertIn("https://openai.com/news", result.messages[1]["content"])
        self.assertIn("compressed", result.messages[1]["content"])
        self.assertEqual(result.messages[2]["role"], "human")
        self.assertEqual(result.messages[2]["content"], "current question")
        self.assertEqual(result.messages[-1]["content"], "Task state:\nnext_action: answer")

    def test_budget_drops_compressed_live_messages_before_task_state(self) -> None:
        result = build_context_messages(
            base_messages=[
                {"role": "human", "content": "older question"},
                {"role": "assistant", "content": "older reasoning"},
                {
                    "role": "tool",
                    "name": "web_fetch",
                    "content": '{"url":"https://openai.com/news","title":"OpenAI News","text":"'
                    + ("A" * 280)
                    + '","truncated":false}',
                },
                {"role": "human", "content": "current question with enough length"},
            ],
            system_text="system prompt",
            summary_text="summary layer",
            user_memory_text=None,
            evidence_cache_text="Evidence cache:\ncached_fetched_urls: https://openai.com/news",
            task_state_text="Task state:\nnext_action: answer",
            reflection_text=None,
            budget=ContextBudget(max_chars=180),
            shorten=lambda text, max_len: text[:max_len],
            message_content=lambda message: str(message.get("content", "")),
            message_role=lambda message: str(message.get("role", "")),
            system_message_factory=None,
            human_message_factory=None,
        )

        self.assertIn("live_messages_compressed", tuple(layer.name for layer in result.dropped_layers))
        self.assertIn("task_state", tuple(layer.name for layer in result.layers))

