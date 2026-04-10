from __future__ import annotations

from unittest import TestCase

from graph_rag_app.token_estimation import HeuristicTokenEstimator, select_token_estimator


class TokenEstimationTests(TestCase):
    def test_select_token_estimator_varies_by_model_family(self) -> None:
        gpt_estimator = select_token_estimator("gpt-4o-mini")
        qwen_estimator = select_token_estimator("qwen3.6-plus")

        self.assertNotEqual(gpt_estimator.message_overhead_tokens, qwen_estimator.message_overhead_tokens)

    def test_estimate_text_tokens_handles_mixed_text(self) -> None:
        estimator = HeuristicTokenEstimator()

        english_tokens = estimator.estimate_text_tokens("OpenAI agents use tools and memory.")
        chinese_tokens = estimator.estimate_text_tokens("请用中文简洁回答")
        mixed_tokens = estimator.estimate_text_tokens("OpenAI 工具调用 JSON payload")

        self.assertGreater(english_tokens, 0)
        self.assertGreater(chinese_tokens, 0)
        self.assertGreater(mixed_tokens, english_tokens // 2)

    def test_estimate_message_tokens_includes_role_overhead(self) -> None:
        estimator = HeuristicTokenEstimator(message_overhead_tokens=4, tool_message_overhead_tokens=8)

        user_tokens = estimator.estimate_message_tokens(
            {"role": "human", "content": "short question"},
            message_content=lambda message: str(message.get("content", "")),
            message_role=lambda message: str(message.get("role", "")),
        )
        tool_tokens = estimator.estimate_message_tokens(
            {"role": "tool", "content": '{"result_count": 1}'},
            message_content=lambda message: str(message.get("content", "")),
            message_role=lambda message: str(message.get("role", "")),
        )

        self.assertGreater(tool_tokens, user_tokens)
