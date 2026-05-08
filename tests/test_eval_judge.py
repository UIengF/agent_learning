from __future__ import annotations

import json
from unittest import TestCase


class EvalJudgeTests(TestCase):
    def test_build_answer_judge_prompt_includes_stable_sections(self) -> None:
        from graph_rag_app.eval_judge import build_answer_judge_prompt

        prompt = build_answer_judge_prompt(
            question="Compare OpenAI and Gemini agent implementations.",
            expected_entities=["OpenAI", "Gemini"],
            must_cover_points=["ecosystem", "tool use"],
            reference_answer="Should compare ecosystem and tool use.",
            sources=[{"source_type": "web", "url": "https://example.com"}],
            answer="They differ in ecosystem and tool use.",
        )

        self.assertIn("Return JSON only", prompt)
        self.assertIn("passed", prompt)
        self.assertIn("score", prompt)
        self.assertIn("grounded", prompt)
        self.assertIn("complete", prompt)
        self.assertIn("Compare OpenAI and Gemini", prompt)

    def test_parse_answer_judge_payload_accepts_valid_json(self) -> None:
        from graph_rag_app.eval_judge import parse_answer_judge_payload

        payload = parse_answer_judge_payload(
            json.dumps(
                {
                    "passed": True,
                    "score": 0.91,
                    "reasons": ["Grounded and complete."],
                    "grounded": True,
                    "complete": True,
                }
            )
        )

        self.assertTrue(payload.passed)
        self.assertEqual(payload.score, 0.91)
        self.assertEqual(payload.reasons, ["Grounded and complete."])
        self.assertTrue(payload.grounded)
        self.assertTrue(payload.complete)

    def test_parse_answer_judge_payload_rejects_invalid_json_shape(self) -> None:
        from graph_rag_app.eval_judge import parse_answer_judge_payload

        with self.assertRaises(ValueError):
            parse_answer_judge_payload('{"foo":"bar"}')
