from __future__ import annotations

from unittest import TestCase

from graph_rag_app.question_frame import build_question_frame, format_question_frame


class QuestionFrameTests(TestCase):
    def test_build_question_frame_extracts_entities_intent_and_focus(self) -> None:
        frame = build_question_frame("openai和gemini在agent实现中有哪些异同点")

        self.assertEqual(frame.target_entities, ("OpenAI", "Gemini"))
        self.assertEqual(frame.task_intent, "compare")
        self.assertEqual(frame.focus_dimensions, ("implementation",))
        self.assertTrue(frame.evidence_scope.prefer_official)
        self.assertIn("cover all target entities", frame.success_criteria)
        self.assertIn("surface meaningful differences and commonalities", frame.success_criteria)
        self.assertIn("address the requested focus dimensions", frame.success_criteria)

    def test_build_question_frame_marks_recency_when_requested(self) -> None:
        frame = build_question_frame("最新的OpenAI agent API变化是什么")

        self.assertEqual(frame.task_intent, "explain")
        self.assertTrue(frame.evidence_scope.require_recency)
        self.assertIn("use up-to-date evidence", frame.success_criteria)

    def test_format_question_frame_renders_stable_sections(self) -> None:
        frame = build_question_frame("为什么LangGraph调用tool失败")
        rendered = format_question_frame(frame)

        self.assertIn("Question frame:", rendered)
        self.assertIn("question: 为什么LangGraph调用tool失败", rendered)
        self.assertIn("target_entities: LangGraph", rendered)
        self.assertIn("task_intent: debug", rendered)
        self.assertIn("focus_dimensions: tool_use", rendered)
