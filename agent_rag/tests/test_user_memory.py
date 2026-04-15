from __future__ import annotations

import tempfile
from pathlib import Path
from unittest import TestCase

from graph_rag_app.user_memory import (
    UserMemory,
    build_user_memory,
    load_user_memory,
    merge_user_memory,
    save_user_memory,
)


class UserMemoryTests(TestCase):
    def test_build_user_memory_extracts_preferences_and_constraints(self) -> None:
        memory = build_user_memory(
            [
                {"role": "human", "content": "Please answer in Chinese and keep it concise."},
                {"role": "human", "content": "Use local evidence first and do not invent facts."},
                {"role": "human", "content": "OpenAI and Gemini agent comparison"},
            ],
            message_content=lambda message: str(message.get("content", "")),
            is_human_message=lambda message: str(message.get("role", "")) == "human",
            shorten=lambda text, max_len: text[:max_len],
        )

        self.assertEqual(memory.preferred_language, "zh")
        self.assertEqual(memory.answer_style, "concise")
        self.assertIn("Use local evidence first and do not invent facts.", memory.stable_constraints)
        self.assertIn("OpenAI and Gemini agent comparison", memory.recurring_topics)

    def test_save_and_load_user_memory_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "user_memory.json"
            memory = UserMemory(preferred_language="zh", answer_style="concise")
            save_user_memory(path, memory, user_id="default_user")

            loaded = load_user_memory(path, user_id="default_user")

        self.assertEqual(loaded.preferred_language, "zh")
        self.assertEqual(loaded.answer_style, "concise")

    def test_merge_user_memory_prefers_new_values_and_unions_lists(self) -> None:
        existing = UserMemory(
            preferred_language="zh",
            stable_constraints=("Use local evidence first",),
            recurring_topics=("OpenAI",),
        )
        observed = UserMemory(
            answer_style="concise",
            stable_constraints=("Do not invent facts",),
            recurring_topics=("Gemini",),
        )

        merged = merge_user_memory(existing, observed)

        self.assertEqual(merged.preferred_language, "zh")
        self.assertEqual(merged.answer_style, "concise")
        self.assertEqual(merged.stable_constraints, ("Use local evidence first", "Do not invent facts"))
        self.assertEqual(merged.recurring_topics, ("OpenAI", "Gemini"))

