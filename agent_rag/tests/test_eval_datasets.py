from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import TestCase


class EvaluationDatasetTests(TestCase):
    def _write_jsonl(self, rows: list[dict]) -> Path:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        path = Path(temp_dir.name) / "cases.jsonl"
        path.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows),
            encoding="utf-8",
        )
        return path

    def test_loads_valid_dataset_with_metadata_and_partial_assertions(self) -> None:
        from graph_rag_app.eval_datasets import load_evaluation_dataset

        dataset_path = self._write_jsonl(
            [
                {
                    "dataset": {
                        "name": "agent-smoke",
                        "version": "2026-04-19",
                        "tags": ["smoke", "comparison"],
                    }
                },
                {
                    "id": "cmp-openai-gemini",
                    "question": "OpenAI 和 Gemini 在 agent 实现中有哪些异同点？",
                    "tags": ["comparison", "web-required"],
                    "expected_entities": ["OpenAI", "Gemini"],
                    "assertions": {
                        "trajectory": {
                            "must_use_tools": ["web_search"],
                        },
                        "sources": {
                            "min_source_count": 1,
                        },
                    },
                },
            ]
        )

        dataset = load_evaluation_dataset(dataset_path)

        self.assertEqual(dataset.metadata.name, "agent-smoke")
        self.assertEqual(dataset.metadata.version, "2026-04-19")
        self.assertEqual(dataset.metadata.tags, ["smoke", "comparison"])
        self.assertEqual(len(dataset.cases), 1)
        self.assertEqual(dataset.cases[0].id, "cmp-openai-gemini")
        self.assertEqual(dataset.cases[0].expected_entities, ["OpenAI", "Gemini"])
        self.assertEqual(dataset.cases[0].assertions.trajectory.must_use_tools, ["web_search"])
        self.assertIsNone(dataset.cases[0].assertions.answer)

    def test_rejects_invalid_case_and_reports_case_location(self) -> None:
        from graph_rag_app.eval_datasets import (
            EvaluationDatasetValidationError,
            load_evaluation_dataset,
        )

        dataset_path = self._write_jsonl(
            [
                {"dataset": {"name": "agent-smoke", "version": "2026-04-19"}},
                {
                    "id": "bad-case",
                    "question": "   ",
                },
            ]
        )

        with self.assertRaises(EvaluationDatasetValidationError) as context:
            load_evaluation_dataset(dataset_path)

        self.assertIn("bad-case", str(context.exception))
        self.assertIn("line 2", str(context.exception))

    def test_filters_cases_by_tag(self) -> None:
        from graph_rag_app.eval_datasets import load_evaluation_dataset

        dataset_path = self._write_jsonl(
            [
                {"dataset": {"name": "agent-smoke", "version": "2026-04-19"}},
                {
                    "id": "local-only",
                    "question": "本地知识库有哪些关于 OpenAI 的资料？",
                    "tags": ["local"],
                },
                {
                    "id": "web-needed",
                    "question": "Gemini agent 官方文档里怎么描述 tool use？",
                    "tags": ["web-required"],
                },
            ]
        )

        dataset = load_evaluation_dataset(dataset_path, include_tags={"web-required"})

        self.assertEqual(len(dataset.cases), 1)
        self.assertEqual(dataset.cases[0].id, "web-needed")

    def test_project_smoke_dataset_covers_local_web_and_comparison_cases(self) -> None:
        from graph_rag_app.eval_datasets import load_evaluation_dataset

        dataset_path = Path("evals/datasets/agent-smoke.jsonl")

        dataset = load_evaluation_dataset(dataset_path)

        self.assertEqual(dataset.metadata.name, "agent-smoke")
        self.assertGreaterEqual(len(dataset.cases), 3)
        self.assertTrue(any("local" in case.tags for case in dataset.cases))
        self.assertTrue(any("web-required" in case.tags for case in dataset.cases))
        self.assertTrue(any("comparison" in case.tags for case in dataset.cases))
