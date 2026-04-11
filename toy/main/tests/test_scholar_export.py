from __future__ import annotations

from datetime import datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

from graph_rag_app.scholar_export import (
    build_scholar_filename,
    render_scholar_search_markdown,
    save_scholar_search_markdown,
)
from graph_rag_app.scholar_search import ScholarSearchResponse
from graph_rag_app.web_types import ScholarHit, ScholarResource


class ScholarExportTests(TestCase):
    def test_render_markdown_includes_frontmatter_and_paper_fields(self) -> None:
        response = ScholarSearchResponse(
            topic="graph rag",
            planned_queries=["query one", "query two"],
            result_count=1,
            results=[
                ScholarHit(
                    title="Paper A",
                    url="https://example.com/a",
                    snippet="Useful snippet",
                    publication_summary="A Smith - 2024 - Example Journal",
                    year=2024,
                    cited_by_count=12,
                    resources=(
                        ScholarResource(
                            title="PDF",
                            link="https://example.com/a.pdf",
                            file_format="PDF",
                        ),
                    ),
                    source_query="query one",
                    rank=1,
                )
            ],
        )

        rendered = render_scholar_search_markdown(
            response,
            generated_at=datetime.fromisoformat("2026-04-12T10:20:30+08:00"),
            count_requested=5,
        )

        self.assertIn('topic: "graph rag"', rendered)
        self.assertIn('generated_at: "2026-04-12T10:20:30+08:00"', rendered)
        self.assertIn("count_requested: 5", rendered)
        self.assertIn("# Scholar Search: graph rag", rendered)
        self.assertIn("## Query Summary", rendered)
        self.assertIn("## 1. Paper A", rendered)
        self.assertIn("- URL: https://example.com/a", rendered)
        self.assertIn("- Year: 2024", rendered)
        self.assertIn("- Cited by: 12", rendered)
        self.assertIn("- Source query: query one", rendered)
        self.assertIn("### Snippet", rendered)
        self.assertIn("Useful snippet", rendered)
        self.assertIn("- PDF | PDF | https://example.com/a.pdf", rendered)

    def test_render_markdown_uses_none_for_missing_optional_fields(self) -> None:
        response = ScholarSearchResponse(
            topic="graph rag",
            planned_queries=[],
            result_count=1,
            results=[
                ScholarHit(
                    title="Paper A",
                    url="https://example.com/a",
                    snippet="",
                    publication_summary="",
                    year=None,
                    cited_by_count=0,
                    resources=(),
                    source_query="query one",
                    rank=1,
                )
            ],
        )

        rendered = render_scholar_search_markdown(
            response,
            generated_at=datetime.fromisoformat("2026-04-12T10:20:30+08:00"),
            count_requested=5,
        )

        self.assertIn("planned_queries: []", rendered)
        self.assertIn("- Planned queries: None", rendered)
        self.assertIn("- Year: Unknown", rendered)
        self.assertIn("- Publication summary: None", rendered)
        self.assertIn("### Snippet\nNone", rendered)
        self.assertIn("### Resources\nNone", rendered)

    def test_build_scholar_filename_sanitizes_topic(self) -> None:
        filename = build_scholar_filename(
            ScholarSearchResponse(topic="Graph RAG: Papers/2026?", planned_queries=[], result_count=0, results=[]),
            now=datetime.fromisoformat("2026-04-12T10:20:30+08:00"),
        )

        self.assertEqual(filename, "2026-04-12-102030-graph-rag-papers-2026.md")

    def test_save_markdown_creates_output_file(self) -> None:
        response = ScholarSearchResponse(
            topic="graph rag",
            planned_queries=["query one"],
            result_count=0,
            results=[],
        )

        with TemporaryDirectory() as temp_dir:
            output_path = save_scholar_search_markdown(
                response,
                output_dir=temp_dir,
                generated_at=datetime.fromisoformat("2026-04-12T10:20:30+08:00"),
                count_requested=5,
            )

            self.assertTrue(output_path.exists())
            self.assertEqual(output_path.parent, Path(temp_dir))
            self.assertIn("# Scholar Search: graph rag", output_path.read_text(encoding="utf-8"))
