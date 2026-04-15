from __future__ import annotations

import unittest

from graph_rag_app.corpus import chunk_text
from graph_rag_app.indexing import SourceDocument, _build_chunk_records
from graph_rag_app.config import IndexBuildConfig


class ChunkingTests(unittest.TestCase):
    def test_overlap_prefers_whole_sentence_boundaries(self) -> None:
        text = (
            "Alpha sentence one. Alpha sentence two.\n\n"
            "Beta sentence one. Beta sentence two.\n\n"
            "Gamma sentence one. Gamma sentence two."
        )

        chunks = chunk_text(text, chunk_size=65, chunk_overlap=30)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(chunks[1].startswith("Alpha sentence two."))
        self.assertFalse(chunks[1].startswith("pha sentence"))
        self.assertTrue(chunks[2].startswith("Beta sentence two."))

    def test_title_enhanced_chunk_includes_source_title_and_section(self) -> None:
        documents = [
            SourceDocument(
                document_id="doc-1",
                source_path="OpenAI/2026-01-23 OpenAI - Unrolling the Codex agent loop.md",
                text="# The agent loop\n\nCodex orchestrates the model and tools.",
                updated_at=1.0,
            )
        ]

        records = _build_chunk_records(documents, IndexBuildConfig(chunk_size=200, chunk_overlap=50), None)

        self.assertEqual(len(records), 1)
        self.assertIn("[SOURCE: OpenAI/2026-01-23 OpenAI - Unrolling the Codex agent loop.md]", records[0].text)
        self.assertIn("[TITLE: Unrolling the Codex agent loop]", records[0].text)
        self.assertIn("[SECTION: The agent loop]", records[0].text)
        self.assertIn("Codex orchestrates the model and tools.", records[0].text)

    def test_short_lead_chunk_is_expanded_with_following_content(self) -> None:
        lead = "Short lead."
        long_body = " ".join(f"detail{i}" for i in range(80))
        documents = [
            SourceDocument(
                document_id="doc-1",
                source_path="OpenAI/example.md",
                text=f"# Overview\n\n{lead}\n\n{long_body}",
                updated_at=1.0,
            )
        ]

        records = _build_chunk_records(documents, IndexBuildConfig(chunk_size=180, chunk_overlap=40), None)

        self.assertGreaterEqual(len(records), 2)
        self.assertIn(lead, records[0].text)
        self.assertIn("detail0", records[0].text)


if __name__ == "__main__":
    unittest.main()

