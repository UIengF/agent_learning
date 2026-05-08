from __future__ import annotations

from unittest import TestCase

from graph_rag_app.retrieval import (
    SearchResult,
    expand_query_for_retrieval,
    rerank_results_by_metadata,
)


class RetrievalRerankTests(TestCase):
    def test_rerank_prefers_matching_source_title_and_section_metadata(self) -> None:
        results = [
            SearchResult(
                chunk_id=1,
                score=1.0,
                text=(
                    "[SOURCE: Anthropic/2026-08-29 Anthropic - A postmortem of three recent issues.md]\n"
                    "[TITLE: A postmortem of three recent issues]\n"
                    "[SECTION: Full Content]\n\n"
                    "Anthropic infrastructure update."
                ),
                source_path="Anthropic/2026-08-29 Anthropic - A postmortem of three recent issues.md",
                section_title="Full Content",
                strategy="hybrid",
            ),
            SearchResult(
                chunk_id=2,
                score=0.72,
                text=(
                    "[SOURCE: OpenAI/2026-03-31 OpenAI - Funding update.md]\n"
                    "[TITLE: OpenAI funding update]\n"
                    "[SECTION: Latest Announcements]\n\n"
                    "OpenAI recent changes and announcements."
                ),
                source_path="OpenAI/2026-03-31 OpenAI - Funding update.md",
                section_title="Latest Announcements",
                strategy="hybrid",
            ),
        ]

        reranked = rerank_results_by_metadata("recent changes OpenAI", results, top_k=2)

        self.assertEqual(reranked[0].chunk_id, 2)
        self.assertEqual(reranked[1].chunk_id, 1)
        self.assertEqual(reranked[0].strategy, "hybrid+metadata")

    def test_rerank_prefers_section_match_within_same_source(self) -> None:
        results = [
            SearchResult(
                chunk_id=1,
                score=0.90,
                text=(
                    "[SOURCE: OpenAI/2026-03-31 OpenAI - Funding update.md]\n"
                    "[TITLE: OpenAI funding update]\n"
                    "[SECTION: Overview]\n\n"
                    "General overview."
                ),
                source_path="OpenAI/2026-03-31 OpenAI - Funding update.md",
                section_title="Overview",
                strategy="hybrid",
            ),
            SearchResult(
                chunk_id=2,
                score=0.72,
                text=(
                    "[SOURCE: OpenAI/2026-03-31 OpenAI - Funding update.md]\n"
                    "[TITLE: OpenAI funding update]\n"
                    "[SECTION: Funding Round Details]\n\n"
                    "Detailed capital and valuation information."
                ),
                source_path="OpenAI/2026-03-31 OpenAI - Funding update.md",
                section_title="Funding Round Details",
                strategy="hybrid",
            ),
        ]

        reranked = rerank_results_by_metadata("OpenAI funding round details", results, top_k=2)

        self.assertEqual(reranked[0].chunk_id, 2)
        self.assertGreater(reranked[0].score, reranked[1].score)

    def test_expand_query_adds_domain_terms_for_semantic_paraphrase(self) -> None:
        query = (
            "Which source discusses the scaffolding needed to let an autonomous "
            "assistant keep working across long jobs?"
        )

        expanded = expand_query_for_retrieval(query)

        self.assertIn("harnesses", expanded)
        self.assertIn("long-running", expanded)
        self.assertIn("agents", expanded)

    def test_rerank_uses_expanded_query_terms_for_metadata(self) -> None:
        results = [
            SearchResult(
                chunk_id=1,
                score=0.90,
                text=(
                    "[SOURCE: OpenAI/2026-01-22 OpenAI - Scaling PostgreSQL.md]\n"
                    "[TITLE: Scaling PostgreSQL]\n"
                    "[SECTION: Full Content]\n\n"
                    "Database scaling notes."
                ),
                source_path="OpenAI/2026-01-22 OpenAI - Scaling PostgreSQL.md",
                section_title="Full Content",
                strategy="hybrid",
            ),
            SearchResult(
                chunk_id=2,
                score=0.72,
                text=(
                    "[SOURCE: Anthropic/2025-11-26 Anthropic - Effective harnesses for long-running agents.md]\n"
                    "[TITLE: Effective harnesses for long-running agents]\n"
                    "[SECTION: Full Content]\n\n"
                    "Long task support."
                ),
                source_path=(
                    "Anthropic/2025-11-26 Anthropic - Effective harnesses "
                    "for long-running agents.md"
                ),
                section_title="Full Content",
                strategy="hybrid",
            ),
        ]

        reranked = rerank_results_by_metadata(
            "Which source discusses scaffolding for long jobs?",
            results,
            top_k=2,
        )

        self.assertEqual(reranked[0].chunk_id, 2)
        self.assertEqual(reranked[0].strategy, "hybrid+metadata")
