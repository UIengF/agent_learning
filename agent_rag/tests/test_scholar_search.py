from __future__ import annotations

from unittest import TestCase

from graph_rag_app.scholar_search import (
    ScholarSearchResponse,
    ScholarSearchService,
    merge_scholar_hits,
    parse_scholar_organic_results,
    plan_scholar_queries,
)
from graph_rag_app.web_types import ScholarHit, ScholarResource


class _PlannerMessage:
    def __init__(self, content: str) -> None:
        self.content = content


class _FakeBackend:
    def __init__(self, responses: dict[str, list[ScholarHit]]) -> None:
        self.responses = responses
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, *, count: int) -> list[ScholarHit]:
        self.calls.append((query, count))
        return list(self.responses.get(query, []))


class ScholarSearchTests(TestCase):
    def test_plan_scholar_queries_uses_planner_output_and_limits_to_four(self) -> None:
        class FakePlanner:
            def invoke(self, prompt: str) -> _PlannerMessage:
                self.prompt = prompt
                return _PlannerMessage(
                    """
                    [
                      "graph neural networks for retrieval augmented generation",
                      "knowledge graph rag reasoning",
                      "graph-based retrieval augmented generation survey",
                      "graph rag benchmark evaluation",
                      "extra query should be dropped"
                    ]
                    """
                )

        planner = FakePlanner()

        queries = plan_scholar_queries("graph rag", planner=planner)

        self.assertEqual(
            queries,
            [
                "graph neural networks for retrieval augmented generation",
                "knowledge graph rag reasoning",
                "graph-based retrieval augmented generation survey",
                "graph rag benchmark evaluation",
            ],
        )
        self.assertIn("graph rag", planner.prompt)

    def test_plan_scholar_queries_falls_back_to_topic_on_planner_failure(self) -> None:
        class FailingPlanner:
            def invoke(self, prompt: str) -> _PlannerMessage:
                raise RuntimeError("planner unavailable")

        queries = plan_scholar_queries("adaptive agents", planner=FailingPlanner())

        self.assertEqual(queries, ["adaptive agents"])

    def test_parse_scholar_organic_results_normalizes_serpapi_fields(self) -> None:
        results = parse_scholar_organic_results(
            {
                "organic_results": [
                    {
                        "title": "Graph RAG for Scientific Discovery",
                        "link": "https://example.com/paper",
                        "snippet": "A practical study of graph-augmented retrieval.",
                        "publication_info": {"summary": "J Smith, A Lee - 2024 - Example Journal"},
                        "inline_links": {"cited_by": {"total": 42}},
                        "resources": [
                            {
                                "title": "example.com",
                                "link": "https://example.com/paper.pdf",
                                "file_format": "PDF",
                            }
                        ],
                    }
                ]
            },
            source_query="graph rag",
        )

        self.assertEqual(
            results,
            [
                ScholarHit(
                    title="Graph RAG for Scientific Discovery",
                    url="https://example.com/paper",
                    snippet="A practical study of graph-augmented retrieval.",
                    publication_summary="J Smith, A Lee - 2024 - Example Journal",
                    year=2024,
                    cited_by_count=42,
                    resources=(
                        ScholarResource(
                            title="example.com",
                            link="https://example.com/paper.pdf",
                            file_format="PDF",
                        ),
                    ),
                    source_query="graph rag",
                    rank=1,
                    source="google_scholar_serpapi",
                )
            ],
        )

    def test_merge_scholar_hits_prefers_query_coverage_before_metadata(self) -> None:
        merged = merge_scholar_hits(
            [
                ScholarHit(
                    title="Graph RAG for Scientific Discovery",
                    url="https://example.com/paper-a",
                    snippet="query one",
                    publication_summary="2024",
                    year=2024,
                    cited_by_count=5,
                    resources=(),
                    source_query="query one",
                    rank=3,
                ),
                ScholarHit(
                    title="Graph RAG for Scientific Discovery",
                    url="https://example.com/paper-a",
                    snippet="query two",
                    publication_summary="2024",
                    year=2024,
                    cited_by_count=5,
                    resources=(),
                    source_query="query two",
                    rank=4,
                ),
                ScholarHit(
                    title="Classic Retrieval Survey",
                    url="https://example.com/paper-b",
                    snippet="classic",
                    publication_summary="2020",
                    year=2020,
                    cited_by_count=500,
                    resources=(),
                    source_query="query one",
                    rank=1,
                ),
            ],
            count=2,
        )

        self.assertEqual(
            [hit.url for hit in merged],
            [
                "https://example.com/paper-a",
                "https://example.com/paper-b",
            ],
        )
        self.assertEqual(merged[0].rank, 1)
        self.assertEqual(merged[1].rank, 2)

    def test_scholar_search_service_combines_planned_queries_and_truncates(self) -> None:
        class FakePlanner:
            def invoke(self, prompt: str) -> _PlannerMessage:
                return _PlannerMessage('["query one", "query two"]')

        backend = _FakeBackend(
            {
                "query one": [
                    ScholarHit(
                        title="Paper A",
                        url="https://example.com/a",
                        snippet="from one",
                        publication_summary="2024",
                        year=2024,
                        cited_by_count=10,
                        resources=(),
                        source_query="query one",
                        rank=1,
                    )
                ],
                "query two": [
                    ScholarHit(
                        title="Paper B",
                        url="https://example.com/b",
                        snippet="from two",
                        publication_summary="2023",
                        year=2023,
                        cited_by_count=8,
                        resources=(),
                        source_query="query two",
                        rank=1,
                    )
                ],
            }
        )
        service = ScholarSearchService(planner=FakePlanner(), backend=backend)

        result = service.search("graph rag", count=1)

        self.assertIsInstance(result, ScholarSearchResponse)
        self.assertEqual(result.topic, "graph rag")
        self.assertEqual(result.planned_queries, ["query one", "query two"])
        self.assertEqual(result.result_count, 1)
        self.assertEqual([hit.url for hit in result.results], ["https://example.com/a"])
        self.assertEqual(backend.calls, [("query one", 1), ("query two", 1)])
