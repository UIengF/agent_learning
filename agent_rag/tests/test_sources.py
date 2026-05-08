from __future__ import annotations

import json
from unittest import TestCase

from graph_rag_app.sources import extract_sources_from_messages


class FakeToolMessage:
    def __init__(self, name: str, content: str) -> None:
        self.name = name
        self.content = content
        self.type = "tool"


class SourceExtractionTests(TestCase):
    def test_extracts_local_rag_sources_from_tool_payload(self) -> None:
        messages = [
            FakeToolMessage(
                "local_rag_retrieve",
                json.dumps(
                    {
                        "results": [
                            {
                                "chunk_id": 3,
                                "score": 0.92,
                                "text": "Agents use tools with grounded evidence.",
                                "document_id": "doc-1",
                                "source_path": "OpenAI/agents.md",
                                "section_title": "Tools",
                                "strategy": "hybrid",
                            }
                        ]
                    }
                ),
            )
        ]

        sources = extract_sources_from_messages(messages)

        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["source_type"], "local")
        self.assertEqual(sources[0]["source_path"], "OpenAI/agents.md")
        self.assertEqual(sources[0]["section_title"], "Tools")
        self.assertEqual(sources[0]["score"], 0.92)
        self.assertEqual(sources[0]["strategy"], "hybrid")
        self.assertEqual(sources[0]["text"], "Agents use tools with grounded evidence.")

    def test_extracts_web_links_from_search_and_fetch_payloads(self) -> None:
        messages = [
            {
                "role": "tool",
                "name": "web_search",
                "content": json.dumps(
                    {
                        "results": [
                            {
                                "title": "Agent Docs",
                                "url": "https://example.com/agents",
                                "snippet": "Agent documentation.",
                                "source": "duckduckgo",
                                "rank": 1,
                            }
                        ]
                    }
                ),
            },
            FakeToolMessage(
                "web_fetch",
                json.dumps(
                    {
                        "url": "https://example.com/agents",
                        "final_url": "https://example.com/agents",
                        "title": "Agent Docs",
                        "text": "Full page body about agents.",
                        "status_code": 200,
                    }
                ),
            ),
        ]

        sources = extract_sources_from_messages(messages)

        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["source_type"], "web")
        self.assertEqual(sources[0]["url"], "https://example.com/agents")
        self.assertEqual(sources[0]["title"], "Agent Docs")
        self.assertEqual(sources[0]["snippet"], "")
        self.assertEqual(sources[0]["text"], "Full page body about agents.")

    def test_ignores_web_search_candidates_without_matching_fetch(self) -> None:
        messages = [
            {
                "role": "tool",
                "name": "web_search",
                "content": json.dumps(
                    {
                        "results": [
                            {
                                "title": "Candidate A",
                                "url": "https://example.com/a",
                                "snippet": "Candidate snippet A.",
                                "source": "duckduckgo",
                                "rank": 1,
                            },
                            {
                                "title": "Candidate B",
                                "url": "https://example.com/b",
                                "snippet": "Candidate snippet B.",
                                "source": "duckduckgo",
                                "rank": 2,
                            },
                        ]
                    }
                ),
            }
        ]

        sources = extract_sources_from_messages(messages)

        self.assertEqual(sources, [])

    def test_keeps_only_fetched_web_source_when_search_has_extra_candidates(self) -> None:
        messages = [
            {
                "role": "tool",
                "name": "web_search",
                "content": json.dumps(
                    {
                        "results": [
                            {
                                "title": "Candidate A",
                                "url": "https://example.com/a",
                                "snippet": "Candidate snippet A.",
                                "source": "duckduckgo",
                                "rank": 1,
                            },
                            {
                                "title": "Candidate B",
                                "url": "https://example.com/b",
                                "snippet": "Candidate snippet B.",
                                "source": "duckduckgo",
                                "rank": 2,
                            },
                        ]
                    }
                ),
            },
            FakeToolMessage(
                "web_fetch",
                json.dumps(
                    {
                        "url": "https://example.com/b",
                        "final_url": "https://example.com/b",
                        "title": "Candidate B",
                        "text": "Chosen page body.",
                        "status_code": 200,
                    }
                ),
            ),
        ]

        sources = extract_sources_from_messages(messages)

        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["source_type"], "web")
        self.assertEqual(sources[0]["url"], "https://example.com/b")
        self.assertEqual(sources[0]["title"], "Candidate B")
        self.assertEqual(sources[0]["text"], "Chosen page body.")

    def test_extracts_scholar_results_as_sources(self) -> None:
        messages = [
            FakeToolMessage(
                "scholar_search",
                json.dumps(
                    {
                        "topic": "graph rag agent evaluation",
                        "planned_queries": [
                            "graph rag agent evaluation survey",
                            "graph retrieval augmented generation evaluation",
                        ],
                        "result_count": 1,
                        "results": [
                            {
                                "title": "Graph retrieval-augmented generation: A survey",
                                "url": "https://arxiv.org/abs/2501.12345",
                                "snippet": "A survey of GraphRAG methods and evaluation settings.",
                                "publication_summary": "arXiv preprint 2025",
                                "year": 2025,
                                "cited_by_count": 128,
                                "resources": [
                                    {
                                        "title": "PDF",
                                        "link": "https://arxiv.org/pdf/2501.12345.pdf",
                                        "file_format": "PDF",
                                    }
                                ],
                                "source_query": "graph rag agent evaluation survey",
                                "rank": 1,
                                "source": "google_scholar_serpapi",
                            }
                        ],
                    }
                ),
            )
        ]

        sources = extract_sources_from_messages(messages)

        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["source_type"], "scholar")
        self.assertEqual(sources[0]["title"], "Graph retrieval-augmented generation: A survey")
        self.assertEqual(sources[0]["url"], "https://arxiv.org/abs/2501.12345")
        self.assertEqual(
            sources[0]["snippet"], "A survey of GraphRAG methods and evaluation settings."
        )
        self.assertEqual(sources[0]["year"], 2025)
        self.assertEqual(sources[0]["cited_by_count"], 128)
        self.assertEqual(sources[0]["source_query"], "graph rag agent evaluation survey")

    def test_filters_out_low_confidence_local_sources(self) -> None:
        messages = [
            FakeToolMessage(
                "local_rag_retrieve",
                json.dumps(
                    {
                        "results": [
                            {
                                "chunk_id": 1,
                                "score": 0.95,
                                "text": "High-confidence local evidence.",
                                "document_id": "doc-1",
                                "source_path": "OpenAI/high.md",
                                "section_title": "Overview",
                                "strategy": "hybrid",
                            },
                            {
                                "chunk_id": 2,
                                "score": 0.45,
                                "text": "Low-confidence local evidence.",
                                "document_id": "doc-2",
                                "source_path": "OpenAI/low.md",
                                "section_title": "Noise",
                                "strategy": "hybrid",
                            },
                        ]
                    }
                ),
            )
        ]

        sources = extract_sources_from_messages(messages)

        self.assertEqual(len(sources), 1)
        self.assertEqual(sources[0]["source_path"], "OpenAI/high.md")

    def test_limits_web_sources_to_three_and_allocates_remaining_slots_to_scholar(self) -> None:
        messages = [
            FakeToolMessage(
                "local_rag_retrieve",
                json.dumps(
                    {
                        "results": [
                            {
                                "chunk_id": 1,
                                "score": 0.97,
                                "text": "Local evidence one.",
                                "document_id": "doc-1",
                                "source_path": "Local/one.md",
                                "section_title": "One",
                                "strategy": "hybrid",
                            },
                            {
                                "chunk_id": 2,
                                "score": 0.91,
                                "text": "Local evidence two.",
                                "document_id": "doc-2",
                                "source_path": "Local/two.md",
                                "section_title": "Two",
                                "strategy": "hybrid",
                            },
                            {
                                "chunk_id": 3,
                                "score": 0.32,
                                "text": "Local low-confidence noise.",
                                "document_id": "doc-3",
                                "source_path": "Local/noise.md",
                                "section_title": "Noise",
                                "strategy": "hybrid",
                            },
                        ]
                    }
                ),
            ),
            FakeToolMessage(
                "web_fetch",
                json.dumps(
                    {
                        "url": "https://example.com/1",
                        "final_url": "https://example.com/1",
                        "title": "Web 1",
                        "text": "Web body 1",
                    }
                ),
            ),
            FakeToolMessage(
                "web_fetch",
                json.dumps(
                    {
                        "url": "https://example.com/2",
                        "final_url": "https://example.com/2",
                        "title": "Web 2",
                        "text": "Web body 2",
                    }
                ),
            ),
            FakeToolMessage(
                "web_fetch",
                json.dumps(
                    {
                        "url": "https://example.com/3",
                        "final_url": "https://example.com/3",
                        "title": "Web 3",
                        "text": "Web body 3",
                    }
                ),
            ),
            FakeToolMessage(
                "web_fetch",
                json.dumps(
                    {
                        "url": "https://example.com/4",
                        "final_url": "https://example.com/4",
                        "title": "Web 4",
                        "text": "Web body 4",
                    }
                ),
            ),
            FakeToolMessage(
                "scholar_search",
                json.dumps(
                    {
                        "topic": "graph rag agent evaluation",
                        "planned_queries": ["graph rag"],
                        "result_count": 4,
                        "results": [
                            {
                                "title": "Scholar 1",
                                "url": "https://arxiv.org/abs/1",
                                "snippet": "Scholar snippet 1",
                                "publication_summary": "Author 1 - 2025",
                                "year": 2025,
                                "cited_by_count": 10,
                                "resources": [],
                                "source_query": "graph rag",
                                "rank": 1,
                                "source": "google_scholar_serpapi",
                            },
                            {
                                "title": "Scholar 2",
                                "url": "https://arxiv.org/abs/2",
                                "snippet": "Scholar snippet 2",
                                "publication_summary": "Author 2 - 2025",
                                "year": 2025,
                                "cited_by_count": 9,
                                "resources": [],
                                "source_query": "graph rag",
                                "rank": 2,
                                "source": "google_scholar_serpapi",
                            },
                            {
                                "title": "Scholar 3",
                                "url": "https://arxiv.org/abs/3",
                                "snippet": "Scholar snippet 3",
                                "publication_summary": "Author 3 - 2025",
                                "year": 2025,
                                "cited_by_count": 8,
                                "resources": [],
                                "source_query": "graph rag",
                                "rank": 3,
                                "source": "google_scholar_serpapi",
                            },
                            {
                                "title": "Scholar 4",
                                "url": "https://arxiv.org/abs/4",
                                "snippet": "Scholar snippet 4",
                                "publication_summary": "Author 4 - 2025",
                                "year": 2025,
                                "cited_by_count": 7,
                                "resources": [],
                                "source_query": "graph rag",
                                "rank": 4,
                                "source": "google_scholar_serpapi",
                            },
                        ],
                    }
                ),
            ),
        ]

        sources = extract_sources_from_messages(messages)

        self.assertEqual(len(sources), 8)
        self.assertEqual(sum(1 for item in sources if item["source_type"] == "local"), 2)
        self.assertEqual(sum(1 for item in sources if item["source_type"] == "web"), 3)
        self.assertEqual(sum(1 for item in sources if item["source_type"] == "scholar"), 3)
        self.assertFalse(any(item.get("url") == "https://example.com/4" for item in sources))
        self.assertFalse(any(item.get("source_path") == "Local/noise.md" for item in sources))
