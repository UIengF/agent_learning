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
