from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest import TestCase

from graph_rag_app.agent import Agent
from graph_rag_app.research_plan import (
    ResearchPlanTool,
    format_research_plan,
    latest_research_plan,
)
from graph_rag_app.skills import LoadSkillTool, SkillRegistry
from graph_rag_app.structured_trace import StructuredTraceWriter


class HarnessFeatureTests(TestCase):
    def test_skill_registry_discovers_inventory_and_loads_body(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = Path(tmpdir) / "skills" / "citation"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text(
                "---\n"
                "name: citation-style\n"
                "description: Use grounded citation formatting.\n"
                "---\n"
                "Always cite retrieved sources.",
                encoding="utf-8",
            )

            registry = SkillRegistry(Path(tmpdir) / "skills")
            inventory = registry.format_inventory()
            payload = json.loads(
                LoadSkillTool(registry=registry).invoke({"name": "citation-style"})
            )

        self.assertIsNotNone(inventory)
        self.assertIn("citation-style", inventory)
        self.assertEqual(payload["name"], "citation-style")
        self.assertEqual(payload["body"], "Always cite retrieved sources.")

    def test_load_skill_reports_unknown_skill_with_available_names(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = Path(tmpdir) / "known"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text("Known body.", encoding="utf-8")
            registry = SkillRegistry(tmpdir)

            payload = json.loads(LoadSkillTool(registry=registry).invoke({"name": "missing"}))

        self.assertEqual(payload["error"], "unknown_skill")
        self.assertEqual(payload["available"], ["known"])

    def test_research_plan_tool_round_trips_into_context_summary(self) -> None:
        result = ResearchPlanTool().invoke(
            {
                "question": "How should RAG use tools?",
                "status": "in_progress",
                "steps": ["Search local docs", "Fetch official docs"],
                "evidence": ["Local docs mention tool use"],
                "gaps": ["Need current public docs"],
                "next_action": "Call web_search",
            }
        )
        plan = latest_research_plan(
            [{"role": "tool", "name": "research_plan_update", "content": result}]
        )
        rendered = format_research_plan(plan)

        self.assertIsNotNone(rendered)
        self.assertIn("How should RAG use tools?", rendered)
        self.assertIn("Need current public docs", rendered)
        self.assertIn("next_action: Call web_search", rendered)

    def test_agent_context_includes_skill_inventory_and_research_plan(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            skill_dir = Path(tmpdir) / "retrieval"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                "---\nname: retrieval-debug\ndescription: Diagnose retrieval failures.\n---\nBody.",
                encoding="utf-8",
            )
            registry = SkillRegistry(tmpdir)
            plan_payload = ResearchPlanTool().invoke(
                {
                    "question": "Why did retrieval fail?",
                    "gaps": ["Need inspect_index output"],
                    "next_action": "Inspect index metadata",
                }
            )
            agent = Agent(system="system", skill_registry=registry)

            messages = agent.build_llm_messages(
                {
                    "messages": [
                        {"role": "human", "content": "Why did retrieval fail?"},
                        {
                            "role": "tool",
                            "name": "research_plan_update",
                            "content": plan_payload,
                        },
                    ]
                }
            )
            contents = [agent._message_content(message) for message in messages]

        self.assertTrue(any("Available skills:" in content for content in contents))
        self.assertTrue(any("retrieval-debug" in content for content in contents))
        self.assertTrue(any("Research plan:" in content for content in contents))
        self.assertTrue(any("Need inspect_index output" in content for content in contents))

    def test_structured_trace_writer_appends_jsonl_events(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            trace_path = Path(tmpdir) / "trace.jsonl"
            writer = StructuredTraceWriter(trace_path)

            writer.append("tool_call", {"name": "local_rag_retrieve", "args": {"query": "rag"}})
            writer.append("final_answer", {"answer_preview": "done"})

            rows = [
                json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertEqual([row["event_type"] for row in rows], ["tool_call", "final_answer"])
        self.assertEqual(rows[0]["name"], "local_rag_retrieve")
