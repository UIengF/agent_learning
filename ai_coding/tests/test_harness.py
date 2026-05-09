from __future__ import annotations

import json
from pathlib import Path

from aicoding_app.context import build_context
from aicoding_app.evidence_cache import EvidenceCache
from aicoding_app.plan import CodingPlan
from aicoding_app.skills import SkillRegistry
from aicoding_app.trace import StructuredTraceWriter


def test_skill_registry_loads_skill_and_reports_unknown(tmp_path: Path) -> None:
    skill_dir = tmp_path / "skills" / "demo"
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\nname: demo\ndescription: demo skill\n---\nUse demo.",
        encoding="utf-8",
    )

    registry = SkillRegistry(tmp_path / "skills")

    assert "demo skill" in registry.format_inventory()
    assert registry.load_payload("demo")["body"] == "Use demo."
    assert registry.load_payload("missing")["error"] == "unknown_skill"


def test_trace_writer_appends_jsonl(tmp_path: Path) -> None:
    writer = StructuredTraceWriter(tmp_path / "traces", "session-a")

    writer.append(
        "tool_call",
        task_id="task-a",
        tool_name="read_file",
        input_summary="app.py",
        output_summary="10 lines",
    )

    line = (tmp_path / "traces" / "session-a.jsonl").read_text(encoding="utf-8").strip()
    payload = json.loads(line)
    assert payload["session_id"] == "session-a"
    assert payload["task_id"] == "task-a"
    assert payload["tool_name"] == "read_file"


def test_evidence_cache_deduplicates_and_context_compresses() -> None:
    cache = EvidenceCache()
    cache.add("file", "app.py", "old content")
    cache.add("file", "app.py", "new content")
    context = build_context(
        history=[{"role": "user", "content": "hello"}],
        plan=CodingPlan(goal="g", edit_steps=["e"]),
        evidence_context=cache.to_context(1000),
        latest_diff="diff --git a/app.py b/app.py",
        max_chars=400,
    )

    assert "new content" in context
    assert "old content" not in context
    assert "Goal: g" in context
