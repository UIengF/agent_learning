from __future__ import annotations

from pathlib import Path

from aicoding_app.evidence_cache import EvidenceCache
from aicoding_app.permissions import WorkspacePolicy
from aicoding_app.plan import CodingPlan
from aicoding_app.skills import SkillRegistry
from aicoding_app.tools import CodingTools
from aicoding_app.trace import StructuredTraceWriter


def _tools(tmp_path: Path, plan: CodingPlan | None = None) -> CodingTools:
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    trace = StructuredTraceWriter(tmp_path / "runtime" / "traces", "demo")
    return CodingTools(
        policy=WorkspacePolicy(workspace=tmp_path, allowed_commands=("git diff",)),
        plan=plan or CodingPlan(),
        evidence_cache=EvidenceCache(),
        skill_registry=SkillRegistry(skills_dir),
        trace_writer=trace,
    )


def test_apply_patch_requires_plan_before_edit(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("old\n", encoding="utf-8")
    tools = _tools(tmp_path)

    result = tools.apply_patch(
        "*** Begin Patch\n*** Update File: app.py\n@@\n-old\n+new\n*** End Patch"
    )

    assert result.startswith("plan_required")
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "old\n"


def test_apply_patch_updates_workspace_file_after_plan(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("old\n", encoding="utf-8")
    plan = CodingPlan(goal="replace value", edit_steps=["replace old with new"])
    tools = _tools(tmp_path, plan)

    result = tools.apply_patch(
        "*** Begin Patch\n*** Update File: app.py\n@@\n-old\n+new\n*** End Patch"
    )

    assert result == "changed files: app.py"
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "new\n"


def test_apply_patch_accepts_unified_diff_after_plan(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("old\n", encoding="utf-8")
    plan = CodingPlan(goal="replace value", edit_steps=["replace old with new"])
    tools = _tools(tmp_path, plan)

    result = tools.apply_patch(
        "*** Begin Patch\n--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-old\n+new\n*** End Patch"
    )

    assert result == "changed files: app.py"
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "new\n"


def test_apply_patch_unified_add_file_does_not_write_hunk_header(tmp_path: Path) -> None:
    plan = CodingPlan(goal="add module", edit_steps=["add app.py"])
    tools = _tools(tmp_path, plan)

    result = tools.apply_patch(
        "*** Begin Patch\n"
        "--- /dev/null\n"
        "+++ b/app.py\n"
        "@@ -0,0 +1,2 @@\n"
        "+def answer():\n"
        "+    return 2\n"
        "*** End Patch"
    )

    assert result == "changed files: app.py"
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "def answer():\n    return 2\n"
    assert "@@" not in (tmp_path / "app.py").read_text(encoding="utf-8")


def test_list_search_read_populate_evidence_cache(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("needle = 1\n", encoding="utf-8")
    tools = _tools(tmp_path)

    assert "src/app.py" in tools.list_files()
    assert "needle" in tools.search_text("needle", "src/*.py")
    assert "1: needle = 1" in tools.read_file("src/app.py")
