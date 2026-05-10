from __future__ import annotations

import json
from pathlib import Path

from aicoding_app.evidence_cache import EvidenceCache
from aicoding_app.harness_state import ToolRunState
from aicoding_app.permissions import WorkspacePolicy
from aicoding_app.plan import CodingPlan
from aicoding_app.skills import SkillRegistry
from aicoding_app.tools import CodingTools
from aicoding_app.trace import StructuredTraceWriter
from aicoding_app.validation import validation_environment_guidance


def _tools(
    tmp_path: Path,
    plan: CodingPlan | None = None,
    *,
    run_state: ToolRunState | None = None,
    require_patch_preview: bool = False,
    allowed_commands: tuple[str, ...] = ("git diff",),
) -> CodingTools:
    skills_dir = tmp_path / "skills"
    skills_dir.mkdir()
    trace = StructuredTraceWriter(tmp_path / "runtime" / "traces", "demo")
    return CodingTools(
        policy=WorkspacePolicy(workspace=tmp_path, allowed_commands=allowed_commands),
        plan=plan or CodingPlan(),
        evidence_cache=EvidenceCache(),
        skill_registry=SkillRegistry(skills_dir),
        trace_writer=trace,
        run_state=run_state,
        require_patch_preview=require_patch_preview,
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


def test_model_driven_apply_patch_requires_successful_preview(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("old\n", encoding="utf-8")
    plan = CodingPlan(goal="replace value", edit_steps=["replace old with new"])
    tools = _tools(tmp_path, plan, require_patch_preview=True)

    result = tools.apply_patch(
        "*** Begin Patch\n*** Update File: app.py\n@@\n-old\n+new\n*** End Patch"
    )

    assert result == "preview_required: call preview_patch successfully before apply_patch"
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "old\n"
    assert tools.run_state.dependency_violations
    assert tools.run_state.dependency_violations[0].tool_name == "apply_patch"


def test_model_driven_successful_preview_allows_apply(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("old\n", encoding="utf-8")
    plan = CodingPlan(goal="replace value", edit_steps=["replace old with new"])
    tools = _tools(tmp_path, plan, require_patch_preview=True)
    patch = "*** Begin Patch\n*** Update File: app.py\n@@\n-old\n+new\n*** End Patch"

    preview = tools.preview_patch(patch)
    result = tools.apply_patch(patch)

    assert preview == "preview changed files: app.py"
    assert result == "changed files: app.py"
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "new\n"
    assert tools.run_state.applied_patches


def test_write_text_file_requires_plan(tmp_path: Path) -> None:
    tools = _tools(tmp_path)

    result = tools.write_text_file("README.md", "# Demo\n")

    assert result == "plan_required: call plan_update with edit_steps before write_text_file"
    assert not (tmp_path / "README.md").exists()


def test_write_text_file_writes_traceable_text_file(tmp_path: Path) -> None:
    plan = CodingPlan(goal="write readme", edit_steps=["write README.md"])
    tools = _tools(tmp_path, plan)

    result = tools.write_text_file("README.md", "# Demo\n\nLong content.\n")

    assert "write preview: README.md" in result
    assert "changed files: README.md" in result
    assert (tmp_path / "README.md").read_text(encoding="utf-8") == "# Demo\n\nLong content.\n"
    assert tools.run_state.written_files == ["README.md"]


def test_write_text_file_blocks_sensitive_files(tmp_path: Path) -> None:
    plan = CodingPlan(goal="write env", edit_steps=["write .env"])
    tools = _tools(tmp_path, plan)

    result = tools.write_text_file(".env", "SECRET=value\n")

    assert result.startswith("write_denied:")
    assert not (tmp_path / ".env").exists()


def test_dependency_state_machine_stops_after_repeated_violations(tmp_path: Path) -> None:
    plan = CodingPlan(goal="replace value", edit_steps=["replace old with new"])
    run_state = ToolRunState(max_dependency_violations=2)
    tools = _tools(tmp_path, plan, run_state=run_state, require_patch_preview=True)
    patch = "*** Begin Patch\n*** Add File: app.py\n+new\n*** End Patch"

    first = tools.apply_patch(patch)
    second = tools.apply_patch(patch)

    assert first == "preview_required: call preview_patch successfully before apply_patch"
    assert second == "preview_required: call preview_patch successfully before apply_patch"
    assert run_state.should_stop()
    assert run_state.stop_reason == "too_many_dependency_violations"


def test_denied_validation_is_recorded(tmp_path: Path) -> None:
    run_state = ToolRunState()
    tools = _tools(tmp_path, run_state=run_state, allowed_commands=("git diff",))

    result = tools.run_validation("python -m pytest tests")

    assert result.startswith("validation_denied:")
    assert run_state.denied_commands[0].command == "python -m pytest tests"
    assert run_state.policy_denials[0].command == "python -m pytest tests"
    assert run_state.validation_attempts[0].status == "denied"


def test_run_command_policy_denial_is_classified(tmp_path: Path) -> None:
    run_state = ToolRunState()
    tools = _tools(tmp_path, run_state=run_state, allowed_commands=("python -m pytest",))

    result = tools.run_command('python -c "print(1)"')

    assert result.startswith("command_denied:")
    assert run_state.denied_commands
    assert run_state.policy_denials


def test_failed_run_command_validation_is_recorded(tmp_path: Path) -> None:
    run_state = ToolRunState()
    tools = _tools(
        tmp_path,
        run_state=run_state,
        allowed_commands=("python -m pytest",),
    )

    result = tools.run_command("python -m pytest missing_tests")

    assert "returncode:" in result
    assert run_state.validation_attempts
    assert run_state.validation_attempts[0].status == "failed"
    assert run_state.code_failures


def test_failed_pytest_environment_guidance_is_returned(tmp_path: Path, monkeypatch) -> None:
    class Completed:
        returncode = 1
        stdout = ""
        stderr = "ModuleNotFoundError: No module named pygments"

    run_state = ToolRunState()
    tools = _tools(
        tmp_path,
        run_state=run_state,
        allowed_commands=("python -m pytest",),
    )
    monkeypatch.setattr("aicoding_app.tools.subprocess.run", lambda *args, **kwargs: Completed())

    result = tools.run_command("python -m pytest test_task_manager.py -v")

    assert "Validation environment failure detected." in result
    assert "environment/tooling problem" in result
    assert "Do not retry the same validation command." in result
    assert "Do not install packages or use python -c / pip / shell control operators" in result
    assert run_state.validation_attempts[0].detail.startswith(
        "Validation environment failure detected."
    )
    assert run_state.environment_failures
    assert not run_state.code_failures
    assert run_state.has_validation_environment_failure()


def test_check_environment_reports_environment_failures(tmp_path: Path, monkeypatch) -> None:
    class Completed:
        def __init__(self, returncode: int, stdout: str = "", stderr: str = ""):
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    def fake_run(command, **kwargs):
        assert kwargs["shell"] is False
        text = " ".join(command)
        if text.endswith("--version") and "-m" not in text:
            return Completed(0, "Python 3.12.4\n")
        if "-m pytest --version" in text:
            return Completed(1, "", "ModuleNotFoundError: No module named pygments")
        if "-m pip --version" in text:
            return Completed(1, "", "No module named pip")
        if "-m ruff --version" in text:
            return Completed(0, "ruff 0.1.0\n")
        if "-m pyright --version" in text:
            return Completed(0, "pyright 1.1.0\n")
        raise AssertionError(text)

    run_state = ToolRunState()
    tools = _tools(tmp_path, run_state=run_state)
    monkeypatch.setattr("aicoding_app.tools.subprocess.run", fake_run)

    result = tools.check_environment()
    payload = json.loads(result)

    assert payload["python"]["ok"] is True
    assert payload["pytest"]["ok"] is False
    assert payload["pytest"]["reason"] == "environment_failure"
    assert payload["pip"]["reason"] == "environment_failure"
    assert run_state.has_validation_environment_failure()
    assert run_state.environment_diagnosed
    assert run_state.should_stop()
    assert run_state.stop_reason == "environment_blocker_detected"


def test_validation_environment_guidance_classifies_pygments_import_error() -> None:
    guidance = validation_environment_guidance(
        "ModuleNotFoundError: No module named pygments"
    )

    assert "Validation environment failure detected." in guidance
    assert "environment/tooling problem" in guidance
    assert "Do not retry the same validation command." in guidance
    assert "Do not install packages or use python -c / pip / shell control operators" in guidance


def test_tool_run_state_recognizes_new_environment_guidance() -> None:
    run_state = ToolRunState()
    guidance = validation_environment_guidance(
        "ModuleNotFoundError: No module named pygments"
    )

    run_state.record_validation(
        command="python -m pytest tests",
        status="failed",
        detail=guidance,
    )

    assert run_state.has_validation_environment_failure()


def test_environment_failure_waits_for_diagnosis_before_stopping() -> None:
    run_state = ToolRunState()

    run_state.record_environment_failure(
        "python -m pytest tests",
        "Validation environment failure detected.",
    )

    assert run_state.terminal_status == "environment_blocker_pending_diagnosis"
    assert not run_state.should_stop()

    run_state.mark_environment_diagnosed()

    assert run_state.should_stop()
    assert run_state.stop_reason == "environment_blocker_detected"


def test_tool_misuse_loop_stops() -> None:
    run_state = ToolRunState()

    run_state.record_tool_misuse("bad_tool", "unknown")
    assert not run_state.should_stop()
    run_state.record_tool_misuse("bad_tool", "unknown again")

    assert run_state.should_stop()
    assert run_state.stop_reason == "tool_misuse_loop"


def test_policy_denial_loop_stops() -> None:
    run_state = ToolRunState()

    run_state.record_policy_denial('python -c "print(1)"', "command is outside whitelist")
    assert not run_state.should_stop()
    for command in ("python -m pip list", "python -m pip freeze", "python -m pip show pytest"):
        run_state.record_policy_denial(command, "command is outside whitelist")

    assert run_state.should_stop()
    assert run_state.stop_reason == "policy_denial_loop"


def test_install_python_package_requires_environment_failure(tmp_path: Path) -> None:
    tools = _tools(tmp_path)

    result = tools.install_python_package("pygments")

    assert result.startswith("install_denied:")


def test_install_python_package_blocks_unsafe_specs(tmp_path: Path) -> None:
    run_state = ToolRunState()
    run_state.record_validation(
        command="python -m pytest tests",
        status="failed",
        detail="Validation environment failure detected.",
    )
    tools = _tools(tmp_path, run_state=run_state)

    assert tools.install_python_package("-r requirements.txt").startswith("install_denied:")
    assert tools.install_python_package("https://example.com/pkg.whl").startswith("install_denied:")
    assert tools.install_python_package("pygments && del x").startswith("install_denied:")


def test_install_python_package_runs_safe_package_after_environment_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    class Completed:
        returncode = 0
        stdout = "installed"
        stderr = ""

    captured = {}

    def fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        return Completed()

    run_state = ToolRunState()
    run_state.record_validation(
        command="python -m pytest tests",
        status="failed",
        detail="Validation environment failure detected.",
    )
    tools = _tools(tmp_path, run_state=run_state)
    monkeypatch.setattr("aicoding_app.tools.subprocess.run", fake_run)

    result = tools.install_python_package("pygments")

    assert "returncode: 0" in result
    assert captured["command"][-4:] == ["-m", "pip", "install", "pygments"]


def test_apply_patch_accepts_unified_diff_after_plan(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("old\n", encoding="utf-8")
    plan = CodingPlan(goal="replace value", edit_steps=["replace old with new"])
    tools = _tools(tmp_path, plan)

    result = tools.apply_patch(
        "*** Begin Patch\n--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-old\n+new\n*** End Patch"
    )

    assert result == "changed files: app.py"
    assert (tmp_path / "app.py").read_text(encoding="utf-8") == "new\n"


def test_patch_hunk_ambiguous_match_is_rejected(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("return 1\nreturn 1\n", encoding="utf-8")
    plan = CodingPlan(goal="replace value", edit_steps=["replace first return"])
    tools = _tools(tmp_path, plan)

    result = tools.preview_patch(
        "*** Begin Patch\n"
        "*** Update File: app.py\n"
        "@@\n"
        "-return 1\n"
        "+return 2\n"
        "*** End Patch"
    )

    assert "patch hunk matches 2 locations" in result


def test_patch_hunk_preserves_tab_only_context_line(tmp_path: Path) -> None:
    (tmp_path / "app.py").write_text("\t\nold\n", encoding="utf-8")
    plan = CodingPlan(goal="replace value", edit_steps=["replace after tab"])
    tools = _tools(tmp_path, plan)

    result = tools.preview_patch(
        "*** Begin Patch\n"
        "*** Update File: app.py\n"
        "@@\n"
        "\t\n"
        "-old\n"
        "+new\n"
        "*** End Patch"
    )

    assert result == "preview changed files: app.py"


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
