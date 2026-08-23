from __future__ import annotations

from pathlib import Path

import pytest

from examples.orchestrations import execute_plan as execute_plan_mod


def test_build_skeleton_wires_impl_context_and_review_dependency() -> None:
    skeleton = execute_plan_mod.build_skeleton("do the thing", "final-solution.md")

    nodes_by_id = {node.id: node for node in skeleton.nodes}
    assert set(nodes_by_id) == {"impl", "review"}

    impl = nodes_by_id["impl"]
    assert impl.role == "implementer"
    assert impl.output_file == "impl_output.md"
    assert impl.context_files == ["final-solution.md"]
    assert impl.depends_on == []

    review = nodes_by_id["review"]
    assert review.role == "reviewer"
    assert review.depends_on == ["impl"]
    assert review.inputs == [{"from": "impl", "field": "output_file"}]
    assert review.reviewer_criteria


def test_build_skeleton_lints_clean() -> None:
    from atomic_agents.linter import lint_skeleton

    skeleton = execute_plan_mod.build_skeleton("do the thing", "plan.md")
    result = lint_skeleton(skeleton)
    assert result.ok, result.to_dict()


def test_execute_plan_copies_plan_into_workspace_and_missing_plan_raises(tmp_path: Path) -> None:
    workspace = tmp_path / "ws"
    workspace.mkdir()

    missing_plan = tmp_path / "does-not-exist.md"
    with pytest.raises(FileNotFoundError):
        execute_plan_mod._copy_plan_to_workspace(str(missing_plan), str(workspace))

    with pytest.raises(IsADirectoryError):
        execute_plan_mod._copy_plan_to_workspace(str(tmp_path), str(workspace))

    plan = tmp_path / "final-solution.md"
    plan.write_text("# plan\n", encoding="utf-8")

    copied_name = execute_plan_mod._copy_plan_to_workspace(str(plan), str(workspace))

    assert copied_name == "final-solution.md"
    assert (workspace / "final-solution.md").read_text(encoding="utf-8") == "# plan\n"


def test_cli_run_dispatches_execute_plan_and_requires_plan_path(monkeypatch, tmp_path: Path) -> None:
    from examples.orchestrations import cli
    from examples.orchestrations.common import OrchestrationResult

    captured: dict = {}

    def _fake_execute_plan(request, plan_path, *, role_runners=None, workspace=None):
        captured["request"] = request
        captured["plan_path"] = plan_path
        captured["role_runners"] = role_runners
        captured["workspace"] = workspace
        return OrchestrationResult(
            template="execute-plan",
            request=request,
            workspace=workspace or "",
            succeeded=True,
            artifacts=[],
            final_path=None,
        )

    monkeypatch.setattr(execute_plan_mod, "execute_plan", _fake_execute_plan)
    monkeypatch.setattr(cli, "execute_plan_mod", execute_plan_mod)

    with pytest.raises(ValueError, match="plan_path is required"):
        cli.run("execute-plan", "do it")

    plan = tmp_path / "plan.md"
    plan.write_text("# plan\n", encoding="utf-8")

    result = cli.run("execute_plan", "do it", plan_path=str(plan), workspace=str(tmp_path))

    assert result.succeeded
    assert captured["plan_path"] == str(plan)
    assert captured["request"] == "do it"
