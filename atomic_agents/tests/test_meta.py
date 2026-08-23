from __future__ import annotations

import pytest

from atomic_agents.linter import Violation, lint_skeleton
from atomic_agents.meta import (
    LintFailedError,
    meta_compile,
    plan_quality_warnings,
    render_explanation,
    suggest_bindings,
)
from atomic_agents.models import Edge, InputRef, Skeleton, SkeletonNode
from atomic_agents.templates import MockCompiler, fill_template, match_template


def test_meta_compile_template_hit_builds_metaplan() -> None:
    plan = meta_compile("实现登录功能", "ducc", MockCompiler())

    assert plan.source == "template"
    assert plan.template_name == "plan-impl-review"
    assert lint_skeleton(plan.skeleton).ok is True
    assert plan.bindings == {
        "implementer": {"runner": "ducc"},
        "planner": {"runner": "ducc"},
        "reviewer": {"runner": "ducc"},
    }
    assert "planner" in plan.explanation
    assert "reviewer" in plan.explanation
    assert isinstance(plan.warnings, list)
    assert plan.lint_repaired is False
    assert plan.to_dict()["skeleton"] == plan.skeleton.to_dict()


def test_meta_compile_free_path_lint_clean_without_repair() -> None:
    request = "把仓库迁移到新构建系统"
    skeleton = make_valid_skeleton()

    plan = meta_compile(request, "codex", MockCompiler({request: skeleton}))

    assert plan.source == "free"
    assert plan.template_name is None
    assert plan.lint_repaired is False
    assert plan.skeleton.to_dict() == skeleton.to_dict()
    assert lint_skeleton(plan.skeleton).ok is True


def test_meta_compile_free_path_repairs_invalid_skeleton() -> None:
    class RepairingCompiler:
        def generate(self, request: str) -> Skeleton:
            return make_skeleton([make_node("impl", write_scope=["src/a.py"], read_only=False)])

        def repair(self, skeleton: Skeleton, violations: list[Violation]) -> Skeleton:
            return make_valid_skeleton()

    plan = meta_compile("把仓库迁移到新构建系统", "ducc", RepairingCompiler())

    assert plan.source == "free"
    assert plan.lint_repaired is True
    assert lint_skeleton(plan.skeleton).ok is True


def test_meta_compile_raises_when_free_path_has_no_repair() -> None:
    request = "把仓库迁移到新构建系统"
    compiler = MockCompiler({request: make_skeleton([make_node("impl", write_scope=["src/a.py"], read_only=False)])})

    with pytest.raises(LintFailedError) as exc_info:
        meta_compile(request, "codex", compiler)

    assert exc_info.value.violations
    assert "writer_without_reviewer_downstream" in {violation.code for violation in exc_info.value.violations}


def test_meta_compile_raises_when_repair_still_fails() -> None:
    class FailingRepairCompiler:
        def generate(self, request: str) -> Skeleton:
            return make_skeleton([make_node("impl", write_scope=["src/a.py"], read_only=False)])

        def repair(self, skeleton: Skeleton, violations: list[Violation]) -> Skeleton:
            return make_skeleton([make_node("impl", write_scope=["src/a.py"], read_only=False)])

    with pytest.raises(LintFailedError) as exc_info:
        meta_compile("把仓库迁移到新构建系统", "ducc", FailingRepairCompiler(), max_repair_rounds=1)

    assert exc_info.value.violations
    assert "writer_without_reviewer_downstream" in {violation.code for violation in exc_info.value.violations}


def test_suggest_bindings_maps_sorted_roles_to_launcher_identity() -> None:
    skeleton = make_skeleton(
        [
            make_node("b", role="writer"),
            make_node("a", role="planner"),
            make_node("c", role="writer"),
            make_node("review", role="reviewer", reviewer_criteria=["A", "B"]),
        ]
    )

    bindings = suggest_bindings(skeleton, "ducc")

    assert list(bindings) == ["planner", "reviewer", "writer"]
    assert bindings == {
        "planner": {"runner": "ducc"},
        "reviewer": {"runner": "ducc"},
        "writer": {"runner": "ducc"},
    }


def test_render_explanation_mentions_template_source_and_reviewer_summary() -> None:
    template = match_template("实现登录功能")
    assert template is not None
    skeleton = fill_template(template, "实现登录功能")

    explanation = render_explanation(skeleton, "template", template.name)

    assert "来源：命中模板 plan-impl-review" in explanation
    assert "适用场景" in explanation
    assert "planner" in explanation
    assert "关键 reviewer 节点：review，共 3 条验收标准。" in explanation


def test_plan_quality_warnings_parallel_write_overlap() -> None:
    skeleton = make_skeleton(
        [
            make_node("write1", write_scope=["src/x.py"], read_only=False),
            make_node("write2", write_scope=["src/x.py", "src/y.py"], read_only=False),
            make_node("review", role="reviewer", deps=["write1", "write2"], reviewer_criteria=["A", "B"]),
        ]
    )

    warnings = plan_quality_warnings(skeleton, "free")

    warning = _warning(warnings, "parallel_write_overlap")
    assert warning.severity == "high"
    assert warning.node_id == "write1"
    assert "src/x.py" in warning.message
    assert {"parallel_write_overlap"} <= _codes(warnings)


def test_plan_quality_warnings_weak_acceptance_for_single_criterion_reviewer() -> None:
    skeleton = make_skeleton(
        [
            make_node("impl", write_scope=["src/a.py"], read_only=False),
            make_node("review", role="reviewer", deps=["impl"], reviewer_criteria=["Only one"]),
        ]
    )

    warnings = plan_quality_warnings(skeleton, "free")

    warning = _warning(warnings, "weak_acceptance")
    assert warning.severity == "warn"
    assert warning.node_id == "review"


def test_plan_quality_warnings_weak_acceptance_for_missing_reviewer() -> None:
    skeleton = make_skeleton([make_node("read")])

    warnings = plan_quality_warnings(skeleton, "free")

    warning = _warning(warnings, "weak_acceptance")
    assert warning.severity == "warn"
    assert warning.node_id is None


def test_plan_quality_warnings_undeclared_write_risk() -> None:
    skeleton = make_skeleton(
        [
            make_node("impl", read_only=False),
            make_node("review", role="reviewer", deps=["impl"], reviewer_criteria=["A", "B"]),
        ]
    )

    warnings = plan_quality_warnings(skeleton, "free")

    warning = _warning(warnings, "undeclared_write_risk")
    assert warning.severity == "warn"
    assert warning.node_id == "impl"


def test_plan_quality_warnings_cost_budget_risk_for_tight_budget() -> None:
    nodes = [make_node(f"n{i}") for i in range(7)]
    skeleton = make_skeleton(nodes, max_total_cost=6.0)

    warnings = plan_quality_warnings(skeleton, "free")

    warning = _warning(warnings, "cost_budget_risk")
    assert warning.severity == "info"
    assert "均摊每节点预算低于 1.0" in warning.message


def test_plan_quality_warnings_cost_budget_risk_for_high_absolute_budget() -> None:
    skeleton = make_skeleton([make_node("read")], max_total_cost=51.0)

    warnings = plan_quality_warnings(skeleton, "free")

    warning = _warning(warnings, "cost_budget_risk")
    assert warning.severity == "info"
    assert "偏高" in warning.message


def test_plan_quality_warnings_clean_template_has_no_high_warning() -> None:
    template = match_template("实现登录功能")
    assert template is not None
    skeleton = fill_template(template, "实现登录功能")

    warnings = plan_quality_warnings(skeleton, "template")

    assert all(warning.severity != "high" for warning in warnings)
    assert "parallel_write_overlap" not in _codes(warnings)
    assert "weak_acceptance" not in _codes(warnings)


def make_node(
    id: str,
    *,
    role: str = "worker",
    deps: list[str] | None = None,
    inputs: list[InputRef] | None = None,
    write_scope: list[str] | None = None,
    read_only: bool = True,
    reviewer_criteria: list[str] | None = None,
    output_file: str | None = None,
) -> SkeletonNode:
    writes = list(write_scope or [])
    if output_file is None:
        if writes:
            resolved_output = writes[0]
        elif role.strip().lower() == "reviewer":
            resolved_output = f"{id}-verdict.json"
        else:
            resolved_output = ""
    else:
        resolved_output = output_file
    return SkeletonNode(
        id=id,
        role=role,
        task=f"Run {id}",
        depends_on=list(deps or []),
        inputs=list(inputs or []),
        write_scope=writes,
        read_only=read_only,
        required_capabilities=["write_files"] if writes else [],
        reviewer_criteria=list(reviewer_criteria or []),
        output_file=resolved_output,
    )


def make_skeleton(
    nodes: list[SkeletonNode],
    *,
    edges: list[Edge] | None = None,
    max_repair_attempts: int = 1,
    max_total_cost: float = 10.0,
) -> Skeleton:
    return Skeleton(
        name="meta-test",
        version=1,
        nodes=nodes,
        edges=edges if edges is not None else _edges_from_dependencies(nodes),
        run_limits={"max_repair_attempts": max_repair_attempts, "max_total_cost": max_total_cost},
        irreversible_ops=[],
    )


def make_valid_skeleton() -> Skeleton:
    return make_skeleton(
        [
            make_node("plan", role="planner"),
            make_node("impl", role="implementer", deps=["plan"], write_scope=["src/a.py"], read_only=False),
            make_node("review", role="reviewer", deps=["impl"], reviewer_criteria=["A", "B"]),
        ]
    )


def _edges_from_dependencies(nodes: list[SkeletonNode]) -> list[Edge]:
    edges: list[Edge] = []
    for node in nodes:
        for dependency in node.depends_on:
            edges.append({"from": dependency, "to": node.id})
    return edges


def _codes(warnings: list[object]) -> set[str]:
    return {getattr(warning, "code") for warning in warnings}


def _warning(warnings: list[object], code: str):
    matches = [warning for warning in warnings if getattr(warning, "code") == code]
    assert matches
    return matches[0]
