from __future__ import annotations

from typing import Any

from atomic_agents.linter import (
    C_BUDGET,
    C_CYCLE,
    C_DEP_UNKNOWN,
    C_DUP_NODE_ID,
    C_DYNAMIC_HOOK,
    C_EDGE_UNKNOWN,
    C_INPUT_FIELD,
    C_INPUT_UNKNOWN,
    C_OUTPUT_FILE,
    C_REVIEWER_COVERAGE,
    C_REVIEWER_CRITERIA,
    lint_skeleton,
)
from atomic_agents.models import Edge, InputRef, Skeleton, SkeletonNode


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
        name="lint-test",
        version=1,
        nodes=nodes,
        edges=edges if edges is not None else _edges_from_dependencies(nodes),
        run_limits={"max_repair_attempts": max_repair_attempts, "max_total_cost": max_total_cost},
        irreversible_ops=[],
    )


def test_valid_skeleton_ok() -> None:
    skeleton = make_skeleton(
        [
            make_node("plan", role="planner"),
            make_node("impl", role="implementer", deps=["plan"], write_scope=["src/a.py"], read_only=False),
            make_node("review", role="reviewer", deps=["impl"], reviewer_criteria=["Implementation is correct"]),
        ]
    )

    result = lint_skeleton(skeleton)

    assert result.ok is True
    assert result.violations == []
    assert result.to_dict() == {"ok": True, "violations": []}


def test_duplicate_node_id_reports_violation() -> None:
    skeleton = make_skeleton(
        [
            make_node("same"),
            make_node("same"),
        ]
    )

    codes = _codes(lint_skeleton(skeleton))

    assert C_DUP_NODE_ID in codes


def test_edge_unknown_node_reports_violation() -> None:
    skeleton = make_skeleton(
        [make_node("a")],
        edges=[{"from": "a", "to": "missing"}],
    )

    codes = _codes(lint_skeleton(skeleton))

    assert C_EDGE_UNKNOWN in codes


def test_depends_on_unknown_reports_violation() -> None:
    skeleton = make_skeleton([make_node("a", deps=["missing"])])

    codes = _codes(lint_skeleton(skeleton))

    assert C_DEP_UNKNOWN in codes


def test_input_from_unknown_reports_violation() -> None:
    skeleton = make_skeleton([make_node("a", inputs=[{"from": "missing", "field": "result"}])])

    codes = _codes(lint_skeleton(skeleton))

    assert C_INPUT_UNKNOWN in codes


def test_input_field_invalid_reports_violation() -> None:
    skeleton = make_skeleton(
        [
            make_node("source"),
            make_node("consumer", deps=["source"], inputs=[{"from": "source", "field": "garbage"}]),
        ]
    )

    codes = _codes(lint_skeleton(skeleton))

    assert C_INPUT_FIELD in codes


def test_cycle_reports_violation() -> None:
    skeleton = make_skeleton(
        [
            make_node("a", deps=["b"]),
            make_node("b", deps=["a"]),
        ]
    )

    codes = _codes(lint_skeleton(skeleton))

    assert C_CYCLE in codes


def test_negative_repair_attempt_budget_reports_violation() -> None:
    skeleton = make_skeleton([make_node("a")], max_repair_attempts=-1)

    codes = _codes(lint_skeleton(skeleton))

    assert C_BUDGET in codes


def test_zero_total_cost_budget_reports_violation() -> None:
    skeleton = make_skeleton([make_node("a")], max_total_cost=0)

    codes = _codes(lint_skeleton(skeleton))

    assert C_BUDGET in codes


def test_writer_without_downstream_reviewer_reports_violation() -> None:
    skeleton = make_skeleton([make_node("impl", write_scope=["src/a.py"], read_only=False)])

    codes = _codes(lint_skeleton(skeleton))

    assert C_REVIEWER_COVERAGE in codes


def test_indirect_downstream_reviewer_satisfies_writer_coverage() -> None:
    skeleton = make_skeleton(
        [
            make_node("impl", write_scope=["src/a.py"], read_only=False),
            make_node("verify", deps=["impl"]),
            make_node("review", role="reviewer", deps=["verify"], reviewer_criteria=["Verified output"]),
        ]
    )

    codes = _codes(lint_skeleton(skeleton))

    assert C_REVIEWER_COVERAGE not in codes


def test_reviewer_missing_criteria_reports_violation() -> None:
    skeleton = make_skeleton([make_node("review", role="reviewer")])

    codes = _codes(lint_skeleton(skeleton))

    assert C_REVIEWER_CRITERIA in codes


def test_dynamic_hook_present_reports_violation() -> None:
    skeleton = make_skeleton([make_node("a")])
    setattr(skeleton, "dynamic_hooks", ["spawn_more_nodes"])

    codes = _codes(lint_skeleton(skeleton))

    assert C_DYNAMIC_HOOK in codes


def test_writer_without_output_file_reports_violation() -> None:
    skeleton = make_skeleton(
        [
            make_node("impl", role="implementer", write_scope=["src/a.py"], read_only=False, output_file=""),
            make_node("review", role="reviewer", deps=["impl"], reviewer_criteria=["ok"]),
        ]
    )

    violations = [v for v in lint_skeleton(skeleton).violations if v.code == C_OUTPUT_FILE]

    assert any(v.node_id == "impl" for v in violations)


def test_reviewer_without_output_file_reports_violation() -> None:
    skeleton = make_skeleton(
        [
            make_node("impl", role="implementer", write_scope=["src/a.py"], read_only=False),
            make_node("review", role="reviewer", deps=["impl"], reviewer_criteria=["ok"], output_file=""),
        ]
    )

    violations = [v for v in lint_skeleton(skeleton).violations if v.code == C_OUTPUT_FILE]

    assert any(v.node_id == "review" for v in violations)


def test_readonly_non_reviewer_without_output_file_is_allowed() -> None:
    skeleton = make_skeleton(
        [
            make_node("scan", role="explorer", read_only=True, output_file=""),
            make_node("impl", role="implementer", deps=["scan"], write_scope=["src/a.py"], read_only=False),
            make_node("review", role="reviewer", deps=["impl"], reviewer_criteria=["ok"]),
        ]
    )

    output_violations = [v.node_id for v in lint_skeleton(skeleton).violations if v.code == C_OUTPUT_FILE]

    assert "scan" not in output_violations


def test_input_field_output_file_is_accepted() -> None:
    skeleton = make_skeleton(
        [
            make_node("impl", role="implementer", write_scope=["src/a.py"], read_only=False),
            make_node(
                "review",
                role="reviewer",
                deps=["impl"],
                inputs=[{"from": "impl", "field": "output_file"}],
                reviewer_criteria=["ok"],
            ),
        ]
    )

    assert C_INPUT_FIELD not in _codes(lint_skeleton(skeleton))


def test_multiple_violations_are_collected_without_short_circuiting() -> None:
    skeleton = make_skeleton(
        [
            make_node(
                "a",
                role="reviewer",
                deps=["missing-dep"],
                inputs=[{"from": "missing-input", "field": "garbage"}],
                write_scope=[],
                read_only=True,
            ),
            make_node("a", write_scope=["src/uncovered.py"]),
        ],
        edges=[{"from": "a", "to": "missing-edge"}],
        max_repair_attempts=-1,
        max_total_cost=0,
    )
    setattr(skeleton, "dynamic_hooks", [{"name": "dynamic"}])

    codes = _codes(lint_skeleton(skeleton))

    assert {
        C_DUP_NODE_ID,
        C_EDGE_UNKNOWN,
        C_DEP_UNKNOWN,
        C_INPUT_UNKNOWN,
        C_INPUT_FIELD,
        C_BUDGET,
        C_REVIEWER_COVERAGE,
        C_REVIEWER_CRITERIA,
        C_DYNAMIC_HOOK,
    }.issubset(codes)


def _codes(result: Any) -> set[str]:
    return {violation.code for violation in result.violations}


def _edges_from_dependencies(nodes: list[SkeletonNode]) -> list[Edge]:
    edges: list[Edge] = []
    for node in nodes:
        for dependency in node.depends_on:
            edges.append({"from": dependency, "to": node.id})
    return edges
