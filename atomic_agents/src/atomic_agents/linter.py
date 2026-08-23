"""P5 semantic linter for skeletons, layered above JSON Schema validation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from atomic_agents.models import Skeleton, SkeletonNode


C_DUP_NODE_ID = "duplicate_node_id"
C_EDGE_UNKNOWN = "edge_unknown_node"
C_DEP_UNKNOWN = "depends_on_unknown"
C_INPUT_UNKNOWN = "input_from_unknown"
C_INPUT_FIELD = "input_field_invalid"
C_CYCLE = "cycle_detected"
C_BUDGET = "budget_invalid"
C_REVIEWER_COVERAGE = "writer_without_reviewer_downstream"
C_REVIEWER_CRITERIA = "reviewer_missing_criteria"
C_DYNAMIC_HOOK = "dynamic_hook_present"
C_OUTPUT_FILE = "writer_missing_output_file"

VALID_INPUT_FIELDS = {"result", "artifacts", "output_file"}


@dataclass(kw_only=True)
class Violation:
    code: str
    message: str
    node_id: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "code": self.code,
            "message": self.message,
            "node_id": self.node_id,
        }


@dataclass(kw_only=True)
class LintResult:
    ok: bool
    violations: list[Violation]

    def __post_init__(self) -> None:
        self.violations = list(self.violations)
        self.ok = not self.violations

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "violations": [violation.to_dict() for violation in self.violations],
        }


def lint_skeleton(skeleton: Skeleton) -> LintResult:
    """Run all semantic checks for a static v1 skeleton."""

    nodes_by_id = _first_nodes_by_id(skeleton.nodes)
    dependents = _dependents_by_id(skeleton.nodes, nodes_by_id)

    violations: list[Violation] = []
    violations.extend(_check_duplicate_node_ids(skeleton))
    violations.extend(_check_edges_reference_known_nodes(skeleton, nodes_by_id))
    violations.extend(_check_dependencies_reference_known_nodes(skeleton, nodes_by_id))
    violations.extend(_check_inputs(skeleton, nodes_by_id))
    violations.extend(_check_cycles(skeleton, nodes_by_id))
    violations.extend(_check_budget(skeleton))
    violations.extend(_check_reviewer_coverage(skeleton, dependents))
    violations.extend(_check_reviewer_criteria(skeleton))
    violations.extend(_check_output_file(skeleton))
    violations.extend(_check_dynamic_hooks(skeleton))

    return LintResult(ok=not violations, violations=violations)


def _check_duplicate_node_ids(skeleton: Skeleton) -> list[Violation]:
    seen: set[str] = set()
    violations: list[Violation] = []

    for node in skeleton.nodes:
        if node.id in seen:
            violations.append(
                Violation(
                    code=C_DUP_NODE_ID,
                    message=f"duplicate node id: {node.id}",
                    node_id=node.id,
                )
            )
            continue
        seen.add(node.id)

    return violations


def _check_edges_reference_known_nodes(skeleton: Skeleton, nodes_by_id: dict[str, SkeletonNode]) -> list[Violation]:
    violations: list[Violation] = []

    for edge in skeleton.edges:
        from_id = edge["from"]
        to_id = edge["to"]

        if from_id not in nodes_by_id:
            violations.append(
                Violation(
                    code=C_EDGE_UNKNOWN,
                    message=f"edge references unknown from node: {from_id}",
                    node_id=from_id,
                )
            )
        if to_id not in nodes_by_id:
            violations.append(
                Violation(
                    code=C_EDGE_UNKNOWN,
                    message=f"edge references unknown to node: {to_id}",
                    node_id=to_id,
                )
            )

    return violations


def _check_dependencies_reference_known_nodes(
    skeleton: Skeleton,
    nodes_by_id: dict[str, SkeletonNode],
) -> list[Violation]:
    violations: list[Violation] = []

    for node in skeleton.nodes:
        for dependency in node.depends_on:
            if dependency not in nodes_by_id:
                violations.append(
                    Violation(
                        code=C_DEP_UNKNOWN,
                        message=f"node {node.id} depends on unknown node: {dependency}",
                        node_id=node.id,
                    )
                )

    return violations


def _check_inputs(skeleton: Skeleton, nodes_by_id: dict[str, SkeletonNode]) -> list[Violation]:
    violations: list[Violation] = []

    for node in skeleton.nodes:
        for input_ref in node.inputs:
            source_id = input_ref["from"]
            field = input_ref["field"]

            if source_id not in nodes_by_id:
                violations.append(
                    Violation(
                        code=C_INPUT_UNKNOWN,
                        message=f"node {node.id} input references unknown node: {source_id}",
                        node_id=node.id,
                    )
                )
            if field not in VALID_INPUT_FIELDS:
                violations.append(
                    Violation(
                        code=C_INPUT_FIELD,
                        message=f"node {node.id} input field is unsupported: {field}",
                        node_id=node.id,
                    )
                )

    return violations


def _check_cycles(skeleton: Skeleton, nodes_by_id: dict[str, SkeletonNode]) -> list[Violation]:
    indegree = {node_id: 0 for node_id in nodes_by_id}
    dependents = {node_id: [] for node_id in nodes_by_id}
    seen_edges: set[tuple[str, str]] = set()
    seen_nodes: set[str] = set()

    for node in skeleton.nodes:
        if node.id in seen_nodes:
            continue
        seen_nodes.add(node.id)
        for dependency in node.depends_on:
            if dependency not in nodes_by_id:
                continue
            edge = (dependency, node.id)
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            indegree[node.id] += 1
            dependents[dependency].append(node.id)

    node_index = _first_node_indexes(skeleton.nodes)
    ready = [
        node.id
        for index, node in enumerate(skeleton.nodes)
        if node_index[node.id] == index and indegree[node.id] == 0
    ]
    visited_count = 0

    while ready:
        node_id = ready.pop(0)
        visited_count += 1
        for dependent in dependents[node_id]:
            indegree[dependent] -= 1
            if indegree[dependent] == 0:
                ready.append(dependent)
        ready.sort(key=node_index.__getitem__)

    if visited_count == len(nodes_by_id):
        return []

    cycle_nodes = [
        node.id
        for index, node in enumerate(skeleton.nodes)
        if node_index[node.id] == index and indegree[node.id] > 0
    ]
    return [
        Violation(
            code=C_CYCLE,
            message=f"cycle detected among nodes: {', '.join(cycle_nodes)}",
        )
    ]


def _check_budget(skeleton: Skeleton) -> list[Violation]:
    violations: list[Violation] = []

    if skeleton.run_limits["max_repair_attempts"] < 0:
        violations.append(
            Violation(
                code=C_BUDGET,
                message="run_limits.max_repair_attempts must be greater than or equal to 0",
            )
        )
    if skeleton.run_limits["max_total_cost"] <= 0:
        violations.append(
            Violation(
                code=C_BUDGET,
                message="run_limits.max_total_cost must be greater than 0",
            )
        )

    return violations


def _check_reviewer_coverage(skeleton: Skeleton, dependents: dict[str, list[str]]) -> list[Violation]:
    nodes_by_id = _first_nodes_by_id(skeleton.nodes)
    violations: list[Violation] = []

    for node in skeleton.nodes:
        if not node.write_scope or _is_reviewer(node):
            continue

        downstream = _reachable_downstream(node.id, dependents)
        if any(_is_reviewer(nodes_by_id[downstream_id]) for downstream_id in downstream if downstream_id in nodes_by_id):
            continue

        violations.append(
            Violation(
                code=C_REVIEWER_COVERAGE,
                message=f"writer node {node.id} has no downstream reviewer",
                node_id=node.id,
            )
        )

    return violations


def _check_reviewer_criteria(skeleton: Skeleton) -> list[Violation]:
    violations: list[Violation] = []

    for node in skeleton.nodes:
        if not _is_reviewer(node):
            continue
        if node.reviewer_criteria:
            continue

        violations.append(
            Violation(
                code=C_REVIEWER_CRITERIA,
                message=f"reviewer node {node.id} must define reviewer criteria",
                node_id=node.id,
            )
        )

    return violations


def _check_dynamic_hooks(skeleton: Skeleton) -> list[Violation]:
    dynamic_hooks = getattr(skeleton, "dynamic_hooks", None)
    if not dynamic_hooks:
        return []

    return [
        Violation(
            code=C_DYNAMIC_HOOK,
            message="dynamic hooks are not allowed in static v1 skeletons",
        )
    ]


def _check_output_file(skeleton: Skeleton) -> list[Violation]:
    """Writer atoms (non-empty write_scope) and reviewers must declare output_file.

    The output_file is the atom's primary text-file product. Writers produce a
    file; reviewers produce a verdict file the scheduler reads to decide
    pass/fail. Read-only non-reviewer nodes may leave output_file empty.
    """

    violations: list[Violation] = []

    for node in skeleton.nodes:
        requires_output = bool(node.write_scope) or _is_reviewer(node)
        if not requires_output:
            continue
        if node.output_file:
            continue

        reason = "reviewer" if _is_reviewer(node) else "writer"
        violations.append(
            Violation(
                code=C_OUTPUT_FILE,
                message=f"{reason} node {node.id} must declare output_file",
                node_id=node.id,
            )
        )

    return violations


def _first_nodes_by_id(nodes: list[SkeletonNode]) -> dict[str, SkeletonNode]:
    nodes_by_id: dict[str, SkeletonNode] = {}
    for node in nodes:
        if node.id not in nodes_by_id:
            nodes_by_id[node.id] = node
    return nodes_by_id


def _dependents_by_id(nodes: list[SkeletonNode], nodes_by_id: dict[str, SkeletonNode]) -> dict[str, list[str]]:
    dependents = {node_id: [] for node_id in nodes_by_id}
    seen_edges: set[tuple[str, str]] = set()
    seen_nodes: set[str] = set()

    for node in nodes:
        if node.id in seen_nodes:
            continue
        seen_nodes.add(node.id)
        for dependency in node.depends_on:
            if dependency not in nodes_by_id:
                continue
            edge = (dependency, node.id)
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            dependents[dependency].append(node.id)

    return dependents


def _first_node_indexes(nodes: list[SkeletonNode]) -> dict[str, int]:
    indexes: dict[str, int] = {}
    for index, node in enumerate(nodes):
        if node.id not in indexes:
            indexes[node.id] = index
    return indexes


def _reachable_downstream(node_id: str, dependents: dict[str, list[str]]) -> list[str]:
    reachable: list[str] = []
    visited: set[str] = set()
    queue = list(dependents.get(node_id, []))

    while queue:
        downstream_id = queue.pop(0)
        if downstream_id in visited:
            continue
        visited.add(downstream_id)
        reachable.append(downstream_id)
        queue.extend(dependents.get(downstream_id, []))

    return reachable


def _is_reviewer(node: SkeletonNode) -> bool:
    return node.role.strip().lower() == "reviewer"


__all__ = [
    "C_BUDGET",
    "C_CYCLE",
    "C_DEP_UNKNOWN",
    "C_DUP_NODE_ID",
    "C_DYNAMIC_HOOK",
    "C_EDGE_UNKNOWN",
    "C_INPUT_FIELD",
    "C_INPUT_UNKNOWN",
    "C_OUTPUT_FILE",
    "C_REVIEWER_COVERAGE",
    "C_REVIEWER_CRITERIA",
    "LintResult",
    "Violation",
    "lint_skeleton",
]
