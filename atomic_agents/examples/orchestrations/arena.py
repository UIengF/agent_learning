"""arena: parallel competing solutions -> judge gate."""

from __future__ import annotations

from . import prompts
from .common import (
    OrchestrationResult,
    ensure_workspace,
    make_skeleton,
    merge_role_runners,
    run_orchestration,
    write_node,
)


def build_skeleton(request: str, n_solutions: int, min_score: int = 7):
    if n_solutions < 2:
        raise ValueError("n_solutions must be >= 2")

    nodes = []
    solution_ids = [f"solution_{i}" for i in range(n_solutions)]
    solution_files = [f"solution-{i}.md" for i in range(n_solutions)]

    for i, (node_id, output_file) in enumerate(zip(solution_ids, solution_files)):
        nodes.append(
            write_node(
                node_id=node_id,
                role="designer",
                task=prompts.arena_solution_task(request, f"方案{i + 1}", 1),
                output_file=output_file,
            )
        )

    nodes.append(
        write_node(
            node_id="judge",
            role="reviewer",
            task=prompts.arena_judge_task(request, solution_files, min_score),
            output_file="verdict.json",
            depends_on=solution_ids,
            inputs=[{"from": node_id, "field": "output_file"} for node_id in solution_ids],
            reviewer_criteria=["最佳方案达标"],
        )
    )

    return make_skeleton("arena", nodes, max_repair_attempts=1)


def arena(
    request: str,
    *,
    n_solutions: int = 3,
    min_score: int = 7,
    role_runners: dict | None = None,
    workspace: str | None = None,
) -> OrchestrationResult:
    if n_solutions < 2:
        raise ValueError("n_solutions must be >= 2")
    ws = ensure_workspace(workspace, "atomic-arena-")
    skeleton = build_skeleton(request, n_solutions, min_score)
    return run_orchestration(
        "arena",
        request,
        skeleton,
        merge_role_runners(role_runners),
        ws,
        final_file="verdict.json",
        extra={"min_score": min_score},
    )


__all__ = ["arena", "build_skeleton"]
