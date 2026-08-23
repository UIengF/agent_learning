"""scatter_gather: parallel exploration -> synthesis -> review gate."""

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

_DIRECTION_HINTS = ["技术可行性", "现有方案", "风险与边界", "成本与维护", "用户场景"]


def _direction_hint(index: int) -> str:
    if index < len(_DIRECTION_HINTS):
        return _DIRECTION_HINTS[index]
    return f"调研方向 {index + 1}（从一个独立角度切入）"


def build_skeleton(request: str, n_explorers: int):
    if n_explorers < 2:
        raise ValueError("n_explorers must be >= 2")

    nodes = []
    explore_ids = [f"explore_{i}" for i in range(n_explorers)]
    explore_files = [f"explore-{i}.md" for i in range(n_explorers)]

    for i, (node_id, output_file) in enumerate(zip(explore_ids, explore_files)):
        nodes.append(
            write_node(
                node_id=node_id,
                role="explorer",
                task=prompts.explore_task(request, _direction_hint(i)),
                output_file=output_file,
            )
        )

    nodes.append(
        write_node(
            node_id="gather",
            role="synthesizer",
            task=prompts.gather_task(request, explore_files),
            output_file="summary.md",
            depends_on=explore_ids,
            inputs=[{"from": node_id, "field": "output_file"} for node_id in explore_ids],
        )
    )
    nodes.append(
        write_node(
            node_id="review",
            role="reviewer",
            task=prompts.gather_review_task(request, "summary.md"),
            output_file="verdict.json",
            depends_on=["gather"],
            inputs=[{"from": "gather", "field": "output_file"}],
            reviewer_criteria=["覆盖所有调研", "结论有据"],
        )
    )

    return make_skeleton("scatter-gather", nodes, max_repair_attempts=1)


def scatter_gather(
    request: str,
    *,
    n_explorers: int = 3,
    role_runners: dict | None = None,
    workspace: str | None = None,
) -> OrchestrationResult:
    if n_explorers < 2:
        raise ValueError("n_explorers must be >= 2")
    ws = ensure_workspace(workspace, "atomic-sg-")
    skeleton = build_skeleton(request, n_explorers)
    return run_orchestration(
        "scatter-gather",
        request,
        skeleton,
        merge_role_runners(role_runners),
        ws,
        final_file="summary.md",
    )


__all__ = ["scatter_gather", "build_skeleton"]
