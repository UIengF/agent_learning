"""pipeline: plan -> implement -> review gate."""

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


def build_skeleton(request: str):
    nodes = [
        write_node(
            node_id="plan",
            role="planner",
            task=prompts.plan_task(request),
            output_file="plan.md",
        ),
        write_node(
            node_id="impl",
            role="implementer",
            task=prompts.impl_task(request, "plan.md"),
            output_file="impl_output.md",
            depends_on=["plan"],
            inputs=[{"from": "plan", "field": "output_file"}],
        ),
        write_node(
            node_id="review",
            role="reviewer",
            task=prompts.pipeline_review_task(request, "impl_output.md"),
            output_file="verdict.json",
            depends_on=["impl"],
            inputs=[{"from": "impl", "field": "output_file"}],
            reviewer_criteria=["实现满足需求", "代码正确健壮"],
        ),
    ]
    return make_skeleton("pipeline", nodes, max_repair_attempts=1)


def pipeline(
    request: str,
    *,
    role_runners: dict | None = None,
    workspace: str | None = None,
) -> OrchestrationResult:
    ws = ensure_workspace(workspace, "atomic-pipeline-")
    skeleton = build_skeleton(request)
    return run_orchestration(
        "pipeline",
        request,
        skeleton,
        merge_role_runners(role_runners),
        ws,
        final_file="impl_output.md",
    )


__all__ = ["pipeline", "build_skeleton"]
