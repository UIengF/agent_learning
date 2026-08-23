"""execute_plan: 读入一份已有的外部计划文档 -> codex 实现 -> reviewer 验收。

与 pipeline 的区别：pipeline 由 planner(ducc) 现场生成计划；execute_plan 跳过
planner，直接把用户指定的外部计划文档（例如 design 模板产出的 final-solution.md）
复制进 workspace 作为 impl 的输入。适用场景：方案已经设计/评审完毕，只需要分派
codex 去执行落地。
"""

from __future__ import annotations

import shutil
from pathlib import Path

from . import prompts
from .common import (
    OrchestrationResult,
    ensure_workspace,
    make_skeleton,
    merge_role_runners,
    run_orchestration,
    write_node,
)


def _copy_plan_to_workspace(plan_path: str, workspace: str) -> str:
    source = Path(plan_path).expanduser()
    if not source.exists():
        raise FileNotFoundError(plan_path)
    if not source.is_file():
        raise IsADirectoryError(plan_path)

    workspace_path = Path(workspace).expanduser().resolve()
    workspace_path.mkdir(parents=True, exist_ok=True)
    target = workspace_path / source.name
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    return target.name


def build_skeleton(request: str, plan_name: str):
    nodes = [
        write_node(
            node_id="impl",
            role="implementer",
            task=prompts.impl_task(request, plan_name),
            output_file="impl_output.md",
            context_files=[plan_name],
        ),
        write_node(
            node_id="review",
            role="reviewer",
            task=prompts.pipeline_review_task(request, "impl_output.md"),
            output_file="verdict.json",
            depends_on=["impl"],
            inputs=[{"from": "impl", "field": "output_file"}],
            reviewer_criteria=["实现满足计划", "代码正确健壮"],
        ),
    ]
    return make_skeleton("execute-plan", nodes, max_repair_attempts=1)


def execute_plan(
    request: str,
    plan_path: str,
    *,
    role_runners: dict | None = None,
    workspace: str | None = None,
) -> OrchestrationResult:
    ws = ensure_workspace(workspace, "atomic-execplan-")
    plan_name = _copy_plan_to_workspace(plan_path, ws)
    skeleton = build_skeleton(request, plan_name)
    return run_orchestration(
        "execute-plan",
        request,
        skeleton,
        merge_role_runners(role_runners),
        ws,
        final_file="impl_output.md",
        extra={"plan_path": plan_path, "plan_name": plan_name},
    )


__all__ = ["execute_plan", "build_skeleton"]
