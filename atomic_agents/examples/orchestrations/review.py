"""review: parallel written reviews -> judge gate."""

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


def _copy_doc_to_workspace(doc_path: str, workspace: str) -> str:
    source = Path(doc_path).expanduser()
    if not source.exists():
        raise FileNotFoundError(doc_path)

    target = Path(workspace) / source.name
    if source.resolve() != target.resolve():
        shutil.copy2(source, target)
    return target.name


def build_skeleton(request: str, doc_name: str, n_reviewers: int, min_score: int = 7):
    if n_reviewers < 2:
        raise ValueError("n_reviewers must be >= 2")

    nodes = []
    reviewer_ids = [f"reviewer_{i}" for i in range(n_reviewers)]
    review_files = [f"review-{i}.md" for i in range(n_reviewers)]

    for i, (node_id, output_file) in enumerate(zip(reviewer_ids, review_files)):
        nodes.append(
            write_node(
                node_id=node_id,
                role="analyst",
                task=prompts.review_task(request, doc_name, perspective_hint=f"视角{i + 1}"),
                output_file=output_file,
            )
        )

    nodes.append(
        write_node(
            node_id="judge",
            role="reviewer",
            task=prompts.review_judge_task(request, review_files, min_score),
            output_file="verdict.json",
            depends_on=reviewer_ids,
            inputs=[{"from": node_id, "field": "output_file"} for node_id in reviewer_ids],
            reviewer_criteria=["分析深度", "结论有据"],
        )
    )

    return make_skeleton("review", nodes, max_repair_attempts=1)


def review(
    request: str,
    doc_path: str,
    *,
    n_reviewers: int = 2,
    min_score: int = 7,
    role_runners: dict | None = None,
    workspace: str | None = None,
) -> OrchestrationResult:
    if n_reviewers < 2:
        raise ValueError("n_reviewers must be >= 2")
    ws = ensure_workspace(workspace, "atomic-review-")
    doc_name = _copy_doc_to_workspace(doc_path, ws)
    skeleton = build_skeleton(request, doc_name, n_reviewers, min_score)
    return run_orchestration(
        "review",
        request,
        skeleton,
        merge_role_runners(role_runners),
        ws,
        final_file="verdict.json",
        extra={"min_score": min_score},
    )


__all__ = ["review", "build_skeleton"]
