"""P5 step2 skeleton templates and compiler protocol.

This module implements the template half of decision 9: template-first
skeleton generation. Templates are proven shapes and not the upper bound;
free-form orchestration is introduced by later P5 steps.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from atomic_agents.models import Skeleton, SkeletonNode


@dataclass(kw_only=True, frozen=True)
class Template:
    name: str
    intent_keywords: tuple[str, ...]
    description: str
    build: Callable[[str], Skeleton]


@runtime_checkable
class SkeletonCompiler(Protocol):
    """Protocol for deterministic or LLM-backed skeleton compilers."""

    def generate(self, request: str) -> Skeleton:
        """Generate a static skeleton for a natural-language request."""


class MockCompiler:
    """Deterministic SkeletonCompiler for tests and P5 wiring smoke flows."""

    def __init__(self, skeletons: Mapping[str, Skeleton] | None = None) -> None:
        self.skeletons = {request: _copy_skeleton(skeleton) for request, skeleton in (skeletons or {}).items()}
        self.calls: list[str] = []

    def generate(self, request: str) -> Skeleton:
        self.calls.append(request)
        if request in self.skeletons:
            return _copy_skeleton(self.skeletons[request])

        skeleton = build_from_template(request)
        if skeleton is not None:
            return skeleton

        return _build_plan_impl_review(request)


def _build_plan_impl_review(request: str) -> Skeleton:
    return Skeleton(
        name="plan-impl-review",
        version=1,
        nodes=[
            SkeletonNode(
                id="plan",
                role="planner",
                task=f"拆解需求为实现步骤：{request}",
                depends_on=[],
                inputs=[],
                write_scope=["docs/plan.md"],
                read_only=False,
                required_capabilities=["write_files"],
                reviewer_criteria=[],
                output_file="docs/plan.md",
            ),
            SkeletonNode(
                id="impl",
                role="implementer",
                task=f"按计划实现：{request}",
                depends_on=["plan"],
                inputs=[{"from": "plan", "field": "result"}],
                write_scope=["src/feature.py"],
                read_only=False,
                required_capabilities=["write_files"],
                reviewer_criteria=[],
                output_file="src/feature.py",
            ),
            SkeletonNode(
                id="review",
                role="reviewer",
                task=f"审查实现是否满足：{request}",
                depends_on=["impl"],
                inputs=[{"from": "impl", "field": "artifacts"}],
                write_scope=["docs/review.md"],
                read_only=False,
                required_capabilities=["write_files"],
                reviewer_criteria=["实现满足需求描述", "无明显回归", "关键路径有测试"],
                output_file="docs/review.md",
            ),
        ],
        edges=[{"from": "plan", "to": "impl"}, {"from": "impl", "to": "review"}],
        run_limits={"max_repair_attempts": 2, "max_total_cost": 100.0},
        irreversible_ops=[],
    )


def _build_scatter_gather(request: str) -> Skeleton:
    return Skeleton(
        name="scatter-gather",
        version=1,
        nodes=[
            SkeletonNode(
                id="explore1",
                role="explorer",
                task=f"调研方向A：{request}",
                depends_on=[],
                inputs=[],
                write_scope=[],
                read_only=True,
                required_capabilities=[],
                reviewer_criteria=[],
                output_file="docs/explore-1.md",
            ),
            SkeletonNode(
                id="explore2",
                role="explorer",
                task=f"调研方向B：{request}",
                depends_on=[],
                inputs=[],
                write_scope=[],
                read_only=True,
                required_capabilities=[],
                reviewer_criteria=[],
                output_file="docs/explore-2.md",
            ),
            SkeletonNode(
                id="gather",
                role="synthesizer",
                task=f"汇总调研结论：{request}",
                depends_on=["explore1", "explore2"],
                inputs=[{"from": "explore1", "field": "result"}, {"from": "explore2", "field": "result"}],
                write_scope=["docs/summary.md"],
                read_only=False,
                required_capabilities=["write_files"],
                reviewer_criteria=[],
                output_file="docs/summary.md",
            ),
            SkeletonNode(
                id="review",
                role="reviewer",
                task=f"审查汇总是否可靠：{request}",
                depends_on=["gather"],
                inputs=[{"from": "gather", "field": "artifacts"}],
                write_scope=["docs/review.md"],
                read_only=False,
                required_capabilities=["write_files"],
                reviewer_criteria=["汇总覆盖所有调研输入", "结论有依据"],
                output_file="docs/review.md",
            ),
        ],
        edges=[
            {"from": "explore1", "to": "gather"},
            {"from": "explore2", "to": "gather"},
            {"from": "gather", "to": "review"},
        ],
        run_limits={"max_repair_attempts": 2, "max_total_cost": 100.0},
        irreversible_ops=[],
    )


TEMPLATES: tuple[Template, ...] = (
    Template(
        name="plan-impl-review",
        intent_keywords=("实现", "开发", "功能", "feature", "implement", "fix", "修复", "bug"),
        description="线性 plan -> impl -> review，适合实现、修复和功能开发。",
        build=_build_plan_impl_review,
    ),
    Template(
        name="scatter-gather",
        intent_keywords=("调研", "对比", "研究", "评估", "research", "compare", "survey", "分析"),
        description="并行只读调研 -> 汇总写入 -> reviewer，适合研究、对比和评估。",
        build=_build_scatter_gather,
    ),
)


def match_template(request: str) -> Template | None:
    normalized = request.lower()
    for template in TEMPLATES:
        if any(keyword in normalized for keyword in template.intent_keywords):
            return template
    return None


def fill_template(template: Template, request: str) -> Skeleton:
    return template.build(request)


def build_from_template(request: str) -> Skeleton | None:
    template = match_template(request)
    if template is None:
        return None
    return fill_template(template, request)


def _copy_skeleton(skeleton: Skeleton) -> Skeleton:
    return Skeleton.from_dict(deepcopy(skeleton.to_dict()))


__all__ = [
    "MockCompiler",
    "SkeletonCompiler",
    "TEMPLATES",
    "Template",
    "build_from_template",
    "fill_template",
    "match_template",
]
