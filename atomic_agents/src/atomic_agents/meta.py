"""P5 step3 meta orchestration for skeleton compilation.

This module implements decision 7 by suggesting bindings from the launcher
identity, and decision 9 by keeping template-first compilation with a free-form
fallback. It only produces a MetaPlan for later approval steps.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from atomic_agents.linter import Violation, lint_skeleton
from atomic_agents.models import Skeleton, SkeletonNode
from atomic_agents.templates import SkeletonCompiler, TEMPLATES, fill_template, match_template


PlanWarningSeverity = Literal["info", "warn", "high"]
MetaPlanSource = Literal["template", "free"]


@dataclass(kw_only=True)
class PlanWarning:
    code: str
    message: str
    severity: PlanWarningSeverity
    node_id: str | None = None

    def to_dict(self) -> dict[str, str | None]:
        return {
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
            "node_id": self.node_id,
        }


@dataclass(kw_only=True)
class MetaPlan:
    skeleton: Skeleton
    source: MetaPlanSource
    template_name: str | None
    bindings: dict[str, dict[str, str]]
    explanation: str
    warnings: list[PlanWarning]
    lint_repaired: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "skeleton": self.skeleton.to_dict(),
            "source": self.source,
            "template_name": self.template_name,
            "bindings": self.bindings,
            "explanation": self.explanation,
            "warnings": [warning.to_dict() for warning in self.warnings],
            "lint_repaired": self.lint_repaired,
        }


class LintFailedError(Exception):
    def __init__(self, violations: list[Violation]) -> None:
        self.violations = list(violations)
        preview = "; ".join(f"{violation.code}: {violation.message}" for violation in self.violations[:3])
        if len(self.violations) > 3:
            preview = f"{preview}; ..."
        super().__init__(f"skeleton lint failed: {preview}")


def meta_compile(
    request: str,
    launcher_identity: str,
    compiler: SkeletonCompiler,
    *,
    max_repair_rounds: int = 2,
) -> MetaPlan:
    """Compile a request into a lint-clean MetaPlan without running approval."""

    template = match_template(request)
    if template is not None:
        skeleton = fill_template(template, request)
        source: MetaPlanSource = "template"
        template_name = template.name
        lint_repaired = False

        lint = lint_skeleton(skeleton)
        if not lint.ok:
            raise LintFailedError(lint.violations)
    else:
        skeleton = compiler.generate(request)
        source = "free"
        template_name = None

        lint = lint_skeleton(skeleton)
        rounds = 0
        repair = getattr(compiler, "repair", None)
        while not lint.ok and rounds < max_repair_rounds and callable(repair):
            skeleton = repair(skeleton, lint.violations)
            lint = lint_skeleton(skeleton)
            rounds += 1

        lint_repaired = rounds > 0
        if not lint.ok:
            raise LintFailedError(lint.violations)

    bindings = suggest_bindings(skeleton, launcher_identity)
    explanation = render_explanation(skeleton, source, template_name)
    warnings = plan_quality_warnings(skeleton, source)

    return MetaPlan(
        skeleton=skeleton,
        source=source,
        template_name=template_name,
        bindings=bindings,
        explanation=explanation,
        warnings=warnings,
        lint_repaired=lint_repaired,
    )


def suggest_bindings(skeleton: Skeleton, launcher_identity: str) -> dict[str, dict[str, str]]:
    """Suggest logical role bindings that follow the launcher's identity."""

    roles = sorted({node.role for node in skeleton.nodes})
    return {role: {"runner": launcher_identity} for role in roles}


def render_explanation(skeleton: Skeleton, source: MetaPlanSource | str, template_name: str | None) -> str:
    """Render a human-readable explanation of why the skeleton is shaped this way."""

    lines: list[str] = []
    if source == "template":
        description = _template_description(template_name)
        lines.append(f"来源：命中模板 {template_name}，适用场景：{description}")
    else:
        lines.append("来源：自由编排，由编译器根据请求生成静态骨架。")

    lines.append("阶段：")
    for node in skeleton.nodes:
        write_state = _write_state(node)
        reviewer_state = "reviewer" if _is_reviewer(node) else "非 reviewer"
        lines.append(f"- {node.role}：{_summarize_task(node.task)}；{write_state}；{reviewer_state}。")

    reviewer_nodes = [node for node in skeleton.nodes if _is_reviewer(node)]
    criteria_count = sum(len(node.reviewer_criteria) for node in reviewer_nodes)
    if reviewer_nodes:
        reviewer_ids = ", ".join(node.id for node in reviewer_nodes)
        lines.append(f"关键 reviewer 节点：{reviewer_ids}，共 {criteria_count} 条验收标准。")
    else:
        lines.append("关键 reviewer 节点：无，共 0 条验收标准。")

    return "\n".join(lines)


def plan_quality_warnings(skeleton: Skeleton, source: str) -> list[PlanWarning]:
    """Collect deterministic quality warnings for a compiled plan."""

    del source
    warnings: list[PlanWarning] = []
    warnings.extend(_parallel_write_overlap_warnings(skeleton))
    warnings.extend(_weak_acceptance_warnings(skeleton))
    warnings.extend(_undeclared_write_risk_warnings(skeleton))
    warnings.extend(_cost_budget_risk_warnings(skeleton))
    return warnings


def _parallel_write_overlap_warnings(skeleton: Skeleton) -> list[PlanWarning]:
    warnings: list[PlanWarning] = []

    for index, left in enumerate(skeleton.nodes):
        left_scope = set(left.write_scope)
        if not left_scope:
            continue
        for right in skeleton.nodes[index + 1 :]:
            right_scope = set(right.write_scope)
            if not right_scope:
                continue
            overlap = sorted(left_scope & right_scope)
            if not overlap:
                continue
            warnings.append(
                PlanWarning(
                    code="parallel_write_overlap",
                    message=f"写范围重叠：{', '.join(overlap)}；节点 {left.id} 与 {right.id} 可能产生写冲突。",
                    severity="high",
                    node_id=left.id,
                )
            )

    return warnings


def _weak_acceptance_warnings(skeleton: Skeleton) -> list[PlanWarning]:
    reviewers = [node for node in skeleton.nodes if _is_reviewer(node)]
    if not reviewers:
        return [
            PlanWarning(
                code="weak_acceptance",
                message="整个骨架没有 reviewer 节点，缺少独立验收阶段。",
                severity="warn",
            )
        ]

    warnings: list[PlanWarning] = []
    for node in reviewers:
        if len(node.reviewer_criteria) >= 2:
            continue
        warnings.append(
            PlanWarning(
                code="weak_acceptance",
                message=f"reviewer 节点 {node.id} 的验收标准少于 2 条。",
                severity="warn",
                node_id=node.id,
            )
        )

    return warnings


def _undeclared_write_risk_warnings(skeleton: Skeleton) -> list[PlanWarning]:
    warnings: list[PlanWarning] = []

    for node in skeleton.nodes:
        if node.read_only or node.write_scope:
            continue
        warnings.append(
            PlanWarning(
                code="undeclared_write_risk",
                message=f"节点 {node.id} 允许写入但未声明 write_scope，drift 检测缺少范围基线。",
                severity="warn",
                node_id=node.id,
            )
        )

    return warnings


def _cost_budget_risk_warnings(skeleton: Skeleton) -> list[PlanWarning]:
    warnings: list[PlanWarning] = []
    node_count = len(skeleton.nodes)
    max_total_cost = skeleton.run_limits["max_total_cost"]

    if node_count > 6 and max_total_cost / node_count < 1.0:
        warnings.append(
            PlanWarning(
                code="cost_budget_risk",
                message=(
                    f"节点数为 {node_count}，max_total_cost={max_total_cost}，"
                    "均摊每节点预算低于 1.0，可能提前触发止损。"
                ),
                severity="info",
            )
        )

    if max_total_cost > 50:
        warnings.append(
            PlanWarning(
                code="cost_budget_risk",
                message=f"max_total_cost={max_total_cost} 偏高，请确认成本上限符合预期。",
                severity="info",
            )
        )

    return warnings


def _template_description(template_name: str | None) -> str:
    for template in TEMPLATES:
        if template.name == template_name:
            return template.description
    return "未知模板场景"


def _write_state(node: SkeletonNode) -> str:
    if node.read_only:
        return "只读"
    if node.write_scope:
        return f"写文件范围 {', '.join(node.write_scope)}"
    return "允许写入但未声明写范围"


def _summarize_task(task: str, *, limit: int = 80) -> str:
    normalized = " ".join(task.split())
    if len(normalized) <= limit:
        return normalized
    return f"{normalized[: limit - 1]}..."


def _is_reviewer(node: SkeletonNode) -> bool:
    return node.role.strip().lower() == "reviewer"


__all__ = [
    "LintFailedError",
    "MetaPlan",
    "PlanWarning",
    "meta_compile",
    "plan_quality_warnings",
    "render_explanation",
    "suggest_bindings",
]
