"""P6 step1 approval presentation layer.

This module maps MetaPlan objects into ApprovalSummary objects, renders
human-readable approval and stop-report text, and defines the approval callback
protocol used by launchers around the two explicit release gates.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from atomic_agents.meta import MetaPlan
from atomic_agents.models import ApprovalSummary, Reviewer, StopReport
from atomic_agents.validation import validate_approval_summary


PER_ATOM_TIMEOUT_SEC = 1800
PlanResponse = Literal["approve", "reject", "edit", "unknown"]


@dataclass(kw_only=True)
class ApprovalDecision:
    approved: bool
    edited: bool = False
    note: str = ""


class ApprovalCallback(Protocol):
    def approve_plan(self, summary: ApprovalSummary) -> ApprovalDecision:
        """Release gate before starting the compiled plan."""

    def approve_irreversible(self, op: str) -> bool:
        """Release gate before executing an irreversible operation."""


def render_approval_summary(plan: MetaPlan) -> ApprovalSummary:
    """Map a MetaPlan to the schema-validated approval summary shown to users."""

    max_total_cost = plan.skeleton.run_limits["max_total_cost"]
    max_repair_attempts = plan.skeleton.run_limits["max_repair_attempts"]

    reviewers: list[Reviewer] = [
        {"node": node.id, "criteria": list(node.reviewer_criteria)}
        for node in sorted(plan.skeleton.nodes, key=lambda item: item.id)
        if _is_reviewer_role(node.role)
    ]
    summary = ApprovalSummary(
        task_restated=_first_explanation_line(plan.explanation),
        stages=[
            f"{node.role}：{_summarize_task(node.task)}（{_write_state(node.read_only, node.write_scope)}）"
            for node in plan.skeleton.nodes
        ],
        files_may_change=sorted({path for node in plan.skeleton.nodes for path in node.write_scope}),
        reviewers=reviewers,
        budget={"max_total_cost": max_total_cost, "per_atom_timeout_sec": PER_ATOM_TIMEOUT_SEC},
        stop_points=[
            f"修复失败超过 {max_repair_attempts} 次即止损",
            f"累计成本超过 {max_total_cost} 即止损",
        ],
        irreversible_ops=list(plan.skeleton.irreversible_ops),
        risk_flags=[
            *(f"[{warning.severity}] {warning.message}" for warning in plan.warnings),
            "写原子按声明 write_scope 分组：不交叠并行、交叠串行；越界写记录但不阻断，"
            "存在并发写未声明同名文件的盲区（v1 无 worktree 硬隔离）",
            "验收仅依赖 reviewer，无 schema 兜底",
        ],
        editable_hints=(
            "你可以：调整阶段拆分、修改验收标准、调整预算/超时、禁用某个不可逆操作，"
            "或驳回让元编排者重新规划。"
        ),
    )
    validate_approval_summary(summary.to_dict())
    return summary


def system_guardrails() -> dict[str, list[str]]:
    """Return the fixed v1 boundary between proactive blocks and audit-only reports."""

    return {
        "blocks_before_harm": [
            "累计成本超预算 → 硬停并出 stop report",
            "不可逆操作（push/PR/外部API）→ 执行前硬确认",
            "修复失败超上限 → 硬停并出 stop report",
        ],
        "reports_after_fact": [
            "原子内部的具体行为（黑盒，只看契约输入输出）",
            "越界写文件（write_scope 外）→ changed-file 扫描检测并记录，默认不阻断（v1 无 worktree 硬隔离，存在并发写未声明同名文件的盲区）",
            "动态新增原子/计划外文件（v1 无动态 hook，留 v1.1；事后 run_summary 可见）",
        ],
    }


def render_approval_text(summary: ApprovalSummary, guardrails: dict[str, list[str]] | None = None) -> str:
    """Render an ApprovalSummary plus guardrail metadata as multi-line Chinese text."""

    resolved_guardrails = guardrails if guardrails is not None else system_guardrails()
    lines = [
        "任务复述",
        summary.task_restated,
        "",
        "阶段",
        *_bullets(summary.stages),
        "",
        "可能改动文件",
        *_bullets(summary.files_may_change),
        "",
        "各 reviewer 验收标准",
        *_reviewer_lines(summary.reviewers),
        "",
        "预算与超时",
        f"- 总预算上限：{summary.budget['max_total_cost']}",
        f"- 每原子超时：{summary.budget['per_atom_timeout_sec']} 秒",
        "",
        "止损点",
        *_bullets(summary.stop_points),
        "",
        "不可逆操作",
        *_bullets(summary.irreversible_ops),
        "",
        "风险提示",
        *_bullets(summary.risk_flags),
        "",
        "系统会阻止什么",
        *_bullets(resolved_guardrails.get("blocks_before_harm", [])),
        "",
        "系统只会事后报告什么",
        *_bullets(resolved_guardrails.get("reports_after_fact", [])),
        "",
        "可改提示",
        summary.editable_hints,
    ]
    return "\n".join(lines)


def render_stop_report_text(report: StopReport) -> str:
    """Render a StopReport as multi-line Chinese text for user notification."""

    lines = [
        "止损原因",
        report.reason,
        "",
        "失败原子",
        report.failed_atom,
        "",
        "尝试次数",
        str(report.attempts),
        "",
        "reviewer 证据",
        *_reviewer_evidence_lines(report),
        "",
        "已改文件",
        *_bullets(report.files_changed),
        "",
        "成本消耗",
        str(report.cost_consumed),
        "",
        "可能原因",
        *_bullets(report.likely_causes),
        "",
        "可选动作",
        *_bullets(report.options),
    ]
    return "\n".join(lines)


class AutoApprove(ApprovalCallback):
    def approve_plan(self, summary: ApprovalSummary) -> ApprovalDecision:
        del summary
        return ApprovalDecision(approved=True)

    def approve_irreversible(self, op: str) -> bool:
        del op
        return True


class AutoReject(ApprovalCallback):
    def approve_plan(self, summary: ApprovalSummary) -> ApprovalDecision:
        del summary
        return ApprovalDecision(approved=False)

    def approve_irreversible(self, op: str) -> bool:
        del op
        return False


@dataclass(kw_only=True)
class CLIApproval(ApprovalCallback):
    input_fn: Callable[[], str] = input
    output_fn: Callable[[str], object] = print

    def approve_plan(self, summary: ApprovalSummary) -> ApprovalDecision:
        self.output_fn(render_approval_text(summary))
        response = _parse_plan_response(self.input_fn())
        if response == "approve":
            return ApprovalDecision(approved=True)
        if response == "edit":
            return ApprovalDecision(approved=False, edited=True, note="user requested edit")
        return ApprovalDecision(approved=False)

    def approve_irreversible(self, op: str) -> bool:
        self.output_fn(f"即将执行不可逆操作：{op}，确认？[y/N]")
        return _parse_yes_no(self.input_fn())


def _first_explanation_line(explanation: str) -> str:
    first_line = explanation.split("\n")[0].strip()
    if first_line:
        return first_line
    return "（无任务说明）"


def _summarize_task(task: str) -> str:
    return " ".join(task.split())[:60]


def _write_state(read_only: bool, write_scope: list[str]) -> str:
    if read_only:
        return "只读"
    if write_scope:
        return f"可能改动 {', '.join(write_scope)}"
    return "允许写入但未声明范围"


def _is_reviewer_role(role: str) -> bool:
    return role.strip().lower() == "reviewer"


def _bullets(items: list[str]) -> list[str]:
    if not items:
        return ["- 无"]
    return [f"- {item}" for item in items]


def _reviewer_lines(reviewers: list[Reviewer]) -> list[str]:
    if not reviewers:
        return ["- 无"]

    lines: list[str] = []
    for reviewer in reviewers:
        lines.append(f"- {reviewer['node']}")
        criteria = reviewer["criteria"]
        if criteria:
            lines.extend(f"  - {criterion}" for criterion in criteria)
        else:
            lines.append("  - 未声明验收标准")
    return lines


def _reviewer_evidence_lines(report: StopReport) -> list[str]:
    if not report.reviewer_evidence:
        return ["- 无"]

    return [
        f"- {item['criterion']}：{item['verdict']}；{item['evidence']}"
        for item in report.reviewer_evidence
    ]


def _parse_plan_response(raw: str) -> PlanResponse:
    normalized = raw.strip().lower()
    if normalized in {"y", "yes"}:
        return "approve"
    if normalized in {"n", "no"}:
        return "reject"
    if normalized in {"e", "edit"}:
        return "edit"
    return "unknown"


def _parse_yes_no(raw: str) -> bool:
    return raw.strip().lower() in {"y", "yes"}


__all__ = [
    "ApprovalCallback",
    "ApprovalDecision",
    "AutoApprove",
    "AutoReject",
    "CLIApproval",
    "render_approval_summary",
    "render_approval_text",
    "render_stop_report_text",
    "system_guardrails",
]
