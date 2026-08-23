from __future__ import annotations

from atomic_agents.approval import (
    AutoApprove,
    AutoReject,
    CLIApproval,
    render_approval_summary,
    render_approval_text,
    render_stop_report_text,
    system_guardrails,
)
from atomic_agents.meta import MetaPlan, PlanWarning, meta_compile
from atomic_agents.models import ApprovalSummary, Skeleton, SkeletonNode, StopReport
from atomic_agents.templates import MockCompiler
from atomic_agents.validation import validate_approval_summary


def test_render_approval_summary_from_template_plan() -> None:
    plan = meta_compile("实现登录功能", "ducc", MockCompiler())

    summary = render_approval_summary(plan)

    assert summary.task_restated
    assert {"docs/plan.md", "src/feature.py"} <= set(summary.files_may_change)
    assert any(reviewer["node"] == "review" and reviewer["criteria"] for reviewer in summary.reviewers)
    assert summary.budget["max_total_cost"] == 100.0
    assert summary.budget["per_atom_timeout_sec"] == 1800
    assert len(summary.stop_points) == 2
    assert "2" in summary.stop_points[0]
    assert "100.0" in summary.stop_points[1]
    validate_approval_summary(summary.to_dict())


def test_render_approval_summary_includes_free_path_warnings_and_v1_boundaries() -> None:
    plan = MetaPlan(
        skeleton=_approval_skeleton(),
        source="free",
        template_name=None,
        bindings={},
        explanation="自由规划第一行\n更多说明",
        warnings=[
            PlanWarning(
                code="manual_risk",
                message="需要人工确认迁移风险",
                severity="high",
                node_id="impl",
            )
        ],
        lint_repaired=False,
    )

    summary = render_approval_summary(plan)

    assert "[high] 需要人工确认迁移风险" in summary.risk_flags
    assert any("不交叠并行" in flag and "越界写记录但不阻断" in flag for flag in summary.risk_flags)
    assert "验收仅依赖 reviewer，无 schema 兜底" in summary.risk_flags


def test_system_guardrails_has_blocking_and_reporting_buckets() -> None:
    guardrails = system_guardrails()

    assert set(guardrails) == {"blocks_before_harm", "reports_after_fact"}
    assert guardrails["blocks_before_harm"]
    assert guardrails["reports_after_fact"]


def test_render_approval_text_contains_sections_guardrails_and_reviewer_criteria() -> None:
    summary = render_approval_summary(meta_compile("实现登录功能", "ducc", MockCompiler()))

    text = render_approval_text(summary)

    for heading in [
        "任务复述",
        "阶段",
        "可能改动文件",
        "各 reviewer 验收标准",
        "预算与超时",
        "止损点",
        "不可逆操作",
        "风险提示",
        "系统会阻止什么",
        "系统只会事后报告什么",
        "可改提示",
    ]:
        assert heading in text
    assert "事后报告" in text
    assert "实现满足需求描述" in text


def test_render_stop_report_text_contains_stop_details() -> None:
    report = StopReport(
        reason="repair_failed",
        failed_atom="impl",
        attempts=2,
        reviewer_evidence=[{"criterion": "tests", "verdict": "fail", "evidence": "3 failing"}],
        files_changed=["src/feature.py"],
        cost_consumed=3.4,
        likely_causes=["测试环境缺依赖"],
        options=["人工修复后继续", "重新规划"],
    )

    text = render_stop_report_text(report)

    assert "repair_failed" in text
    assert "impl" in text
    assert "3.4" in text


def test_cli_approval_plan_decisions() -> None:
    summary = _approval_summary()

    yes_callback = CLIApproval(input_fn=lambda: "y", output_fn=lambda text: None)
    no_callback = CLIApproval(input_fn=lambda: "n", output_fn=lambda text: None)
    edit_callback = CLIApproval(input_fn=lambda: "edit", output_fn=lambda text: None)

    assert yes_callback.approve_plan(summary).approved is True
    assert no_callback.approve_plan(summary).approved is False

    edit_decision = edit_callback.approve_plan(summary)
    assert edit_decision.approved is False
    assert edit_decision.edited is True


def test_cli_approval_outputs_text_and_confirms_irreversible_ops() -> None:
    outputs: list[str] = []
    yes_callback = CLIApproval(input_fn=lambda: "y", output_fn=outputs.append)
    no_callback = CLIApproval(input_fn=lambda: "no", output_fn=outputs.append)

    assert yes_callback.approve_irreversible("git_push") is True
    assert no_callback.approve_irreversible("git_push") is False
    assert "不可逆操作" in outputs[0]


def test_auto_approve_and_reject_callbacks() -> None:
    summary = _approval_summary()

    assert AutoApprove().approve_plan(summary).approved is True
    assert AutoApprove().approve_irreversible("git_push") is True
    assert AutoReject().approve_plan(summary).approved is False
    assert AutoReject().approve_irreversible("git_push") is False


def _approval_summary() -> ApprovalSummary:
    return render_approval_summary(meta_compile("实现登录功能", "ducc", MockCompiler()))


def _approval_skeleton() -> Skeleton:
    return Skeleton(
        name="approval-test",
        version=1,
        nodes=[
            SkeletonNode(
                id="impl",
                role="implementer",
                task="实施迁移并更新配置",
                depends_on=[],
                inputs=[],
                write_scope=["src/config.py"],
                read_only=False,
                required_capabilities=["write_files"],
                reviewer_criteria=[],
                output_file="src/config.py",
            ),
            SkeletonNode(
                id="review",
                role="reviewer",
                task="审查迁移结果",
                depends_on=["impl"],
                inputs=[{"from": "impl", "field": "artifacts"}],
                write_scope=[],
                read_only=True,
                required_capabilities=[],
                reviewer_criteria=["配置变更完整", "无明显回归"],
                output_file="docs/review.md",
            ),
        ],
        edges=[{"from": "impl", "to": "review"}],
        run_limits={"max_repair_attempts": 3, "max_total_cost": 12.0},
        irreversible_ops=["git_push"],
    )
