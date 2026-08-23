# 完成审计

## 原则

声明完成前必须有证据。计划状态、代码差异、投入时长或 agent 报告都不属于完成证据。

## 构建证据矩阵

将每项完成定义标准映射到一个或多个检查：

| 标准 | 证据 | 结果 | 契约 revision | 产物 revision / 观察窗口 | 剩余风险 |
|---|---|---|---|---|---|
| DOD-1 | 精确测试或观察到的行为 | pass/fail | 契约内容哈希或版本 | Git commit、内容哈希、版本 ID 或 UTC 窗口 | 剩余限制 |

先使用能够证明声明的最小检查；若共享行为或集成风险需要，再增加更广泛的检查。

## 证据质量

良好证据应：

- 与标准和当前 `contract_revision` 直接对应；
- 产生于相关最终变更之后；
- 足以呈现失败信息和退出状态；
- 可复现或可检查；
- 明确说明部分覆盖和不确定性。

将每项证据绑定到产生该声明的契约 revision，以及所有相关代码或产物的 Git commit、内容哈希或不可变版本 ID。对于外部或持续变化的状态，记录 UTC `observation_window`、查询范围和观察对象。只有时间戳而没有这些绑定，不足以证明当前目标。

`checked_at` 用于排序同一标准的多次证据，不默认采用统一的 24 小时过期规则。只有契约明确要求绝对时效时才向审计脚本传入 `--max-age-hours`；稳定性、性能或线上观察应使用契约规定的 `observation_window`。

契约、相关产物或外部状态发生变化后，受影响的旧证据立即失效，即使时间戳很新。修复审查发现同样属于相关变更：修复后必须在新 revision 上重新运行受影响的 DOD 验证和最终审查，不能沿用修复前的通过结果。

示例：

- 行为：复现原始工作流或运行端到端测试；
- 兼容性：运行相关回归测试套件；
- 可构建性：运行实际构建，而不只是 lint；
- 视觉质量：在要求的视口中渲染并检查；
- 调研：引用检索到的来源，并区分推断；
- 文档质量：验证必需章节和读者任务；
- 外部操作：验证最终远程状态。

## 最终交叉审核

最终 Goal 必须对同一冻结产物 revision 运行 OCR 和一个独立 subagent 审核。两者使用相同的中立 Review Packet，分别检查契约覆盖、回归风险和未验证假设；独立 subagent 在给出结论前不读取 OCR 结论。主 Agent 负责核验、合并和裁决发现，不能按票数判断正确性。

OCR 适用于有可审查 Git diff 的代码、配置或脚本变更。使用完整 commit SHA 或固定 range，并保存 `--format json --audience agent` 输出。若目标没有适用的代码 diff，必须将 OCR 记录为 `not_applicable`，说明目标类型、检查过的产物和不适用原因，再由独立 subagent 与对应领域验证覆盖；不得伪造 OCR 已通过。临时不可用、凭据缺失、超时或运行失败不属于 `not_applicable`。

解析 OCR 结果时必须同时检查：

- 进程退出状态和 JSON 是否完整可解析；
- `status`，区分 `success`、`completed_with_warnings`、`completed_with_errors` 和 `skipped`；
- `warnings` 中的逐文件失败和其他非致命错误；
- `summary.files_reviewed` 与预览及预期变更文件清单是否一致，识别过滤、删除、二进制、超大 diff 或其他未审查项；
- `comments` 中每项发现的路径、行号、严重度、证据和处置状态。

`success` 只表示运行成功，不等于审查通过。`completed_with_errors` 或 `skipped` 不能满足门禁；`completed_with_warnings` 必须逐项解决覆盖缺口。所有实质评论都必须记录为已修复、以证据驳回或经授权接受风险。修复后旧 OCR、subagent 和受影响验证证据均失效，必须基于新 revision 复审。

### Capability Protocol v1 独立性

所有 Goal Evidence 文档必须声明 `"capability_protocol": "goal-capability/v1"` 并附带
ParticipantLedger。formation owner 以及每位 researcher、architect、design lead、design synthesizer、
implementer、direction reviewer、里程碑 reviewer 和 final reviewer 都记录 participant ID、lineage、契约
revision、至少一个 artifact 引用、是否继承历史及起止时间。final reviewer 的审查结果必须
显式声明冻结产物的 `artifact_revision`，不得借用证据文档顶层 revision 补缺。

异常方向 reviewer 必须 `history_inherited=false`，且 lineage 不得与形成、调研、设计、
实现或其他参与者重叠。最终独立 reviewer 必须在冻结最终 revision 后新建、
`history_inherited=false`，且 lineage
不得与任何调研、设计、实现或早期审查参与者重叠。无法取得稳定 session ID 时可使用
canonical task path；lineage 必须是无首尾空白、重复分隔符、尾随分隔符或 `.`/`..` 段的
绝对任务路径。最终 reviewer 的 `started_at` 必须严格晚于当前 revision 最后一项 DOD
证据的 `checked_at`。仍无法证明独立或时间顺序时标记 unverified，不能完成。

协议 v1 的每项 DOD 证据必须声明 `evidence_kind`。`research_report`、`final_design`、
`goal_brief`、`goal_compilation`、`goal_package`、`goal_prompt`、`user_decision_record` 和
`integration_record` 只属于过程输入，不能满足行为 DOD。若 Goal
交付物本身是研究报告或设计稿，使用独立 reviewer 产生的 `artifact_acceptance` 证据验证
内容完整性、来源质量和一致性，producer 自评仍不算证据。

## 最终门禁

完成前：

1. 重新阅读契约并逐项列出所有必需标准。
2. 固定并记录当前 `contract_revision`、相关产物 revision 和必要的观察窗口。
3. 将因契约、产物、外部状态或修复变更而失效的证据作废。
4. 补充运行缺失或已失效的验证。
5. 对最终 revision 完成 OCR 与独立 subagent 交叉审核，或合法记录 OCR `not_applicable`。
6. 解析并处置所有审核状态、warnings 和 comments。
7. 确认每项必需标准的状态均为 `pass`。
8. 确认没有遗留的必需工作、审批或外部操作。
9. 若用户需要批准、合并、部署或接管高风险、跨系统或陌生变更，完成必要的理解交接并取得契约要求的确认。
10. 报告重要偏差和剩余不确定性。
11. 之后才能将原生或文件持久化目标更新为 complete。

若无法运行某项检查，将该标准标记为 `unverified` 并说明原因。除非契约明确允许替代审查方法且该方法通过，否则不要声明目标完成。

## 理解与接管

当最终批准者需要操作、维护或承担高风险结果时，提供足以接管的变更说明：

- 用户可见行为和关键执行路径；
- 重要设计决定、被放弃的方案和实施偏差；
- 失败模式、监控信号、回退方式和剩余风险；
- 需要批准者继续承担或明确接受的未知项。

根据风险选择最低成本的理解确认：让用户确认关键决定、复述操作与回退路径，或在复杂且后果重大的情况下使用简短测验。只有契约、组织流程或待执行动作要求批准时，理解确认才是完成门禁；其他情况下它属于交接质量证据，不要强制所有 Goal 进行测验。

## 证据 JSON

```json
{
  "goal_id": "passkey-sign-in",
  "contract_revision": "sha256:contract-content-hash",
  "capability_protocol": "goal-capability/v1",
  "revision": "git:abc123",
  "evidence": [
    {
      "criterion_id": "DOD-1",
      "evidence_kind": "implementation_evidence",
      "status": "pass",
      "checked_at": "2026-07-15T08:00:00Z",
      "contract_revision": "sha256:contract-content-hash",
      "artifact_revision": "git:abc123",
      "observation_window": null,
      "source": "npm test -- passkey.e2e.ts",
      "summary": "1 项测试通过；退出码为 0"
    }
  ],
  "final_review": {
    "ocr": {
      "verdict": "pass",
      "status": "success",
      "exit_code": 0,
      "artifact_revision": "git:abc123",
      "source": "ocr review --commit abc123 --format json --audience agent",
      "warnings": [],
      "comments": [],
      "unresolved_findings": []
    },
    "subagent": {
      "verdict": "pass",
      "participant_id": "final-review-task",
      "lineage": "/root/final-review-task",
      "artifact_revision": "git:abc123",
      "source": "independent review packet result",
      "summary": "未发现未处置的实质问题",
      "unresolved_findings": []
    }
  },
  "participant_ledger": [
    {
      "participant_id": "final-review-task",
      "role": "final_reviewer",
      "lineage": "/root/final-review-task",
      "contract_revision": "sha256:contract-content-hash",
      "artifact_ids": ["completion-review-1"],
      "history_inherited": false,
      "started_at": "2026-07-15T08:10:00Z",
      "ended_at": "2026-07-15T08:15:00Z"
    }
  ],
  "findings": [],
  "pending_approvals": [],
  "remaining_required_work": []
}
```

允许的证据状态为 `pass`、`fail`、`blocked` 和 `unverified`。只有 `pass` 满足完成门禁。

OCR 不适用时，将 `final_review.ocr.verdict` 写为 `not_applicable`，将原始 `status` 写为 `skipped`，并提供非空 `reason` 和检查来源；不要在证据项中把它写成 `pass`。每项 DOD 证据必须包含匹配的 `contract_revision`，并按证据类型包含 `artifact_revision`、非空 `observation_window`，或两者兼有。

对于使用 Git 的工作区，基于当前修订版本运行确定性审计：

```bash
python3 scripts/audit_completion.py goal.json evidence.json \
  --expected-revision "git:$(git rev-parse HEAD)"
```

对于非 Git 产物，改用稳定的内容哈希或不可变版本 ID。
