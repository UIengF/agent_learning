# 能力路由与交接协议

## 目的

`goal-prompt` 是 Goal 形成阶段的 owner；用户确认 GoalPackage 后，`goal-orchestrator`
成为运行阶段 Goal、契约、计划、权限、用户决定和完成声明的唯一 owner。所有权只发生
一次明确交接，不同时存在两个状态 owner。并行调研与并行方案设计是两阶段均可按需使用的
只读能力，不能直接写入 Goal 状态或把自己的产出声明为完成证据。

本文件是 `goal-capability/v1` 公共字段、路由和集成语义的唯一规范。
`parallel-research` 与 `parallel-design-council` 只定义各自 payload 和角色拓扑。
`goal-prompt` 定义 GoalPackage、GoalPackageRevision 和 Prompt payload；它只拥有形成过程，
不拥有运行时 Goal 状态。完整边界见 [goal-package.md](goal-package.md)。

## 编排策略

用户未指定时采用：

```yaml
orchestration_policy:
  mode: auto
  allow: [research, design]
  require: []
  ask_before_delegate: false
```

- `mode` 只能是 `auto` 或 `user_selected`。
- `allow` 和 `require` 的成员只能是 `research`、`design`，且 `require` 必须是
  `allow` 的子集。
- `allow: []` 表示禁用专门并行能力。
- `ask_before_delegate: true` 表示每次委托前都要取得用户确认。
- 用户明确选择 `neither`、`research`、`design` 或 `both` 时优先遵循；但不能绕过
  权限、安全、并发、角色可用性或宿主限制。

策略只控制形成或实现方式时属于计划信息。若策略改变范围、权限、结果或 DOD，必须由
用户决定；运行期据此修订 Goal 契约。形成和执行中出现新证据时都允许重新路由并记录原因。

`goal-prompt` 是形成阶段控制器，不是第五种 research/design 路由。它按本策略选择是否
调研或设计，最终生成 GoalPackage。运行期由 `goal-orchestrator` 使用相同路由做窄范围纠偏。

## 四种路由

模糊需求本身不构成并行理由。先进行最低成本的真实情况检查，再选择：

| 路由 | 适用条件 |
|---|---|
| `neither` | 事实充分、路径明确、局部、低风险且易回退 |
| `research` | 至少两个独立且高影响的事实问题；查清后技术路径基本唯一 |
| `design` | 事实基线充分，但至少有两个会实质影响接口、安全、迁移、维护成本或 DOD 的可行路径 |
| `both` | 关键事实未知，且预计查清后仍存在实质设计分歧 |

`both` 必须按 `research -> 当前阶段 owner 集成/用户决定 -> 冻结 DesignBrief -> design`
串行运行。不要同时运行调研与设计，也不要为形式上的多样性调用昂贵能力。

只有一个已知位置的问题时由当前阶段 owner 直接核验。只有单一路径、明确项目惯例或低风险
局部改动时直接形成计划。用户明确要求某一路由时，在能力和权限允许的前提下执行。

## 调用模式

- `direct`：用户显式调用专门 Skill，保持其公开入口行为。
- `delegated_by_goal_prompt`：Goal 尚未创建时，formation owner 依据 `formation_id` 和
  package revision 委托只读调研或设计。
- `delegated_by_goal`：Goal owner 依据本协议构造 brief，并直接按专门 Skill 的角色
  拓扑派生 Agent。这不等于启用隐式 Skill 触发，也不依赖运行时支持 Skill-to-Skill 调用。

进入 delegated 模式前，读取实际安装的
[`goal-prompt`](../../goal-prompt/SKILL.md)、
[`parallel-research`](../../parallel-research/SKILL.md) 或
[`parallel-design-council`](../../parallel-design-council/SKILL.md) 及其 payload reference；
本文件只拥有公共 envelope 和治理规则。专门 Skill 或所需自定义角色不可用时按本文件的
降级规则处理，不要根据记忆仿造其拓扑。

两个专门 Skill 的 `allow_implicit_invocation` 必须保持 `false`。只有当前 formation owner
能使用 `delegated_by_goal_prompt`；只有已激活 Goal 的 runtime owner 能使用
`delegated_by_goal`。

## CapabilityEnvelope@1

委托和返回产物都使用 JSON 对象。公共字段如下：

```json
{
  "protocol": "goal-capability/v1",
  "artifact_kind": "research_brief",
  "artifact_id": "artifact-unique-id",
  "request_id": "request-unique-id",
  "invocation_mode": "delegated_by_goal_prompt",
  "goal_id": null,
  "formation_id": "formation-id",
  "contract_revision": "package:2",
  "goal_phase": "formation",
  "input_fingerprint": "sha256:<64 lowercase hex characters>",
  "snapshot_refs": [
    {"locator": "repo-or-resource", "revision_or_hash": "git:abc123"}
  ],
  "permission_ceiling": {
    "read_paths": ["/allowed/path"],
    "write_allowed": false,
    "network_allowed": false
  },
  "participant_ids": ["canonical-task-or-session-id"],
  "created_at": "2026-07-17T08:00:00Z",
  "payload": {}
}
```

允许的 `artifact_kind`：

- `research_brief`、`research_report`；
- `design_brief`、`final_design`；
- `goal_intake`、`goal_brief`、`goal_compile_request`、`goal_compilation`、`goal_package`、`goal_prompt`；
- `user_decision_record`、`integration_record`、`participant_ledger`。

`direct` 模式下 `goal_id`、`formation_id`、`contract_revision` 和 `goal_phase` 可以为 `null`。
`delegated_by_goal_prompt` 要求非空 `formation_id`、package revision，`goal_id: null` 且
`goal_phase: formation`。`delegated_by_goal` 要求非空 `goal_id`、Goal contract revision，
`goal_phase` 为 `active` 或 `verifying`。两种 delegated 模式都要求非空的
fingerprint、snapshot、permission ceiling 和 participants。

`delegated_by_goal_prompt` 只允许 research/design 请求、响应及其 integration/participant
记录；GoalBrief、GoalCompilation、GoalPackage 和 GoalPrompt 由 formation owner 直接产生。
`delegated_by_goal` 不允许 `goal_intake` 或 `goal_brief`，运行期只能在 `active` 且绑定非空
`changed_fields` 和 `user_decision_refs` 后请求 GoalCompilation/GoalPackageRevision，不能
重新夺回形成阶段所有权或创建第二个 Goal。

研究和设计的 `write_allowed` 必须为 `false`。权限上限只能等于或严于当前 formation 或
Goal 权限；返回产物必须回显相同的 `request_id`、formation/Goal binding、revision、
fingerprint 和 snapshot。扩大网络、读取或写入权限必须返回当前 owner，由用户授权并生成
相应 package 或 contract revision。

委托前校验 brief，返回后同时校验结构和 request binding：

```bash
python3 scripts/validate_handoff.py research-brief.json \
  --expected-kind research_brief --expected-contract-revision contract:2

python3 scripts/validate_handoff.py research-report.json \
  --request research-brief.json --expected-kind research_report
```

使用 `--allowed-read-path`（可重复）和 `--allow-network` 将委托权限限制在 Goal 已授权
范围内。校验脚本通过只证明结构和绑定成立，不证明研究事实或设计判断正确。

## 产物分类

| 产物 | 固定 `artifact_class` | 可直接满足行为 DOD |
|---|---|---|
| ResearchReport | `advisory_not_decision_not_completion_evidence` | 否 |
| FinalDesign | `adjudicated_design_not_implementation_evidence` | 否 |
| GoalBrief | `advisory_goal_brief_not_goal_state` | 否 |
| GoalCompilation | `candidate_goal_contract_not_goal_state` | 否 |
| GoalPackage | `approved_goal_package_not_goal_state` | 否 |
| GoalPrompt | `rendered_goal_prompt_not_goal_state` | 否 |
| UserDecisionRecord | `user_authorization_not_completion_evidence` | 否 |
| IntegrationRecord | `integration_metadata_not_completion_evidence` | 否 |
| ImplementationEvidence | `implementation_evidence` | 是，仍须验证 |
| ArtifactAcceptance | `artifact_acceptance` | 是，用于目标产物本身为调研或设计时 |
| CompletionReview | `completion_review` | 只作为独立审核门禁 |

ResearchReport 只能更新事实、推断、矛盾和未知项。FinalDesign 只能成为初始 GoalPackage
或运行期架构和计划的输入。
只有用户的明确决定记录可以授权产品意图、范围、权限、风险接受或 DOD 的实质变化。

`GoalBrief` 是对用户意图、现实基线、范围、DOD 和未决问题的候选澄清；
`GoalCompilation` 是由形成材料编译出的候选 GoalPackage；`GoalPackage` 是用户确认后的
只读交接包；`GoalPrompt` 是面向外部执行器的可复制文本。它们都不能直接写入 native Goal
或文件模式的 `goal.json`，确认后的 package 仍须由 runtime owner 核验和导入。

## 用户决定

遇到以下事项时，Agent 共识不能替代用户：

- 产品意图或目标结果；
- 范围、权限或外部副作用变化；
- 不可逆迁移或高回退成本路径；
- 安全、数据、合规或验收风险接受；
- DOD 或成功标准的实质变化；
- 多个方案体现不同用户偏好，且证据不能判定优劣。

`user_decision_record` payload 至少包含 `decision_id`、`user_statement`、`decision`、
`scope`、`decided_by`、`decided_at`、`applies_to_revision`，以及可选的
`supersedes` / `superseded_by`。需要用户决定时进入 `waiting_for_decision`，不要标为 blocked。

## IntegrationRecord@1

专门能力不得直接修改 GoalPackage 或 Goal。当前阶段 owner 核验返回产物后创建：

```json
{
  "source_artifact_id": "artifact-id",
  "source_fingerprint": "sha256:...",
  "input_revision": "contract:2",
  "disposition": "accepted",
  "mapped_items": [
    {
      "source_item_id": "RF-1",
      "target_kind": "finding",
      "target_ref": "U-1"
    }
  ],
  "reason": "why the result applies",
  "before_revision": "contract:2",
  "after_revision": "contract:2",
  "owner_id": "goal-owner-id"
}
```

`disposition` 只能是 `accepted`、`partially_accepted`、`rejected` 或 `stale`。
目标、DOD、约束、权限、非目标或重要未知项处置发生变化时更新 contract revision；
仅实现路径变化时更新计划。

## ParticipantLedger@1

使用 delegated 协议时，记录每位参与者：

```json
{
  "participant_id": "task-or-session-id",
  "role": "researcher",
  "lineage": "canonical-task-path-or-session-lineage",
  "contract_revision": "contract:2",
  "artifact_ids": ["artifact-id"],
  "history_inherited": false,
  "started_at": "2026-07-17T08:00:00Z",
  "ended_at": "2026-07-17T08:05:00Z"
}
```

角色包括 `formation_owner`、`researcher`、`architect`、`design_lead`、`design_synthesizer`、`implementer`、
`direction_reviewer`、`milestone_reviewer` 和 `final_reviewer`。lineage 必须使用 canonical
绝对任务路径，不得包含首尾空白、重复/尾随分隔符或 `.`/`..` 段。最终 reviewer 必须是
冻结 revision 且当前 DOD 证据记录后创建、不继承历史的新 lineage，且未参与同一 Goal 的
调研、设计、实现或早期审查。

## 新鲜度与重入

每个 brief 明确依赖的 contract fields 和 snapshot。revision 或 snapshot 改变且触及依赖时，
产物立即成为 `stale`；变化不相关时当前阶段 owner 必须记录 applicability check。stale 产物不能
进入计划、实现或完成审计。

- ResearchReport 的 coverage 为 partial/missing 时，相应 unknown 不得标记 resolved。
- FinalDesign 返回 `missing_evidence` 时回到窄调研；返回 `user_decision` 时进入用户门；
  `implementation_detail` 写入计划。
- 相同输入、相同失败最多重试一次；再次尝试必须改变 brief、证据或方法。

## 降级

- researcher 不可用时，direct 模式 fail closed；delegated 模式只能把单个窄查询交回
  当前阶段 owner，本来需要广泛调查的问题保持 unresolved。
- council 的隔离角色拓扑不可用时不得伪装为单 Agent council；保持当前阶段并说明能力缺失。
- 专门能力请求超过 permission ceiling 时拒绝并进入授权门。
- 无法证明最终 reviewer 独立时保持 unverified，不得完成。
