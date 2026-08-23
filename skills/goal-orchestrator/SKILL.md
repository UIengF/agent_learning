---
name: goal-orchestrator
description: "运行和治理已经形成的长期 Goal。用户明确要求创建、启动、追踪、继续、暂停或恢复 Goal，查看长期目标状态，要求执行期 subagent 编排、方向纠偏、里程碑审查或完成审计时使用；若尚无用户确认的 GoalPackage，先使用 goal-prompt 完成背景调研、目标/方案/初始里程碑设计。普通一次性任务、仅生成 /goal 文本或仅做初始方案时不要触发。"
---

# 目标编排器（Goal Orchestrator）

作为运行阶段唯一 owner，导入用户确认的 GoalPackage，维护 Goal 状态和可变计划，持续用
全局证据检查执行方向，在异常或高影响事件时编排独立 subagent，并在完成前执行独立审计。
中文请求默认使用中文回应。

## 硬边界

`goal-prompt` 负责形成阶段：背景调研、用户澄清、初始方案、初始里程碑、GoalPackage 和
`/goal` 文本。本 Skill 不复制这套形成流程。

本 Skill 从以下时点开始负责：

```text
用户确认 GoalPackage
  -> goal-orchestrator 核验并导入
  -> 创建或激活 runtime Goal
  -> 执行、纠偏、审查、恢复和完成审计
```

如果没有确认的 GoalPackage，先使用 `goal-prompt`。若用户只要求生成 `/goal`，输出后停止，
不要创建 Goal。完整交接规则见 [references/goal-package.md](references/goal-package.md)。

## 唯一状态源

1. 宿主提供原生 Goal 时，以原生 Goal 状态为唯一真源。
2. 原生能力不可用，或确需跨会话文件时，使用 `.goals/<goal-id>/goal.json` 作为唯一状态源。
3. GoalPackage 和 `/goal` 是只读输入或导出物，不能反向覆盖运行时状态。
4. 用户明确要求创建但未启动时保持 `ready`；明确要求创建并启动时可在 package 确认后
   直接进入 `active`，不重复询问。

状态、暂停、恢复和阻塞只按
[references/lifecycle-and-steering.md](references/lifecycle-and-steering.md) 处理。

## 导入 GoalPackage

导入前检查：

- package 的 `formation_id`、revision、source refs 和用户确认有效；
- handoff intent 与用户原始请求一致；
- outcome、DOD、范围、非目标、权限和停止规则完整且无冲突；
- 初始方案和里程碑有证据来源，未把假设写成事实；
- 重要未知项已解决，或有合规的用户风险接受记录；
- 不存在尚未决定且会改变结果、范围、权限、DOD 或外部行为的问题。

GoalPackage 是候选输入。runtime owner 核验后创建正式 Goal ID 和 contract revision，记录
来源 package，但不得暗中修改用户确认的结果。如果包不完整，返回 formation 阶段修正；
不要在运行时悄悄补造目标和验收标准。

## 默认治理

用户没有另行说明时：

- 本机非删除操作可直接执行；删除或等价破坏性操作，以及连接或操作用户/组织管理的
  外部服务器，必须先获得批准；
- 不主动设置 token、时间、重试或并发预算；用户明确给出时原样保留；
- 主 Agent 是 Goal、当前里程碑、权限、计划、集成结果和完成声明的唯一 owner；
- 运行期研究和设计保持只读，subagent 不能直接写 Goal 状态；
- 最终完成必须有当前 revision 的 DOD 证据、OCR（适用时）和全新 lineage 的独立 reviewer。

宿主沙箱、审批策略和用户授权始终优先；启动长期 Goal 不扩大权限。

## 运行期能力路由

初始研究和设计已经由 GoalPackage 形成阶段完成。运行期只在新证据要求纠偏时调用：

- `research`：至少两个新的独立高影响事实问题；
- `design`：事实充分但出现多个会实质影响接口、安全、迁移、维护或 DOD 证据的路径；
- `both`：先 research，owner 集成事实并处理用户决定，再冻结 DesignBrief 后 design；
- `neither`：问题局部、路径明确、低风险且易回退，由 owner 直接处理。

用户可以要求、禁止或要求委托前确认。运行期委托使用 `delegated_by_goal`，绑定当前 Goal
ID、contract revision、snapshot、permission ceiling 和 participant lineage。完整规则见
[references/capability-routing-and-handoffs.md](references/capability-routing-and-handoffs.md)。

## 建立运行计划

把 GoalPackage 的初始里程碑导入为第一版计划。每个里程碑记录：

- 可观察结果和关联 DOD；
- 依赖、风险和关键假设；
- owner 和互斥写入范围；
- 验证方法、预期结果和产物 revision；
- checkpoint 与回退方式。

目标契约保持稳定，运行计划允许变化。runtime owner 可以重排、拆分、合并或替换里程碑，
调整实现路径、subagent 分工和验证方式，只要不改变结果、范围、权限、DOD 或风险接受。
记录调整理由和证据，不把初始方案当成不可修改的圣旨。

## 全局决策检查

在每个重要决策和里程碑循环前，用
[references/steering-and-global-checks.md](references/steering-and-global-checks.md) 的
`steering_check` 从全局和第一性原则检查：

1. 当前动作推进哪个 Goal 结果和 DOD；
2. 本轮产生什么新的可观察证据；
3. 当前方案依赖什么假设，如何证伪；
4. 这是 Goal 级进展还是仅改善局部指标；
5. 成本、复杂度、耦合、风险和可逆性如何变化；
6. 是否存在尚未比较的实质替代路径；
7. 下一步最小、可验证、可回退的动作是什么。

正常、局部且可逆的决策由主 Agent按实际证据判断，不固定调用 subagent。Prompt 只提供
思考框架，不能成为证据或完成声明。

## 里程碑执行循环

同一时间默认推进一个里程碑；内部工作只有在无先后依赖、接口明确、写入范围互斥且合并
成本较低时才并行。每次循环：

1. 核对 Goal、当前里程碑、真实工作区、依赖和其他 writer；
2. 执行全局 `steering_check` 并重评风险；
3. 检查权限，决定主 Agent 直接执行或委托互不影响的实现切片；
4. 执行一个最小可回退增量；
5. 运行针对性构建、测试、行为或外部状态验证；
6. 核验原始结果，记录证据增量、失败、假设变化和计划偏差；
7. 判断继续、调整计划、触发异常方向审查、请求用户决定或进入 `verifying`。

subagent 的“完成”只表示其切片已交付。主 Agent必须检查实际差异和原始验证，完成一个
里程碑也不代表整个 Goal 完成。详细实现与审查规则见
[references/milestone-execution-and-review.md](references/milestone-execution-and-review.md)。

## 异常和高影响方向审查

普通执行不固定调用 reviewer。出现以下事件时，必须冻结当前决策材料并创建独立方向
reviewer：同一根因重复失败、连续循环无 DOD 证据、里程碑完成却无用户可见改善、关键假设
被推翻、workaround/复杂度持续增加、回归或证据矛盾、架构/迁移/安全/权限/不可逆决策，
或主 Agent 无法说明从当前路径到最终结果的可信路线。

reviewer 只读检查 `global_progress`、`local_only_improvement`、`dead_end_risk`、失效假设、
缺失证据、替代路径和推荐动作。Goal owner 核验后选择：

```text
continue | replan | research | design | rollback | ask_user | blocked
```

方向审查是 `active` 内部子阶段，不新增顶层状态。`ask_user` 进入
`waiting_for_decision`；只有真实外部依赖才进入 `blocked`。方向 reviewer 不能替代最终独立
审查，也不能自行扩大范围、权限或接受风险。

## 运行期调整与重新形成

将变化分类：

- 新事实：更新 findings，必要时重新 research/design 和计划；
- 实现路径、里程碑或验证方式：只更新计划；
- 结果、范围、非目标、权限、DOD 或风险接受：进入 `waiting_for_decision`；
- 用户确认契约级变化后：可请求 `goal-prompt` 生成 GoalPackageRevision，owner 核验并创建
  新 contract revision，作废受影响证据；
- 预期结果被替换：原 Goal 标记 `superseded`，形成并创建新 Goal；
- 暂停、恢复或状态询问：先核对宿主和真实工作区，再按 lifecycle 处理。

运行期重新调用 `goal-prompt` 只是候选包重编译，Goal 状态所有权不转移。

## 完成审计

进入 `verifying` 后：

1. 为每项 DOD 获取绑定当前 contract 和最终产物 revision 的新鲜证据；
2. 固定审查对象和中立 Review Packet；
3. 运行 OCR（适用时）和冻结 revision 后创建的全新独立 subagent reviewer；
4. 主 Agent核验、合并和裁决所有 warnings/comments/findings；
5. 修复会使相关证据失效，必须重新验证和复审；
6. 只有全部 DOD 为 pass、无未处置发现、待批准动作或剩余必需工作时才标记 complete。

不要用计划状态、代码差异、耗时、里程碑完成或 agent 报告推断完成。完整门禁只按
[references/completion-audit.md](references/completion-audit.md) 执行。

## 持久化与恢复

文件回退模式默认从确认的 GoalPackage 确定性导入：

```bash
python3 scripts/import_goal_package.py <goal-package.json> \
  --root <workspace> --goal-id <slug>
```

它创建 `.goals/<slug>/goal.json`、`plan.md`、`findings.md`、`progress.md` 和
`evidence.json`。恢复时先检查真实工作区和版本控制状态，再读取契约、当前里程碑、近期
进展和相关发现，运行最小冒烟检查后从最小有效增量继续。详见
[references/persistence.md](references/persistence.md)。

需要把已通过 Ready 门禁的运行时契约交给外部执行器时，可使用：

```bash
python3 scripts/render_goal_prompt.py .goals/<slug>/goal.json --format text
```

导出文本是只读快照，不能反向覆盖 Goal。

## 交接

先报告真实状态，再说明已完成结果和产物、新鲜验证、独立审查、重要偏差、剩余不确定性、
待批准事项和下一项安全动作。验证缺失、审查失败或证据失效时，不使用暗示成功的措辞。

## 资源索引

- GoalPackage 交接：[references/goal-package.md](references/goal-package.md)
- 生命周期：[references/lifecycle-and-steering.md](references/lifecycle-and-steering.md)
- 运行期方向检查：[references/steering-and-global-checks.md](references/steering-and-global-checks.md)
- 能力协议：[references/capability-routing-and-handoffs.md](references/capability-routing-and-handoffs.md)
- 目标契约：[references/goal-contract.md](references/goal-contract.md)
- 运行期未知项：[references/unknown-discovery.md](references/unknown-discovery.md)
- 里程碑执行与审查：[references/milestone-execution-and-review.md](references/milestone-execution-and-review.md)
- 持久化与恢复：[references/persistence.md](references/persistence.md)
- 完成审计：[references/completion-audit.md](references/completion-audit.md)
