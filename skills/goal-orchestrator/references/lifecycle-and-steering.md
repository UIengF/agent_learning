# 生命周期与引导

## 状态模型

本状态模型只属于用户确认 GoalPackage 后的运行阶段。形成过程由 `goal-prompt` 以
`formation_id` 和 package revision 管理，不伪造 runtime Goal 状态。

```text
draft -> ready -> active -> verifying -> complete

active    -> waiting_for_decision -> active
active    -> paused                -> active
active    -> blocked               -> active
verifying -> waiting_for_decision -> verifying
verifying -> paused                -> verifying
verifying -> blocked               -> verifying
verifying -> active  （验证失败、验证路径变化或相关产物被修改）

任何非终态 --用户取消--> cancelled
任何非终态 --结果被替换--> superseded
```

`resume_to` 是进入等待、暂停或阻塞前的可运行状态，只能是 `active` 或 `verifying`。恢复时先核验工作区、契约和最小冒烟结果，再回到 `resume_to`；若恢复检查发现需要修改产物，则进入 `active`。

状态语义和转换条件如下：

| 状态 | 语义 | 进入或退出条件 |
|---|---|---|
| `draft` | 契约仍在定义，允许存在高影响未知项 | 契约通过完整质量门槛后进入 `ready` |
| `ready` | 契约可激活，但尚未开始长期执行 | 仅在用户明确启动后进入 `active` |
| `active` | 正在推进当前里程碑 | 全部必需实现增量完成且验证路径就绪后进入 `verifying` |
| `verifying` | 按完成定义执行最终证据审计 | 验证失败或产物发生相关修改时回到 `active`；全部门禁通过后进入 `complete` |
| `waiting_for_decision` | 缺少会实质改变结果、权限、范围或外部行为的用户决策 | 收到决策并更新受影响的契约或计划后回到 `resume_to` |
| `paused` | 用户或宿主主动暂停，但未形成真实阻塞 | 用户或宿主恢复且恢复检查通过后回到 `resume_to` |
| `blocked` | 存在当前权限和可用手段无法解除的外部依赖 | 外部条件确实改变并通过恢复检查后回到 `resume_to` |
| `complete` | 所有完成标准均有新鲜通过证据，且无必需工作剩余 | 终态；新增范围创建新目标 |
| `cancelled` | 用户明确停止目标，不再要求达成原结果 | 终态，不得宣称完成 |
| `superseded` | 原预期结果被另一目标替换 | 终态，记录替代目标，不得沿用旧目标证据 |

若宿主的原生状态不同，以宿主实际支持的状态和控件为准。不得为了匹配上表而伪造宿主不支持的状态。宿主不支持 `draft` 或 `ready` 时，在创建原生 Goal 前于会话中完成这两个阶段；宿主不支持 `verifying`、`waiting_for_decision`、`paused` 或 `blocked` 时，保持宿主真实状态并在计划或进度记录中表达阶段和 `resume_to`。宿主不支持 `cancelled` 或 `superseded` 时，只记录终止语义并停止推进，不得假写为 `complete`、`blocked` 或其他受支持状态。

## 原生 Goal 模式

宿主提供持久化 Goal 控件时，优先使用原生模式。

- 创建前先核验用户确认的 GoalPackage；缺失形成材料时返回 `goal-prompt`，不在运行时
  临时编造目标或初始方案。
- 在原生 objective 中保存预期结果、约束和完成标准。
- 原生 Goal 是唯一状态真源；以原生状态及暂停、恢复、编辑控件为准，文件不得镜像、覆盖或反向推断原生状态。
- 在同一任务或对话中处理引导和进度请求。
- 将相互独立的并行目标放在不同任务和可写工作区中。
- 在新证据覆盖全部必需标准前，不要标记为 complete。
- 严格遵循宿主的阻塞语义；困难或不确定本身不构成阻塞。

启动目标不会授予额外权限。沙箱、审批、外部写入、破坏性操作、购买和范围扩展的边界保持不变。

## 文件持久化模式

原生状态不可用，或跨会话任务需要持久化产物时，使用文件持久化模式。将状态保存在 `.goals/<goal-id>/` 下，并以 `goal.json` 为权威来源。

不要维护与原生 Goal 状态冲突的文件状态。在原生模式下，文件仅用于辅助证据和工作记忆；即使文件中存在历史 `status`，也必须忽略它，不能把它同步回原生 Goal。

## 引导分类

更新状态前，先对用户的新输入分类：

| 变更 | 更新内容 |
|---|---|
| 新事实或产物 | 发现，必要时更新计划 |
| 实现路径变化 | 计划 |
| 新约束或权限 | 契约，并记录变更说明 |
| 完成标准变化 | 契约和证据矩阵 |
| 预期结果变化 | 将原目标终止为 `superseded`，并定义替代目标 |
| 状态询问 | 先报告；除非用户要求暂停，否则继续 |

以最新的明确指令为准。契约级变更必须生成新的 `contract_revision` 并记录原因。保留仍有效的既有证据；受契约或实现变更影响的证据必须作废，且新证据应绑定当前契约修订和相关产物修订。

## 等待、暂停与阻塞

- 需要用户作出实质性选择时，使用 `waiting_for_decision`。
- 用户或宿主暂停工作但未认定陷入僵局时，使用 `paused`。
- 仅当宿主规则下确有未解决依赖时，使用 `blocked`。
- 不要仅因任务困难、缓慢或适合进一步澄清，就称其为阻塞。
- 进入上述状态前记录 `resume_to`、已做尝试、相关证据，以及恢复所需的确切决策或外部变化。
- 恢复时先检查原生 Goal 状态、当前契约修订、工作区和版本控制状态，并运行必要的最小冒烟检查；条件未满足时保持原状态。
- 从 `verifying` 恢复后，只要需要修改相关产物或验证路径，就进入 `active` 并作废受影响证据，不直接返回 `verifying`。

## 增量推进

每次选择一个连贯的里程碑。结束工作会话时应确保：

- 环境整洁且易于理解；
- 已运行当前测试或冒烟检查；
- 已记录重要决策和偏差；
- 已明确下一个里程碑；
- 不存在未明确记录的隐藏半成品变更。

## 调研与设计子阶段

运行期 Research、design 和 `steering_review` 是 `active` 或 `verifying` 内部的子阶段，
不新增或伪造原生 Goal 状态。初始 research/design 属于 `goal-prompt` 的 formation 阶段；
运行期只在新事实、异常或高影响方向事件要求纠偏时选择 neither/research/design/both。

每份 delegated 产物绑定 contract revision、输入 fingerprint 和实际产物 snapshot。
若变化触及其依赖字段，将产物标记为 stale 并停止使用；若变化无关，记录 applicability
检查。设计暴露 `missing_evidence` 时回到调研，暴露 `user_decision` 时进入
`waiting_for_decision`，普通 implementation detail 进入计划。

方向审查输出 `continue/replan/research/design/rollback` 时保持 `active`，输出 `ask_user`
时进入 `waiting_for_decision`；只有真实外部依赖才能进入 `blocked`。契约级用户决定后可
请求 `goal-prompt` 重编 GoalPackageRevision，但 runtime Goal 的状态 owner 不转移。
