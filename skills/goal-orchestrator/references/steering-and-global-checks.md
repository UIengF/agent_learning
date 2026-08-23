# 运行期方向检查与异常审查

## 目的

正常执行以结构化 Prompt 持续提醒主 Agent 站在 Goal 全局和第一性原则上决策；只有
出现异常或高影响方向事件时，才调用独立 subagent。Prompt 不能代替证据，subagent
不能代替用户决定或 Goal owner。

## 每个重要决策的轻量检查

主 Agent 在里程碑或其他重要决策前记录最小的 `steering_check`：

```text
Goal/DOD: 当前决策推进哪个结果和标准？
Evidence delta: 本轮新增了什么可观察证据？
Assumptions: 当前路径依赖什么，什么可以证伪它？
Global effect: 这是 Goal 级进展，还是仅改善局部指标？
Tradeoffs: 成本、复杂度、耦合、性能、安全和可逆性如何变化？
Alternatives: 是否存在尚未比较的实质可行路径？
Next action: 下一步最小、可验证、可回退的动作是什么？
```

检查结论必须引用当前工作区、测试、运行行为、研究或设计 artifact，不得只写直觉。

## 异常和高影响事件

出现以下任一条件时，主 Agent 应暂停当前方向并调用独立方向 reviewer：

- 同一根因重复失败，或有限重试后仍无新证据；
- 连续执行循环没有新的 DOD 证据；
- 里程碑完成但用户可见结果没有改善；
- 关键假设被新证据推翻；
- workaround、特殊分支、复杂度或风险持续增加；
- 发生回归、不可解释行为或证据互相矛盾；
- 架构、迁移、安全、鉴权、并发、权限或不可逆操作；
- 主 Agent 无法给出从当前路径到最终结果的可信路线；
- reviewer 对方向有未解决的实质分歧。

## reviewer 输出

方向 reviewer 使用固定且中立的 Review Packet，至少判断：

```text
global_progress: yes | no | uncertain
local_only_improvement: true | false | uncertain
dead_end_risk: low | medium | high
invalidated_assumptions: []
missing_evidence: []
alternative_paths: []
recommended_action: continue | replan | research | design | rollback | ask_user | blocked
user_decision_required: true | false
evidence: []
```

reviewer 只能给出方向意见。Goal owner 根据证据决定继续、调整、重新调研、重新设计、
回退、等待用户或报告外部阻塞。

方向 reviewer 必须是空历史、`history_inherited=false` 的独立 lineage，且不得与 formation、
research、design、implementation 或其他 reviewer lineage 重叠。相同 Agent 的自我复核不算
独立方向审查。

## 决策分级

- `local` 且可逆：主 Agent 自检即可；
- `milestone` 或跨模块：按风险选择独立 reviewer；
- `goal_level`、高风险或不可逆：必须独立 reviewer，必要时请求用户；
- `contract_level`：必须由用户决定，Agent 共识不能替代。

## 状态语义

方向检查是 `active` 内部子阶段，不新增顶层 Goal 状态。其结果映射为：

```text
continue/replan -> active
research/design -> active 内部重新路由
rollback -> active，并作废受影响证据
ask_user -> waiting_for_decision
blocked -> blocked，仅限真实外部依赖
```

最终 `verifying` 仍必须对冻结产物执行独立完成审计；方向 reviewer 不能替代最终审查。
