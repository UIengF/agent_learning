# 里程碑执行与审查

## 责任模型

主 Agent 是运行时 Goal、当前里程碑和集成结果的唯一责任人，负责全局方向检查、风险分级、权限检查、任务拆分、共享接口、实际差异检查、统一验证、发现裁决和状态更新。

subagent 可以执行独立调查、隔离的实现切片、测试、文档或独立审查。subagent 的“完成”只表示其分配任务已交付，不能直接关闭里程碑或 Goal。

通过能力协议返回的 ResearchReport 和 FinalDesign 同样只是输入。主 Agent 必须按
[capability-routing-and-handoffs.md](capability-routing-and-handoffs.md) 检查 revision、
snapshot、coverage、权限和适用性，并记录 IntegrationRecord。FinalDesign 缺少硬约束时，
退回同一 design lead 做一次窄修正或重新运行 council；主 Agent不得查看候选后自行拼出
未经裁决的第三个设计。

OCR 是结构化代码 reviewer，不是方向或完成判官。测试、构建、运行行为和外部最终状态是直接证据。

## 决定是否并行

主 Agent 根据依赖关系、写入边界和合并成本自主决定并行数。

只在以下条件同时成立时并行实现：

- 工作之间没有先后依赖；
- 文件、模块或外部状态的写入范围互不重叠；
- 接口和验收标准已经明确；
- 合并成本低于并行收益；
- 每项工作都有一个明确 owner。

紧耦合、小型或共享核心状态的工作由主 Agent 直接执行。默认在同一工作区分配互斥写入范围；只有需要隔离重叠实验、分支或仓库状态时才创建独立 worktree。不得让多个 writer 同时修改相同文件、迁移状态、锁文件或目标记录。

主 Agent 应保留关键路径和公共接口的直接理解，不能退化为只转发 subagent 报告的调度器。

## 风险评估

在规划里程碑、开始执行、解决关键未知项、范围改变和验证失败后重新评估风险。

主 Agent 根据事实写出风险理由，不使用乘法分数代替判断。至少说明：可能失败的行为、受影响对象、当前未知项及证据、可逆性和回退成本、直接验证能覆盖与不能覆盖的部分。

| 等级 | 证据特征 | 默认门禁 |
|---|---|---|
| 低 | 影响局部、易回退、行为已有直接证据且精确验证覆盖主要失败路径 | 全局 steering prompt 加直接验证；不默认调用 reviewer |
| 中 | 用户可见或跨模块、存在部分关键假设、回退有成本或验证不能覆盖全部关系 | 全局 steering prompt 加直接验证；出现异常触发条件时调用独立 reviewer |
| 高 | 影响广泛、不可逆、外部副作用明显、关键行为未知或缺少可靠直接验证 | 先解决关键未知项；直接验证，加独立方向 reviewer；代码风险需要时再加 OCR |

涉及生产、外部服务器、真实用户数据、删除或不可逆操作、安全、鉴权、权限、迁移、公共 API、共享基础设施、并发或一致性时，至少按高风险处理。需要批准的动作在批准前不得执行。风险定级不能降低宿主政策或用户权限边界。

当证据同时符合多个等级，或主 Agent 与 reviewer 有分歧时，先按较高等级执行，直到新的直接证据支持降级。记录理由和证据变化，不记录没有解释力的数值。

## 里程碑执行循环

1. 固定当前里程碑的 DOD 覆盖、owner、写入范围和验收标准。
2. 核对真实工作区和依赖，确认没有其他 writer 冲突。
3. 运行全局 `steering_check`，获取影响下一决策的最小证据并重评风险。
4. 执行最小可回退增量。
5. 运行针对性构建、测试或行为检查。
6. 检查异常和高影响触发条件；仅在触发时发起独立方向或代码审查。
7. 主 Agent 检查原始产物和发现，修复已确认问题并重跑受影响验证。
8. 记录结果、偏差、剩余风险和下一动作。

里程碑完成只说明增量成立，不代表整个 Goal 完成。

## 异常与高影响方向审查

普通里程碑依靠结构化全局 Prompt 和直接证据，不为形式完整固定创建 subagent。方向审查
触发条件、输入和输出只按 [steering-and-global-checks.md](steering-and-global-checks.md)
执行。方向 reviewer 检查 Goal 对齐、局部最优、死胡同风险、失效假设和替代路径，不承担
实现工作，也不能修改 Goal 状态。

里程碑出现异常后的修复必须由 reviewer 对修复后的同一固定 revision 重新检查相关方向或
代码风险。普通测试失败先诊断，不把一次失败自动升级为异常审查。

## 固定审查对象

代码审查优先使用不可漂移的完整 revision。记录：

```text
base_sha
head_sha
merge_base_sha
head_tree_sha
patch_sha256
dirty_state
ocr_version
model
rule_hash
review_packet_hash
```

完成门禁不要依赖运行期间仍可变化的工作区 diff。优先审查 checkpoint commit 或不可变 patch；不能提交时，计算 tree/patch 哈希并禁止并发写入。`--from/--to` 会审查 merge base 到 head 的差异，因此必须记录 `merge_base_sha`。

## 中立 Review Packet

需要并行 OCR 与独立 subagent 时，两者使用同一份事实性 Packet，在各自完成前不得看到
对方结论。方向 reviewer 也使用中立 Packet。Packet 包含：

- Goal objective、相关 DOD、约束和 non-goals；
- 固定的 base/head/merge-base revision 或不可变产物版本；
- 变更文件、diffstat、接口和数据流影响；
- 已运行验证及原始结果；
- 已确认外部约束、开放未知项和允许副作用；
- 当前路径依赖的关键假设、替代路径和本轮 DOD 证据增量；
- 要求 reviewer 输出触发条件、可观察影响、证据位置、严重度和置信度。

不要包含作者自评、预期“通过”、其他 reviewer 结论、预设严重度或为实现辩护的文字。

代码 Goal 的推荐命令：

```bash
ocr review --repo <repo> \
  --from <base_sha> --to <head_sha> \
  --background-file <review-packet.md> \
  --format json --audience agent
```

使用 `--commit` 时仍显式提供 Packet，避免让 commit message 成为唯一业务背景。

## OCR 结果处理

退出码 `0` 只表示 OCR 运行完成。必须解析：

- `status`：区分 `success`、`completed_with_warnings`、`completed_with_errors` 和 `skipped`；
- `warnings`：识别失败、超大文件、跳过或覆盖缺口；
- `comments`：逐项核验，而不是按数量判断通过。

对于包含受支持代码变更的里程碑，`skipped`、未解释的 warning 或未覆盖文件不能作为通过。最终 Goal 的 OCR 适用性、`not_applicable` 和完成判定只按 [completion-audit.md](completion-audit.md) 执行。

## 合并与裁决发现

先保存 reviewer 原始输出和来源，再规范化：

```text
finding_id
reviewer_or_session
artifact_revision
path_or_component
failure_condition
observable_impact
evidence
severity
confidence
status
```

只有根因、触发条件和修复方向均相同时才去重。同一位置的不同风险分别保留。严重度按可达性、影响和发生概率裁定，不按多数投票或机械取最高值。

分歧处理顺序：检查固定产物和规范，构造最小复现或定向测试，再请求第三个独立 reviewer。只有涉及产品取舍、授权或接受风险时才请求用户决定。

发现状态使用 `confirmed`、`rejected`、`needs_evidence` 和 `accepted_risk`。`accepted_risk` 必须有用户明确授权；另一个 reviewer 没有发现问题不能否定已有 finding。

## 修复后重新审查

任何相关代码、配置、schema、依赖、产物或 DOD 变化都会使受影响证据失效。修复后至少执行：

- 原 finding 的定向验证；
- 修改区域的新鲜审查；
- 受影响回归测试。

使用 `goal-capability/v1` 时，参与过 research、architecture、synthesis、implementation
或 milestone review 的 participant lineage 不得充当最终独立 reviewer。研究、设计、
用户决定和集成记录不能作为行为型 DOD 的通过证据。

OCR `--resume` 只用于同一固定 revision 的中断恢复。revision、模型、规则或跨文件不变量变化后启动新审查；最终交叉审核按 [completion-audit.md](completion-audit.md) 执行。
