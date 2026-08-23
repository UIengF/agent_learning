# GoalPackage 形成与运行交接

## 责任边界

`goal-prompt` 是形成阶段的 owner。它负责把用户意图、背景事实、初始方案和里程碑
编译为用户可确认的 GoalPackage；它不创建运行时 Goal，也不维护执行状态。

用户确认 GoalPackage 后，`goal-orchestrator` 接管运行时所有权，创建或导入唯一的 native
Goal / `.goals/<goal-id>/goal.json`，并负责执行、调整、审查和完成审计。

```text
formation owner: goal-prompt
  -> GoalPackage candidate
  -> user confirmation
runtime owner: goal-orchestrator
  -> Goal contract and state
  -> execution and steering
```

## GoalPackage 内容

GoalPackage 是只读交接材料，至少包含：

- `objective` 和可观察 `outcome`；
- `definition_of_done` 及每项验证方法；
- 背景事实、证据 locator 和推断；
- `scope`、`exclusions`、约束和权限边界；
- 初始方案、备选方案和被拒绝的理由；
- 初始里程碑、依赖、风险、验证和回退方式；
- 未决问题、停止规则和编排策略；
- `formation_id`、package revision、来源 artifact 和用户决定引用。
- formation owner 及所有被采用 research/design 参与者的 ParticipantLedger。

调研和设计可以作为形成阶段的可选只读输入。GoalPackage 不等于 Goal 状态，不能单独
证明行为型 DOD，也不能反向覆盖已激活的 Goal。

## 用户确认与启动

- 用户只要求生成 `/goal`：`creation_authorized=false`、`start_authorized=false`，输出 Prompt；
- 用户明确要求创建但未要求启动：creation 为 true、start 为 false，导入并保持 `ready`；
- 用户明确要求创建并启动：两个 authorization 均为 true，导入并进入 `active`；
- `goal-prompt` 的确认不授权未列出的写入、外部操作、权限扩大或破坏性动作。

## 运行期重新形成

执行中只改变实现路径、里程碑或验证方式时，由 `goal-orchestrator` 直接更新计划。
若目标结果、范围、权限、DOD 或风险接受发生变化，先进入 `waiting_for_decision`；用户
决定后可以重新调用 `goal-prompt` 生成 `GoalPackageRevision`。编排器核验并接管新的
contract revision，运行时 owner 不发生转移。

## 持久化

`goal-prompt` 不创建或维护 `.goals/<id>/`、`state.md`、`progress.md` 或 `evidence.json`。
若用户要求保存形成材料，可将只读包存放在明确标记的 `.goal-packages/<package-id>/`，
但它不是状态源。运行时状态只由 native Goal 或 Goal Orchestrator 的文件模式维护。

文件模式使用 `scripts/import_goal_package.py` 导入确认包。导入器拒绝 `prompt_only`，根据
`ready/activate` 创建对应状态，并在 Goal contract 中保存只读 source package provenance。
导入器只接受 `invocation_mode: direct`、尚未绑定 runtime Goal ID 的初始形成包；运行期
GoalPackageRevision 必须更新原 Goal，不能借导入器创建第二个 Goal。
