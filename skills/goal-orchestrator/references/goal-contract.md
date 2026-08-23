# 目标契约（Goal Contract）

## 目的

定义目标终态和治理规则，但不预先规定每个执行步骤。在允许实施计划调整的同时，保持契约稳定。

契约采用版本化管理。`version` 表示 JSON schema 版本，`contract_revision` 表示当前目标契约的非空字符串修订标识；两者不得混用。首次形成草案时使用 `contract:1`，此后可使用 `contract:2`、内容哈希或其他单调且可区分的标识。任何会改变结果、完成标准、约束、非目标、权限、重要未知项处置或停止规则的变更都必须生成新的 `contract_revision`，并记录变更原因。只改变实现路径或里程碑时更新计划，不改变 `contract_revision`。

## 必填字段

### 目标（Objective）

使用一句适合原生 Goal 界面的简洁表述。描述结果，而非活动。

较弱：`研究认证并改进代码。`

较强：`增加通行密钥登录，保留密码登录，并让全部认证回归测试通过。`

### 结果（Outcome）

描述用户可见或外部可观察的最终状态，并明确受影响的系统或产物。

### 完成定义（Definition Of Done）

列出少量可独立验证的标准，并为每项标准分配稳定 ID，例如 `DOD-1`。

每项标准都必须回答：

- 必须满足什么可观察状态？
- 什么证据可以证明它？
- 证据属于客观验证、评审判断，还是两者结合？

避免使用“高质量”“健壮”或“正确完成”等缺少度量方式或评审准则的标准。

### 约束（Constraints）

记录安全、政策、兼容性、资源、业务、证据和副作用限制。用户提供的精确值必须原样保留。

### 非目标（Non-Goals）

明确不得纳入目标的相邻工作，防止看似有益的范围扩张。

### 权限（Permissions）

区分：

- 无需确认即可执行的操作；
- 需要确认的操作；
- 禁止执行的操作。

长期执行绝不会扩大宿主沙箱、审批策略或用户授权范围。

未被更具体契约覆盖时，采用以下默认规则：

- 对本机数据的读取、创建、编辑、构建、测试和其他非删除操作，可在现有宿主权限内直接执行；
- 删除本机数据前必须获得 approve，包括删除文件、目录、数据库记录、缓存或不可逆覆盖；
- 连接或操作用户或组织管理的外部服务器、远程数据库、云资源或生产环境前必须获得 approve，包括读取、写入、部署、调用管理接口或改变远程状态；公开资料的只读检索仍遵循宿主网络权限；
- approve 只授权当次明确说明的动作、对象和范围，不自动扩展为后续同类操作；
- 用户给出的更严格限制优先；宿主沙箱、审批策略和禁止项始终优先。

权限判断必须基于动作的实际副作用，而不是命令名称。无法确认操作仅影响本机且不删除数据时，按 `requires_confirmation` 处理。

### 未知项（Unknowns）

只记录具有实质影响的未知项，并为每项指定负责人或解决路径。不要把普通实施细节堆入未知项。

未知项可标记为 `open`、`resolved` 或 `accepted_risk`。`accepted_risk` 不是智能体自行关闭问题的捷径，必须满足：

- 由用户或契约中明确授权的决策者接受；
- 记录风险、影响和现有证据，并在 `acceptance` 中记录 `accepted_by`、`accepted_at`、`scope` 和 `review_condition`；
- 不得绕过 `forbidden`、宿主审批、安全、隐私、法律或合规边界；
- 不得把未通过的必需完成标准改写成已完成；若风险改变完成定义、约束或权限，先生成新的 `contract_revision` 并作废受影响证据。

激活前，高影响未知项必须为 `resolved` 或合规的 `accepted_risk`；普通实施细节可以留到计划中处理。

### 停止规则（Stop Rules）

定义何时：

- 答复或完成；
- 调整方法后重试；
- 采用备用方案；
- 请求关键决策；
- 暂停；
- 报告真实阻塞。

### 可选编排策略（Orchestration Policy）

用户可以指定是否使用专门并行调研或方案设计；未指定时由形成阶段或运行阶段的当前 owner
按真实任务自动判断。策略结构和路由规则见
[capability-routing-and-handoffs.md](capability-routing-and-handoffs.md)。默认是
`mode: auto`、允许 research/design、两者均非必需且委托前无需额外确认。

编排策略不是文件模式契约的必填字段。仅改变调查或设计方法时记录在计划中；若用户的
选择同时改变范围、权限、结果或 DOD，才写入契约并生成新的 `contract_revision`。

契约和发现可以记录可选的 `basis_refs`、`decision_refs` 或 capability artifact 引用，
但这些引用不成为第二状态源，也不能让专门能力直接修改契约。

`goal-prompt` 形成的 GoalBrief 和 GoalCompilation 是候选输入；用户确认后成为只读
GoalPackage。runtime owner 必须核验其中的路径、命令、DOD、权限、用户决定和形成证据，
再导入 native Goal 或 `goal.json`。只有通过 Ready 门禁的运行时契约才能用
`scripts/render_goal_prompt.py` 再导出外部 `/goal`；导出的 Prompt 记录源 Goal ID 和
contract revision，但不能反向覆盖契约。

## 文件持久化 JSON 结构

```json
{
  "version": 1,
  "contract_revision": "contract:1",
  "goal_id": "passkey-sign-in",
  "status": "draft",
  "objective": "增加通行密钥登录，同时保留密码登录。",
  "outcome": "用户可以注册并使用通行密钥，且仍能使用密码登录。",
  "definition_of_done": [
    {
      "id": "DOD-1",
      "criterion": "用户可以注册并使用通行密钥登录。",
      "verification": {
        "method": "end-to-end test",
        "command": "npm test -- passkey.e2e.ts",
        "expected": "exit 0"
      }
    }
  ],
  "constraints": ["保留密码登录"],
  "non_goals": ["替换身份提供方"],
  "permissions": {
    "allowed_without_confirmation": ["读取、创建和编辑本机数据", "运行本机构建和非破坏性测试"],
    "requires_confirmation": ["删除本机数据", "连接或操作用户或组织管理的外部服务器"],
    "forbidden": ["访问真实用户凭据"]
  },
  "unknowns": [],
  "stop_rules": ["每项 DOD 标准都有新鲜的通过证据后才能完成"],
  "final_review_required": true,
  "contract_changes": [
    {
      "revision": "contract:1",
      "changed_at": "2026-07-15T08:00:00Z",
      "changed_by": "user",
      "reason": "initial contract"
    }
  ]
}
```

上述 `status` 仅用于文件持久化模式。宿主提供原生 Goal 时，原生 Goal 是唯一状态真源，不要用契约文件镜像或覆盖原生状态。

## 质量门槛

激活前确认：

- 目标描述了结果；
- 完成定义可度量且充分；
- 每项标准都有验证方法；
- 约束和非目标不与结果冲突；
- 权限覆盖可能产生的副作用；
- 权限明确包含本机数据、删除操作和外部服务器操作的边界；
- 重要未知项都有解决路径，或满足规则的 `accepted_risk` 记录；
- `contract_revision` 与契约变更记录一致，受影响的旧证据已经作废；
- 无需暗中重定义成功标准即可完成目标。

如果结果仍不明确，继续停留在规划阶段，并且只询问一个影响最大的关键问题。不要展示问卷，也不要激活模糊目标。将优先级较低的开放问题保留在契约草案中，直到前一个答案使其变得相关。
