# CLAUDE.md — atomic-agents

> 给在本仓库工作的 AI agent 的项目级指引。**这是约定，按它工作。** 设计意图的权威来源是 `docs/DESIGN.md`；运行时实际行为的权威来源是 `README.md` §运行时行为 + 本文件。

## 这是什么

**原子化多 agent 协作运行时**：用户一句话 → 元编排者生成静态协作骨架（DAG）→ 运行时把 codex / ducc 当作可互换的**黑盒原子**调度执行 → 产出 + 可审计 lockfile。本质是「有界批处理 DAG 机器」。

独立项目，运行时不依赖 teammate。

## 环境（硬约束，先读）

- **必须用 `.venv/bin/python3`**，不是系统 `python3`（系统的没有 pytest/jsonschema）。所有命令前缀 `.venv/bin/python3`。
- Python 3.14（`.python-version`），`requires-python >=3.10`。依赖仅 `jsonschema`；dev 加 `pytest`。
- macOS：**没有 `timeout` 命令**，要限时用 `perl -e 'alarm N; exec @ARGV' ...`。
- 跑测试：`.venv/bin/python3 -m pytest -q`（配置在 `pyproject.toml`，`pythonpath=["src"]`）。当前 **185 passed**。
- 两个 runner 的默认路径：codex → `vendor/codex/scripts/ask_codex.sh`（仓库内）；ducc → `~/.comate/baidu-cc/bin/ducc`。

## 代码地图

```
src/atomic_agents/
  models.py        # 数据契约：AtomContract / Skeleton / SkeletonNode / AtomResult / StopReport
                   #   Status = success|failed|blocked|timeout|transient
  scheduler.py     # ★核心：StaticScheduler — 拓扑/分批/并行/drift/retry/瞬时退避/止损/lockfile emit
  adapters/
    __init__.py    # RunnerAdapter 协议 + run_with_timeout(idle-timeout看门狗) + is_transient_error
                   #   + build_output_directive / build_context_files_directive
    codex.py       # CodexAdapter（包 ask_codex.sh）；token→USD 折算
    ducc.py        # DuccAdapter（ducc -p --output-format json）；真实 USD
  drift.py         # changed-file 快照 + detect_drift（git porcelain / mtime 双模式）
  locks.py         # ConsultingFileLock（按 write_scope 排队的写互斥，非权限校验）
  lockfile.py      # 只追加事件流 writer/reader（统一 envelope）
  reviewer.py      # ReviewResult / 结构化 verdict
  linter.py        # 骨架语义 lint；VALID_INPUT_FIELDS（inputs[].field 单一真相源）
  templates.py     # 模板库 + match_template
  meta.py          # meta_compile：NL→MetaPlan（模板优先+自由兜底）+ 质量警告
  llm_compiler.py  # LLM 自由编排骨架 + repair
  approval.py      # 审批摘要/stop-report 渲染 + 审批回调协议
  validation.py    # JSON Schema 校验（schemas/*.json）
examples/orchestrations/   # 5 个编排模板 + CLI（design/review/arena/pipeline/scatter_gather）
  common.py        # 角色→runner 路由 + run_orchestration + ensure_workspace
  prompts.py       # 所有 prompt + PROGRESS_HEARTBEAT + SELF_REFLECTION_CHECKLIST
tests/             # 全 mock/fake runner；真实 LLM 链路无自动化测试
schemas/           # 5 个 JSON Schema（contract/skeleton/lockfile_event/approval/stop_report）
docs/DESIGN.md     # 权威设计（9 项锁定决策）
```

## 9 项锁定决策（不可推翻，见 `DESIGN.md §2`）

1. atom = 一次完整黑盒会话，一次性交付契约；三道护栏（超时 / 预算 / 内部轮次）。
2. 编排者只调度不产出内容；reviewer 逻辑常驻、物理每次新起、结构化 verdict。
3. v1 只做静态骨架 + 线性 retry；动态 hook（add_atom/goto）推迟 v1.1。
4. lockfile 只追加审计、不重放。
5. v1 关闭权限硬控（prompt 软约束，`required_capabilities` 保留但运行时忽略）。
6. write_scope + 咨询写锁 + changed-file drift（**v1 实现已修订，见下**）。
7. 元编排者跟随启动身份（ducc 里启动就是 ducc）。
8. reviewer 输出结构化（criteria↔verdict↔evidence↔confidence）。
9. 元编排者：模板优先 + 自由兜底 + 解释 + 质量警告。

## 运行时行为修订（2026-06-29 健壮性修复，与 DESIGN 原文有出入处以此为准）

改这些区域前先读懂修订，别退回旧语义：

- **写并行（路1，偏离决策6原文）**：写原子**按声明 write_scope 分组——不交叠并行、交叠串行**，不是「默认串行」。编排扇出依赖它。`write_scope` 是调度提示 + drift 基线 + 写锁键，**不是安全边界**。
- **越界 drift = 记录不停**：`_detect_batch_drift` / `_resolve_drift_choice` 默认 `"continue"`，越界只写 `scope_drift` 事件 + `run_summary.unplanned_files_touched`，**不连坐、不止损**。`drift_handler` 注入 `"stop"` 可恢复硬停（兼容）。**别再让批次 drift `_stop_run` 连坐**——那是已修的 P1。
- **瞬时错误退避**：网关 503/429 等经 `is_transient_error` 判 `status="transient"`，scheduler 在 `_invoke_contract` 就地退避重试（`TRANSIENT_MAX_RETRIES` / `TRANSIENT_BACKOFF_SECONDS`），**独立于 `max_repair_attempts`、不烧配额**；耗尽归因 `infrastructure_unavailable`。别把 transient 混进 `atom_failed`。
- **workspace 必绝对**：`ensure_workspace` + `StaticScheduler.__init__` 两层 `resolve()`。别移除——相对路径会让子进程秒挂。
- **idle 心跳**：长任务写 prompt 必须带 `PROGRESS_HEARTBEAT`（防 codex 被误判 idle 杀掉）。加新写产出 prompt 记得拼上；裁判类短 JSON 不拼。
- **inputs[].field 单一真相源**：`linter.VALID_INPUT_FIELDS`。改字段只动这一处，`build_contract` 和 LLM prompt 都引用它。

## 改代码时

- **改 `scheduler.py` 要格外小心**：它是上帝类（~1200 行），`_settle_regular_node` 与 `_settle_reviewer_node` 是两套相似的 retry/budget/feedback 循环——历史上两个 P1 bug 都出在这里。改任一处想清楚对另一处的影响。
- **改完必跑 `pytest`**，全绿再交。新功能/修 bug 配回归测试。关键回归测试：
  - `tests/test_scheduler.py` — 批次 drift 不连坐、disjoint 并行
  - `tests/test_e2e_scheduler.py` — drift 记录/停下两路
  - `tests/test_retry.py` — 瞬时退避恢复 / 耗尽归因
  - `tests/test_orchestration_robustness.py` — workspace 绝对化 + 心跳注入面
- **lockfile payload 是自由对象**（schema `additionalProperties:true`），加字段不破坏校验，但读侧要容忍缺失。
- **成本单位是 USD**：ducc 报真实 USD，codex 按 token×`usd_per_mtok` 折算。`max_total_cost` = 美元。

## 排障

- 编排卡住/失败：读 lockfile `$TMPDIR/atomic-orch-locks-*/run-*/run.lock.jsonl`（**不在 workspace 内**），看 `atom_finished` 的 `payload.error`、`run_stopped` 的 `reason` + stop-report。
- codex/ducc 节点秒挂报 `Workspace does not exist`：workspace 没归一化成绝对路径（应已两层兜底，若复现先查这里）。
- 节点被误判 idle 杀掉：该 prompt 缺 `PROGRESS_HEARTBEAT`。
- 网关 503/`credentials exhausted`：是基础设施瞬时不可用，已自动退避；持续不可用会以 `infrastructure_unavailable` 停下，不是任务问题。

## 已知未做（不是 bug，是 v1 边界）

- 并发写**未声明**同名文件无硬保护（靠汇报兜底，留 v1.1 worktree）。
- per-atom `max_cost` 未强制（adapter 不读 `limits.max_cost`）；编排默认 `max_total_cost=1000.0` 偏松。
- reviewer 触发上游 writer 重试的预算嵌套未收敛。
- 测试几乎全 mock，真实 LLM 链路无自动化验证。
- 动态 hook、硬权限/能力校验、worktree 隔离、自动重放 → v1.1/v2。

## 协作约定

非平凡任务优先编排协作（见用户全局 CLAUDE.md）。但**修 atomic-agents 自身的 bug 时，编排工具可能正不可用，直接动手**——这是约定的例外。
