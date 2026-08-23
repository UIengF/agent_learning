# atomic-agents

> 原子化多 agent 协作运行时 — 用一句话描述需求，由元编排者生成可控的协作骨架，运行时把 codex / ducc 当作可互换的黑盒原子调度执行。

**独立项目，与 `~/.claude/skills/teammate` 解耦**：本项目运行时不依赖 teammate 的任何代码、状态机或共享文件层。teammate 仅是孕育本设计的协作工具，非本项目的依赖。

## 是什么

一台 **「有界批处理 DAG 机器」**：

```
用户一句话
  → 元编排者（模板优先 + 自由兜底，全程解释 + 质量警告）生成【静态骨架】
  → 行动级审批①（看得懂、可改、可驳回）
  → 运行时黑盒调度 codex/ducc 严格照图跑
     · 写原子按声明 write_scope 分组：不交叠并行 / 交叠串行 + 咨询写锁
     · changed-file drift 检测：越界写默认【记录但不停】，如实汇报
     · 瞬时网关错误（503/429）退避重试，不烧 repair 配额
     · reviewer 结构化验收（criteria↔evidence↔confidence）
     · 线性 retry / 止损
  → 产出 + 可审计 run lockfile
```

完整设计见 [`docs/DESIGN.md`](docs/DESIGN.md)（权威来源）。运行时行为修订见下方 **§运行时行为（v1 当前实现）**。

## 核心概念

- **atom（原子）**：一次完整黑盒 agent 会话，内部多轮自迭代，完成时一次性交付契约。
- **skeleton（骨架）**：声明式 DAG，v1 为静态（开跑前画死，运行时不增不改）。
- **orchestrator（编排者）**：只读契约、只调度、不产出内容；身份可换（codex/ducc）。
- **runtime-binding**：逻辑角色 → 具体 runner，运行时绑定，锁进 lockfile。
- **meta-orchestrator（元编排者）**：自然语言 → 骨架；模板优先 + 自由兜底。

## v1 范围

里程碑 **P0 → P0.5 → P1 → P1.5 → P2 → P3 → P5 → P6**：

| 阶段 | 内容 |
|------|------|
| P0 | JSON schema + mock runner + lockfile writer/reader |
| P0.5 | 元编排者探针（一次性、风险前置） |
| P1 / P1.5 | runner adapters（codex+ducc）+ 真实 smoke test |
| P2 | 静态 DAG runtime（写串行/读并行 + 写锁 + drift 暂停） |
| P3 | reviewer gate（结构化 verdict）+ 线性 retry/止损 |
| P5 | Meta-Orchestrator（模板优先 + 自由兜底 + 解释 + 质量警告）|
| P6 | 审批 UX（①③⑥⑦，行动级确认）|

**不含**：动态 hook（v1.1）；硬权限/能力校验、worktree 隔离、流式、自动重放（v2）。

## 开发

```bash
pip install -e ".[dev]"
pytest
```

## 运行时行为（v1 当前实现）

> 以下是经 2026-06-29 健壮性修复后的**实际运行时语义**，部分对 `DESIGN.md` 的原始决策做了修订（见 DESIGN 决策6 修订说明）。`DESIGN.md` 仍是设计意图的权威来源，本节是实现现状的权威来源。逐项根因/修法/测试见 [`docs/ROBUSTNESS-FIXES-2026-06.md`](docs/ROBUSTNESS-FIXES-2026-06.md)。

### 写并行与 drift（路1）

- **写原子按声明 `write_scope` 分组**：不交叠 → 并行；交叠 → 串行（咨询写锁 `ConsultingFileLock` 排队）。这取代了 DESIGN 原定的「写原子默认串行」——编排扇出（多立场设计 / 多评审者）依赖不交叠写并行。
- `write_scope` 是 **调度提示 + drift 基线 + 写锁占用键**，**不是安全边界**。
- **越界写（写了 `write_scope` 外的文件）默认【记录但不停 run】**：写进 lockfile `scope_drift` 事件 + `run_summary.unplanned_files_touched`，如实汇报，不判失败、不止损。批次内合规节点的成果不连坐丢失。
- 调用方仍可注入 `drift_handler` 返回 `"stop"` 强制止损（向后兼容）。
- **已知盲区（诚实声明）**：不阻止两个并行原子同时写一个**未声明的同名文件**——这是声明制的固有盲区，靠事后汇报兜底，物理根治留 v1.1 worktree。

### 瞬时（基础设施）错误退避重试

- adapter 把网关 5xx/429 或明确的网关临时错误（`credentials exhausted` / `server-side issue` / `rate limit` / `retry later` 等）判为 `status="transient"`，与「任务失败」区分开。
- scheduler 对 `transient` **就地退避重试**（默认 3 次，5s/15s/45s），**与 `max_repair_attempts` 完全独立、不烧 repair 配额、不计成本**。
- 退避耗尽仍不可用 → 归一化为 `failed`，stop reason = `infrastructure_unavailable`，stop-report **如实归因为基础设施问题**（而非误导成「需求需澄清」）。
- 常量在 `scheduler.py` 顶部：`TRANSIENT_MAX_RETRIES` / `TRANSIENT_BACKOFF_SECONDS`。

### workspace 路径

- `--workspace` 传相对路径也安全：`ensure_workspace` 与 `StaticScheduler.__init__` **两层归一化为绝对路径**（`Path(...).expanduser().resolve()`）。根治了相对路径被子进程 `cwd` 二次解析导致 codex/ducc 报 `Workspace does not exist` 秒挂的坑。

### idle-timeout 心跳

- 运行时靠「workspace 文件 mtime 增长 OR 子进程 stdout 字节增长」判活。codex 经 `script` 块缓冲、产出到结束才落盘，长任务期间不碰文件会被误判 idle。
- 所有**长任务写产出** prompt 统一注入 `PROGRESS_HEARTBEAT`（`examples/orchestrations/prompts.py`），要求周期性 append `_progress.md` 刷新 mtime。裁判/验收节点（只写短 JSON）不注入。

### 契约字段

- 合法的 `inputs[].field` 单一真相源为 `linter.VALID_INPUT_FIELDS = {result, artifacts, output_file}`。`build_contract`、linter、LLM 编译 prompt 三处都引用它，不再漂移。按文件交接时优先用 `output_file`。

## 编排模板 CLI

6 个内置编排模板（`examples/orchestrations/`），后台跑 + 轮询：

```bash
cd /Users/uleng/Code/atomic-agents
.venv/bin/python3 -m examples.orchestrations.cli \
  <design|review|arena|pipeline|scatter-gather|execute-plan> "<需求>" \
  [--n N] [--count ROLE=N] [--doc PATH] [--plan PATH] [--min-score S] \
  [--role ROLE=codex|ducc|codex,ducc] \
  [--designer codex|ducc] [--reviewer codex|ducc] [--implementer codex|ducc] \
  --workspace /绝对/路径/输出目录
```

| 模板 | 形状 | 默认数量 |
|------|------|---------|
| `design` | 检索→选立场→N立场并行设计→红队2轮→合成 | 3 立场、3 researcher |
| `review` | N 评审者并行读同一文档→裁判打分 | 2 评审者（`--doc` 必填）|
| `arena` | N 方案竞争→裁判打分→低于阈值重跑 | 3 方案 |
| `pipeline` | plan→impl(codex)→验收 | — |
| `scatter-gather` | N 路调研并行→汇总→审查 | 3 路 |
| `execute-plan` | 读取已有计划→impl(codex)→验收 | —（`--plan` 必填）|

- `--count` 可重复：`design` 支持 `designer`（>=2）和 `researcher`（>=1），`review` 支持 `analyst`（>=2），`arena` 支持 `designer`（>=2），`scatter-gather` 支持 `explorer`（>=2）。`--n` 保留为各模板主并行角色的兼容别名。
- `--role` 可重复，支持上表模板中的任意已知角色；逗号分隔多个 runner 时，同角色 agent 按声明顺序轮换，例如 `--role analyst=codex,ducc`。

- **真实跑慢**：多 agent + 联网，常数分钟级。
- **`--workspace` 建议传绝对路径**（虽已两层兜底）；产出默认落 `./atomic-orch-out/`。
- lockfile/stop-report 落 `$TMPDIR/atomic-orch-locks-*/run-*/run.lock.jsonl`（`tempfile.mkdtemp`，不在 workspace 内），排障时读它的 `atom_finished` payload。

## 定位声明

v1 是**审计型运行时（audit runtime）**，不是安全沙箱：通过写锁 + drift 检测（记录/可选停下）+ 结构化审批/审查 + 瞬时错误退避 + 全程 lockfile 把风险**可见化、可决策化**，但不提供硬隔离/硬权限保证。并发写未声明同名文件、per-atom 成本硬上限等留待后续版本。
