"""三路独立评审 atomic-agents 核心运行时（1 ducc + 2 codex），同题不互见，产出后 Claude 交叉裁决。

范围：src/atomic_agents/ 下 16 个核心文件（models/scheduler/adapters/drift/locks/
lockfile/reviewer/linter/templates/meta/llm_compiler/approval/validation/run），
已拷贝进 workspace（examples/orchestrations 编排模板与 docs/DESIGN.md 均不在评审范围）。
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
EX_ROOT = REPO_ROOT / "examples"
for p in (str(SRC_ROOT), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from examples.orchestrations.common import (  # noqa: E402
    ensure_workspace, make_skeleton, run_orchestration, write_node,
)
from examples.orchestrations.prompts import PROGRESS_HEARTBEAT, SELF_REFLECTION_CHECKLIST  # noqa: E402

WORKSPACE = str(REPO_ROOT / "atomic-orch-out" / "self-review-runtime")

_FILES = [
    "src__atomic_agents__models.py",
    "src__atomic_agents__scheduler.py",
    "src__atomic_agents__adapters____init__.py",
    "src__atomic_agents__adapters__codex.py",
    "src__atomic_agents__adapters__ducc.py",
    "src__atomic_agents__drift.py",
    "src__atomic_agents__locks.py",
    "src__atomic_agents__lockfile.py",
    "src__atomic_agents__reviewer.py",
    "src__atomic_agents__linter.py",
    "src__atomic_agents__templates.py",
    "src__atomic_agents__meta.py",
    "src__atomic_agents__llm_compiler.py",
    "src__atomic_agents__approval.py",
    "src__atomic_agents__validation.py",
    "src__atomic_agents__run.py",
]

BACKGROUND = """\
【项目背景】atomic-agents 是一个原子化多 agent 协作运行时：用户一句话需求 →
元编排者生成静态协作骨架（DAG）→ 运行时把 codex / ducc 当作可互换的黑盒原子
调度执行 → 产出 + 可审计 lockfile。本质是"有界批处理 DAG 机器"，v1 阶段。

9 项设计上锁定的决策（评审时可作为"是否吻合设计意图"的参照，但不要求你认同
这些决策本身，若认为某决策本身有问题也可以提出）：
1. atom = 一次完整黑盒会话，一次性交付契约；三道护栏（超时/预算/内部轮次）。
2. 编排者只调度不产出内容；reviewer 逻辑常驻、物理每次新起、结构化 verdict。
3. v1 只做静态骨架 + 线性 retry；动态 hook（add_atom/goto）推迟 v1.1。
4. lockfile 只追加审计、不重放。
5. v1 关闭权限硬控（prompt 软约束，required_capabilities 保留但运行时忽略）。
6. write_scope + 咨询写锁 + changed-file drift ——已修订为：写原子按声明
   write_scope 分组，不交叠并行、交叠串行；越界写默认记录但不停 run（不再
   连坐整批止损）。write_scope 是调度提示+drift基线+写锁键，不是安全边界。
7. 元编排者跟随启动身份。
8. reviewer 输出结构化（criteria↔verdict↔evidence↔confidence）。
9. 元编排者：模板优先+自由兜底+解释+质量警告。

近期已知修复（2026-06-29）：批次 drift 不再连坐止损（_detect_batch_drift/
_resolve_drift_choice）；网关 503/429 等瞬时错误判 transient 就地退避重试
（3次 5/15/45s），独立于 max_repair_attempts、不烧配额；workspace 两层
resolve 绝对化；idle-timeout 心跳 PROGRESS_HEARTBEAT 注入长任务写 prompt。

【已知未做（不是 bug，是 v1 边界，评审时不需要重复指出这些】：
- 并发写未声明同名文件无硬保护（靠汇报兜底，留 v1.1 worktree）。
- per-atom max_cost 未强制（adapter 不读 limits.max_cost）。
- reviewer 触发上游 writer 重试的预算嵌套未收敛。
- 测试几乎全 mock，真实 LLM 链路无自动化验证。
- 动态 hook、硬权限/能力校验、worktree 隔离、自动重放 → v1.1/v2 范围。
"""

_FILE_LIST = "\n".join(f"- {name}" for name in _FILES)

TASK_HEADER = f"""\
你是资深 Python 系统架构评审专家。你的工作目录（workspace）下有以下 {len(_FILES)} 个
源文件，是 atomic-agents 项目 src/atomic_agents/ 下的全部核心运行时代码（原始包路径
已编码进文件名，"__"代表原路径的"/"；不含 examples 编排模板、不含 docs 设计文档——只
评审这些文件本身的实现）：
{_FILE_LIST}

请先读取工作目录下这些文件的完整内容，再开始评审。

{BACKGROUND}

【评审任务】通读全部 16 个文件，找出：

A. 当前可修复的缺陷（bug / 边界条件 / 逻辑漏洞 / 并发安全 / 资源泄漏 / 与自身
   文档注释矛盾 / 错误处理缺失或过度）。每条缺陷要求：
   - 具体文件名 + 大致位置（函数名/行为描述，你能定位到哪里就写到哪里）
   - 复现条件或触发场景（不要泛泛而谈"可能有问题"）
   - 严重度：[致命] 会导致错误结果/数据丢失/挂死 vs [中等] 影响正确性但有
     绕过方式 vs [轻微] 代码质量/可维护性问题
   - 建议修复方向

B. 未来演进方向。区分两类：
   - 若发现当前架构中期就会遇到的硬瓶颈（不是"锦上添花"，是"不改会撞墙"），
     指出具体是什么、为什么现在的设计撑不住、大致改法方向。
   - 常规意义上"下一步该往哪走"的建议（不必须是硬瓶颈，但要言之有物、
     具体到某个模块或能力，不要空泛的"增加测试覆盖率"之类套话）。

【要求】
- 只针对这 16 个文件本身的实现，不要评价 examples/ 编排模板或 docs/ 文档写得
  怎样（那些不在你的评审范围）。
- 已在背景里列出的"已知未做"不需要重复提出，除非你发现它们比背景描述的更
  严重或范围不同。
- 不要因为不确定就回避；标注确信度，但要给出你的判断。
- 找不到某类问题就明确说"未发现"，不要为了显得全面而凑数。

只输出你的评审 markdown，不要输出与本任务无关的内容。
""" + PROGRESS_HEARTBEAT + SELF_REFLECTION_CHECKLIST


def build_skeleton():
    nodes = [
        write_node(
            node_id="review_ducc",
            role="analyst",
            task=TASK_HEADER,
            output_file="review-ducc.md",
        ),
        write_node(
            node_id="review_codex_a",
            role="analyst",
            task=TASK_HEADER,
            output_file="review-codex-a.md",
        ),
        write_node(
            node_id="review_codex_b",
            role="analyst",
            task=TASK_HEADER,
            output_file="review-codex-b.md",
        ),
    ]
    # 三者互不可见彼此产出：不设 depends_on/inputs，纯并行独立评审。
    return make_skeleton("self-review-runtime", nodes, max_repair_attempts=0, max_total_cost=100.0)


def run() -> bool:
    ws = ensure_workspace(WORKSPACE, "atomic-review-runtime-")
    skeleton = build_skeleton()

    explicit_runners = {
        "review_ducc": "ducc",
        "review_codex_a": "codex",
        "review_codex_b": "codex",
    }

    result = run_orchestration(
        "self-review-runtime",
        BACKGROUND,
        skeleton,
        {},
        ws,
        final_file=None,
        explicit_atom_runners=explicit_runners,
    )
    print(f"=== SELF-REVIEW-RUNTIME DONE succeeded={result.succeeded} ===", flush=True)
    print("artifacts: " + ", ".join(result.artifacts), flush=True)
    if result.run_result is not None:
        rr = result.run_result
        print("run_id: " + rr.run_id, flush=True)
        print("lock_dir: " + rr.lock_dir, flush=True)
        print("failed_nodes: " + ", ".join(rr.failed_nodes), flush=True)
        print("blocked_nodes: " + ", ".join(rr.blocked_nodes), flush=True)
    return result.succeeded


if __name__ == "__main__":
    ok = run()
    raise SystemExit(0 if ok else 1)
