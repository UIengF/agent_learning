"""Schema-iteration benchmark 编排：阶段1 多路联网调研 → research-report.md；
阶段2 多并行方案设计 → 各方案 + final-benchmark-design.md。

codex 多分配兜底（链接不稳定），ducc 固定 1 路。后台跑 + 轮询。
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
    ensure_workspace, make_skeleton, merge_role_runners, run_orchestration, write_node,
)
from examples.orchestrations.prompts import PROGRESS_HEARTBEAT, SELF_REFLECTION_CHECKLIST  # noqa: E402

WORKSPACE = "/Users/uleng/Code/atomic-agents/atomic-orch-out/schema-benchmark"

# ── 用户的真实痛点 + 硬约束（注入每个 agent 的 prompt）────────────────────────
BACKGROUND = """\
【业务背景】用户在做「有驾 Car Graph」——一个 HugeGraph 知识图谱，用于 GraphRAG 场景，
领域是汽车说明书。当前致命的低效闭环：每次改 schema，都必须用新 schema 重新跑一遍 LLM
抽取整批文档、重建一份新图数据库，才能测效果。没有 benchmark 去对比新旧 schema 优劣。

三个并存痛点：
1. 重抽取太慢/太贵——抽取全靠 LLM，每改一次 schema 就重花一遍 token + 时间。
2. 没有金标准——没有标准答案/标注集，无法判断新 schema 是变好还是变坏，只能人肉看。
3. 结果不可对比——两次测试环境不一致，分数没法横向比，缺一个固定 benchmark 套件。

【最终交付目标】一套可落地方案，让 schema 迭代能【快速、可对比、有金标准】地评估，
终结「改 schema 必须全量重抽重建图」的低效循环。
"""

HARD_CONSTRAINTS = """\
【已确认硬约束与决策（不可推翻，方案必须吻合）】
1. 评测覆盖三层（全链路都要能测）：
   - 抽取/建图质量（同批文档用新 schema 抽出的实体/边数量、覆盖率、抽取准确率）。
   - 召回层质量（给定 query，图召回的子图/路径相关性）。
   - 端到端问答质量（GraphRAG 最终答案的准召/正确性）。
2. 金标准路线 = (a)+(b) 并用：
   - (a) 人工标注 QA 对：挑 N 个文档，人工写「问题 → 标准答案 + 该命中的实体/边/原文证据」。
   - (b) 冻结一个「黄金 schema + 黄金图」当基准：选定认可的 schema 版本，其抽出的图作为
     ground truth，新 schema 与它对比（测相对变化）。
   - 须说明 (a)(b) 如何配合：(a) 锚定绝对正确性，(b) 做零标注的快速相对回归。
3. 抽取管线现状 = 一步到位：当前 LLM 抽取 prompt 里直接带 schema 抽，抽取与 schema 深度
   耦合，没有中间表示层。因此方案必须直面核心抉择：【是否引入 schema 无关的中间事实表示层】
   ——让 LLM 先一次性抽出与具体 schema 解耦的原始事实（缓存它），改 schema 时只重跑
   「中间事实 → 新 schema 图」的纯代码映射，从而不重花 token。这是省抽取成本的关键。须把
   「改造抽取管线引入中间层」作为重点方案方向之一，论证可行性、迁移成本与收益；同时给出
   「若不改造管线」的退路方案。
4. 可对比的输入 query 集：主要从业务文档挖典型问题构造；用户手头有少量现成问题作种子。
5. schema 变更分两档处理（重要简化）：
   - 轻量变更（加字段、加索引、调 nullable）——抽取产物结构基本不变，须支持快速复用对比。
   - 结构大改（加减顶点/边 label、改主键/sortKeys）——接受退化为全量重抽，方案【不需要】
     强行支持结构变更下的语义对齐。即：分两档，结构大改走全量重抽即可。
6. 手头可复用资产（除此之外几乎没有）：完整业务文档（汽车说明书，可作固定测试集输入）；
   抽取全部由 LLM 完成；极少量现成参考问题；【没有】现成大规模标注/QA 对/既有 benchmark 工具。
7. 底座技术约束（HugeGraph）：一个 edge label 只能绑一对 source/target；SINGLE 边静默去重
   覆盖；一个 search 索引只能 by 单个 TEXT 属性；sortKeys 字段非空。涉及建图/索引须吻合。
"""


# ════════════════════════════════════════════════════════════════════════════
# 阶段 1：多路联网调研 → research-report.md
# ════════════════════════════════════════════════════════════════════════════
_RESEARCH_DIRECTIONS = [
    ("GraphRAG / KG 评测方法",
     "业界怎么评 GraphRAG 与知识图谱的【抽取质量、召回质量、端到端问答质量】：具体指标、"
     "公开数据集、现成工具（如 RAGAS 及其 context precision/recall、faithfulness、answer "
     "correctness 等；ragas/TruLens/DeepEval/Microsoft GraphRAG 自带评测；KG 抽取的 "
     "precision/recall/F1 算法）。每个指标说明怎么算、适配本汽车说明书 GraphRAG 场景的可借鉴点。"),
    ("Schema/数据管线回归基准",
     "数据库/数据管线的 schema 变更如何做回归 benchmark；有没有 golden dataset / golden "
     "schema 式的【相对回归】实践（snapshot testing、golden-file testing、dbt 的 data test / "
     "schema test、approval testing、契约测试）。怎么把『冻结黄金基准 + 新版本 diff』工程化。"),
    ("LLM 抽取的 schema 解耦 / 中间表示",
     "开放信息抽取（OpenIE）、先抽 triples/中间事实再映射到目标 schema 的做法；抽取产物缓存与"
     "复用的工程实践（如 LangChain LLMGraphTransformer、Microsoft GraphRAG 的 claims/中间产物、"
     "REBEL、OpenIE6、schema-guided vs schema-free extraction）。这种中间层的收益、坑、迁移成本。"),
    ("金标准构造省力法",
     "小样本人工标注 + LLM-as-judge 辅助扩充、用 LLM 生成候选标注再人工校验、active learning / "
     "human-in-the-loop 标注、合成 QA 对生成（如用文档自动生成 question-answer-evidence 三元组）"
     "等【降低标注成本】的方法与已知质量陷阱（LLM-judge 偏差、自洽性校验）。"),
    ("GraphRAG 召回层评测细化",
     "针对图召回（子图检索/路径检索/多跳）的相关性评测：如何定义 ground-truth 子图/路径、"
     "怎么算召回层 precision/recall/nDCG、有没有 retrieval-only 的中间评测（不跑到最终答案就能"
     "评召回好坏），以及 query 集如何从领域文档自动挖掘典型问题（含多跳/对比/规格查询类）。"),
]


def _research_task(direction_name: str, direction_detail: str) -> str:
    return (
        "你是知识图谱与 GraphRAG 评测方向的检索调研者，必须【联网检索】"
        "（官方文档、论文 arXiv、知名开源 GitHub 仓库、权威工程博客），不要只凭记忆。\n\n"
        f"{BACKGROUND}\n{HARD_CONSTRAINTS}\n"
        f"【你负责的调研方向】{direction_name}\n{direction_detail}\n\n"
        "产出要求（结构化 markdown）：\n"
        "1. 该方向业界有哪些公认做法 / 官方推荐 / 现成开源实现（逐条给【来源链接】与要点）。\n"
        "2. 关键指标/方法的【具体定义与计算方式】，能落地的写清算法或公式。\n"
        "3. 每条结论提炼【对本汽车说明书 GraphRAG schema 评测场景的可借鉴点 / 不适用之处】。\n"
        "4. 已知坑、适用边界、不确定性显式标注。\n"
        "务必先联网查证、标来源链接、简洁可落地。"
        + PROGRESS_HEARTBEAT
    )


def _research_merge_task() -> str:
    files = "、".join(f"research-{i}.md" for i in range(len(_RESEARCH_DIRECTIONS)))
    return (
        f"你是调研汇总者。context 里有 {len(_RESEARCH_DIRECTIONS)} 份独立联网调研：{files}，"
        "分别覆盖 GraphRAG/KG 评测方法、schema/数据管线回归基准、LLM 抽取的 schema 解耦/中间表示、"
        "金标准构造省力法、GraphRAG 召回层评测细化。\n\n"
        f"{BACKGROUND}\n{HARD_CONSTRAINTS}\n"
        "把这些调研去重、消解冲突，合并成一份结构化 markdown 调研报告 research-report.md。要求：\n"
        "1. 保留【所有有价值的来源链接】，链接放在对应结论附近。\n"
        "2. 按主题组织（评测指标三层 / 回归基准范式 / 中间表示与抽取解耦 / 金标准省力法 / "
        "召回评测 / query 集构造），多份结论一致时合并表达。\n"
        "3. 冲突或分歧显式标注：分歧点、各自依据、你的取舍判断与不确定性。\n"
        "4. 区分来源类型（官方文档/论文/开源实现/工程博客）说明可信度与适用边界。\n"
        "5. 末尾输出【对本场景设计的启示】总表：哪些做法可直接采用、哪些需针对本需求改良、"
        "哪些风险必须保留——后续方案设计 agent 会直接基于它工作。\n"
        "只输出合并后的 markdown，不要遗漏来源链接。"
        + PROGRESS_HEARTBEAT
    )


def build_research_skeleton():
    # 5 路调研：codex 多分配兜底（0,1,2,4 = codex），ducc 固定 1 路（3）。
    runner_plan = {
        "research_0": "codex", "research_1": "codex", "research_2": "codex",
        "research_3": "ducc", "research_4": "codex",
        "research_merge": "ducc",
    }
    nodes = []
    for i, (name, detail) in enumerate(_RESEARCH_DIRECTIONS):
        nodes.append(write_node(
            node_id=f"research_{i}", role="researcher",
            task=_research_task(name, detail), output_file=f"research-{i}.md",
        ))
    nodes.append(write_node(
        node_id="research_merge", role="researcher", task=_research_merge_task(),
        output_file="research-report.md",
        depends_on=[f"research_{i}" for i in range(len(_RESEARCH_DIRECTIONS))],
        inputs=[{"from": f"research_{i}", "field": "output_file"}
                for i in range(len(_RESEARCH_DIRECTIONS))],
    ))
    return make_skeleton("schema-benchmark-research", nodes), runner_plan


# ════════════════════════════════════════════════════════════════════════════
# 阶段 2：3 并行方案设计 → 各方案 → 合成 final-benchmark-design.md
# ════════════════════════════════════════════════════════════════════════════
_DESIGN_STANCES = [
    ("中间层激进派",
     "押注【引入 schema 无关中间事实表示层】：LLM 一次抽取缓存中间事实，schema 迭代只重跑"
     "纯代码映射。把这条作为主路径论证到底——中间事实的数据结构、映射层设计、token 节省量级、"
     "迁移成本、轻量变更的零重抽复用机制。但仍要诚实给出它的失效边界。"),
    ("务实退路派",
     "押注【最小改造、复用现状】：尽量不动现有「prompt 带 schema 一步抽」的管线，靠缓存原始"
     "LLM 输出、固定测试集、黄金图相对 diff 来拿到大部分收益。论证不引入中间层时如何仍做到"
     "快速可对比，迁移成本最低，何时这条路够用、何时不得不升级到中间层。"),
    ("评测严谨派",
     "押注【金标准与指标体系的严谨性、可对比性】：三层指标的精确定义与阈值、人工标注 QA 对的"
     "字段结构/规模/标注规范、黄金 schema+图基准的选定与冻结、LLM-as-judge 的偏差控制与校验、"
     "测试环境固定与版本化，让分数真正可横向比、可回归。"),
]


def _design_task(stance_name: str, stance_focus: str) -> str:
    return (
        f"你是【{stance_name}】方案设计者。视角：{stance_focus}\n\n"
        f"{BACKGROUND}\n{HARD_CONSTRAINTS}\n"
        "context 里有联网调研报告 research-report.md，参考其最佳实践并在此之上改良"
        "（勿无视也勿照抄）。\n\n"
        "请基于调研报告 + 上述硬约束，设计一套【完整、可落地】的 Schema 迭代评测 Benchmark 方案。"
        "必须逐条覆盖以下 8 点，每点给具体设计而非泛泛而谈：\n"
        "1. 整体架构：如何把「抽取→建图→召回→问答」评测闭环固定成可重复的 benchmark 套件"
        "（含数据流、目录结构、运行入口、产物形态）。\n"
        "2. 抽取成本问题的正面回答：是否引入 schema 无关中间表示层？给出【改造方案】+"
        "【不改造的退路】，含中间事实数据结构、映射层、迁移成本与 token 节省量级估算。\n"
        "3. 金标准设计：(a) 人工标注 QA 对的规模/字段结构/标注规范；(b) 黄金 schema+图基准的"
        "选定与冻结方式；(a)(b) 如何配合（绝对正确性 vs 零标注相对回归）。\n"
        "4. 三层评测指标的具体定义：抽取层/召回层/端到端层各用什么指标、怎么算、阈值怎么定。\n"
        "5. 两档变更处理流程：轻量变更（加字段/索引/nullable）走快速复用对比的【具体机制】；"
        "结构大改（加减 label/改主键 sortKeys）走全量重抽的流程。\n"
        "6. query 测试集构造：怎么从业务文档挖典型问题（多跳/对比/规格查询）+ 用好少量现成问题。\n"
        "7. 可对比性保证：如何固定测试环境让分数可横向比（固定文档集/固定 query 集/固定指标/"
        "版本化基准/确定性控制）。\n"
        "8. 落地路径：MVP 先做什么、分几期、每期产出与验收标准。\n\n"
        "须吻合 HugeGraph 硬约束（edge label 单 source/target、SINGLE 边去重、search 单 TEXT、"
        "sortKeys 非空）。坚持你的立场倾向，但方案要完整、自洽、诚实标注残留风险。"
        + PROGRESS_HEARTBEAT + SELF_REFLECTION_CHECKLIST
    )


def _redteam_task(design_files: list[str]) -> str:
    files = "、".join(design_files)
    return (
        f"你是红队挑战者。context 里有 {len(design_files)} 份针对【Schema 迭代评测 Benchmark】"
        f"的方案（{files}）。用最强反方论证压力测试每一份，逼出最脆弱环节。\n\n"
        f"{BACKGROUND}\n{HARD_CONSTRAINTS}\n"
        "规则：\n"
        "1. 攻击前提而非结论：列出每份隐含的未验证假设（尤其『中间表示层真能省 token 且不丢"
        "信息』『黄金图能当 ground truth』『LLM-judge 可信』『从文档挖的 query 有代表性』），"
        "逐个问『若不成立会怎样』。\n"
        "2. 每个质询可证伪+可落地：给具体失败场景（某类 schema 变更/某类 query/某种文档结构）、"
        "触发条件、最小复现路径。禁止空话。\n"
        "3. 区分严重度：标 [致命/会推翻] vs [可修补/局部]。\n"
        "4. 建设性收尾：每个致命问题给至少一个修复方向或更优替代，说明残留代价。\n"
        "5. 自我设限：找不到致命问题就明说『未发现颠覆性缺陷』，不要凑数。\n"
        "按方案分节输出 markdown 质询报告。"
        + PROGRESS_HEARTBEAT
    )


def _design_v2_task(stance_name: str, stance_focus: str, challenge_file: str, prev_file: str) -> str:
    return (
        f"你是【{stance_name}】方案设计者，视角：{stance_focus}\n\n"
        f"{BACKGROUND}\n{HARD_CONSTRAINTS}\n"
        f"这是第 2 版。context 里有调研报告 research-report.md、红队对上一版的质询 {challenge_file}、"
        f"以及你的上一版 {prev_file}。\n"
        "认真回应质询：致命问题必须修复或显式反驳并说理；可修补缺陷尽量改。坚持立场但让方案更强。\n"
        "仍须完整覆盖原 8 点（整体架构 / 抽取成本中间层抉择 / 金标准 / 三层指标 / 两档变更 / "
        "query 集 / 可对比性 / 落地路径），并吻合 HugeGraph 硬约束。输出完整定稿 markdown。"
        + PROGRESS_HEARTBEAT + SELF_REFLECTION_CHECKLIST
    )


def _synthesis_task(final_files: list[str], challenge_files: list[str]) -> str:
    designs = "、".join(final_files)
    challenges = "、".join(challenge_files)
    return (
        f"你是方案合成者。context 里有 {len(final_files)} 份最终方案（{designs}）、红队质询"
        f"（{challenges}）、以及调研报告 research-report.md。\n\n"
        f"{BACKGROUND}\n{HARD_CONSTRAINTS}\n"
        "目标不是『选一个』，而是合成一份比任何单一方案都强的最终方案 final-benchmark-design.md：\n"
        "1. 先评后合：用统一维度（省抽取成本有效性 / 金标准可信度 / 三层指标可落地性 / 可对比性 / "
        "迁移成本 / 工程复杂度）简评每份。\n"
        "2. 逐维度取长：每维度挑最好做法并说明，按维度拼装，不整体二选一。\n"
        "3. 显式裁决核心冲突：尤其【是否引入 schema 无关中间表示层】——明确给出推荐结论"
        "（引入 / 不引入 / 分期引入）、判据、代价、何时反过来；以及金标准 (a)(b) 的配合权重。\n"
        "4. 吸收红队修复。\n"
        "5. 最终 markdown 必须含：①完整融合方案（覆盖原 8 点，可直接照着落地）②各从哪份吸收了"
        "什么 ③【仍需用户拍板的开放决策点】清单（逐条写清选项、影响、你的倾向）④MVP 落地路线图。"
        "写充分、可执行、吻合 HugeGraph 硬约束。"
        + PROGRESS_HEARTBEAT
    )


def build_design_skeleton():
    # 3 立场：codex 2 路兜底（0,1），ducc 1 路（2）。红队/合成各分配。
    n = len(_DESIGN_STANCES)
    runner_plan: dict[str, str] = {}
    stance_runner = ["codex", "codex", "ducc"]
    for i in range(n):
        for v in (1, 2):
            runner_plan[f"design_{i}_v{v}"] = stance_runner[i]
    runner_plan.update({"redteam_r1": "codex", "synth": "ducc"})

    nodes = []
    v1 = [f"design-{i}-v1.md" for i in range(n)]
    for i, (name, focus) in enumerate(_DESIGN_STANCES):
        nodes.append(write_node(
            node_id=f"design_{i}_v1", role="designer",
            task=_design_task(name, focus), output_file=v1[i],
        ))
    nodes.append(write_node(
        node_id="redteam_r1", role="critic", task=_redteam_task(v1),
        output_file="challenge-r1.md",
        depends_on=[f"design_{i}_v1" for i in range(n)],
        inputs=[{"from": f"design_{i}_v1", "field": "output_file"} for i in range(n)],
    ))
    v2 = [f"design-{i}-v2.md" for i in range(n)]
    for i, (name, focus) in enumerate(_DESIGN_STANCES):
        nodes.append(write_node(
            node_id=f"design_{i}_v2", role="designer",
            task=_design_v2_task(name, focus, "challenge-r1.md", v1[i]),
            output_file=v2[i], depends_on=["redteam_r1"],
            inputs=[{"from": "redteam_r1", "field": "output_file"},
                    {"from": f"design_{i}_v1", "field": "output_file"}],
        ))
    nodes.append(write_node(
        node_id="synth", role="synthesizer",
        task=_synthesis_task(v2, ["challenge-r1.md"]),
        output_file="final-benchmark-design.md",
        depends_on=[f"design_{i}_v2" for i in range(n)] + ["redteam_r1"],
        inputs=[{"from": f"design_{i}_v2", "field": "output_file"} for i in range(n)]
               + [{"from": "redteam_r1", "field": "output_file"}],
    ))
    return make_skeleton("schema-benchmark-design", nodes), runner_plan


# ════════════════════════════════════════════════════════════════════════════
def run_phase(phase: str) -> bool:
    ws = ensure_workspace(WORKSPACE, "schema-benchmark-")
    role_runners = merge_role_runners(None)
    if phase == "research":
        skeleton, plan = build_research_skeleton()
        final = "research-report.md"
        name = "schema-benchmark-research"
    elif phase == "design":
        skeleton, plan = build_design_skeleton()
        final = "final-benchmark-design.md"
        name = "schema-benchmark-design"
    else:
        raise ValueError(f"unknown phase {phase!r}")

    print(f"=== PHASE {phase} START ws={ws} ===", flush=True)
    print("runner_plan: " + ", ".join(f"{k}={v}" for k, v in plan.items()), flush=True)
    result = run_orchestration(
        name, BACKGROUND, skeleton, role_runners, ws,
        final_file=final, explicit_atom_runners=plan,
    )
    print(f"=== PHASE {phase} DONE succeeded={result.succeeded} ===", flush=True)
    print("final_path: " + str(result.final_path), flush=True)
    print("artifacts: " + ", ".join(result.artifacts), flush=True)
    if result.run_result is not None:
        rr = result.run_result
        print("run_id: " + rr.run_id, flush=True)
        print("lock_dir: " + rr.lock_dir, flush=True)
        print("failed_nodes: " + ", ".join(rr.failed_nodes), flush=True)
        print("blocked_nodes: " + ", ".join(rr.blocked_nodes), flush=True)
    return result.succeeded


if __name__ == "__main__":
    phase_arg = sys.argv[1] if len(sys.argv) > 1 else "research"
    ok = run_phase(phase_arg)
    raise SystemExit(0 if ok else 1)
