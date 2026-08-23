"""Prompt library for all orchestration templates.

Self-reflection checklist + stance library + design prompts distilled from
researched best practices (Self-Refine / Reflexion / Chain-of-Verification /
Society-of-Minds debate / red-team self-critique). Review/arena/pipeline/
scatter-gather prompts added on top.

v1.1: added type-specific production quality dimensions (PRODUCTION_DIMENSIONS),
review checklists (REVIEW_CHECKLISTS), a red-team attack dimension table
(RED_TEAM_DIMENSIONS), and a synthesis adjudication checklist
(SYNTHESIS_ADJUDICATION). Distilled from Google Engineering Practices, OWASP
(Code Review Guide + LLM Top 10), Anthropic/OpenAI prompting guides,
Chain-of-Verification, Self-Refine, STRIDE, FMEA, MCDA/AHP/ATAM, and ADR/MADR.
"""

from __future__ import annotations

# ── 层1：每个产出 agent 内部自反思清单 (atom 内黑盒"想到满意"才交付) ──────────
SELF_REFLECTION_CHECKLIST = """\
在写出最终产出前，先在心里静默做几轮自我反思与修订（不要展示完整推理过程，
只输出反思后定稿）。反复逐条拷问自己，直到自己满意为止：

1. 核心问题是什么？我是否真正解决了核心问题，而不是一个相近但更容易的问题？
2. 隐含了哪些假设？哪些没有依据、最可能错？若某条不成立会怎样？
3. 从第一性原理重新推导，结论是否仍成立，还是只是套常见模板？
4. 有没有更简单的解？是否把问题复杂化了？
5. 最坏情况 / 边界条件 / 失败场景是什么？在这些情况下会不会崩？
6. 一个严苛的专家或红队会怎样攻击它？这些攻击我是否已处理或显式承认？
7. 我遗漏了什么？真正关心的约束（成本、时间、风险、可维护性、可执行性）覆盖了吗？
8. 有没有把"可能性"说成"确定性"，把经验建议说成通用规律？
9. 若有检索到的最佳实践，我是否参考并超越，而不是无视或简单复制？
10. 最能提升质量的一处修改是什么？先改掉它，再定稿。
"""

# ── 层1.1：按产出物类型的通用质量维度（v1.1 新增）──────────────────────────────
# 通用自反思清单不区分产出物类型；这里补充"这类产出物本身该显式检查什么"，
# 拼在 SELF_REFLECTION_CHECKLIST 之前，作为更具体的第一遍检查。
# 来源：OpenAI/Anthropic prompt engineering 指南、Chain-of-Verification (ACL 2024
# Findings)、Self-Refine (arXiv 2303.17651)。
PRODUCTION_DIMENSIONS: dict[str, str] = {
    "code": """\
产出代码实现前，额外检查：
1. 是否明确处理了失败路径、异常来源和降级策略，而不只覆盖 happy path。
2. 是否列出并处理边界输入、空输入、重复输入、非法输入、规模极限和权限不足等场景。
3. 是否避免引入与需求无关的抽象、配置、框架或跨模块重构。
4. 是否把错误处理放在调用边界、IO 边界和外部依赖边界，而不是只在最外层吞异常。
5. 是否有可运行的最小验证方式（测试、命令、示例输入输出或手动验收步骤）。
6. 是否说明哪些行为已验证、哪些只是按代码推断，不要把未运行的实现说成已通过。
7. 是否保持与既有接口、数据结构、文件边界兼容，避免无意扩大改动面。
""",
    "design": """\
产出设计/方案前，额外检查：
1. 是否解释了为什么选择当前方案，而不是只描述"怎么做"。
2. 是否列出至少 1-2 个被放弃的替代方案，并说明放弃原因、适用边界和代价。
3. 是否显式标注关键假设，并说明假设不成立时方案会如何失效。
4. 是否给出失败模式、触发条件、影响范围和可观察信号，而不是笼统写"有风险"。
5. 是否说明如何验证方案有效（指标、实验、测试、灰度或人工验收方式）。
6. 是否区分短期可落地设计与长期演进设计，避免把愿景伪装成当前可执行计划。
7. 是否覆盖时间、成本、兼容性、迁移风险、可维护性和可回滚性等主要约束。
""",
    "research": """\
产出调研结论前，额外检查：
1. 是否按来源权威性分级：官方文档/标准/论文优先，开源实现次之，博客论坛作补充。
2. 是否标注来源新鲜度，尤其对易变信息（API/框架版本、产品行为）给出日期或版本。
3. 是否区分事实、作者观点、工程经验和本次推断，不把二手解释当一手事实。
4. 是否针对关键结论做跨来源验证，而不是只引用第一个支持自己结论的来源。
5. 是否主动寻找反例、限制条件或竞争假设，避免确认偏误。
6. 是否标注不确定性和置信度，并说明降低不确定性还需要查什么。
7. 是否把调研结论转成对后续设计/实现的可执行启示，不停留在资料摘录。
""",
    "synthesis": """\
产出合成/汇总结论前，额外检查：
1. 是否先建立统一评价维度再比较各输入，而不是按文件顺序逐段复述。
2. 是否按维度吸收最强部分，说明"从哪份输入吸收了什么"，避免机械拼接。
3. 是否显式列出冲突点、各方依据、最终裁决和裁决代价，不用模糊措辞绕开分歧。
4. 是否识别输入间的重复、互补和矛盾，删除低信息重复内容。
5. 是否保留少数派但高风险的观点，尤其可能推翻结论的边界条件或失败模式。
6. 是否标注仍未解决的问题和残留不确定性，不让合成后显得所有冲突都已消失。
7. 是否给出下一步验证动作，让合成结果可被验收或继续推进。
""",
}

# 产出者通用通病（跨类型，附在每类产出维度之后）：
PRODUCER_PITFALLS = """\
产出者常见通病（自查）：
- 过度自信：把未验证推断写成确定事实，不标注置信度和不确定性。
- 套话替代分析：写"健壮、可扩展、最佳实践"，但不给失败场景、触发条件和验证方法。
- 忽略隐性约束：只优化答案漂亮程度，漏掉兼容性、迁移成本、权限、时间和可回滚性。
- 只修辞不裁决：面对冲突输入时用折中话术回避选择，没有说明取舍代价。
- 过度工程：为简单任务引入额外层次、配置和抽象，反而降低可读性与可验收性。
"""


def production_dimensions(kind: str) -> str:
    """按产出物类型返回额外质量维度 + 通用通病自查（v1.1）。"""

    dimension = PRODUCTION_DIMENSIONS.get(kind, "")
    return dimension + PRODUCER_PITFALLS


# ── 层1.2：按被评审产出物类型的通用审查清单（v1.1 新增）─────────────────────────
# 现有 review_judge_task/pipeline_review_task/gather_review_task/arena_judge_task
# 只约束 JSON verdict 的字段形状，没有"具体要审查什么维度"。这里补充可复用的检查
# 清单，拼进各 judge/review prompt。来源：Google Engineering Practices code review、
# OWASP Code Review Guide、Langfuse/LangChain 的 LLM-as-judge rubric 设计建议。
REVIEW_CHECKLISTS: dict[str, str] = {
    "code": """\
审查代码实现类产出时，额外检查：
1. 实现是否满足原始需求和真实目标，而不是只满足表面描述。
2. 核心逻辑在正常路径、边界条件、异常输入、并发或状态切换下是否都保持正确。
3. 错误处理是否明确、可恢复，不会吞掉关键异常或输出误导性的成功结果。
4. 安全风险是否被检查：输入验证、权限边界、敏感信息、注入面、不可信依赖。
5. 测试是否覆盖关键行为、失败路径和回归风险，且测试本身会在代码坏掉时失败。
6. 代码复杂度是否必要，是否存在过度抽象、未来猜测式设计或难维护的控制流。
7. 命名、接口、注释是否帮助理解"为什么这样做"，而不是掩盖不清晰的代码。
8. 改动是否与现有系统风格、模块边界保持一致，没有让整体代码健康倒退。
""",
    "design": """\
审查方案/设计类产出时，额外检查：
1. 方案是否直接回应原始目标和约束，而不是转向一个更容易解决的相邻问题。
2. 关键假设是否显式列出，并说明假设不成立时的失败模式和降级路径。
3. 技术可行性是否有足够依据：依赖、接口、数据、权限、运行环境、团队能力。
4. 成本、复杂度、交付周期和长期维护负担是否被纳入权衡，而不是只描述理想收益。
5. 安全、隐私、合规、滥用和操作风险是否被作为一等约束处理。
6. 扩展性是否基于明确规模假设，避免为未知未来需求引入过度设计。
7. 是否比较了至少一个合理替代方案，并解释为什么当前方案更适合。
8. 结论是否可验收：包含可执行步骤、验证信号和不通过时的修正方向。
""",
    "research": """\
审查调研/汇总类产出时，额外检查：
1. 来源是否可信，并区分官方文档、论文、工程实践、博客等不同证据等级。
2. 覆盖范围是否匹配原始问题，避免只收集支持某一结论的材料。
3. 多来源之间的冲突是否被显式标注，并说明各自依据、适用边界和取舍判断。
4. 关键结论是否能追溯到具体来源或本地证据，而不是只凭模型概括。
5. 是否清楚区分事实、推断、建议和不确定性，避免把可能性写成确定性。
6. 调研是否覆盖反例、失败案例、限制条件和不适用场景。
7. 汇总是否真正合并去重并消解差异，而不是机械拼接多份输入。
8. 最终建议是否可执行，且说明需要进一步验证的点。
""",
}

# 评审者通用陷阱（跨被评审类型，附在各审查清单之后）：
REVIEWER_PITFALLS = """\
评审者常见陷阱（自查）：
1. 只检查 JSON 格式是否合规，却没有判断 criteria.evidence 是否真的支撑 verdict。
2. 被流畅、自信、篇幅较长的回答误导，忽略事实错误或遗漏（冗长偏差）。
3. 对第一个或最后一个候选产生位置偏差，尤其在多方案比较场景中。
4. 把"没有发现问题"当作"通过"，但没有证明已覆盖关键路径、边界和反例。
5. 用单一绝对分数替代逐项 rubric 判断，导致失败项被平均分掩盖。
"""


def review_checklist(kind: str) -> str:
    """按被评审产出物类型返回额外审查维度 + 评审者陷阱自查（v1.1）。"""

    checklist = REVIEW_CHECKLISTS.get(kind, "")
    return checklist + REVIEWER_PITFALLS


def verdict_directive(
    *,
    has_score: bool,
    criterion_label: str,
    feedback_audience: str,
    score_meaning: str = "",
    min_score: int | None = None,
    blocking_note: str = "",
) -> str:
    """统一构造 JSON verdict 输出指令（v1.1，去重 4 个评审者函数里几乎相同的字段说明）。

    has_score: 是否要求给 0-10 分数（review_judge/arena_judge 需要，pipeline/gather
        只需 bool passed）。
    criterion_label: criteria 数组每项 criterion 字段该填什么语义（比如"评审维度"
        vs "方案名"），告诉 runner 这个字段名代表什么。
    feedback_audience: feedback 建议面向谁（比如"评审者"/"实现者"）。
    score_meaning: score 字段的额外说明（比如"最佳方案的得分"），has_score=False 时忽略。
    min_score: 通过阈值；has_score=True 时用于 passed 字段说明，未传则写"min_score"占位。
    blocking_note: blocking_findings 字段的额外说明；留空则不加括注。
    """

    score_field = ""
    passed_field = "passed(bool), "
    if has_score:
        meaning = f", {score_meaning}" if score_meaning else ""
        score_field = f"score(0-10 整数{meaning}), "
        threshold = min_score if min_score is not None else "min_score"
        passed_field = f"passed(score>={threshold} 的布尔), "
    blocking_field = f"blocking_findings({blocking_note}), " if blocking_note else "blocking_findings, "
    return (
        "把审查结论写成 JSON verdict 到你的输出文件，字段：\n"
        f"{score_field}{passed_field}"
        f"criteria(每项 {{criterion:{criterion_label},verdict:pass|fail,evidence,confidence}}), "
        f"{blocking_field}reviewer_session(可为 null), "
        f"feedback(给{feedback_audience}的改进建议)。只写这个 JSON，不要其他内容。"
    )


# ── 层1.3：红队攻击维度表（v1.1 新增）────────────────────────────────────────
# redteam_task 原本只有原则性规则（攻击前提、可证伪、分严重度），缺少系统性的
# "该覆盖哪几类风险"清单，容易漏判某类攻击面。来源：STRIDE 威胁建模、FMEA、
# MITRE CWE-20/362/345/400、OWASP LLM Top 10、Google SRE 监控原则、
# Microsoft Retry/Circuit Breaker、SemVer。
RED_TEAM_DIMENSIONS = """\
红队质询时，按以下维度逐项扫一遍（不要求每类都强行找问题，但发现问题必须给出
触发条件、最小复现场景、严重度和修复方向）：
- 正确性：输入、边界、状态机、业务规则是否会产生错误结果？
- 并发与竞态：共享状态、文件、缓存、队列、checkpoint 是否会被并发破坏？
- 数据完整性：是否会接受伪造、过期、重复、部分写入或不一致数据？
- 成本与资源：上下文、调用次数、重试、并发、存储、CPU/内存是否有上限？
- 安全与越权：是否可能 prompt injection、敏感信息泄露、工具越权或权限提升？
- 兼容与迁移：旧配置、旧 schema、旧产物、旧调用方是否会静默失败？
- 可观测性：失败时能否知道坏在哪里、为什么坏、影响多大、如何恢复？
- 依赖失效：外部 API、网络、模型、文件系统、数据库不可用时如何降级？
- 过度信任：是否盲信某个 agent、模型输出或评审结论，缺少证据闭环？
- 供应链与配置：依赖、模型、工具权限、环境变量变化时是否可检测和可复现？
"""

# ── 层1.4：方案合成裁决清单（v1.1 新增）──────────────────────────────────────
# synthesis_task 原本只要求"先评后合、逐维度取长、显式裁决冲突"，缺少具体裁决
# 方法论。来源：MCDA/AHP 多准则决策、ATAM 架构权衡分析、ADR/MADR 决策记录法。
SYNTHESIS_ADJUDICATION = """\
遇到互斥冲突时，不得直接选一个方案。必须：
1. 写清冲突点和备选项，标出硬约束，先淘汰违反硬约束的选项。
2. 用统一维度评分，并说明权重依据；总分只用于排序，高权重维度的致命短板应
   触发否决或补救，不能被平均分掩盖。
3. 对高争议维度做成对比较，显式说明偏好（例如"正确性比性能重要多少"）。
4. 用具体失败/规模/迁移场景验证 trade-off，不用抽象形容词裁决。
5. 记录选择的正负后果和反向条件（什么条件下应选择另一边）。
6. 吸收红队修复建议，逐项说明处理状态：已修复/被其他方案覆盖/有意识接受/
   需要用户裁决。未处理的致命问题不得被合成文本冲淡。
7. 给出最终裁决的验证方法（测试、指标、演练、故障注入）和残留风险。
"""

# ── 层1.5：可用性/可维护性质询维度表（v1.2 新增）─────────────────────────────
# redteam_task 覆盖 10 个风险维度，但安全/并发/成本类占多数，缺少专门盯"这个
# 设计好不好用、好不好维护"的视角——设计初期这类问题往往比安全问题更常见、
# 更早决定方案生死。与红队并行、独立产出，不共享上下文，避免视角互相污染。
# 来源：Nielsen 十大可用性原则、Google Engineering Practices 可维护性标准、
# ISO/IEC 25010 软件质量模型（可用性/可维护性子特征）。
USABILITY_DIMENSIONS = """\
可用性/可维护性质询时，按以下维度逐项扫一遍（不要求每类都强行找问题，但发现
问题必须给出具体场景、影响范围和改进方向）：
- 心智负担：使用者/后续维护者要记住多少隐含规则才能正确使用？能否降低到无需记忆？
- 一致性：是否与已有约定（命名、接口、错误处理风格）保持一致，减少意外行为？
- 可发现性：关键能力/限制/前提条件是否显而易见，还是需要读源码才能发现？
- 错误可恢复性：用户/调用方犯错时，是否有清晰提示和恢复路径，而不是静默失败？
- 演进成本：需求变化时，这个设计需要改几个地方？是否有单一改动点？
- 调试友好度：出问题时，能否快速定位到根因，而不需要深入内部实现？
- 文档/自解释性：设计本身（命名、结构）是否降低了对外部文档的依赖？
"""

# ── 进度心跳（硬性要求）──────────────────────────────────────────────────────
# 运行时用「workspace 文件 mtime 增长 OR 子进程 stdout 字节增长」判活；codex 经 script
# 块缓冲、产出文件到结束才落盘，长任务期间若不碰文件会被误判 idle 而被杀。所有【写产出】
# 节点统一注入此心跳：周期性 append _progress.md 刷新 mtime，防止空闲超时误杀。
PROGRESS_HEARTBEAT = (
    "\n过程留痕（硬性要求，防空闲超时）：\n"
    "在产出最终结果的过程中，必须周期性地把阶段性进度追加写入 workspace 下的 _progress.md。"
    "_progress.md 是固定过程文件名，且不等于最终产出文件名；它只用于刷新工作区文件 mtime，"
    "不会被当成交付物，也不参与 inputs。\n"
    "第一步先 echo/创建 _progress.md，写入本次要覆盖的大纲或步骤清单。"
    "之后每完成一个章节/步骤/关键决策，就立刻 append 对应进度到 _progress.md（带时间或步骤名），"
    "不要把写文件憋到最后。必须保证每隔几分钟就有一次对 _progress.md 的写入。\n"
    "_progress.md 只是过程留痕；最终完整产出仍要按既有【产出要求】写入指定的最终产出文件。\n"
)

# ── 立场库：5 组互斥张力对 ────────────────────────────────────────────────────
STANCE_LIBRARY = [
    {"pair": "速度/简洁 ↔ 健壮/严谨",
     "a": {"name": "速度简洁派", "focus": "最小可行、尽快交付、避免过度设计"},
     "b": {"name": "健壮严谨派", "focus": "边界处理、错误恢复、长期可维护"}},
    {"pair": "创新/重构 ↔ 保守/兼容",
     "a": {"name": "创新重构派", "focus": "推翻重来、引入更优范式"},
     "b": {"name": "保守兼容派", "focus": "复用现有、最小改动、控制迁移风险"}},
    {"pair": "性能/规模 ↔ 可读/开发体验",
     "a": {"name": "性能规模派", "focus": "吞吐、延迟、成本、可扩展性"},
     "b": {"name": "可读体验派", "focus": "抽象清晰、易调试、开发者友好"}},
    {"pair": "乐观执行 ↔ 红队/失败",
     "a": {"name": "乐观执行派", "focus": "happy path、快速落地"},
     "b": {"name": "失败防御派", "focus": "边界、攻击面、故障模式与降级"}},
    {"pair": "用户/产品 ↔ 工程/约束",
     "a": {"name": "用户产品派", "focus": "体验、需求覆盖、产品价值"},
     "b": {"name": "工程约束派", "focus": "可行性、技术债、资源与时间约束"}},
]


def research_task(request: str) -> str:
    return (
        "你是检索调研者，必须【联网检索】(GitHub、官方文档、论文、知名工程博客)。\n"
        f"需求：\n---\n{request}\n---\n\n"
        "调研：这个需求是否已有成熟最佳实践/官方推荐方案/GitHub 现成实现？产出：\n"
        "1. 是否有公认做法(有则给来源链接与要点；没有明确说明)。\n"
        "2. 相关官方文档 / 知名开源实现 / 设计模式(带链接)。\n"
        "3. 这些方案的关键取舍、已知坑、适用边界。\n"
        "4. 对后续设计的启示(哪些可直接用、哪些需针对本需求改良)。\n"
        "写成结构化 markdown，务必先联网查证，不要只凭记忆，标来源、简洁可落地。"
        + PROGRESS_HEARTBEAT + "\n" + production_dimensions("research")
    )


def research_merge_task(request: str, research_files: list[str]) -> str:
    files = "、".join(research_files)
    return (
        f"你是调研汇总者。context 里有 {len(research_files)} 份独立联网检索调研：{files}。\n"
        f"需求：\n---\n{request}\n---\n\n"
        "请把这些调研去重、消解冲突，并合并成一份结构化 markdown 调研报告。要求：\n"
        "1. 保留所有有价值的来源链接，并把链接放在对应结论附近。\n"
        "2. 多份调研结论一致时合并表达，避免重复堆砌。\n"
        "3. 存在冲突或分歧时显式标注：分歧点、各自依据、你的取舍判断与不确定性。\n"
        "4. 区分官方文档、论文、开源实现、工程博客等来源类型，说明可信度和适用边界。\n"
        "5. 最后输出“对后续设计的启示”：哪些做法可直接采用，哪些需要针对本需求改良，哪些风险必须保留。\n"
        "只输出合并后的 markdown，不要遗漏来源链接。"
        + PROGRESS_HEARTBEAT
    )


def stance_selection_task(request: str, n_stances: int) -> str:
    pairs = "\n".join(
        f"  - {s['pair']}: {s['a']['name']}({s['a']['focus']}) vs {s['b']['name']}({s['b']['focus']})"
        for s in STANCE_LIBRARY
    )
    return (
        "你是方案设计的元编排者。先读 context 里的 research.md，再读需求，决定派几个"
        "【立场不同的设计者】并行出方案。张力来自【优化互斥指标】，不靠措辞。\n"
        f"需求：\n---\n{request}\n---\n\n可选互斥张力对：\n{pairs}\n\n"
        f"输出一个恰好 {n_stances} 个元素的 JSON 数组，每个："
        '{"stance_name":"立场名","focus":"重点优化什么","prompt_hint":"给该立场的一句视角提示"}。'
        "只输出 JSON 数组，不要其他文字、不要 markdown 围栏。"
    )


def design_task(request, stance, research_file, version, challenge_files=None, prev_version_file=None):
    head = (
        f"你是【{stance['stance_name']}】设计者，视角：{stance.get('focus','')}。"
        f"{stance.get('prompt_hint','')}\n需求：\n---\n{request}\n---\n\n"
        f"context 里有调研报告 {research_file}，参考其最佳实践并在此之上改良(勿无视也勿照抄)。\n"
    )
    if version == 1:
        body = "这是第一版。给出完整设计：核心思路、关键决策、为什么这么做、关键风险。\n"
    else:
        files = "、".join(challenge_files or [])
        body = (
            f"这是第 {version} 版。context 里有对上一版的两组独立质询（{files}，分别来自对抗性"
            f"评审专家和可用性/可维护性评审专家）和你的上一版 {prev_version_file}。\n"
            "认真回应两组质询：致命问题必须修复或显式反驳并说理；可修补缺陷尽量改。"
            "两组质询同等重要，不要只顾安全/健壮而忽略可用性/可维护性，坚持立场但让方案更强。\n"
        )
    process_trace = PROGRESS_HEARTBEAT
    return head + body + process_trace + "\n" + production_dimensions("design") + "\n" + SELF_REFLECTION_CHECKLIST


def redteam_task(request, design_files, round_no):
    files = "、".join(design_files)
    return (
        "你是对抗性评审专家，与可用性/可维护性评审专家并行独立质询，不看对方的产出。"
        f"context 里有若干份针对同一需求的产出（{files}）。用最强反方论证压力测试每一份，"
        "逼出最脆弱环节。\n"
        f"需求：\n---\n{request}\n---\n\n规则：\n"
        "1. 攻击前提而非结论：列出每份隐含的未验证假设，逐个问“若不成立会怎样”。\n"
        "2. 每个质询可证伪+可落地：给具体失败场景(输入/边界/规模)、触发条件、最小复现路径。禁止空话。\n"
        "3. 区分严重度：标 [致命/会推翻] vs [可修补/局部]，不要平铺。\n"
        "4. 建设性收尾：每个致命问题给至少一个修复方向或更优替代，说明残留代价。\n"
        "5. 自我设限：找不到致命问题就明说“未发现颠覆性缺陷”，不要凑数造伪问题。\n"
        f"这是第 {round_no} 轮，按方案分节输出 markdown 质询报告。\n"
        + RED_TEAM_DIMENSIONS
        + PROGRESS_HEARTBEAT
    )


def usability_review_task(request, design_files, round_no):
    files = "、".join(design_files)
    return (
        "你是可用性/可维护性评审专家，与对抗性评审专家并行独立质询，不看对方的产出。"
        f"context 里有若干份针对同一需求的产出（{files}）。\n"
        f"需求：\n---\n{request}\n---\n\n规则：\n"
        "1. 聚焦可用性和可维护性，不重复对抗性评审专家负责的安全/并发/资源类问题。\n"
        "2. 每个质询给具体场景（谁在什么情况下会遇到）、影响和改进方向。禁止空话。\n"
        "3. 区分严重度：标 [必须改] vs [建议改进]，不要平铺。\n"
        "4. 自我设限：找不到问题就明说“未发现显著可用性/可维护性缺陷”，不要凑数造伪问题。\n"
        f"这是第 {round_no} 轮，按方案分节输出 markdown 质询报告。\n"
        + USABILITY_DIMENSIONS
        + PROGRESS_HEARTBEAT
    )


def synthesis_task(request, final_files, challenge_files):
    designs = "、".join(final_files)
    challenges = "、".join(challenge_files) if challenge_files else "（无）"
    return (
        f"你是方案合成者。context 里有 N 份最终方案({designs})及对抗性评审专家与可用性/可维护性"
        f"评审专家两组独立质询({challenges})。\n"
        f"需求：\n---\n{request}\n---\n\n目标不是“选一个”，而是合成比任何单一方案都强的最终方案：\n"
        "1. 先评后合：用统一维度(正确性/健壮性/性能/复杂度/可维护性/可用性)简评每份，先评估再下结论。\n"
        "2. 逐维度取长：每维度挑最好做法并说明，按维度拼装，不整体二选一。\n"
        "3. 显式裁决冲突：互斥处(如简洁vs健壮)明确选哪边、代价、何时反过来。\n"
        "4. 同等吸收两组评审专家的修复建议，不要因为篇幅或轮次先后而偏重某一组。\n"
        "5. 产出最终 markdown：①融合方案 ②各从哪吸收了什么 ③仍存在的已知取舍/风险。写充分可执行。\n"
        + SYNTHESIS_ADJUDICATION
        + PROGRESS_HEARTBEAT
    )


# ── review 模板 ───────────────────────────────────────────────────────────────
def review_task(request, doc_file, perspective_hint=""):
    return (
        "你是独立评审者。context / 工作目录里有待评审文档 "
        f"{doc_file}。请仔细比较/评审其内容。\n"
        f"评审要求：\n---\n{request}\n---\n\n"
        f"{('评审视角：' + perspective_hint + '。') if perspective_hint else ''}"
        "给出结构化评审 markdown：覆盖度/合理性/优缺点/关键问题/明确结论与依据。"
        "证据要可对照原文，结论不要笼统。\n" + PROGRESS_HEARTBEAT + SELF_REFLECTION_CHECKLIST
    )


def review_judge_task(request, review_files, min_score):
    files = "、".join(review_files)
    return (
        f"你是审查员。context 里有若干份评审({files})与原始评审要求。审查这些评审的质量"
        "(分析深度、是否抓住关键、结论是否有据)。\n"
        f"评审要求背景：\n---\n{request}\n---\n\n"
        + review_checklist("research") + "\n"
        + verdict_directive(
            has_score=True, criterion_label="评审维度", feedback_audience="评审者",
            min_score=min_score, blocking_note="passed=False 时非空",
        )
    )


# ── arena 模板 ────────────────────────────────────────────────────────────────
def arena_solution_task(request, solution_label, version, challenge_file=None, prev_file=None):
    head = (
        f"你是方案竞争者【{solution_label}】。为以下需求给出一份有竞争力的完整方案。\n"
        f"需求：\n---\n{request}\n---\n\n"
    )
    if version == 1:
        body = "给出完整方案：核心思路、关键决策、为什么、风险。力求比对手更强。\n"
    else:
        body = (
            f"这是第 {version} 版。context 里有评审对上一版的反馈 {challenge_file} 和你的上一版 {prev_file}。\n"
            "针对反馈改进，提高方案得分。\n"
        )
    return head + body + "\n" + PROGRESS_HEARTBEAT + SELF_REFLECTION_CHECKLIST


def arena_judge_task(request, solution_files, min_score):
    files = "、".join(solution_files)
    return (
        f"你是评审裁判。context 里有 N 份竞争方案({files})。\n"
        f"需求：\n---\n{request}\n---\n\n"
        + review_checklist("design") + "\n"
        "比较所有方案，选出最佳并给【整体最高分方案的分数】。\n"
        + verdict_directive(
            has_score=True, score_meaning="最佳方案的得分", criterion_label="方案名",
            feedback_audience="所有竞争者", min_score=min_score,
            blocking_note="若 score<阈值, 列出最佳方案仍需改进的点",
        )
    )


# ── pipeline 模板 ─────────────────────────────────────────────────────────────
def plan_task(request):
    return (
        "你是规划者。把下面的需求拆解成清晰的实现计划(步骤、模块、接口、关键决策、风险)。\n"
        f"需求：\n---\n{request}\n---\n\n写成结构化 markdown 计划，供实现者据此编码。\n"
        + PROGRESS_HEARTBEAT + production_dimensions("design") + "\n" + SELF_REFLECTION_CHECKLIST
    )


def impl_task(request, plan_file):
    return (
        f"你是实现者(代码编写)。context 里有实现计划 {plan_file}。按计划实现代码。\n"
        f"需求：\n---\n{request}\n---\n\n"
        "把完整实现(代码 + 必要说明)写入你的输出文件。代码要可运行、有错误处理。\n"
        + PROGRESS_HEARTBEAT + production_dimensions("code") + "\n" + SELF_REFLECTION_CHECKLIST
    )


def pipeline_review_task(request, impl_file, min_score=7):
    del min_score  # 保留参数是历史签名，pipeline_review_task 走 bool passed，未消费阈值。
    return (
        f"你是验收审查者。context 里有实现产物 {impl_file} 和原始需求。审查实现是否满足需求、"
        "代码是否正确健壮。\n"
        f"需求：\n---\n{request}\n---\n\n"
        + review_checklist("code") + "\n"
        + verdict_directive(
            has_score=False, criterion_label="审查维度", feedback_audience="实现者",
            blocking_note="未通过项, 非空则 passed=false",
        )
    )


# ── scatter_gather 模板 ───────────────────────────────────────────────────────
def explore_task(request, direction_hint):
    return (
        f"你是调研者，负责方向：{direction_hint}。围绕需求做这个方向的深入调研。\n"
        f"需求：\n---\n{request}\n---\n\n"
        "写成结构化 markdown 调研结论(发现、依据、对该方向的判断)。若能联网请联网查证。\n"
        + PROGRESS_HEARTBEAT + production_dimensions("research") + "\n" + SELF_REFLECTION_CHECKLIST
    )


def gather_task(request, explore_files):
    files = "、".join(explore_files)
    return (
        f"你是汇总者。context 里有多路调研结论({files})。把它们汇总成一份连贯的综合结论。\n"
        f"需求：\n---\n{request}\n---\n\n"
        "要求：覆盖所有调研输入、消解冲突、给出整体结论与依据。写成 markdown。\n"
        + PROGRESS_HEARTBEAT + production_dimensions("synthesis") + "\n" + SELF_REFLECTION_CHECKLIST
    )


def gather_review_task(request, gather_file):
    return (
        f"你是审查者。context 里有汇总结论 {gather_file} 和原始需求。审查汇总是否覆盖各路输入、"
        "结论是否有据。\n" + review_checklist("research") + "\n"
        + verdict_directive(has_score=False, criterion_label="审查维度", feedback_audience="调研者")
    )


__all__ = [
    "SELF_REFLECTION_CHECKLIST", "STANCE_LIBRARY",
    "PRODUCTION_DIMENSIONS", "PRODUCER_PITFALLS", "production_dimensions",
    "REVIEW_CHECKLISTS", "REVIEWER_PITFALLS", "review_checklist", "verdict_directive",
    "RED_TEAM_DIMENSIONS", "SYNTHESIS_ADJUDICATION", "USABILITY_DIMENSIONS",
    "research_task", "stance_selection_task", "design_task", "redteam_task",
    "usability_review_task", "synthesis_task",
    "review_task", "review_judge_task",
    "arena_solution_task", "arena_judge_task",
    "plan_task", "impl_task", "pipeline_review_task",
    "explore_task", "gather_task", "gather_review_task",
]
