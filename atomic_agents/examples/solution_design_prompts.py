"""Prompt library for the solution-design example orchestration.

Content distilled from researched best practices (real web search via codex/ducc):
- Self-Refine / Reflexion / Chain-of-Verification (arxiv 2303.17651 / 2303.11366 / 2309.11495)
- Multi-agent debate "Society of Minds" (arxiv 2305.14325) — tension comes from
  stance差异 itself, not flowery wording.
- Red-team self-critique (arxiv 2202.03286), Constitutional AI critique posture.
- Anthropic multi-agent: separation of concerns, explicit goals/format/boundary,
  stopping conditions.

These are plain f-string builders; the orchestration assembles them into atom tasks.
"""

from __future__ import annotations

# ── 层1：每个设计 agent 内部自反思清单（atom 内黑盒"想到满意"才交付）────────────
# 来源：Self-Refine 生成-反馈-改写 + Chain-of-Verification 草稿-验证问题-独立回答-修订
#       + Socratic / first-principles + red-team self-critique.
SELF_REFLECTION_CHECKLIST = """\
在写出最终方案前，先在心里静默做几轮自我反思与修订（不要展示完整推理过程，
只输出反思后定稿的方案）。反复逐条拷问自己，直到自己满意为止：

1. 我是否真正解决了需求的核心问题，而不是一个相近但更容易的问题？
2. 这个方案隐含了哪些假设？其中哪些没有依据、最可能是错的？若某条假设不成立会怎样？
3. 从第一性原理重新推导，这个方案是否仍然成立，还是只是在套常见模板？
4. 有没有更简单的解？我是不是把问题复杂化了？
5. 最坏情况 / 边界条件 / 失败场景是什么？方案在这些情况下会不会崩？
6. 一个严苛的专家或红队会怎样攻击这个方案？这些攻击点我是否已经处理或显式承认？
7. 我遗漏了什么？需求里真正关心的约束（成本、时间、风险、可维护性、可执行性）覆盖了吗？
8. 我有没有把"可能性"说成"确定性"，把经验建议说成通用规律？
9. 如果有检索到的最佳实践/现有方案，我是否参考并超越了它，而不是无视或简单复制？
10. 最能提升这个方案质量的一处修改是什么？先改掉它，再定稿。
"""

# ── 立场库：5 组互斥张力对（调研结论：张力来自优化互斥指标）────────────────────
STANCE_LIBRARY = [
    {"pair": "速度/简洁 ↔ 健壮/严谨",
     "a": {"name": "速度简洁派", "focus": "最小可行、尽快交付、避免过度设计"},
     "b": {"name": "健壮严谨派", "focus": "边界处理、错误恢复、长期可维护"}},
    {"pair": "创新/重构 ↔ 保守/兼容",
     "a": {"name": "创新重构派", "focus": "推翻重来、引入更优范式、不被现状束缚"},
     "b": {"name": "保守兼容派", "focus": "复用现有、最小改动、控制迁移风险"}},
    {"pair": "性能/规模 ↔ 可读/开发体验",
     "a": {"name": "性能规模派", "focus": "吞吐、延迟、成本、可扩展性"},
     "b": {"name": "可读体验派", "focus": "抽象清晰、易调试、开发者友好"}},
    {"pair": "乐观执行 ↔ 红队/失败",
     "a": {"name": "乐观执行派", "focus": "happy path、快速落地、先跑起来"},
     "b": {"name": "失败防御派", "focus": "专找边界、攻击面、故障模式与降级"}},
    {"pair": "用户/产品 ↔ 工程/约束",
     "a": {"name": "用户产品派", "focus": "体验、需求覆盖、产品价值"},
     "b": {"name": "工程约束派", "focus": "可行性、技术债、资源与时间约束"}},
]


def research_task(request: str) -> str:
    return (
        "你是检索调研者，必须【联网检索】（GitHub、官方文档、论文、知名工程博客）。\n"
        f"用户的设计需求是：\n---\n{request}\n---\n\n"
        "请调研：这个需求是否已经有成熟的最佳实践、官方推荐方案、或 GitHub 上的现成实现/库？\n"
        "重点产出：\n"
        "1. 是否已有公认最佳实践/标准做法（有则给出来源链接与要点；没有则明确说明）。\n"
        "2. 相关的官方文档 / 知名开源实现 / 设计模式（带链接）。\n"
        "3. 这些现有方案的关键设计取舍、已知坑、适用边界。\n"
        "4. 对后续方案设计的启示（哪些可直接采用、哪些需要针对本需求改良）。\n"
        "写成一份结构化 markdown 调研报告，务必先联网查证，不要只凭记忆。简洁、可落地、标来源。"
    )


def stance_selection_task(request: str, max_stances: int) -> str:
    pairs = "\n".join(f"  - {s['pair']}：{s['a']['name']}({s['a']['focus']}) vs {s['b']['name']}({s['b']['focus']})"
                      for s in STANCE_LIBRARY)
    return (
        "你是方案设计的元编排者。先读 context 里的调研报告 research.md，再读下面的需求，"
        "然后决定派几个【立场不同的设计者】并行出方案。\n"
        f"需求：\n---\n{request}\n---\n\n"
        "立场设计原则（来自多 agent 辩论最佳实践）：张力来自【优化互斥指标】，不是让他们吵架。"
        "从以下互斥张力对里挑选最贴合本需求的立场（也可微调命名/聚焦点）：\n"
        f"{pairs}\n\n"
        f"请输出一个 JSON 数组（2 到 {max_stances} 个立场），每个元素：\n"
        '{"stance_name": "立场名", "focus": "这个立场重点优化什么", '
        '"prompt_hint": "给这个立场设计者的一句话视角提示"}\n'
        "只输出这个 JSON 数组，不要其他文字、不要 markdown 代码围栏。"
        "选 2-3 个通常最有效；只有需求确实多维复杂才用更多。"
    )


def design_task(request: str, stance: dict, research_file: str, version: int,
                challenge_file: str | None = None, prev_version_file: str | None = None) -> str:
    head = (
        f"你是【{stance['stance_name']}】设计者，视角：{stance.get('focus', '')}。"
        f"{stance.get('prompt_hint', '')}\n"
        f"用户需求：\n---\n{request}\n---\n\n"
        f"请基于你的立场为该需求设计一份方案。context 里有调研报告 {research_file}，"
        "请参考其中的最佳实践并在此之上改良（不要无视，也不要简单照抄）。\n"
    )
    if version == 1:
        body = "这是第一版方案。给出完整设计：核心思路、关键决策、为什么这么做、关键风险。\n"
    else:
        body = (
            f"这是第 {version} 版。context 里有红队对上一版的质询 {challenge_file} 和你的上一版 {prev_version_file}。\n"
            "请【认真回应红队质询】：被指出的致命问题必须修复或显式反驳并说明理由；"
            "可修补的缺陷尽量改进。坚持你的立场，但要让方案更强。给出修订后的完整方案。\n"
        )
    return head + body + "\n" + SELF_REFLECTION_CHECKLIST


def redteam_task(request: str, design_files: list[str], round_no: int) -> str:
    files = "、".join(design_files)
    return (
        "你是红队挑战者。context 里有若干份针对同一需求的设计方案（"
        f"{files}）。你的任务不是否定，而是用最强的反方论证压力测试每一份方案，逼出最脆弱的环节。\n"
        f"需求：\n---\n{request}\n---\n\n"
        "规则（务必遵守）：\n"
        "1. 攻击前提而非结论：先列出每份方案隐含的未经验证假设，逐个问“如果这条不成立会怎样”。\n"
        "2. 每个质询必须【可证伪 + 可落地】：给出具体失败场景（输入/边界/规模）、触发条件、"
        "最小复现路径。禁止“可能有风险”这类空话。\n"
        "3. 区分严重度：标记 [致命/会推翻方案] vs [可修补/局部缺陷]，不要平铺。\n"
        "4. 建设性收尾：对每个致命问题，给出至少一个修复方向或更优替代，并说明修复后的残留代价。\n"
        "5. 自我设限：如果某方案找不到致命问题，明确说“未发现颠覆性缺陷”，不要为凑数制造伪问题。\n"
        f"这是第 {round_no} 轮质询。按方案分节输出质询报告（markdown）。"
    )


def synthesis_task(request: str, final_design_files: list[str], challenge_files: list[str]) -> str:
    designs = "、".join(final_design_files)
    challenges = "、".join(challenge_files)
    return (
        "你是方案合成者。context 里有 N 份不同立场的最终方案（"
        f"{designs}）以及红队两轮质询（{challenges}）。\n"
        f"需求：\n---\n{request}\n---\n\n"
        "你的目标不是“选出最好的一个”，而是合成一个比任何单一方案都更强的最终方案。步骤：\n"
        "1. 先评后合：用统一维度（正确性/健壮性/性能/复杂度/可维护性）简评每份方案，先写评估再下结论。\n"
        "2. 逐维度取长：对每个维度，挑表现最好的方案的做法并说明为什么；按维度拼装，不要整体二选一。\n"
        "3. 显式裁决冲突：当两方案在某维度互斥（如简洁 vs 健壮），明确选哪边、代价是什么、"
        "什么条件下应反过来选。\n"
        "4. 吸收红队修复：把红队指出的致命问题对应的修复合并进最终方案。\n"
        "5. 产出最终 markdown：① 融合后的完整方案 ② 它从每个来源各吸收了什么 ③ 仍存在的已知取舍/风险。\n"
        "这是交付给用户的最终方案，请写充分、可执行。"
    )


__all__ = [
    "SELF_REFLECTION_CHECKLIST", "STANCE_LIBRARY",
    "research_task", "stance_selection_task", "design_task", "redteam_task", "synthesis_task",
]
