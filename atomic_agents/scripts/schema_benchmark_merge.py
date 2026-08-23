"""补跑：把已落盘的 N 份 research-*.md 合并成 research-report.md。

阶段1 research_3（ducc）因网关 'Upstream body read failed' 失败触发止损，
research_merge 被 blocked。瞬时错误识别已修；这里只补跑 merge 单节点，
复用已成功落盘的调研（research-0/1/2/4）。
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for p in (str(REPO_ROOT / "src"), str(REPO_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from examples.orchestrations.common import (  # noqa: E402
    ensure_workspace, make_skeleton, merge_role_runners, run_orchestration, write_node,
)
from examples.orchestrations.prompts import PROGRESS_HEARTBEAT  # noqa: E402
from scripts.schema_benchmark_orch import BACKGROUND, HARD_CONSTRAINTS, WORKSPACE  # noqa: E402


def _merge_task(research_files: list[str]) -> str:
    files = "、".join(research_files)
    return (
        f"你是调研汇总者。context 里有 {len(research_files)} 份独立联网调研：{files}，"
        "分别覆盖：GraphRAG/KG 评测方法（指标三层）、schema/数据管线回归基准（golden dataset/"
        "snapshot/契约测试）、LLM 抽取的 schema 解耦/中间事实表示层、GraphRAG 召回层评测细化"
        "（含金标准省力法 LLM-as-judge / 合成 QA / active-learning 散见各份）。\n\n"
        f"{BACKGROUND}\n{HARD_CONSTRAINTS}\n"
        "把这些调研去重、消解冲突，合并成一份结构化 markdown 调研报告 research-report.md。要求：\n"
        "1. 保留【所有有价值的来源链接】，链接放在对应结论附近。\n"
        "2. 按主题组织：①三层评测指标（抽取/召回/端到端，给具体定义与公式）②回归基准范式"
        "（golden schema/图 + 相对 diff）③中间表示与抽取解耦（Fact Store 思路、收益与坑）"
        "④金标准省力法（小样本标注 + LLM-judge 扩充/校准）⑤召回层评测（qrels/子图/路径指标）"
        "⑥query 集构造。多份结论一致时合并表达。\n"
        "3. 冲突或分歧显式标注：分歧点、各自依据、你的取舍判断与不确定性。\n"
        "4. 区分来源类型（官方文档/论文/开源实现/工程博客）说明可信度与适用边界。\n"
        "5. 末尾输出【对本场景设计的启示】总表：哪些做法可直接采用、哪些需针对本需求改良、"
        "哪些风险必须保留——后续方案设计会直接基于它工作。\n"
        "只输出合并后的 markdown，不要遗漏来源链接。"
        + PROGRESS_HEARTBEAT
    )


def main() -> bool:
    ws = ensure_workspace(WORKSPACE, "schema-benchmark-")
    # 只取真实落盘且非空的调研文件
    present = [f"research-{i}.md" for i in (0, 1, 2, 4)
               if (Path(ws) / f"research-{i}.md").exists()]
    print(f"=== MERGE START ws={ws} inputs={present} ===", flush=True)

    node = write_node(
        node_id="research_merge", role="researcher", task=_merge_task(present),
        output_file="research-report.md",
    )
    # merge 单节点无上游 DAG 依赖（上游已落盘在 workspace 根，runner 以 cwd=workspace
    # 启动，可直接读取）。prompt 已显式列出文件清单。

    skeleton = make_skeleton("schema-benchmark-research-merge", [node])
    result = run_orchestration(
        "schema-benchmark-research-merge", BACKGROUND, skeleton,
        merge_role_runners(None), ws,
        final_file="research-report.md",
        explicit_atom_runners={"research_merge": "ducc"},
    )
    print(f"=== MERGE DONE succeeded={result.succeeded} ===", flush=True)
    print("final_path: " + str(result.final_path), flush=True)
    if result.run_result is not None:
        print("lock_dir: " + result.run_result.lock_dir, flush=True)
        print("failed_nodes: " + ", ".join(result.run_result.failed_nodes), flush=True)
    return result.succeeded


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
