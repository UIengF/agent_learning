from __future__ import annotations

import os
from pathlib import Path

from examples.orchestrations import prompts
from examples.orchestrations.common import ensure_workspace


def test_ensure_workspace_normalizes_relative_path_to_absolute(tmp_path: Path, monkeypatch) -> None:
    # 相对 --workspace 会被 adapter 在子进程 cwd 下二次解析致 codex/ducc 秒挂；
    # ensure_workspace 必须归一化为绝对路径。
    monkeypatch.chdir(tmp_path)

    ws = ensure_workspace("./rel-ws/sub", "x-")

    assert os.path.isabs(ws)
    assert Path(ws).is_dir()
    assert Path(ws).resolve() == (tmp_path / "rel-ws" / "sub").resolve()


def test_ensure_workspace_default_tempdir_is_absolute() -> None:
    ws = ensure_workspace(None, "atomic-test-")
    assert os.path.isabs(ws)
    assert Path(ws).is_dir()


_HEARTBEAT_MARKER = "防空闲超时"


def test_long_running_writer_prompts_carry_progress_heartbeat() -> None:
    # 非 design 模板的写产出节点过去没有 _progress.md 心跳，慢 codex 节点会被 idle 误杀。
    # 现在所有【写产出】prompt 统一注入 PROGRESS_HEARTBEAT。
    writer_prompts = [
        prompts.research_task("x"),
        prompts.research_merge_task("x", ["a.md"]),
        prompts.design_task("x", {"stance_name": "s"}, "r.md", 1),
        prompts.redteam_task("x", ["a.md"], 1),
        prompts.synthesis_task("x", ["a.md"], ["c.md"]),
        prompts.review_task("x", "doc.md"),
        prompts.arena_solution_task("x", "方案1", 1),
        prompts.plan_task("x"),
        prompts.impl_task("x", "plan.md"),
        prompts.explore_task("x", "方向"),
        prompts.gather_task("x", ["e.md"]),
    ]
    for prompt in writer_prompts:
        assert _HEARTBEAT_MARKER in prompt


def test_short_judge_prompts_do_not_carry_heartbeat() -> None:
    # 裁判/验收节点只写短 JSON verdict，不做长任务，不需要心跳（避免噪声）。
    judge_prompts = [
        prompts.review_judge_task("x", ["r.md"], 7),
        prompts.arena_judge_task("x", ["s.md"], 7),
        prompts.pipeline_review_task("x", "impl.md"),
        prompts.gather_review_task("x", "g.md"),
    ]
    for prompt in judge_prompts:
        assert _HEARTBEAT_MARKER not in prompt
