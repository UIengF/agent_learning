"""design: 3 路并行检索 → 汇总 → ducc 动态选立场 → N 立场并行 → 红队+可用性并行质询 2 轮 → 合成."""

from __future__ import annotations

from .common import (
    OrchestrationResult, ensure_workspace, extract_json_array, get_adapter,
    invoke_atom, make_skeleton, merge_role_runners, run_orchestration, write_node,
)
from . import prompts

_STANCE_RUNNERS = ("ducc", "codex", "codex", "codex")
_RESEARCH_RUNNERS = ("codex", "codex", "ducc")


def _select_stances(request: str, workspace: str, n_stances: int, runner: str = "ducc") -> list[dict]:
    adapter = get_adapter(runner)
    result = invoke_atom(adapter, prompts.stance_selection_task(request, n_stances), workspace,
                         "orchestrator", context_files=["research.md"], timeout_sec=1800)
    try:
        raw = extract_json_array(result.result)
        stances = [_normalize(s, i) for i, s in enumerate(raw)]
        if len(stances) >= 2:
            return stances[:n_stances]
    except (ValueError, Exception):
        pass
    return _fallback(n_stances)


def _normalize(s: object, i: int) -> dict:
    if not isinstance(s, dict):
        return {"stance_name": f"立场{i+1}", "focus": "独立视角", "prompt_hint": "保持视角一致。"}
    return {
        "stance_name": str(s.get("stance_name") or s.get("name") or f"立场{i+1}"),
        "focus": str(s.get("focus") or "独立视角"),
        "prompt_hint": str(s.get("prompt_hint") or s.get("hint") or s.get("focus") or ""),
    }


def _fallback(n: int) -> list[dict]:
    out = []
    for pair in prompts.STANCE_LIBRARY:
        for side in ("a", "b"):
            e = pair[side]
            out.append({"stance_name": e["name"], "focus": e["focus"],
                        "prompt_hint": f"优先从“{e['focus']}”角度设计。"})
            if len(out) >= max(2, n):
                return out[:max(2, n)]
    return out[:max(2, n)]


def build_research_skeleton(request: str, n_researchers: int = 3):
    if n_researchers < 1:
        raise ValueError("n_researchers must be >= 1")
    research_files = [f"research-{i}.md" for i in range(n_researchers)]
    nodes = [
        write_node(
            node_id=f"research_{i}",
            role="researcher",
            task=prompts.research_task(request),
            output_file=output_file,
        )
        for i, output_file in enumerate(research_files)
    ]
    nodes.append(write_node(
        node_id="research_merge",
        role="researcher",
        task=prompts.research_merge_task(request, research_files),
        output_file="research.md",
        depends_on=[f"research_{i}" for i in range(n_researchers)],
        inputs=[{"from": f"research_{i}", "field": "output_file"} for i in range(n_researchers)],
    ))
    return make_skeleton("design-research", nodes)


def research_runner_plan(n_researchers: int = 3) -> dict[str, str]:
    plan = {
        f"research_{i}": _RESEARCH_RUNNERS[min(i, len(_RESEARCH_RUNNERS) - 1)]
        for i in range(n_researchers)
    }
    plan["research_merge"] = "ducc"
    return plan


def _default_atom_runners(skeleton, defaults: dict[str, str], overrides: dict | None) -> dict[str, str]:
    """Keep built-in routing except for roles explicitly supplied by the caller."""
    overridden_roles = {role.strip().lower() for role in (overrides or {})}
    node_roles = {node.id: node.role.strip().lower() for node in skeleton.nodes}
    return {
        atom_id: runner
        for atom_id, runner in defaults.items()
        if node_roles.get(atom_id) not in overridden_roles
    }


def design_runner_plan(stances: list[dict]) -> dict[str, str]:
    plan: dict[str, str] = {}
    for i in range(len(stances)):
        runner = _STANCE_RUNNERS[i] if i < len(_STANCE_RUNNERS) else _STANCE_RUNNERS[-1]
        for version in (1, 2, 3):
            plan[f"design_{i}_v{version}"] = runner
    plan.update({
        # 每轮质询两个视角交叉分配 runner：同一轮两个视角用不同 runner，避免同一个
        # runner 连续跑两个角色可能带来的思路惯性；两轮之间也互换，避免固定偏向。
        "redteam_r1": "codex", "usability_r1": "ducc",
        "redteam_r2": "ducc", "usability_r2": "codex",
        "synth": "ducc",
    })
    return plan


def build_skeleton(request: str, stances: list[dict], research_file: str):
    nodes = []
    v1 = [f"design-stance{i}-v1.md" for i in range(len(stances))]
    for i, st in enumerate(stances):
        nodes.append(write_node(node_id=f"design_{i}_v1", role="designer",
            task=prompts.design_task(request, st, research_file, 1), output_file=v1[i],
            context_files=[research_file]))

    r1_deps = [f"design_{i}_v1" for i in range(len(stances))]
    r1_inputs = [{"from": node_id, "field": "output_file"} for node_id in r1_deps]
    nodes.append(write_node(node_id="redteam_r1", role="critic",
        task=prompts.redteam_task(request, v1, 1), output_file="challenge-r1.md",
        depends_on=r1_deps, inputs=r1_inputs))
    nodes.append(write_node(node_id="usability_r1", role="critic",
        task=prompts.usability_review_task(request, v1, 1), output_file="usability-r1.md",
        depends_on=r1_deps, inputs=r1_inputs))

    v1_challenge_files = ["challenge-r1.md", "usability-r1.md"]
    v2 = [f"design-stance{i}-v2.md" for i in range(len(stances))]
    for i, st in enumerate(stances):
        nodes.append(write_node(node_id=f"design_{i}_v2", role="designer",
            task=prompts.design_task(request, st, research_file, 2, v1_challenge_files, v1[i]),
            output_file=v2[i], depends_on=["redteam_r1", "usability_r1"],
            inputs=[{"from": "redteam_r1", "field": "output_file"},
                    {"from": "usability_r1", "field": "output_file"},
                    {"from": f"design_{i}_v1", "field": "output_file"}],
            context_files=[research_file]))

    r2_deps = [f"design_{i}_v2" for i in range(len(stances))]
    r2_inputs = [{"from": node_id, "field": "output_file"} for node_id in r2_deps]
    nodes.append(write_node(node_id="redteam_r2", role="critic",
        task=prompts.redteam_task(request, v2, 2), output_file="challenge-r2.md",
        depends_on=r2_deps, inputs=r2_inputs))
    nodes.append(write_node(node_id="usability_r2", role="critic",
        task=prompts.usability_review_task(request, v2, 2), output_file="usability-r2.md",
        depends_on=r2_deps, inputs=r2_inputs))

    v2_challenge_files = ["challenge-r2.md", "usability-r2.md"]
    v3 = [f"design-stance{i}-v3.md" for i in range(len(stances))]
    for i, st in enumerate(stances):
        nodes.append(write_node(node_id=f"design_{i}_v3", role="designer",
            task=prompts.design_task(request, st, research_file, 3, v2_challenge_files, v2[i]),
            output_file=v3[i], depends_on=["redteam_r2", "usability_r2"],
            inputs=[{"from": "redteam_r2", "field": "output_file"},
                    {"from": "usability_r2", "field": "output_file"},
                    {"from": f"design_{i}_v2", "field": "output_file"}],
            context_files=[research_file]))

    nodes.append(write_node(node_id="synth", role="synthesizer",
        task=prompts.synthesis_task(request, v3, ["challenge-r1.md", "usability-r1.md",
                                                   "challenge-r2.md", "usability-r2.md"]),
        output_file="final-solution.md",
        depends_on=[f"design_{i}_v3" for i in range(len(stances))] + ["redteam_r2", "usability_r2"],
        inputs=[{"from": f"design_{i}_v3", "field": "output_file"} for i in range(len(stances))]
               + [{"from": "redteam_r2", "field": "output_file"},
                  {"from": "usability_r2", "field": "output_file"}]))
    return make_skeleton("design", nodes)


def design(request: str, *, n_stances: int = 3, n_researchers: int = 3,
           role_runners: dict | None = None,
           workspace: str | None = None) -> OrchestrationResult:
    if n_stances < 2:
        raise ValueError("n_stances must be >= 2")
    if n_researchers < 1:
        raise ValueError("n_researchers must be >= 1")
    ws = ensure_workspace(workspace, "atomic-design-")
    merged_role_runners = merge_role_runners(role_runners)

    research_skeleton = build_research_skeleton(request, n_researchers)
    research_result = run_orchestration(
        "design-research",
        request,
        research_skeleton,
        merged_role_runners,
        ws,
        final_file="research.md",
        explicit_atom_runners=_default_atom_runners(
            research_skeleton, research_runner_plan(n_researchers), role_runners
        ),
    )
    if not research_result.succeeded:
        research_result.template = "design"
        research_result.final_path = None
        research_result.extra = {
            "stances": [],
            "research_succeeded": False,
            "research_run_result": research_result.run_result,
        }
        return research_result

    orchestrator_runner = merged_role_runners["orchestrator"]
    if not isinstance(orchestrator_runner, str):
        orchestrator_runner = str(orchestrator_runner[0])
    stances = _select_stances(request, ws, n_stances, orchestrator_runner)
    print("design stances: " + ", ".join(s["stance_name"] for s in stances))
    skeleton = build_skeleton(request, stances, "research.md")
    design_result = run_orchestration(
        "design",
        request,
        skeleton,
        merged_role_runners,
        ws,
        final_file="final-solution.md",
        extra={"stances": stances},
        explicit_atom_runners=_default_atom_runners(
            skeleton, design_runner_plan(stances), role_runners
        ),
    )
    design_result.succeeded = research_result.succeeded and design_result.succeeded
    design_result.atom_runner_plan = {
        **research_result.atom_runner_plan,
        **design_result.atom_runner_plan,
    }
    design_result.extra = {
        **design_result.extra,
        "research_succeeded": research_result.succeeded,
        "research_run_result": research_result.run_result,
    }
    return design_result


__all__ = [
    "design",
    "build_research_skeleton",
    "build_skeleton",
    "research_runner_plan",
    "design_runner_plan",
]
