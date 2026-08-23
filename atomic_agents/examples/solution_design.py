"""Solution-design orchestration example for atomic-agents.

This example intentionally keeps all review-like work as regular writer atoms:
red-team and synthesis nodes produce Markdown reports, not scheduler verdicts.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

EXAMPLES_ROOT = Path(__file__).resolve().parent
REPO_ROOT = EXAMPLES_ROOT.parents[0]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from atomic_agents.adapters import RunnerAdapter
from atomic_agents.adapters.ducc import DuccAdapter
from atomic_agents.models import AtomContract, AtomResult, InputRef, Skeleton, SkeletonNode
from atomic_agents.run import RunResult, run_skeleton
from solution_design_prompts import (
    STANCE_LIBRARY,
    design_task,
    redteam_task,
    research_task,
    stance_selection_task,
    synthesis_task,
)


class RoleRoutingAdapter:
    """Route atom contracts by logical role while satisfying RunnerAdapter."""

    def __init__(self, routes: dict[str, RunnerAdapter], default: RunnerAdapter) -> None:
        self.routes = {role.strip().lower(): adapter for role, adapter in routes.items()}
        self.default = default
        self.feature_profile = default.feature_profile

    def invoke(self, contract: AtomContract, timeout_sec: int) -> AtomResult:
        role = contract.logical_role.strip().lower()
        adapter = self.routes.get(role, self.default)
        return adapter.invoke(contract, timeout_sec)


def _extract_json_array(text: str) -> list:
    """Extract a JSON array from tolerant LLM output."""

    stripped = text.strip()
    fence_match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        stripped = fence_match.group(1).strip()

    start = stripped.find("[")
    end = stripped.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("missing JSON array")

    try:
        value = json.loads(stripped[start : end + 1])
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON array: {exc}") from exc

    if not isinstance(value, list):
        raise ValueError("JSON value is not an array")
    return value


@dataclass(kw_only=True)
class SolutionDesignResult:
    request: str
    workspace: str
    stances: list[dict]
    final_solution_path: str | None
    artifacts: list[str]
    run_result: RunResult | None
    succeeded: bool


def _invoke_readonly(
    adapter: RunnerAdapter,
    task: str,
    workspace: str,
    role: str,
    output_file: str = "",
    timeout_sec: int = 1800,
    context_files: list[str] | None = None,
) -> AtomResult:
    timestamp = datetime.now(timezone.utc).isoformat()
    contract = AtomContract(
        task=task,
        inputs=[],
        context_files=list(context_files or []),
        workspace=workspace,
        read_only=not bool(output_file),
        write_scope=[output_file] if output_file else [],
        required_capabilities=["write_files"] if output_file else [],
        status="success",
        result="",
        artifacts=[],
        output_file=output_file,
        output_schema_ref=None,
        handoff={"completed": [], "pending": [], "decisions": [], "risks": []},
        consult=None,
        atom_id=role,
        correlation_id="solution-design-preflight",
        logical_role=role,
        resolved_runner=None,
        session_id=None,
        hop_count=1,
        limits={"max_cost": 3.0, "timeout_sec": timeout_sec, "max_internal_turns": 20},
        cost=0.0,
        duration_sec=0.0,
        timestamps={"started_at": timestamp, "finished_at": timestamp},
    )
    return adapter.invoke(contract, timeout_sec)


def _research(adapter: RunnerAdapter, request: str, workspace: str) -> str:
    result = _invoke_readonly(
        adapter,
        research_task(request),
        workspace,
        "researcher",
        output_file="research.md",
        timeout_sec=1800,
    )
    research_path = Path(workspace) / "research.md"
    if not research_path.exists():
        research_path.parent.mkdir(parents=True, exist_ok=True)
        fallback = result.result or result.error or "Research adapter did not return content."
        research_path.write_text(fallback, encoding="utf-8")
    return "research.md"


def _select_stances(adapter: RunnerAdapter, request: str, workspace: str, max_stances: int) -> list[dict]:
    result = _invoke_readonly(
        adapter,
        stance_selection_task(request, max_stances),
        workspace,
        "stance_selector",
        context_files=["research.md"],
        timeout_sec=1800,
    )
    try:
        raw_stances = _extract_json_array(result.result)
        if not 2 <= len(raw_stances) <= max_stances:
            raise ValueError(f"expected 2..{max_stances} stances, got {len(raw_stances)}")
        return [_normalize_stance(stance, index) for index, stance in enumerate(raw_stances)]
    except ValueError:
        return _fallback_stances(max_stances)


def design_solution(
    request: str,
    *,
    workspace: str | None = None,
    research_adapter: RunnerAdapter | None = None,
    design_adapter: RunnerAdapter | None = None,
    redteam_adapter: RunnerAdapter | None = None,
    synth_adapter: RunnerAdapter | None = None,
    max_stances: int = 5,
    rounds: int = 2,
) -> SolutionDesignResult:
    if rounds != 2:
        raise NotImplementedError("solution_design currently uses a fixed 2-round red-team skeleton")
    if max_stances < 2:
        raise ValueError("max_stances must be at least 2")

    resolved_workspace = workspace or tempfile.mkdtemp(prefix="solution-design-")
    Path(resolved_workspace).mkdir(parents=True, exist_ok=True)

    research_runner = research_adapter or DuccAdapter()
    design_runner = design_adapter or DuccAdapter()
    redteam_runner = redteam_adapter or DuccAdapter()
    synth_runner = synth_adapter or DuccAdapter()
    default_runner = DuccAdapter()

    research_md = _research(research_runner, request, resolved_workspace)
    stances = _select_stances(research_runner, request, resolved_workspace, max_stances)
    print("Selected stances: " + ", ".join(stance["stance_name"] for stance in stances))

    skeleton = _build_solution_skeleton(request, stances, research_md)
    adapter = RoleRoutingAdapter(
        {
            "designer": design_runner,
            "redteam": redteam_runner,
            "synthesizer": synth_runner,
        },
        default=default_runner,
    )

    locks_dir = tempfile.mkdtemp(prefix="solution-design-locks-")
    run_result = run_skeleton(skeleton, adapter, workspace=resolved_workspace, lock_base_dir=locks_dir)

    artifacts = sorted(
        path.relative_to(resolved_workspace).as_posix()
        for path in Path(resolved_workspace).rglob("*.md")
        if path.is_file()
    )
    final_path = Path(resolved_workspace) / "final-solution.md"
    return SolutionDesignResult(
        request=request,
        workspace=resolved_workspace,
        stances=stances,
        final_solution_path=str(final_path) if final_path.exists() else None,
        artifacts=artifacts,
        run_result=run_result,
        succeeded=run_result.succeeded and final_path.exists(),
    )


def _build_solution_skeleton(request: str, stances: list[dict], research_file: str) -> Skeleton:
    nodes: list[SkeletonNode] = []

    v1_files = []
    for index, stance in enumerate(stances):
        output_file = f"design-stance{index}-v1.md"
        v1_files.append(output_file)
        nodes.append(
            _write_node(
                node_id=f"design_{index}_v1",
                role="designer",
                task=design_task(request, stance, research_file, 1),
                output_file=output_file,
            )
        )

    nodes.append(
        _write_node(
            node_id="redteam_r1",
            role="redteam",
            task=redteam_task(request, v1_files, 1),
            output_file="challenge-r1.md",
            depends_on=[f"design_{index}_v1" for index in range(len(stances))],
            inputs=[{"from": f"design_{index}_v1", "field": "output_file"} for index in range(len(stances))],
        )
    )

    v2_files = []
    for index, stance in enumerate(stances):
        prev_file = f"design-stance{index}-v1.md"
        output_file = f"design-stance{index}-v2.md"
        v2_files.append(output_file)
        nodes.append(
            _write_node(
                node_id=f"design_{index}_v2",
                role="designer",
                task=design_task(
                    request,
                    stance,
                    research_file,
                    2,
                    challenge_file="challenge-r1.md",
                    prev_version_file=prev_file,
                ),
                output_file=output_file,
                depends_on=["redteam_r1"],
                inputs=[
                    {"from": "redteam_r1", "field": "output_file"},
                    {"from": f"design_{index}_v1", "field": "output_file"},
                ],
            )
        )

    nodes.append(
        _write_node(
            node_id="redteam_r2",
            role="redteam",
            task=redteam_task(request, v2_files, 2),
            output_file="challenge-r2.md",
            depends_on=[f"design_{index}_v2" for index in range(len(stances))],
            inputs=[{"from": f"design_{index}_v2", "field": "output_file"} for index in range(len(stances))],
        )
    )

    v3_files = []
    for index, stance in enumerate(stances):
        prev_file = f"design-stance{index}-v2.md"
        output_file = f"design-stance{index}-v3.md"
        v3_files.append(output_file)
        nodes.append(
            _write_node(
                node_id=f"design_{index}_v3",
                role="designer",
                task=design_task(
                    request,
                    stance,
                    research_file,
                    3,
                    challenge_file="challenge-r2.md",
                    prev_version_file=prev_file,
                ),
                output_file=output_file,
                depends_on=["redteam_r2"],
                inputs=[
                    {"from": "redteam_r2", "field": "output_file"},
                    {"from": f"design_{index}_v2", "field": "output_file"},
                ],
            )
        )

    nodes.append(
        _write_node(
            node_id="synth",
            role="synthesizer",
            task=synthesis_task(request, v3_files, ["challenge-r1.md", "challenge-r2.md"]),
            output_file="final-solution.md",
            depends_on=[f"design_{index}_v3" for index in range(len(stances))] + ["redteam_r2"],
            inputs=[
                *[{"from": f"design_{index}_v3", "field": "output_file"} for index in range(len(stances))],
                {"from": "redteam_r2", "field": "output_file"},
            ],
        )
    )

    return Skeleton(
        name="solution-design",
        version=1,
        nodes=nodes,
        edges=_edges_from_dependencies(nodes),
        run_limits={"max_repair_attempts": 1, "max_total_cost": 1000.0},
        irreversible_ops=[],
    )


def _write_node(
    *,
    node_id: str,
    role: str,
    task: str,
    output_file: str,
    depends_on: list[str] | None = None,
    inputs: list[InputRef] | None = None,
) -> SkeletonNode:
    return SkeletonNode(
        id=node_id,
        role=role,
        task=task,
        depends_on=list(depends_on or []),
        inputs=list(inputs or []),
        write_scope=[output_file],
        read_only=False,
        required_capabilities=["write_files"],
        reviewer_criteria=[],
        output_file=output_file,
    )


def _edges_from_dependencies(nodes: list[SkeletonNode]) -> list[dict[str, str]]:
    edges: list[dict[str, str]] = []
    for node in nodes:
        for dependency in node.depends_on:
            edges.append({"from": dependency, "to": node.id})
    return edges


def _normalize_stance(stance: object, index: int) -> dict:
    if not isinstance(stance, dict):
        return {
            "stance_name": f"立场{index + 1}",
            "focus": "从一个独立视角优化方案",
            "prompt_hint": "保持该视角的一致性，同时吸收证据修正方案。",
        }
    name = str(stance.get("stance_name") or stance.get("name") or f"立场{index + 1}")
    focus = str(stance.get("focus") or "从一个独立视角优化方案")
    prompt_hint = str(stance.get("prompt_hint") or stance.get("hint") or focus)
    return {"stance_name": name, "focus": focus, "prompt_hint": prompt_hint}


def _fallback_stances(max_stances: int) -> list[dict]:
    stances = []
    for pair in STANCE_LIBRARY[:2]:
        for side in ("a", "b"):
            entry = pair[side]
            stances.append(
                {
                    "stance_name": entry["name"],
                    "focus": entry["focus"],
                    "prompt_hint": f"优先从“{entry['focus']}”角度设计。",
                }
            )
            if len(stances) >= min(max_stances, 2):
                return stances
    return stances


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    request = args[0] if args else "设计一个 Python 进程内的 LRU 缓存，支持 TTL 过期和线程安全"
    workspace = str(Path(__file__).resolve().parent / "solution-design-out")

    result = design_solution(request, workspace=workspace)

    print("\nStances:")
    for stance in result.stances:
        print(f"- {stance['stance_name']}: {stance['focus']}")
    print(f"Succeeded: {result.succeeded}")
    print("Artifacts:")
    for artifact in result.artifacts:
        print(f"- {artifact}")
    print(f"Final solution: {result.final_solution_path}")
    return 0 if result.succeeded else 1


if __name__ == "__main__":
    raise SystemExit(main())
