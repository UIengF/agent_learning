"""Shared building blocks for the named orchestration templates.

Extracted/generalized from the original solution_design example so all five
templates (design / review / arena / pipeline / scatter_gather) reuse one
role→runner router, one read-only invoke helper, and one run wrapper.
"""

from __future__ import annotations

import json
import re
import sys
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

EXAMPLES_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EXAMPLES_ROOT.parent
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from atomic_agents.adapters import RunnerAdapter
from atomic_agents.adapters.codex import CodexAdapter
from atomic_agents.adapters.ducc import DuccAdapter
from atomic_agents.models import AtomContract, AtomResult, InputRef, Skeleton, SkeletonNode
from atomic_agents.run import RunResult, run_skeleton


# ── runner registry ──────────────────────────────────────────────────────────
# Single source of truth for "runner name → adapter". codex/ducc both tested to
# go online and to write contract.output_file. Adapters are constructed lazily
# and cached so one orchestration reuses the same instances.
_ADAPTER_CACHE: dict[str, RunnerAdapter] = {}


def get_adapter(runner_name: str) -> RunnerAdapter:
    name = runner_name.strip().lower()
    if name not in _ADAPTER_CACHE:
        if name == "codex":
            _ADAPTER_CACHE[name] = CodexAdapter()
        elif name == "ducc":
            _ADAPTER_CACHE[name] = DuccAdapter()
        else:
            raise ValueError(f"unknown runner {runner_name!r} (expected 'codex' or 'ducc')")
    return _ADAPTER_CACHE[name]


# ── default role → runner policy (overridable via role_runners) ───────────────
# 用户原则: 代码编写一定 codex; 并行设计/审查 ducc+codex 结合(轮流); 编排侧 ducc。
# 值为单个 runner 名(该角色所有实例同一 runner) 或 list(多实例按 index 轮流)。
DEFAULT_ROLE_RUNNERS: dict[str, object] = {
    "implementer": "codex",        # 代码编写固定 codex
    "coder": "codex",
    "designer": ["ducc", "codex"], # 并行设计轮流
    "reviewer": ["ducc", "codex"], # 审查/评判轮流
    "judge": ["ducc", "codex"],
    "critic": ["ducc", "codex"],   # 红队/质询(审查侧)轮流
    "analyst": ["ducc", "codex"],  # 并行评审者(review 模板, 不卡关)轮流
    "researcher": "ducc",          # 编排侧
    "planner": "ducc",
    "synthesizer": "ducc",
    "orchestrator": "ducc",
    "explorer": ["ducc", "codex"], # 调研也可结合
}

_DEFAULT_RUNNER = "ducc"  # 未在表里的角色兜底


class RoleRoutingAdapter:
    """Route each atom to a runner by (role, per-role instance index).

    The scheduler calls invoke(contract). We pick the adapter by the contract's
    logical_role; when a role maps to a list of runners, atoms of that role are
    assigned round-robin by the order they were declared (resolved up-front in
    build_router via an explicit role→[runner-per-node] plan keyed by atom_id).
    """

    def __init__(self, atom_runner: dict[str, str], default: RunnerAdapter) -> None:
        # atom_runner: atom_id → runner_name (fully resolved up front)
        self._atom_runner = dict(atom_runner)
        self._default = default
        self.feature_profile = default.feature_profile

    def invoke(self, contract: AtomContract, timeout_sec: int) -> AtomResult:
        runner_name = self._atom_runner.get(contract.atom_id)
        adapter = get_adapter(runner_name) if runner_name else self._default
        return adapter.invoke(contract, timeout_sec)


def resolve_runner_for_role(role: str, instance_index: int, role_runners: dict[str, object]) -> str:
    """Resolve one node's runner from policy: single name, or round-robin list."""
    spec = role_runners.get(role.strip().lower())
    if spec is None:
        return _DEFAULT_RUNNER
    if isinstance(spec, str):
        return spec
    if isinstance(spec, (list, tuple)) and spec:
        return str(spec[instance_index % len(spec)])
    return _DEFAULT_RUNNER


def merge_role_runners(overrides: dict[str, object] | None) -> dict[str, object]:
    """Default policy with user overrides applied (override wins per role)."""
    merged = dict(DEFAULT_ROLE_RUNNERS)
    if overrides:
        for role, runner in overrides.items():
            merged[role.strip().lower()] = runner
    return merged


def build_atom_runner_plan(nodes: list[SkeletonNode], role_runners: dict[str, object]) -> dict[str, str]:
    """Assign each node a concrete runner, round-robin within each role."""
    role_counter: dict[str, int] = {}
    plan: dict[str, str] = {}
    for node in nodes:
        role = node.role.strip().lower()
        idx = role_counter.get(role, 0)
        plan[node.id] = resolve_runner_for_role(role, idx, role_runners)
        role_counter[role] = idx + 1
    return plan


# ── node + skeleton helpers ───────────────────────────────────────────────────
def write_node(
    *,
    node_id: str,
    role: str,
    task: str,
    output_file: str,
    depends_on: list[str] | None = None,
    inputs: list[InputRef] | None = None,
    context_files: list[str] | None = None,
    reviewer_criteria: list[str] | None = None,
) -> SkeletonNode:
    """A writer/reviewer node: read_only=False, writes its declared output_file."""
    return SkeletonNode(
        id=node_id,
        role=role,
        task=task,
        depends_on=list(depends_on or []),
        inputs=list(inputs or []),
        write_scope=[output_file],
        read_only=False,
        required_capabilities=["write_files"],
        reviewer_criteria=list(reviewer_criteria or []),
        output_file=output_file,
        context_files=list(context_files or []),
    )


def edges_from_dependencies(nodes: list[SkeletonNode]) -> list[dict[str, str]]:
    edges: list[dict[str, str]] = []
    for node in nodes:
        for dependency in node.depends_on:
            edges.append({"from": dependency, "to": node.id})
    return edges


def make_skeleton(
    name: str,
    nodes: list[SkeletonNode],
    *,
    max_repair_attempts: int = 1,
    max_total_cost: float = 1000.0,
) -> Skeleton:
    return Skeleton(
        name=name,
        version=1,
        nodes=nodes,
        edges=edges_from_dependencies(nodes),
        run_limits={"max_repair_attempts": max_repair_attempts, "max_total_cost": max_total_cost},
        irreversible_ops=[],
    )


# ── read-only / pre-skeleton invoke (research, stance selection) ──────────────
def invoke_atom(
    adapter: RunnerAdapter,
    task: str,
    workspace: str,
    role: str,
    *,
    output_file: str = "",
    context_files: list[str] | None = None,
    timeout_sec: int = 1800,
) -> AtomResult:
    """One-off atom outside run_skeleton (Phase 0/1 helpers)."""
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
        correlation_id="orchestration-preflight",
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


# ── tolerant JSON extraction ──────────────────────────────────────────────────
def _strip_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else stripped


def extract_json_array(text: str) -> list:
    stripped = _strip_fence(text)
    start, end = stripped.find("["), stripped.rfind("]")
    if start == -1 or end == -1 or end < start:
        raise ValueError("missing JSON array")
    value = json.loads(stripped[start : end + 1])
    if not isinstance(value, list):
        raise ValueError("not a JSON array")
    return value


def extract_json_object(text: str) -> dict:
    stripped = _strip_fence(text)
    start, end = stripped.find("{"), stripped.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("missing JSON object")
    value = json.loads(stripped[start : end + 1])
    if not isinstance(value, dict):
        raise ValueError("not a JSON object")
    return value


# ── run wrapper + result ──────────────────────────────────────────────────────
@dataclass(kw_only=True)
class OrchestrationResult:
    template: str
    request: str
    workspace: str
    succeeded: bool
    artifacts: list[str]
    final_path: str | None
    atom_runner_plan: dict[str, str] = field(default_factory=dict)
    run_result: RunResult | None = None
    extra: dict = field(default_factory=dict)


def run_orchestration(
    template: str,
    request: str,
    skeleton: Skeleton,
    role_runners: dict[str, object],
    workspace: str,
    *,
    final_file: str | None = None,
    extra: dict | None = None,
    explicit_atom_runners: dict[str, str] | None = None,
) -> OrchestrationResult:
    """Resolve per-node runners, run the skeleton, collect produced files."""
    atom_runner_plan = build_atom_runner_plan(skeleton.nodes, role_runners)
    if explicit_atom_runners:
        atom_runner_plan.update(explicit_atom_runners)
    router = RoleRoutingAdapter(atom_runner_plan, default=get_adapter(_DEFAULT_RUNNER))
    locks_dir = tempfile.mkdtemp(prefix="atomic-orch-locks-")
    run_result = run_skeleton(skeleton, router, workspace=workspace, lock_base_dir=locks_dir)

    artifacts = sorted(
        path.relative_to(workspace).as_posix()
        for path in Path(workspace).rglob("*")
        if path.is_file()
    )
    final_path = None
    if final_file:
        candidate = Path(workspace) / final_file
        final_path = str(candidate) if candidate.exists() else None

    return OrchestrationResult(
        template=template,
        request=request,
        workspace=workspace,
        succeeded=run_result.succeeded and (final_path is not None if final_file else True),
        artifacts=artifacts,
        final_path=final_path,
        atom_runner_plan=atom_runner_plan,
        run_result=run_result,
        extra=extra or {},
    )


def ensure_workspace(workspace: str | None, prefix: str) -> str:
    ws = workspace or tempfile.mkdtemp(prefix=prefix)
    # 归一化为绝对路径：相对 --workspace 会被 adapter 在子进程 cwd 下二次解析，导致
    # codex/ducc 报 "Workspace does not exist" 秒挂（见 scheduler 同款兜底）。
    ws = str(Path(ws).expanduser().resolve())
    Path(ws).mkdir(parents=True, exist_ok=True)
    return ws


__all__ = [
    "RoleRoutingAdapter", "DEFAULT_ROLE_RUNNERS", "OrchestrationResult",
    "get_adapter", "merge_role_runners", "resolve_runner_for_role",
    "build_atom_runner_plan", "write_node", "make_skeleton", "edges_from_dependencies",
    "invoke_atom", "extract_json_array", "extract_json_object",
    "run_orchestration", "ensure_workspace",
]
