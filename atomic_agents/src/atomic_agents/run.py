"""Top-level P5/P6 runtime entrypoints.

Natural-language requests are compiled into ``MetaPlan`` objects by
``meta_compile``. This module only stitches the compiled skeleton to execution:
Skeleton/MetaPlan + one adapter + lockfile + workspace + optional approval into
``StaticScheduler.run``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from atomic_agents.adapters import RunnerAdapter
from atomic_agents.approval import ApprovalCallback, ApprovalSummary, render_approval_summary
from atomic_agents.lockfile import LockWriter
from atomic_agents.meta import MetaPlan
from atomic_agents.models import Skeleton
from atomic_agents.scheduler import DriftHandler, NodeState, StaticScheduler


@dataclass(kw_only=True)
class RunResult:
    run_id: str
    states: dict[str, NodeState]
    succeeded: bool
    stop_reason: str | None = None
    lock_dir: str

    @property
    def failed_nodes(self) -> list[str]:
        return [node_id for node_id, state in self.states.items() if state.status == "failed"]

    @property
    def blocked_nodes(self) -> list[str]:
        return [node_id for node_id, state in self.states.items() if state.status == "blocked"]

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "states": {
                node_id: {
                    "node_id": state.node_id,
                    "status": state.status,
                    "result": state.result.to_dict() if state.result is not None else None,
                }
                for node_id, state in self.states.items()
            },
            "succeeded": self.succeeded,
            "stop_reason": self.stop_reason,
            "lock_dir": self.lock_dir,
            "failed_nodes": self.failed_nodes,
            "blocked_nodes": self.blocked_nodes,
        }


def run_skeleton(
    skeleton: Skeleton,
    adapter: RunnerAdapter,
    *,
    run_id: str | None = None,
    workspace: str = ".",
    lock_base_dir: str | Path | None = None,
    approval: ApprovalCallback | None = None,
    approval_summary: ApprovalSummary | None = None,
    drift_handler: DriftHandler | None = None,
) -> RunResult:
    """Run a compiled skeleton through the static scheduler with one adapter."""

    resolved_run_id = run_id or f"run-{uuid4().hex[:8]}"
    writer = LockWriter(resolved_run_id, base_dir=lock_base_dir) if lock_base_dir is not None else LockWriter(resolved_run_id)
    scheduler = StaticScheduler(
        skeleton=skeleton,
        adapter=adapter,
        lock=writer,
        run_id=resolved_run_id,
        workspace=workspace,
        approval=approval,
        approval_summary=approval_summary,
        drift_handler=drift_handler,
    )
    states = scheduler.run()
    succeeded = scheduler.stop_reason is None and all(state.status == "succeeded" for state in states.values())
    return RunResult(
        run_id=resolved_run_id,
        states=states,
        succeeded=succeeded,
        stop_reason=scheduler.stop_reason,
        lock_dir=str(writer.run_dir),
    )


def run_meta_plan(
    plan: MetaPlan,
    adapter: RunnerAdapter,
    *,
    run_id: str | None = None,
    workspace: str = ".",
    lock_base_dir: str | Path | None = None,
    approval: ApprovalCallback | None = None,
    drift_handler: DriftHandler | None = None,
) -> RunResult:
    """Run a MetaPlan, auto-rendering the plan approval summary when needed."""

    approval_summary = render_approval_summary(plan) if approval is not None else None
    return run_skeleton(
        plan.skeleton,
        adapter,
        run_id=run_id,
        workspace=workspace,
        lock_base_dir=lock_base_dir,
        approval=approval,
        approval_summary=approval_summary,
        drift_handler=drift_handler,
    )


__all__ = ["RunResult", "run_meta_plan", "run_skeleton"]
