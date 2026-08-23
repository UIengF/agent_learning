"""Static DAG scheduler for the P2 atomic-agents runtime layer."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import platform
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal

from atomic_agents.adapters import RunnerAdapter
from atomic_agents.approval import ApprovalCallback, ApprovalDecision
from atomic_agents.drift import Snapshot, changed_files, detect_drift, snapshot
from atomic_agents.linter import VALID_INPUT_FIELDS
from atomic_agents.locks import ConsultingFileLock
from atomic_agents.lockfile import LockWriter
from atomic_agents.models import ApprovalSummary, AtomContract, AtomResult, InputRef, Skeleton, SkeletonNode, StopReport
from atomic_agents.reviewer import CriterionVerdict, ReviewResult, build_review_feedback, review_to_lock_payload
from atomic_agents.validation import validate_stop_report


NodeRuntimeStatus = Literal["pending", "running", "succeeded", "failed", "blocked"]
DriftChoice = Literal["continue", "stop"]
DriftHandler = Callable[[str, list[str]], str]

# 瞬时（基础设施）错误退避重试：与 max_repair_attempts（质量修复）完全独立——
# 网关 503/429 等不是任务失败，不烧 repair 配额。默认最多 3 次、退避 5/15/45 秒。
# 效果优先：不计入成本预算（这些尝试基本没真实消费 token）。
TRANSIENT_MAX_RETRIES = 3
TRANSIENT_BACKOFF_SECONDS = (5.0, 15.0, 45.0)


@dataclass
class NodeState:
    node_id: str
    status: NodeRuntimeStatus
    result: AtomResult | None = None


@dataclass
class DriftRecord:
    declared: list[str]
    actual: list[str]
    drift: list[str]
    user_choice: DriftChoice


def build_contract(
    node: SkeletonNode,
    skeleton: Skeleton,
    upstream_results: dict[str, AtomResult],
    run_id: str,
    workspace: str = ".",
) -> AtomContract:
    """Build an atom contract from a skeleton node and completed upstream outputs."""

    contract_inputs: list[InputRef] = []
    upstream_output_files: list[str] = []

    for context_file in node.context_files or []:
        if context_file not in upstream_output_files:
            upstream_output_files.append(context_file)

    for input_ref in node.inputs:
        source_id = input_ref["from"]
        field = input_ref["field"]
        if source_id not in upstream_results:
            raise ValueError(f"missing upstream result for input {source_id}.{field}")
        if field not in VALID_INPUT_FIELDS:
            raise ValueError(f"unsupported input field {field!r} for node {node.id}")

        contract_inputs.append({"from": source_id, "field": field})

        # Decision 2: a downstream atom reads its upstream atom's product as a
        # file. Surface every referenced upstream output_file via context_files
        # so the runner can read it (codex -f / ducc --add-dir).
        upstream_output_file = upstream_results[source_id].output_file
        if upstream_output_file and upstream_output_file not in upstream_output_files:
            upstream_output_files.append(upstream_output_file)

    timestamp = _utc_now()
    return AtomContract(
        task=node.task,
        inputs=contract_inputs,
        context_files=upstream_output_files,
        workspace=workspace,
        read_only=node.read_only,
        write_scope=list(node.write_scope),
        required_capabilities=list(node.required_capabilities),
        status="success",
        result="",
        artifacts=[],
        output_file=node.output_file,
        output_schema_ref=None,
        handoff={"completed": [], "pending": [], "decisions": [], "risks": []},
        consult=None,
        atom_id=node.id,
        correlation_id=run_id,
        logical_role=node.role,
        resolved_runner=None,
        session_id=None,
        hop_count=1,
        limits={
            "max_cost": _per_atom_cost(skeleton),
            "timeout_sec": 1800,
            "max_internal_turns": 30,
        },
        cost=0.0,
        duration_sec=0.0,
        timestamps={"started_at": timestamp, "finished_at": timestamp},
    )


class StaticScheduler:
    """Run a fixed skeleton DAG with read parallelism and scope-aware writes."""

    def __init__(
        self,
        skeleton: Skeleton,
        adapter: RunnerAdapter,
        lock: LockWriter,
        run_id: str,
        reviewer: object | None = None,
        file_lock: ConsultingFileLock | None = None,
        drift_handler: DriftHandler | None = None,
        workspace: str = ".",
        approval: ApprovalCallback | None = None,
        approval_summary: ApprovalSummary | None = None,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        self.skeleton = skeleton
        self.adapter = adapter
        self.lock = lock
        self.run_id = run_id
        # Deprecated compatibility parameter. Reviewers are DAG nodes that
        # produce verdict files; injected reviewer objects are ignored.
        del reviewer
        self.file_lock = file_lock or ConsultingFileLock()
        self.drift_handler = drift_handler
        # 归一化为绝对路径：adapter 用 cwd=contract.workspace 启动子进程，相对路径在不同
        # cwd 下被二次解析会让 codex/ducc 报 "Workspace does not exist" 秒挂。在调度层统一
        # 兜底，任何调用方（含非编排路径）都安全。
        self.workspace = str(Path(workspace).expanduser().resolve())
        self.approval = approval
        self._approval_summary = approval_summary
        self._sleep_fn = sleep_fn

        self._nodes_by_id = _nodes_by_id(skeleton.nodes)
        self._dependents, self._topo_order = self._build_topology()
        self._states = {node_id: NodeState(node_id=node_id, status="pending") for node_id in self._topo_order}
        self._topo_index = {node_id: index for index, node_id in enumerate(self._topo_order)}
        self._attempts = {node_id: 0 for node_id in self._topo_order}
        self._files_changed: set[str] = set()
        self._unplanned_files: set[str] = set()
        self._total_cost = 0.0
        self._stopped = False
        self._stop_reason: str | None = None
        self._batch_pre_snapshot: Snapshot | None = None
        self._lock_mutex = threading.Lock()
        self._adapter_name = _adapter_name(adapter)
        self._adapter_version = _adapter_version(adapter, self._adapter_name)

    def run(self) -> dict[str, NodeState]:
        self._run_started()
        self._binding_locked()
        if not self._run_approval_gates():
            self._run_summary(total_cost=self._total_cost, path_taken=[])
            return self._states

        upstream_results: dict[str, AtomResult] = {}
        path_taken: list[str] = []

        while not self._stopped and not self._is_terminal():
            self._block_failed_dependents()
            if self._is_terminal():
                break

            runnable = self._runnable_nodes()
            if not runnable:
                pending = [node_id for node_id, state in self._states.items() if state.status == "pending"]
                raise RuntimeError(f"scheduler stalled with pending nodes: {', '.join(pending)}")

            for batch in self._batches_for(runnable):
                if self._stopped:
                    break
                self._mark_running(batch)
                outcomes = self._execute_batch(batch, upstream_results)
                outcomes.sort(key=lambda outcome: self._topo_index[outcome[0].id])
                for node, result in outcomes:
                    # _detect_batch_drift (called inside _execute_batch above) may have
                    # already force-failed one owner node and set self._stopped when a
                    # drift_handler requests "stop". That owner has no settling left to
                    # do (its state is already terminal with result=None by design) but
                    # its batch siblings genuinely finished and must still be settled —
                    # skipping them here would silently discard real, already-produced
                    # results. So only skip the owner itself, not the whole remaining batch.
                    if self._stopped and self._states[node.id].status == "failed":
                        continue
                    self._settle_node(node, result, upstream_results, path_taken)

        self._run_summary(total_cost=self._total_cost, path_taken=path_taken)
        return self._states

    def _build_topology(self) -> tuple[dict[str, list[str]], list[str]]:
        indegree = {node_id: 0 for node_id in self._nodes_by_id}
        dependents = {node_id: [] for node_id in self._nodes_by_id}
        seen_edges: set[tuple[str, str]] = set()

        for node in self.skeleton.nodes:
            for dependency in node.depends_on:
                if dependency not in self._nodes_by_id:
                    raise ValueError(f"node {node.id} depends on unknown node {dependency}")
                edge = (dependency, node.id)
                if edge in seen_edges:
                    continue
                seen_edges.add(edge)
                indegree[node.id] += 1
                dependents[dependency].append(node.id)

        node_index = {node.id: index for index, node in enumerate(self.skeleton.nodes)}
        ready = [node.id for node in self.skeleton.nodes if indegree[node.id] == 0]
        topo_order: list[str] = []

        while ready:
            node_id = ready.pop(0)
            topo_order.append(node_id)
            for dependent in dependents[node_id]:
                indegree[dependent] -= 1
                if indegree[dependent] == 0:
                    ready.append(dependent)
            ready.sort(key=node_index.__getitem__)

        if len(topo_order) != len(self.skeleton.nodes):
            cycle_nodes = [node_id for node_id, degree in indegree.items() if degree > 0]
            raise ValueError(f"skeleton has a cycle: {', '.join(cycle_nodes)}")

        return dependents, topo_order

    def _block_failed_dependents(self) -> None:
        changed = True
        while changed:
            changed = False
            for node_id in self._topo_order:
                state = self._states[node_id]
                if state.status != "pending":
                    continue

                node = self._nodes_by_id[node_id]
                blocked_by = [
                    dependency
                    for dependency in node.depends_on
                    if self._states[dependency].status in ("failed", "blocked")
                ]
                if not blocked_by:
                    continue

                state.status = "blocked"
                self._atom_blocked(node, blocked_by)
                changed = True

    def _runnable_nodes(self) -> list[SkeletonNode]:
        runnable: list[SkeletonNode] = []
        for node_id in self._topo_order:
            state = self._states[node_id]
            node = self._nodes_by_id[node_id]
            if state.status == "pending" and all(self._states[dependency].status == "succeeded" for dependency in node.depends_on):
                runnable.append(node)
        return runnable

    def _batches_for(self, runnable: list[SkeletonNode]) -> list[list[SkeletonNode]]:
        read_batch = [node for node in runnable if not node.write_scope]
        write_nodes = [node for node in runnable if node.write_scope]

        batches: list[list[SkeletonNode]] = []
        if read_batch:
            batches.append(read_batch)

        write_batches: list[list[SkeletonNode]] = []
        occupied_by_batch: list[list[str]] = []
        for node in write_nodes:
            scope = _node_write_scope(node, node.output_file)
            for index, occupied in enumerate(occupied_by_batch):
                if _scopes_overlap(scope, occupied):
                    continue
                write_batches[index].append(node)
                occupied.extend(scope)
                break
            else:
                write_batches.append([node])
                occupied_by_batch.append(list(scope))

        batches.extend(write_batches)
        return batches

    def _mark_running(self, batch: list[SkeletonNode]) -> None:
        for node in batch:
            self._states[node.id].status = "running"

    def _execute_batch(
        self,
        batch: list[SkeletonNode],
        upstream_results: dict[str, AtomResult],
    ) -> list[tuple[SkeletonNode, AtomResult]]:
        if len(batch) == 1:
            node = batch[0]
            return [self._execute_node(node, upstream_results, self._next_attempt_id(node.id))]

        max_workers = len(batch)
        attempt_ids = {node.id: self._next_attempt_id(node.id) for node in batch}
        write_scopes = {
            node.id: _node_write_scope(node, _attempt_output_file(node.output_file, attempt_ids[node.id]))
            for node in batch
        }
        # Per-batch drift attribution (decision: charge out-of-scope writes to the
        # whole batch, not to each sibling). Inside a parallel write batch, the
        # global workspace snapshot cannot tell which sibling wrote a stray file,
        # so we suppress per-node drift here and run one combined drift check over
        # the batch's union scope after all members finish (see below).
        self._batch_pre_snapshot = snapshot(self.workspace)
        outcomes: list[tuple[SkeletonNode, AtomResult]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    self._execute_node,
                    node,
                    upstream_results,
                    attempt_ids[node.id],
                    None,
                    _peer_write_scope(node, batch, write_scopes),
                    True,  # suppress_drift: defer to the combined per-batch check
                )
                for node in batch
            ]
            for future in concurrent.futures.as_completed(futures):
                outcomes.append(future.result())

        self._detect_batch_drift(batch, write_scopes)
        return outcomes

    def _execute_node(
        self,
        node: SkeletonNode,
        upstream_results: dict[str, AtomResult],
        attempt_id: int,
        feedback: str | None = None,
        ignore_write_scope: list[str] | None = None,
        suppress_drift: bool = False,
    ) -> tuple[SkeletonNode, AtomResult]:
        contract = build_contract(node, self.skeleton, upstream_results, self.run_id, self.workspace)
        effective_output = _attempt_output_file(node.output_file, attempt_id)
        contract.output_file = effective_output
        if feedback:
            contract.task = _task_with_feedback(contract.task, feedback)
        self._atom_input(node, contract, attempt_id)

        drift_record: DriftRecord | None = None
        if node.write_scope:
            result, drift_record = self._execute_write_node(
                node, contract, attempt_id, ignore_write_scope, suppress_drift
            )
        else:
            self._atom_started(node, attempt_id)
            result = self._invoke_contract(contract)

        result = _verify_output_file(self.workspace, effective_output, result)
        self._atom_finished(node, result, attempt_id)
        if drift_record is not None:
            self._scope_drift(node, drift_record, attempt_id)
        return node, result

    def _execute_write_node(
        self,
        node: SkeletonNode,
        contract: AtomContract,
        attempt_id: int,
        ignore_write_scope: list[str] | None = None,
        suppress_drift: bool = False,
    ) -> tuple[AtomResult, DriftRecord | None]:
        write_scope = _node_write_scope(node, contract.output_file)

        pre = snapshot(self.workspace)
        with self.file_lock.hold(write_scope):
            self._atom_started(node, attempt_id)
            result = self._invoke_contract(contract)
        # In a parallel write batch the per-node drift decision is unreliable
        # (the global snapshot mixes siblings' writes). The caller does one
        # combined per-batch drift check instead — see _detect_batch_drift.
        if suppress_drift:
            return result, None
        post = snapshot(self.workspace)
        drift = detect_drift(self.workspace, pre, post, write_scope, ignore_paths=ignore_write_scope)
        if not drift:
            return result, None

        actual = changed_files(pre, post)
        choice = self._resolve_drift_choice(node.id, drift)
        drift_record = DriftRecord(
            declared=write_scope,
            actual=actual,
            drift=drift,
            user_choice=choice,
        )
        if choice == "continue":
            return result, drift_record

        return _result_with_scope_drift_error(result, drift), drift_record

    def _detect_batch_drift(
        self,
        batch: list[SkeletonNode],
        write_scopes: dict[str, list[str]],
    ) -> None:
        """Combined drift check for a parallel write batch.

        路1：write_scope 是调度提示而非安全边界。并行批次的全局快照无法判定是哪个
        sibling 写了 scope 外的文件，所以越界归因为【批次级】（payload 显式注明无法
        定位到单节点），并默认【记录但不停】——只 emit scope_drift 事件，不把任何
        sibling 判失败、不连坐整批。批内合规且成功的节点照常在 run() 里被 settle，
        成果保留。仅当调用方注入的 drift_handler 显式返回 stop 时才止损（兼容）。
        """

        union_scope: list[str] = []
        for node in batch:
            for path in write_scopes[node.id]:
                if path not in union_scope:
                    union_scope.append(path)

        pre = self._batch_pre_snapshot
        post = snapshot(self.workspace)
        drift = detect_drift(self.workspace, pre, post, union_scope)
        if not drift:
            return

        actual = changed_files(pre, post)
        # 批次级越界：用 topo 最小节点仅作 lockfile atom_id 锚点，不代表它就是肇事者。
        owner = min(batch, key=lambda candidate: self._topo_index[candidate.id])
        choice = self._resolve_drift_choice(owner.id, drift)
        drift_record = DriftRecord(
            declared=union_scope,
            actual=actual,
            drift=drift,
            user_choice=choice,
        )
        self._scope_drift(
            owner,
            drift_record,
            self._attempts[owner.id],
            batch_scope=[node.id for node in batch],
        )
        # 默认 choice=="continue"：只记录，run 不停，sibling 不连坐。
        # 仅显式 handler 要求 stop 时才止损。
        if choice != "continue":
            state = self._states[owner.id]
            state.result = _result_with_scope_drift_error(state.result, drift) if state.result else state.result
            state.status = "failed"
            self._stop_run("scope_drift", owner, self._attempts[owner.id], None)

    def _settle_node(
        self,
        node: SkeletonNode,
        result: AtomResult,
        upstream_results: dict[str, AtomResult],
        path_taken: list[str],
    ) -> None:
        if self._is_reviewer_node(node):
            self._settle_reviewer_node(node, result, upstream_results, path_taken)
            return

        self._settle_regular_node(node, result, upstream_results, path_taken)

    def _settle_regular_node(
        self,
        node: SkeletonNode,
        result: AtomResult,
        upstream_results: dict[str, AtomResult],
        path_taken: list[str],
    ) -> None:
        state = self._states[node.id]
        attempt_id = self._attempts[node.id]

        while True:
            self._total_cost += result.cost
            self._record_result_observations(result)
            state.result = result

            if result.status == "success":
                state.status = "succeeded"
                upstream_results[node.id] = result
                path_taken.append(node.id)
                if self._budget_exceeded():
                    self._stop_run("over_budget", node, attempt_id, None)
                return

            if self._budget_exceeded():
                state.status = "failed"
                self._stop_run("over_budget", node, attempt_id, None)
                return

            feedback = _result_failure_feedback(result)
            stop_reason = _failure_stop_reason(result)

            if attempt_id < self._max_repair_attempts():
                attempt_id = self._next_attempt_id(node.id)
                state.status = "running"
                self._retry_scheduled(node, attempt_id, feedback)
                _retry_node, result = self._execute_node(node, upstream_results, attempt_id, feedback)
                continue

            state.status = "failed"
            self._stop_run(stop_reason, node, attempt_id, None)
            return

    def _settle_reviewer_node(
        self,
        node: SkeletonNode,
        result: AtomResult,
        upstream_results: dict[str, AtomResult],
        path_taken: list[str],
    ) -> None:
        state = self._states[node.id]
        attempt_id = self._attempts[node.id]
        last_review: ReviewResult | None = None

        while True:
            self._total_cost += result.cost
            self._record_result_observations(result)
            state.result = result

            if result.status == "success":
                last_review = self._read_verdict(node, result)
                self._review_finished(node, last_review, attempt_id)
                if last_review.passed:
                    state.status = "succeeded"
                    upstream_results[node.id] = result
                    path_taken.append(node.id)
                    if self._budget_exceeded():
                        self._stop_run("over_budget", node, attempt_id, last_review)
                    return

                if self._budget_exceeded():
                    state.status = "failed"
                    self._stop_run("over_budget", node, attempt_id, last_review)
                    return

                feedback = last_review.feedback or build_review_feedback(last_review)
                stop_reason = "repair_failed"
            else:
                last_review = None
                if self._budget_exceeded():
                    state.status = "failed"
                    self._stop_run("over_budget", node, attempt_id, None)
                    return

                feedback = _result_failure_feedback(result)
                stop_reason = _failure_stop_reason(result)

            if attempt_id < self._max_repair_attempts():
                if last_review is not None and not self._retry_upstream_writers(
                    node,
                    upstream_results,
                    path_taken,
                    feedback,
                    last_review,
                ):
                    return

                attempt_id = self._next_attempt_id(node.id)
                state.status = "running"
                self._retry_scheduled(node, attempt_id, feedback)
                _retry_node, result = self._execute_node(node, upstream_results, attempt_id, feedback)
                continue

            state.status = "failed"
            self._stop_run(stop_reason, node, attempt_id, last_review)
            return

    def _retry_upstream_writers(
        self,
        node: SkeletonNode,
        upstream_results: dict[str, AtomResult],
        path_taken: list[str],
        feedback: str,
        review: ReviewResult,
    ) -> bool:
        upstream_writers = [
            self._nodes_by_id[dependency]
            for dependency in node.depends_on
            if self._nodes_by_id[dependency].write_scope
        ]

        for writer in upstream_writers:
            writer_attempt = self._attempts[writer.id]
            if writer_attempt >= self._max_repair_attempts():
                self._states[node.id].status = "failed"
                self._stop_run("repair_failed", node, self._attempts[node.id], review)
                return False

            next_attempt = self._next_attempt_id(writer.id)
            self._states[writer.id].status = "running"
            self._retry_scheduled(writer, next_attempt, feedback)
            _writer_node, writer_result = self._execute_node(writer, upstream_results, next_attempt, feedback)
            self._settle_regular_node(writer, writer_result, upstream_results, path_taken)
            if self._stopped or self._states[writer.id].status != "succeeded":
                return False

        return True

    def _read_verdict(self, node: SkeletonNode, result: AtomResult) -> ReviewResult:
        verdict_ref = result.output_file
        if not verdict_ref:
            return _invalid_verdict_review("verdict file not declared")

        verdict_path = Path(verdict_ref)
        if not verdict_path.is_absolute():
            verdict_path = Path(self.workspace) / verdict_path

        try:
            with verdict_path.open("r", encoding="utf-8") as verdict_file:
                data = json.load(verdict_file)
            if not isinstance(data, dict):
                raise ValueError("verdict JSON must be an object")
            return ReviewResult.from_dict(data)
        except Exception as exc:
            return _invalid_verdict_review(f"verdict file unreadable for {node.id}: {verdict_ref}: {exc}")

    def _invoke_contract(self, contract: AtomContract) -> AtomResult:
        # 瞬时（基础设施）错误退避重试：adapter 把网关 503/429 等判为 status=="transient"。
        # 这类不是任务失败，所以在此就地退避重试（最多 TRANSIENT_MAX_RETRIES 次），与
        # max_repair_attempts（质量修复）完全独立、不烧 repair 配额。退避耗尽仍 transient
        # 则把状态归一化为 failed，并保留「最后一次仍是瞬时错误」标记供调度器如实归因。
        attempt = 0
        while True:
            result = self._invoke_once(contract)
            if result.status != "transient":
                return result

            if attempt >= TRANSIENT_MAX_RETRIES:
                # 退避用尽仍不可用：归一化为 failed，但用 transient_exhausted 标记真因。
                error = result.error or "transient infrastructure error"
                return AtomResult(
                    status="failed",
                    result=result.result,
                    artifacts=result.artifacts,
                    session_id=result.session_id,
                    cost=result.cost,
                    duration_sec=result.duration_sec,
                    raw_events_path=result.raw_events_path,
                    error=f"transient_exhausted: {error}",
                    output_file=result.output_file,
                    output_sha256=result.output_sha256,
                )

            backoff_index = min(attempt, len(TRANSIENT_BACKOFF_SECONDS) - 1)
            self._sleep_fn(TRANSIENT_BACKOFF_SECONDS[backoff_index])
            attempt += 1

    def _invoke_once(self, contract: AtomContract) -> AtomResult:
        started = time.monotonic()
        try:
            result = self.adapter.invoke(contract, timeout_sec=contract.limits["timeout_sec"])
        except Exception as exc:  # pragma: no cover - defensive boundary for third-party adapters.
            result = AtomResult(
                status="failed",
                result="",
                artifacts=[],
                session_id=None,
                cost=0.0,
                duration_sec=time.monotonic() - started,
                raw_events_path=None,
                error=f"{exc.__class__.__name__}: {exc}",
                output_file="",
                output_sha256=None,
            )
        return result

    def _resolve_drift_choice(self, node_id: str, drift_files: list[str]) -> DriftChoice:
        # 路1：write_scope 是调度提示，不是安全边界。越界写默认【记录但不停】——
        # 写进 scope_drift 事件 + run_summary.unplanned_files_touched 如实汇报，不止损。
        # 调用方仍可注入 drift_handler 显式要求 stop（向后兼容）。
        if self.drift_handler is None:
            return "continue"

        choice = self.drift_handler(node_id, list(drift_files))
        if choice == "continue":
            return "continue"
        return "stop"

    def _next_attempt_id(self, node_id: str) -> int:
        attempt_id = self._attempts[node_id] + 1
        self._attempts[node_id] = attempt_id
        return attempt_id

    def _max_repair_attempts(self) -> int:
        return int(self.skeleton.run_limits.get("max_repair_attempts", 0))

    def _budget_exceeded(self) -> bool:
        return self._total_cost > float(self.skeleton.run_limits.get("max_total_cost", 0.0))

    def _record_result_observations(self, result: AtomResult) -> None:
        for artifact in result.artifacts:
            path = artifact.get("path")
            if path:
                self._files_changed.add(path)

    def _is_reviewer_node(self, node: SkeletonNode) -> bool:
        return node.role.strip().lower() == "reviewer"

    def _run_approval_gates(self) -> bool:
        if self.approval is None:
            return True

        if self._approval_summary is not None:
            decision: ApprovalDecision = self.approval.approve_plan(self._approval_summary)
            if not decision.approved:
                self._reject_run("user_rejected")
                return False

            with self._lock_mutex:
                self.lock.user_approved(
                    {
                        "gate": "plan",
                        "approved": True,
                        "edited": decision.edited,
                    }
                )

        irreversible_ops = list(self.skeleton.irreversible_ops)
        for op in irreversible_ops:
            if not self.approval.approve_irreversible(op):
                self._reject_run("irreversible_denied", op=op)
                return False

        if irreversible_ops:
            with self._lock_mutex:
                self.lock.user_approved(
                    {
                        "gate": "irreversible",
                        "approved": True,
                        "ops": irreversible_ops,
                    }
                )

        return True

    def _reject_run(self, reason: str, op: str | None = None) -> None:
        stop_report = StopReport(
            reason=reason,
            failed_atom="",
            attempts=0,
            reviewer_evidence=[],
            files_changed=sorted(self._files_changed),
            cost_consumed=self._total_cost,
            likely_causes=_reject_causes(reason, op),
            options=["重新规划后再跑", "调整骨架/预算后重试", "放弃本次 run"],
        ).to_dict()
        validate_stop_report(stop_report)

        stop_report_path = self.lock.run_dir / "stop-report.json"
        with stop_report_path.open("w", encoding="utf-8") as report_file:
            json.dump(stop_report, report_file, ensure_ascii=False, indent=2)
            report_file.write("\n")

        with self._lock_mutex:
            self.lock.run_stopped(
                {
                    "reason": reason,
                    "stop_report_ref": None,
                    "stop_report": stop_report,
                }
            )

        self._stopped = True
        self._block_unfinished_after_stop(reason)

    def _stop_run(
        self,
        reason: str,
        node: SkeletonNode,
        attempts: int,
        review: ReviewResult | None,
    ) -> None:
        stop_report = StopReport(
            reason=reason,
            failed_atom=node.id,
            attempts=attempts,
            reviewer_evidence=_reviewer_evidence(review),
            files_changed=sorted(self._files_changed),
            cost_consumed=self._total_cost,
            likely_causes=_likely_causes(reason),
            options=_stop_options(reason),
        ).to_dict()
        validate_stop_report(stop_report)

        stop_report_path = self.lock.run_dir / "stop-report.json"
        with stop_report_path.open("w", encoding="utf-8") as report_file:
            json.dump(stop_report, report_file, ensure_ascii=False, indent=2)
            report_file.write("\n")

        with self._lock_mutex:
            self.lock.run_stopped(
                {
                    "reason": reason,
                    "stop_report_ref": None,
                    "stop_report": stop_report,
                }
            )

        self._stopped = True
        self._stop_reason = reason
        self._block_unfinished_after_stop(reason)

    @property
    def stop_reason(self) -> str | None:
        """Non-None when the run was terminated by ``_stop_run`` rather than reaching a normal terminal state."""

        return self._stop_reason

    def _is_terminal(self) -> bool:
        return all(state.status in ("succeeded", "failed", "blocked") for state in self._states.values())

    def _run_started(self) -> None:
        with self._lock_mutex:
            self.lock.run_started({"launcher": self._adapter_name})

    def _binding_locked(self) -> None:
        bindings: dict[str, dict[str, str]] = {}
        for node in self.skeleton.nodes:
            bindings.setdefault(node.role, {"runner": self._adapter_name})

        with self._lock_mutex:
            self.lock.binding_locked({"binding": self._adapter_name, "bindings": bindings})

    def _atom_input(self, node: SkeletonNode, contract: AtomContract, attempt_id: int) -> None:
        with self._lock_mutex:
            self.lock.atom_input(
                {"snapshot": contract.to_dict()},
                atom_id=node.id,
                attempt_id=attempt_id,
                adapter_version=self._adapter_version,
            )

    def _atom_started(self, node: SkeletonNode, attempt_id: int) -> None:
        with self._lock_mutex:
            self.lock.atom_started(
                {
                    "runner": self._adapter_name,
                    "read_only": node.read_only,
                    "write_scope": list(node.write_scope),
                },
                atom_id=node.id,
                attempt_id=attempt_id,
                adapter_version=self._adapter_version,
            )

    def _atom_finished(self, node: SkeletonNode, result: AtomResult, attempt_id: int) -> None:
        with self._lock_mutex:
            self.lock.atom_finished(
                _result_payload(result),
                atom_id=node.id,
                attempt_id=attempt_id,
                adapter_version=self._adapter_version,
            )

    def _scope_drift(
        self,
        node: SkeletonNode,
        drift_record: DriftRecord,
        attempt_id: int,
        batch_scope: list[str] | None = None,
    ) -> None:
        self._files_changed.update(drift_record.actual)
        # 路1 裂缝①兜底：累计越界（计划外）文件，run_summary 如实汇报。
        self._unplanned_files.update(drift_record.drift)
        payload: dict[str, object] = {
            "declared": drift_record.declared,
            "actual": drift_record.actual,
            "drift": drift_record.drift,
            "user_choice": drift_record.user_choice,
        }
        # 批次级越界：atom_id 仅作锚点，无法定位单个肇事 sibling，显式标注涉及的整批。
        if batch_scope is not None:
            payload["attribution"] = "batch"
            payload["batch_atoms"] = list(batch_scope)
        with self._lock_mutex:
            self.lock.scope_drift(
                payload,
                atom_id=node.id,
                attempt_id=attempt_id,
                adapter_version=self._adapter_version,
            )

    def _review_finished(self, node: SkeletonNode, review: ReviewResult, attempt_id: int) -> None:
        with self._lock_mutex:
            self.lock.review_finished(
                review_to_lock_payload(review),
                atom_id=node.id,
                attempt_id=attempt_id,
                adapter_version=self._adapter_version,
            )

    def _retry_scheduled(self, node: SkeletonNode, attempt_id: int, feedback: str) -> None:
        with self._lock_mutex:
            self.lock.retry_scheduled(
                {
                    "attempt": attempt_id,
                    "feedback_summary": _feedback_summary(feedback),
                },
                atom_id=node.id,
                attempt_id=attempt_id,
                adapter_version=self._adapter_version,
            )

    def _atom_blocked(self, node: SkeletonNode, blocked_by: list[str]) -> None:
        upstream_status = {dependency: self._states[dependency].status for dependency in blocked_by}
        with self._lock_mutex:
            self.lock.atom_finished(
                {
                    "status": "blocked",
                    "reason": "dependency_failed",
                    "blocked_by": list(blocked_by),
                    "upstream_status": upstream_status,
                },
                atom_id=node.id,
                adapter_version=self._adapter_version,
            )

    def _block_unfinished_after_stop(self, reason: str) -> None:
        for node_id in self._topo_order:
            state = self._states[node_id]
            if state.status not in ("pending", "running"):
                continue

            state.status = "blocked"
            with self._lock_mutex:
                self.lock.atom_finished(
                    {
                        "status": "blocked",
                        "reason": "run_stopped",
                        "blocked_by": [reason],
                    },
                    atom_id=node_id,
                    adapter_version=self._adapter_version,
                )

    def _run_summary(self, *, total_cost: float, path_taken: list[str]) -> None:
        with self._lock_mutex:
            self.lock.run_summary(
                {
                    "total_cost": total_cost,
                    "path_taken": list(path_taken),
                    "dynamic_atoms_added": [],
                    "unplanned_files_touched": sorted(self._unplanned_files),
                    "env": {
                        "adapter": self._adapter_name,
                        "adapter_version": self._adapter_version,
                        "python": platform.python_version(),
                        "platform": platform.platform(),
                    },
                }
            )


def _nodes_by_id(nodes: list[SkeletonNode]) -> dict[str, SkeletonNode]:
    nodes_by_id: dict[str, SkeletonNode] = {}
    for node in nodes:
        if node.id in nodes_by_id:
            raise ValueError(f"duplicate node id: {node.id}")
        nodes_by_id[node.id] = node
    return nodes_by_id


def _node_write_scope(node: SkeletonNode, output_file: str) -> list[str]:
    scope = list(node.write_scope)
    if output_file and output_file not in scope:
        scope.append(output_file)
    return scope


def _peer_write_scope(
    node: SkeletonNode,
    batch: list[SkeletonNode],
    write_scopes: dict[str, list[str]],
) -> list[str]:
    paths: list[str] = []
    for peer in batch:
        if peer.id == node.id:
            continue
        for path in write_scopes[peer.id]:
            if path not in paths:
                paths.append(path)
    return paths


def _scopes_overlap(scope_a: list[str], scope_b: list[str]) -> bool:
    normalized_a = [_normalize_scope_path(path) for path in scope_a]
    normalized_b = [_normalize_scope_path(path) for path in scope_b]
    return any(
        _scope_paths_overlap(left, right)
        for left in normalized_a
        for right in normalized_b
        if left and right
    )


def _normalize_scope_path(path: str) -> str:
    normalized = path.strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    normalized = normalized.strip("/")
    if normalized == ".":
        return ""
    return normalized


def _scope_paths_overlap(left: str, right: str) -> bool:
    return left == right or left.startswith(f"{right}/") or right.startswith(f"{left}/")


def _result_payload(result: AtomResult) -> dict[str, object]:
    return {
        "status": result.status,
        "result_sha256": hashlib.sha256(result.result.encode("utf-8")).hexdigest(),
        "artifacts": result.artifacts,
        "session_id": result.session_id,
        "cost": result.cost,
        "duration_sec": result.duration_sec,
        "raw_events_path": result.raw_events_path,
        "error": result.error,
        "last_activity": result.last_activity,
    }


def _result_with_scope_drift_error(result: AtomResult, drift: list[str]) -> AtomResult:
    error = f"scope_drift: changed files outside write_scope: {', '.join(drift)}"
    if result.error:
        error = f"{result.error}\n{error}"

    return AtomResult(
        status="failed",
        result=result.result,
        artifacts=result.artifacts,
        session_id=result.session_id,
        cost=result.cost,
        duration_sec=result.duration_sec,
        raw_events_path=result.raw_events_path,
        error=error,
        output_file=result.output_file,
        output_sha256=result.output_sha256,
    )


def _attempt_output_file(output_file: str, attempt_id: int) -> str:
    """Use the declared path for attempt 1; suffix later attempts before the extension."""

    if not output_file or attempt_id <= 1:
        return output_file
    root, ext = os.path.splitext(output_file)
    return f"{root}.attempt{attempt_id}{ext}"


def _verify_output_file(workspace: str, effective_output: str, result: AtomResult) -> AtomResult:
    """Verify declared output exists and record its path and digest."""

    if not effective_output:
        return AtomResult(
            status=result.status,
            result=result.result,
            artifacts=result.artifacts,
            session_id=result.session_id,
            cost=result.cost,
            duration_sec=result.duration_sec,
            raw_events_path=result.raw_events_path,
            error=result.error,
            output_file="",
            output_sha256=None,
        )

    output_path = Path(effective_output)
    if not output_path.is_absolute():
        output_path = Path(workspace) / output_path

    if output_path.is_file():
        return AtomResult(
            status=result.status,
            result=result.result,
            artifacts=result.artifacts,
            session_id=result.session_id,
            cost=result.cost,
            duration_sec=result.duration_sec,
            raw_events_path=result.raw_events_path,
            error=result.error,
            output_file=effective_output,
            output_sha256=_sha256_file(output_path),
        )

    error = f"declared output_file not produced: {effective_output}"
    if result.error:
        error = f"{result.error}\n{error}"
    return AtomResult(
        status="failed",
        result=result.result,
        artifacts=result.artifacts,
        session_id=result.session_id,
        cost=result.cost,
        duration_sec=result.duration_sec,
        raw_events_path=result.raw_events_path,
        error=error,
        output_file=effective_output,
        output_sha256=None,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _task_with_feedback(task: str, feedback: str) -> str:
    return f"{task}\n\n[上一轮反馈]\n{feedback}"


def _result_failure_feedback(result: AtomResult) -> str:
    if result.error:
        return result.error
    return f"Atom returned status={result.status}"


def _failure_stop_reason(result: AtomResult) -> str:
    # 退避耗尽的瞬时错误（adapter→_invoke_contract 归一化为 failed 并加 transient_exhausted
    # 前缀）如实归因为基础设施不可用，而非 atom_failed，避免误导用户去改需求。
    if result.error and result.error.startswith("transient_exhausted"):
        return "infrastructure_unavailable"
    return "atom_failed"


def _invalid_verdict_review(evidence: str) -> ReviewResult:
    return ReviewResult(
        passed=False,
        criteria=[
            CriterionVerdict(
                criterion="verdict_file",
                verdict="fail",
                evidence=evidence,
                confidence="high",
            )
        ],
        blocking_findings=["verdict_file"],
        reviewer_session=None,
        feedback="",
    )


def _reviewer_evidence(review: ReviewResult | None) -> list[dict[str, str]]:
    if review is None:
        return []

    return [
        {
            "criterion": criterion.criterion,
            "verdict": criterion.verdict,
            "evidence": criterion.evidence,
        }
        for criterion in review.criteria
        if criterion.verdict == "fail"
    ]


def _feedback_summary(feedback: str, limit: int = 240) -> str:
    summary = " ".join(feedback.split())
    if len(summary) <= limit:
        return summary
    return f"{summary[: limit - 3]}..."


def _likely_causes(reason: str) -> list[str]:
    if reason == "repair_failed":
        return ["reviewer 持续未通过", "需求或验收标准需要人工澄清"]
    if reason == "atom_failed":
        return ["原子执行持续失败", "运行环境或输入契约可能需要修复"]
    if reason == "infrastructure_unavailable":
        return [
            "推理网关/runner 暂时不可用（如 503 credentials exhausted、限流、网关超时），退避重试后仍未恢复",
            "这是基础设施问题，非任务或需求问题；稍后网关恢复再重跑即可",
        ]
    if reason == "over_budget":
        return ["累计成本超过预算", "原子拆分或修复次数可能需要收敛"]
    return ["运行触发止损"]


def _stop_options(reason: str) -> list[str]:
    if reason == "infrastructure_unavailable":
        return ["等待网关/runner 恢复后重跑", "更换可用的 runner 或网关后重试", "放弃本次 run"]
    return ["人工修复后继续", "改需求重跑", "放弃本次 run"]


def _reject_causes(reason: str, op: str | None) -> list[str]:
    if reason == "user_rejected":
        return ["用户在开跑前驳回了计划"]
    if reason == "irreversible_denied":
        return [f"用户拒绝了不可逆操作：{op}"]
    return ["用户审批未通过"]


def _per_atom_cost(skeleton: Skeleton) -> float:
    if not skeleton.nodes:
        return float(skeleton.run_limits.get("max_total_cost", 0.0))
    return float(skeleton.run_limits.get("max_total_cost", 0.0)) / len(skeleton.nodes)


def _adapter_name(adapter: RunnerAdapter) -> str:
    profile = getattr(adapter, "feature_profile", None)
    name = getattr(profile, "name", None)
    if isinstance(name, str) and name:
        return name

    adapter_version = getattr(adapter, "adapter_version", None)
    if isinstance(adapter_version, str) and adapter_version:
        return adapter_version.split("-", 1)[0]
    return adapter.__class__.__name__


def _adapter_version(adapter: RunnerAdapter, adapter_name: str) -> str:
    version = getattr(adapter, "adapter_version", None)
    if isinstance(version, str) and version:
        return version
    return adapter_name


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = ["NodeState", "StaticScheduler", "build_contract"]
