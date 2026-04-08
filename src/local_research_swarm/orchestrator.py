from __future__ import annotations

import asyncio
import re
import time

from local_research_swarm.models import (
    AgentAction,
    AgentRegistry,
    AgentRole,
    ArtifactRecord,
    PlannedTask,
    RunRecord,
    RunStatus,
    TaskRecord,
    TaskStatus,
    TraceEvent,
)
from local_research_swarm.persistence import SQLiteRunStore


def has_markdown_heading(text: str, heading: str) -> bool:
    return bool(re.search(rf"(?mi)^\s{{0,3}}#{{1,6}}\s+{re.escape(heading)}\s*$", text))


def has_label_section(text: str, heading: str) -> bool:
    return bool(re.search(rf"(?mi)^\s*{re.escape(heading)}\s*:\s*", text))


def has_section_marker(text: str, heading: str) -> bool:
    return has_markdown_heading(text, heading) or has_label_section(text, heading)


def draft_quality_issues(draft_markdown: str) -> list[str]:
    issues: list[str] = []
    if not draft_markdown.strip():
        issues.append("empty report")
    if len(re.findall(r"https?://\S+", draft_markdown)) < 3:
        issues.append("fewer than 3 source urls")
    stripped = draft_markdown.rstrip()
    if stripped and not re.search(r"[.!?\)]\s*$", stripped):
        issues.append("report may be truncated")
    return issues


def draft_quality_warnings(draft_markdown: str) -> list[str]:
    warnings: list[str] = []
    if not has_section_marker(draft_markdown, "Findings"):
        warnings.append("missing findings marker")
    if not has_section_marker(draft_markdown, "Sources"):
        warnings.append("missing sources marker")
    if not has_section_marker(draft_markdown, "Conclusion"):
        warnings.append("missing conclusion marker")
    return warnings


class Orchestrator:
    def __init__(self, *, store: SQLiteRunStore, agents: AgentRegistry) -> None:
        self.store = store
        self.agents = agents

    async def submit(
        self,
        *,
        goal: str,
        max_rounds: int | None = None,
        max_parallel_agents: int | None = None,
        budget_tokens: int | None = None,
        budget_cost_usd: float | None = None,
        defer: bool = False,
    ) -> RunRecord:
        run = self.store.create_run(
            goal=goal,
            max_rounds=max_rounds or self.store.config.default_max_rounds,
            max_parallel_agents=max_parallel_agents or self.store.config.default_max_parallel_agents,
            budget_tokens=budget_tokens or self.store.config.default_budget_tokens,
            budget_cost_usd=budget_cost_usd or self.store.config.default_budget_cost_usd,
            auto_execute=not defer,
        )
        self.store.save_checkpoint(run.id, phase="created", payload={"goal": goal, "defer": defer})
        if defer:
            return run
        return await self.execute(run.id)

    async def resume(self, run_id: str) -> RunRecord:
        self.store.reset_in_progress_tasks(run_id)
        return await self.execute(run_id)

    def abort(self, run_id: str) -> RunRecord:
        run = self.store.set_run_status(run_id, RunStatus.ABORTED)
        self.store.save_checkpoint(run_id, phase="aborted", payload={"status": run.status.value})
        self.store.append_trace(
            TraceEvent.new(
                run_id=run_id,
                task_id="",
                agent_role=AgentRole.CRITIC,
                action=AgentAction.STOP,
                message="Run aborted by operator.",
            )
        )
        return self.store.get_run(run_id)

    async def execute(self, run_id: str) -> RunRecord:
        run = self.store.get_run(run_id)
        if run.status in {RunStatus.COMPLETED, RunStatus.ABORTED}:
            return run
        run.status = RunStatus.RUNNING
        run.current_agent = AgentRole.PLANNER.value
        run.last_error = ""
        self.store.update_run(run)

        try:
            while True:
                run = self.store.get_run(run_id)
                if self._budget_exhausted(run):
                    return self._stop_for_budget(run)

                all_tasks = self.store.list_tasks(run.id)
                pending_tasks = self.store.list_tasks(run.id, status=TaskStatus.PENDING)
                if not all_tasks:
                    await self._plan(run)
                    continue

                if pending_tasks:
                    finished = await self._research_pending(run, pending_tasks)
                    if finished.status in {RunStatus.FAILED, RunStatus.ABORTED, RunStatus.WAITING_REVIEW}:
                        return finished

                evidence = self.store.list_artifacts(run.id, kind="evidence")
                if not evidence:
                    return self._wait_for_review(run, "No evidence artifacts available to synthesize.")

                draft = await self._synthesize(run, evidence)
                issues = self._draft_quality_issues(draft.content)
                warnings = draft_quality_warnings(draft.content)
                if warnings:
                    self.store.append_trace(
                        TraceEvent.new(
                            run_id=run.id,
                            task_id="",
                            agent_role=AgentRole.CRITIC,
                            action=AgentAction.CRITIQUE,
                            message=f"Local quality gate warnings: {', '.join(warnings)}.",
                        )
                    )
                if issues:
                    run = self.store.get_run(run.id)
                    if run.round_index >= run.max_rounds:
                        return self._wait_for_review(run, f"Local quality gate failed: {'; '.join(issues)}")
                    run.round_index += 1
                    self.store.update_run(run)
                    for task in self._repair_tasks_for_issues(issues):
                        self.store.upsert_task(self._task_from_planned(run.id, run.round_index, task))
                    self.store.append_trace(
                        TraceEvent.new(
                            run_id=run.id,
                            task_id="",
                            agent_role=AgentRole.CRITIC,
                            action=AgentAction.CRITIQUE,
                            message=f"Local quality gate requested targeted repairs: {', '.join(issues)}.",
                        )
                    )
                    self.store.save_checkpoint(
                        run.id,
                        phase="revision_requested",
                        payload={"issues": issues, "round_index": run.round_index, "source": "local_quality_gate"},
                    )
                    continue
                run = self.store.get_run(run.id)
                review = await self._critique(run, draft.content, evidence)
                if review.approved:
                    run = self.store.get_run(run.id)
                    run.status = RunStatus.COMPLETED
                    run.current_agent = ""
                    self.store.update_run(run)
                    self.store.save_checkpoint(run.id, phase="completed", payload={"result_path": run.result_path})
                    return self.store.get_run(run.id)

                run = self.store.get_run(run.id)
                if run.round_index >= run.max_rounds:
                    return self._wait_for_review(run, review.feedback)
                run.round_index += 1
                self.store.update_run(run)
                for task in review.follow_up_tasks:
                    self.store.upsert_task(self._task_from_planned(run.id, run.round_index, task))
                self.store.save_checkpoint(
                    run.id,
                    phase="revision_requested",
                    payload={
                        "feedback": review.feedback,
                        "follow_up_count": len(review.follow_up_tasks),
                        "round_index": run.round_index,
                    },
                )
        except Exception as exc:
            run = self.store.get_run(run_id)
            run.status = RunStatus.FAILED
            run.current_agent = ""
            run.last_error = str(exc)
            self.store.update_run(run)
            self.store.save_checkpoint(run.id, phase="failed", payload={"error": str(exc)})
            return self.store.get_run(run.id)

    async def _plan(self, run: RunRecord) -> None:
        run.current_agent = AgentRole.PLANNER.value
        self.store.update_run(run)
        started = time.perf_counter()
        outcome = await self.agents.planner.plan(run.goal, run.round_index, self.store.list_artifacts(run.id))
        duration = int((time.perf_counter() - started) * 1000)
        self.store.increment_usage(run.id, outcome.token_usage, outcome.cost_usd)
        for planned in outcome.tasks:
            self.store.upsert_task(self._task_from_planned(run.id, run.round_index, planned))
        self.store.append_trace(
            TraceEvent.new(
                run_id=run.id,
                task_id="",
                agent_role=AgentRole.PLANNER,
                action=AgentAction.PLAN,
                message=outcome.notes or f"Planner produced {len(outcome.tasks)} tasks.",
                token_usage=outcome.token_usage,
                cost_usd=outcome.cost_usd,
                duration_ms=duration,
            )
        )
        self.store.save_checkpoint(
            run.id,
            phase="planned",
            payload={"task_count": len(outcome.tasks), "round_index": run.round_index},
        )

    async def _research_pending(self, run: RunRecord, tasks: list[TaskRecord]) -> RunRecord:
        run.current_agent = AgentRole.RESEARCHER.value
        self.store.update_run(run)

        for index in range(0, len(tasks), run.max_parallel_agents):
            batch = tasks[index:index + run.max_parallel_agents]
            for task in batch:
                self.store.mark_task_status(task.id, TaskStatus.IN_PROGRESS)
            results = await asyncio.gather(
                *(self._research_task(run.id, self.store.get_task(task.id)) for task in batch),
                return_exceptions=True,
            )
            for task, result in zip(batch, results, strict=True):
                if isinstance(result, Exception):
                    self.store.mark_task_status(task.id, TaskStatus.PENDING, last_error=str(result))
                    failed = self.store.get_run(run.id)
                    failed.status = RunStatus.FAILED
                    failed.current_agent = ""
                    failed.last_error = str(result)
                    self.store.update_run(failed)
                    self.store.save_checkpoint(failed.id, phase="failed", payload={"error": str(result), "task_id": task.id})
                    return failed
        self.store.save_checkpoint(
            run.id,
            phase="researched",
            payload={"evidence_count": len(self.store.list_artifacts(run.id, kind='evidence'))},
        )
        return self.store.get_run(run.id)

    async def _research_task(self, run_id: str, task: TaskRecord) -> None:
        started = time.perf_counter()
        outcome = await self.agents.researcher.research(self.store.get_run(run_id).goal, task, self.store.list_artifacts(run_id))
        duration = int((time.perf_counter() - started) * 1000)
        self.store.increment_usage(run_id, outcome.token_usage, outcome.cost_usd)
        evidence = ArtifactRecord.new(
            run_id=run_id,
            task_id=task.id,
            kind="evidence",
            title=task.title,
            content=outcome.summary,
            source_url=outcome.citations[0]["url"] if outcome.citations else "",
            citation_label=f"[{len(self.store.list_artifacts(run_id, kind='evidence')) + 1}]",
            citations=outcome.citations,
            metadata={"query": task.payload.get("query", "")},
        )
        self.store.save_artifact(evidence)
        self.store.mark_task_status(task.id, TaskStatus.DONE)
        self.store.append_trace(
            TraceEvent.new(
                run_id=run_id,
                task_id=task.id,
                agent_role=AgentRole.RESEARCHER,
                action=AgentAction.RESEARCH,
                message=f"Completed research for {task.title}.",
                token_usage=outcome.token_usage,
                cost_usd=outcome.cost_usd,
                duration_ms=duration,
            )
        )

    async def _synthesize(self, run: RunRecord, artifacts: list[ArtifactRecord]) -> ArtifactRecord:
        run.current_agent = AgentRole.SYNTHESIZER.value
        self.store.update_run(run)
        started = time.perf_counter()
        outcome = await self.agents.synthesizer.synthesize(run.goal, artifacts)
        duration = int((time.perf_counter() - started) * 1000)
        self.store.increment_usage(run.id, outcome.token_usage, outcome.cost_usd)
        artifact = ArtifactRecord.new(
            run_id=run.id,
            task_id="",
            kind="report",
            title="Research report",
            content=outcome.markdown,
            citations=[citation for item in artifacts for citation in item.citations],
        )
        self.store.save_artifact(artifact)
        self.store.append_trace(
            TraceEvent.new(
                run_id=run.id,
                task_id="",
                agent_role=AgentRole.SYNTHESIZER,
                action=AgentAction.SYNTHESIZE,
                message="Synthesized report draft.",
                token_usage=outcome.token_usage,
                cost_usd=outcome.cost_usd,
                duration_ms=duration,
            )
        )
        self.store.save_checkpoint(run.id, phase="synthesized", payload={"report_path": artifact.file_path})
        return artifact

    async def _critique(self, run: RunRecord, draft_markdown: str, artifacts: list[ArtifactRecord]):
        run = self.store.get_run(run.id)
        run.current_agent = AgentRole.CRITIC.value
        self.store.update_run(run)
        started = time.perf_counter()
        outcome = await self.agents.critic.critique(run.goal, draft_markdown, artifacts, run.round_index)
        duration = int((time.perf_counter() - started) * 1000)
        self.store.increment_usage(run.id, outcome.token_usage, outcome.cost_usd)
        self.store.append_trace(
            TraceEvent.new(
                run_id=run.id,
                task_id="",
                agent_role=AgentRole.CRITIC,
                action=AgentAction.CRITIQUE,
                message=outcome.feedback,
                token_usage=outcome.token_usage,
                cost_usd=outcome.cost_usd,
                duration_ms=duration,
            )
        )
        return outcome

    def _task_from_planned(self, run_id: str, round_index: int, planned: PlannedTask) -> TaskRecord:
        return TaskRecord.new(
            run_id=run_id,
            role=AgentRole.RESEARCHER,
            title=planned.title,
            description=planned.description or planned.query,
            round_index=round_index,
            payload={"query": planned.query},
        )

    def _budget_exhausted(self, run: RunRecord) -> bool:
        return run.spent_tokens >= run.budget_tokens or run.spent_cost_usd >= run.budget_cost_usd

    def _draft_quality_issues(self, draft_markdown: str) -> list[str]:
        return draft_quality_issues(draft_markdown)

    def _has_markdown_heading(self, text: str, heading: str) -> bool:
        return has_section_marker(text, heading)

    def _repair_tasks_for_issues(self, issues: list[str]) -> list[PlannedTask]:
        tasks: list[PlannedTask] = []
        if "missing findings" in issues or "report may be truncated" in issues:
            tasks.append(
                PlannedTask(
                    title="Repair incomplete report sections",
                    query="recover missing findings and finalize concise markdown report",
                    description="Patch truncated or incomplete report sections with concise evidence-backed text.",
                )
            )
        if "missing sources" in issues or "fewer than 3 source urls" in issues:
            tasks.append(
                PlannedTask(
                    title="Add verifiable sources",
                    query="find 3 authoritative urls supporting the current report claims",
                    description="Add exactly three verifiable source URLs for the key claims.",
                )
            )
        if "missing conclusion" in issues:
            tasks.append(
                PlannedTask(
                    title="Add concise conclusion",
                    query="write a concise evidence-based recommendation for the comparison",
                    description="Provide a short decision-oriented conclusion grounded in the evidence.",
                )
            )
        return tasks[:3]

    def _stop_for_budget(self, run: RunRecord) -> RunRecord:
        self.store.append_trace(
            TraceEvent.new(
                run_id=run.id,
                task_id="",
                agent_role=AgentRole.CRITIC,
                action=AgentAction.STOP,
                message="Run paused because the budget limit was exhausted.",
            )
        )
        return self._wait_for_review(run, "Budget limit exhausted.")

    def _wait_for_review(self, run: RunRecord, reason: str) -> RunRecord:
        run.status = RunStatus.WAITING_REVIEW
        run.current_agent = ""
        run.last_error = reason
        self.store.update_run(run)
        self.store.save_checkpoint(run.id, phase="waiting_review", payload={"reason": reason})
        return self.store.get_run(run.id)
