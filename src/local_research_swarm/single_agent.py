from __future__ import annotations

import time

from local_research_swarm.agents import PlannerAgent
from local_research_swarm.models import AgentAction, AgentRole, ArtifactRecord, PlanOutcome, RunRecord, RunStatus, TraceEvent
from local_research_swarm.orchestrator import draft_quality_issues, draft_quality_warnings
from local_research_swarm.persistence import SQLiteRunStore


class SingleAgentRuntime:
    def __init__(
        self,
        *,
        store: SQLiteRunStore,
        llm_backend,
        search_tool,
        web_reader,
        search_limit: int = 3,
    ) -> None:
        self.store = store
        self.llm_backend = llm_backend
        self.search_tool = search_tool
        self.web_reader = web_reader
        self.search_limit = max(3, search_limit)
        self.planner = PlannerAgent(llm_backend=llm_backend)

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
            runtime_mode="single_agent",
            max_rounds=max_rounds or self.store.config.default_max_rounds,
            max_parallel_agents=max_parallel_agents or self.store.config.default_max_parallel_agents,
            budget_tokens=budget_tokens or self.store.config.default_budget_tokens,
            budget_cost_usd=budget_cost_usd or self.store.config.default_budget_cost_usd,
            auto_execute=not defer,
        )
        self.store.save_checkpoint(
            run.id,
            phase="created",
            payload={"goal": goal, "defer": defer, "runtime_mode": "single_agent"},
        )
        if defer:
            return run
        return await self.execute(run.id)

    async def resume(self, run_id: str) -> RunRecord:
        return await self.execute(run_id)

    def abort(self, run_id: str) -> RunRecord:
        run = self.store.set_run_status(run_id, RunStatus.ABORTED)
        self.store.save_checkpoint(run_id, phase="aborted", payload={"status": run.status.value})
        self.store.append_trace(
            TraceEvent.new(
                run_id=run_id,
                task_id="",
                agent_role=AgentRole.SINGLE_AGENT,
                action=AgentAction.STOP,
                message="Single-agent run aborted by operator.",
            )
        )
        return self.store.get_run(run_id)

    async def execute(self, run_id: str) -> RunRecord:
        run = self.store.get_run(run_id)
        if run.status is RunStatus.ABORTED:
            return run
        run.status = RunStatus.RUNNING
        run.current_agent = AgentRole.SINGLE_AGENT.value
        run.last_error = ""
        self.store.update_run(run)

        try:
            if self._budget_exhausted(run):
                return self._stop_for_budget(run)

            plan = await self._plan(run)
            if self._budget_exhausted(self.store.get_run(run.id)):
                return self._stop_for_budget(self.store.get_run(run.id))

            pages = await self._research(run, plan)
            if self._budget_exhausted(self.store.get_run(run.id)):
                return self._stop_for_budget(self.store.get_run(run.id))
            if not pages:
                return self._wait_for_review(self.store.get_run(run.id), "No pages collected for single-agent research.")

            draft = await self._synthesize(run, pages)
            issues = draft_quality_issues(draft.content)
            warnings = draft_quality_warnings(draft.content)
            if warnings:
                self.store.append_trace(
                    TraceEvent.new(
                        run_id=run.id,
                        task_id="",
                        agent_role=AgentRole.SINGLE_AGENT,
                        action=AgentAction.CRITIQUE,
                        message=f"Local quality gate warnings: {', '.join(warnings)}.",
                    )
                )
            if issues:
                return self._wait_for_review(self.store.get_run(run.id), f"Local quality gate failed: {'; '.join(issues)}")

            run = self.store.get_run(run.id)
            run.status = RunStatus.COMPLETED
            run.current_agent = ""
            self.store.update_run(run)
            self.store.save_checkpoint(run.id, phase="completed", payload={"result_path": run.result_path})
            return self.store.get_run(run.id)
        except Exception as exc:
            run = self.store.get_run(run_id)
            run.status = RunStatus.FAILED
            run.current_agent = ""
            run.last_error = str(exc)
            self.store.update_run(run)
            self.store.save_checkpoint(run.id, phase="failed", payload={"error": str(exc)})
            return self.store.get_run(run.id)

    async def _plan(self, run: RunRecord) -> PlanOutcome:
        run.current_agent = AgentRole.SINGLE_AGENT.value
        self.store.update_run(run)
        started = time.perf_counter()
        outcome = await self.planner.plan(run.goal, run.round_index, self.store.list_artifacts(run.id))
        duration = int((time.perf_counter() - started) * 1000)
        self.store.increment_usage(run.id, outcome.token_usage, outcome.cost_usd)
        self.store.append_trace(
            TraceEvent.new(
                run_id=run.id,
                task_id="",
                agent_role=AgentRole.SINGLE_AGENT,
                action=AgentAction.PLAN,
                message=outcome.notes or f"Single-agent planner produced {len(outcome.tasks)} queries.",
                token_usage=outcome.token_usage,
                cost_usd=outcome.cost_usd,
                duration_ms=duration,
            )
        )
        self.store.save_checkpoint(
            run.id,
            phase="planned",
            payload={"query_count": len(outcome.tasks), "round_index": run.round_index, "runtime_mode": run.runtime_mode},
        )
        return outcome

    async def _research(self, run: RunRecord, plan: PlanOutcome):
        started = time.perf_counter()
        pages = []
        seen_urls: set[str] = set()
        for planned in plan.tasks[:3]:
            results = await self.search_tool.search(planned.query, limit=self.search_limit)
            for result in results:
                if result.url in seen_urls:
                    continue
                try:
                    page = await self.web_reader.read(result.url)
                except Exception:
                    continue
                seen_urls.add(result.url)
                pages.append(page)
                evidence = ArtifactRecord.new(
                    run_id=run.id,
                    task_id="",
                    kind="evidence",
                    title=page.title,
                    content=page.content,
                    source_url=page.url,
                    citation_label=f"[{len(self.store.list_artifacts(run.id, kind='evidence')) + 1}]",
                    citations=[{"title": page.title, "url": page.url, "snippet": page.content[:140]}],
                    metadata={"query": planned.query},
                )
                self.store.save_artifact(evidence)
        duration = int((time.perf_counter() - started) * 1000)
        self.store.append_trace(
            TraceEvent.new(
                run_id=run.id,
                task_id="",
                agent_role=AgentRole.SINGLE_AGENT,
                action=AgentAction.RESEARCH,
                message=f"Collected {len(pages)} pages for single-agent synthesis.",
                duration_ms=duration,
            )
        )
        self.store.save_checkpoint(
            run.id,
            phase="researched",
            payload={"page_count": len(pages), "evidence_count": len(self.store.list_artifacts(run.id, kind='evidence'))},
        )
        return pages

    async def _synthesize(self, run: RunRecord, pages) -> ArtifactRecord:
        started = time.perf_counter()
        raw = await self.llm_backend.single_agent_report(run.goal, pages)
        duration = int((time.perf_counter() - started) * 1000)
        token_usage = int(raw.get("token_usage", 0))
        cost_usd = float(raw.get("cost_usd", 0.0))
        markdown = str(raw.get("markdown", ""))
        self.store.increment_usage(run.id, token_usage, cost_usd)
        artifact = ArtifactRecord.new(
            run_id=run.id,
            task_id="",
            kind="report",
            title="Single-agent research report",
            content=markdown,
            citations=[
                {"title": page.title, "url": page.url, "snippet": page.content[:140]}
                for page in pages[:3]
            ],
        )
        self.store.save_artifact(artifact)
        self.store.append_trace(
            TraceEvent.new(
                run_id=run.id,
                task_id="",
                agent_role=AgentRole.SINGLE_AGENT,
                action=AgentAction.SYNTHESIZE,
                message="Synthesized single-agent report draft.",
                token_usage=token_usage,
                cost_usd=cost_usd,
                duration_ms=duration,
            )
        )
        self.store.save_checkpoint(run.id, phase="synthesized", payload={"report_path": artifact.file_path})
        return artifact

    def _budget_exhausted(self, run: RunRecord) -> bool:
        return run.spent_tokens >= run.budget_tokens or run.spent_cost_usd >= run.budget_cost_usd

    def _stop_for_budget(self, run: RunRecord) -> RunRecord:
        self.store.append_trace(
            TraceEvent.new(
                run_id=run.id,
                task_id="",
                agent_role=AgentRole.SINGLE_AGENT,
                action=AgentAction.STOP,
                message="Single-agent run paused because the budget limit was exhausted.",
            )
        )
        return self._wait_for_review(run, "Budget limit exhausted.")

    def _wait_for_review(self, run: RunRecord, reason: str) -> RunRecord:
        self.store.append_trace(
            TraceEvent.new(
                run_id=run.id,
                task_id="",
                agent_role=AgentRole.SINGLE_AGENT,
                action=AgentAction.STOP,
                message=reason,
            )
        )
        run.status = RunStatus.WAITING_REVIEW
        run.current_agent = ""
        run.last_error = reason
        self.store.update_run(run)
        self.store.save_checkpoint(run.id, phase="waiting_review", payload={"reason": reason})
        return self.store.get_run(run.id)
