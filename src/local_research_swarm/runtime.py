from __future__ import annotations

import asyncio

from local_research_swarm.agents import CriticAgent, PlannerAgent, ResearcherAgent, SynthesizerAgent
from local_research_swarm.config import AppConfig
from local_research_swarm.models import AgentRegistry
from local_research_swarm.orchestrator import Orchestrator
from local_research_swarm.persistence import SQLiteRunStore
from local_research_swarm.providers import (
    BingSearchTool,
    DeterministicLLMBackend,
    DeterministicSearchTool,
    DeterministicWebReader,
    DuckDuckGoSearchTool,
    FallbackSearchTool,
    HttpWebReader,
    OpenAICompatibleLLMBackend,
)
from local_research_swarm.single_agent import SingleAgentRuntime


class SingleAgentService:
    def __init__(self, *, config: AppConfig, store: SQLiteRunStore, runtime: SingleAgentRuntime) -> None:
        self.config = config
        self.store = store
        self.runtime = runtime

    def submit(
        self,
        goal: str,
        *,
        max_rounds: int | None = None,
        max_parallel_agents: int | None = None,
        budget_tokens: int | None = None,
        budget_cost_usd: float | None = None,
        defer: bool = False,
    ):
        return asyncio.run(
            self.runtime.submit(
                goal=goal,
                max_rounds=max_rounds,
                max_parallel_agents=max_parallel_agents,
                budget_tokens=budget_tokens,
                budget_cost_usd=budget_cost_usd,
                defer=defer,
            )
        )

    def resume(self, run_id: str):
        return asyncio.run(self.runtime.resume(run_id))

    def abort(self, run_id: str):
        return self.runtime.abort(run_id)


class SwarmService:
    def __init__(
        self,
        *,
        config: AppConfig,
        store: SQLiteRunStore,
        orchestrator: Orchestrator,
        single_service: SingleAgentService,
    ) -> None:
        self.config = config
        self.store = store
        self.orchestrator = orchestrator
        self.single_service = single_service

    @classmethod
    def from_config(cls, config: AppConfig) -> "SwarmService":
        config.ensure_directories()
        store = SQLiteRunStore(config)
        if config.profile == "cloud" and config.api_key:
            llm_backend = OpenAICompatibleLLMBackend(
                api_key=config.api_key,
                base_url=config.base_url,
                model=config.model,
            )
            search_tool = FallbackSearchTool(DuckDuckGoSearchTool(), BingSearchTool())
            web_reader = HttpWebReader(retries=2)
        else:
            llm_backend = DeterministicLLMBackend()
            search_tool = DeterministicSearchTool()
            web_reader = DeterministicWebReader()
        registry = AgentRegistry(
            planner=PlannerAgent(llm_backend=llm_backend),
            researcher=ResearcherAgent(
                llm_backend=llm_backend,
                search_tool=search_tool,
                web_reader=web_reader,
                search_limit=config.default_search_limit,
            ),
            synthesizer=SynthesizerAgent(llm_backend=llm_backend),
            critic=CriticAgent(llm_backend=llm_backend),
        )
        single_runtime = SingleAgentRuntime(
            store=store,
            llm_backend=llm_backend,
            search_tool=search_tool,
            web_reader=web_reader,
            search_limit=config.default_search_limit,
        )
        return cls(
            config=config,
            store=store,
            orchestrator=Orchestrator(store=store, agents=registry),
            single_service=SingleAgentService(config=config, store=store, runtime=single_runtime),
        )

    def submit(
        self,
        goal: str,
        *,
        mode: str = "swarm",
        max_rounds: int | None = None,
        max_parallel_agents: int | None = None,
        budget_tokens: int | None = None,
        budget_cost_usd: float | None = None,
        defer: bool = False,
    ):
        if mode == "single":
            return self.single_service.submit(
                goal,
                max_rounds=max_rounds,
                max_parallel_agents=max_parallel_agents,
                budget_tokens=budget_tokens,
                budget_cost_usd=budget_cost_usd,
                defer=defer,
            )
        if mode != "swarm":
            raise ValueError(f"Unsupported mode: {mode}")
        return asyncio.run(
            self.orchestrator.submit(
                goal=goal,
                max_rounds=max_rounds,
                max_parallel_agents=max_parallel_agents,
                budget_tokens=budget_tokens,
                budget_cost_usd=budget_cost_usd,
                defer=defer,
            )
        )

    def resume(self, run_id: str):
        run = self.store.get_run(run_id)
        if run.runtime_mode == "single_agent":
            return self.single_service.resume(run_id)
        return asyncio.run(self.orchestrator.resume(run_id))

    def status(self, run_id: str):
        return self.store.get_run(run_id)

    def trace(self, run_id: str):
        return self.store.list_trace(run_id)

    def result(self, run_id: str) -> str:
        return self.store.get_result_artifact(run_id).content

    def abort(self, run_id: str):
        run = self.store.get_run(run_id)
        if run.runtime_mode == "single_agent":
            return self.single_service.abort(run_id)
        return self.orchestrator.abort(run_id)
