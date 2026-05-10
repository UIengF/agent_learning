from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Type

from pydantic import BaseModel, Field, PrivateAttr

from .agent_executor import ToolExecutor
from .agent_policy import ToolPolicyEngine
from .agent_reflection import ReflectionRecorder
from .config import AppConfig, DEFAULT_TOP_K, build_app_config
from .context_budget import ContextBudget
from .context_builder import ContextBuildResult, build_context_messages
from .context_metrics import format_context_metrics
from .evidence_cache import build_evidence_cache, format_evidence_cache
from .indexing import load_index
from .permissions import build_permission_policy
from .question_frame import QuestionFrame, build_question_frame, format_question_frame
from .research_plan import ResearchPlanTool, format_research_plan, latest_research_plan
from .retrieval import Retriever, normalize_search_result
from .scholar_search import build_scholar_search_service
from .session_summary import build_session_summary, format_session_summary
from .scholar_tools import ScholarSearchTool
from .skills import LoadSkillTool, SkillRegistry
from .structured_trace import StructuredTraceWriter
from .task_state import build_task_state, format_task_state
from .token_estimation import HeuristicTokenEstimator, select_token_estimator
from .user_memory import UserMemory, format_user_memory, load_user_memory
from .web_fetch import fetch_url
from .web_runtime import build_configured_web_search_backend
from .web_tools import WebFetchTool, WebSearchTool

try:
    import operator
    from typing import Annotated, NotRequired, TypedDict

    from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
    from langchain_core.tools import BaseTool
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, StateGraph

    LANGGRAPH_AVAILABLE = True
except ImportError:  # pragma: no cover - keep helper tests importable without full runtime deps
    AnyMessage = Any

    @dataclass
    class _FallbackMessage:
        content: str = ""
        type: str = "assistant"

    @dataclass
    class HumanMessage(_FallbackMessage):
        type: str = "human"

    @dataclass
    class SystemMessage(_FallbackMessage):
        type: str = "system"

    @dataclass
    class ToolMessage(_FallbackMessage):
        tool_call_id: str = ""
        name: str = ""
        type: str = "tool"

    BaseTool = object  # type: ignore[assignment]
    ChatOpenAI = None
    StateGraph = None
    END = None
    Annotated = list  # type: ignore[assignment]
    TypedDict = dict  # type: ignore[assignment]
    operator = None
    LANGGRAPH_AVAILABLE = False


PROMPT = """\
You are a research assistant that answers questions using grounded evidence.

Use `local_rag_retrieve` first for information that may already exist in the local knowledge base.
Use `web_search` only for recent public information or when local evidence is missing.
Use `scholar_search` for requests about papers, literature reviews, citations, related work, or academic surveys.
Do not answer detailed factual questions from search snippets alone. If a snippet suggests the needed evidence is on a page, call `web_fetch` before relying on it.
Scholar results can be used directly as grounded paper metadata without fetching the paper page first.

Do not invent facts that are not supported by tool results. If evidence is incomplete, say what is missing.
If results contain multiple plausible answers, dates, or scenarios, call that out clearly before answering.
Unless the user explicitly asks for brevity, prefer a detailed, structured response.
For comparison questions, cover similarities, differences, implementation details, tradeoffs, and practical implications when the evidence supports them.
When evidence is sufficient, synthesize it into a multi-paragraph answer instead of a terse summary.
"""

if LANGGRAPH_AVAILABLE:

    class AgentState(TypedDict):
        messages: Annotated[list[AnyMessage], operator.add]
        current_question: NotRequired[str]
        research_plan: NotRequired[str]
        question_frame: NotRequired[str]
        task_state: NotRequired[str]
else:
    AgentState = dict[str, Any]


class LocalRAGInput(BaseModel):
    query: str = Field(..., description="Query used to search the local knowledge base.")
    top_k: int = Field(
        DEFAULT_TOP_K, ge=1, le=8, description="Number of relevant passages to return."
    )


class LocalRAGRetrieveTool(BaseTool):
    name: str = "local_rag_retrieve"
    description: str = "Retrieve relevant passages from the local knowledge base."
    args_schema: Type[BaseModel] = LocalRAGInput

    _store: Retriever = PrivateAttr()
    _min_evidence_score: float = PrivateAttr(default=0.0)

    def __init__(self, store: Retriever, **kwargs):
        min_evidence_score = float(kwargs.pop("min_evidence_score", 0.0))
        super().__init__(**kwargs)
        self._store = store
        self._min_evidence_score = max(0.0, min_evidence_score)

    def _run(self, query: str, top_k: int = DEFAULT_TOP_K) -> str:
        raw_results = self._store.search(query, top_k=top_k)
        results = []
        for item in raw_results:
            normalized = normalize_search_result(item)
            if normalized.score >= self._min_evidence_score:
                results.append(normalized)

        payload = {
            "query": query,
            "result_count": len(results),
            "reason": None if results else "insufficient_evidence",
            "results": [asdict(item) for item in results],
        }
        return json.dumps(payload, ensure_ascii=False)


class Agent:
    def __init__(
        self,
        model=None,
        tools=None,
        checkpointer: Any | None = None,
        system: str = "",
        log_path: Path | None = None,
        ensure_log_file: Any | None = None,
        append_log: Any | None = None,
        shorten_text: Any | None = None,
        max_rounds: int = 3,
        max_recent_messages: int | None = None,
        recent_full_turns: int | None = None,
        max_context_chars: int | None = None,
        max_context_tokens: int | None = None,
        live_messages_compression_enabled: bool = True,
        live_messages_keep_turns: int = 1,
        live_messages_max_fetch_chars: int = 180,
        live_messages_max_search_results: int = 3,
        user_memory: UserMemory | None = None,
        token_estimator: HeuristicTokenEstimator | None = None,
        skill_registry: SkillRegistry | None = None,
        trace_writer: StructuredTraceWriter | None = None,
    ):
        self.system = system
        self.log_path = log_path or Path("runtime") / "logs" / "graph_rag.log"
        self.model_call_count = 0
        self.tool_call_count = 0
        self.current_round = 0
        self.max_rounds = max(1, int(max_rounds))
        self.max_recent_messages = (
            max(1, int(max_recent_messages)) if max_recent_messages is not None else None
        )
        self.recent_full_turns = (
            max(1, int(recent_full_turns)) if recent_full_turns is not None else 3
        )
        self.max_context_chars = (
            max(200, int(max_context_chars)) if max_context_chars is not None else 12000
        )
        self.max_context_tokens = (
            max(100, int(max_context_tokens)) if max_context_tokens is not None else None
        )
        self.live_messages_compression_enabled = bool(live_messages_compression_enabled)
        self.live_messages_keep_turns = max(1, int(live_messages_keep_turns))
        self.live_messages_max_fetch_chars = max(80, int(live_messages_max_fetch_chars))
        self.live_messages_max_search_results = max(1, int(live_messages_max_search_results))
        self.user_memory = user_memory or UserMemory()
        self.token_estimator = token_estimator or HeuristicTokenEstimator()
        self.skill_registry = skill_registry
        self.trace_writer = trace_writer
        self.tools = {tool.name: tool for tool in (tools or [])}
        self.base_model = model
        self.graph = None
        self._ensure_log_file = ensure_log_file
        self._append_log = append_log
        self._shorten_text = shorten_text
        self.tool_policy = ToolPolicyEngine(append_log=self._append)
        self.tool_executor = ToolExecutor(
            tools=self.tools,
            tool_policy=self.tool_policy,
            append_log=self._append,
            trace=self._trace,
            shorten=self._shorten,
            increment_tool_call_count=self._increment_tool_call_count,
        )
        self.reflection_recorder = ReflectionRecorder(
            tool_policy=self.tool_policy,
            append=self._append,
            append_json=self._append_json,
            shorten=self._shorten,
            build_task_state=self._build_task_state,
            build_evidence_cache=self._build_evidence_cache,
            get_current_round=lambda: self.current_round,
        )

        if self.base_model is not None and tools is not None and LANGGRAPH_AVAILABLE:
            graph = StateGraph(AgentState)
            graph.add_node("llm", self.call_openai)
            graph.add_node("action", self.take_action)
            graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
            graph.add_edge("action", "llm")
            graph.set_entry_point("llm")
            try:
                self.graph = graph.compile(checkpointer=checkpointer)
            except TypeError:
                self.graph = graph.compile()
            self.model = model.bind_tools(tools)
        else:
            self.model = None

        if self._ensure_log_file is not None:
            self._ensure_log_file(self.log_path)

    def __getattr__(self, name: str) -> Any:
        # Policy engine methods
        policy_methods = {
            "_remove_site_filter",
            "_parse_tool_payload",
            "_url_domain",
            "_message_content",
            "_tool_name",
            "_failed_web_fetch_urls",
            "_failed_web_fetch_domains",
            "_format_fetch_failure_guidance",
            "_official_search_urls",
            "_ranked_web_search_urls",
            "_web_fetch_fallback_urls",
            "_latest_tool_snapshot",
            "_current_question",
            "_format_scholar_title_guardrail",
            "_apply_official_first_fetch_policy",
            "_apply_failed_domain_search_policy",
        }
        if name in policy_methods:
            return getattr(self.tool_policy, name)

        # Executor static methods
        executor_static = {
            "_tool_limit_message_text": "tool_limit_message_text",
            "_finalize_without_tool_calls": "finalize_without_tool_calls",
        }
        if name in executor_static:
            return getattr(ToolExecutor, executor_static[name])

        # Reflection methods
        reflection_methods = {
            "build_reflection_prompt": "build_reflection_prompt",
            "log_round_header": "log_round_header",
            "_log_question_frame": "log_question_frame",
            "_build_reflection_record": "build_reflection_record",
            "_log_reflection_record": "log_reflection_record",
        }
        if name in reflection_methods:
            return getattr(self.reflection_recorder, reflection_methods[name])
        raise AttributeError(f"{self.__class__.__name__!s} object has no attribute {name!r}")

    @staticmethod
    def _message_role(message: AnyMessage | dict) -> str:
        if isinstance(message, dict):
            return str(message.get("role", "unknown"))
        message_type = getattr(message, "type", None)
        if message_type:
            return str(message_type)
        return message.__class__.__name__

    def _append(self, text: str) -> None:
        if self._append_log is not None:
            self._append_log(self.log_path, text)

    def _append_json(self, title: str, payload: dict[str, Any]) -> None:
        self._append(f"{title}\n{json.dumps(payload, ensure_ascii=False, sort_keys=True)}")

    def _trace(self, event_type: str, payload: dict[str, Any]) -> None:
        if self.trace_writer is not None:
            self.trace_writer.append(event_type, payload)

    def _increment_tool_call_count(self) -> int:
        self.tool_call_count += 1
        return self.tool_call_count

    def _shorten(self, text: str, max_len: int) -> str:
        if self._shorten_text is None:
            compact = " ".join(text.split())
            return compact if len(compact) <= max_len else compact[: max_len - 3] + "..."
        return self._shorten_text(text, max_len)

    @staticmethod
    def _is_tool_message(message: AnyMessage | dict) -> bool:
        if ToolMessage is not None and isinstance(message, ToolMessage):
            return True
        if isinstance(message, dict):
            return str(message.get("role", "")) == "tool"
        return str(getattr(message, "type", "")) == "tool"

    @staticmethod
    def _is_human_message(message: AnyMessage | dict) -> bool:
        if HumanMessage is not None and isinstance(message, HumanMessage):
            return True
        if isinstance(message, dict):
            return str(message.get("role", "")).lower() in {"human", "user"}
        return str(getattr(message, "type", "")).lower() in {"human", "user"}

    def _current_call_messages(self, state: AgentState) -> list[AnyMessage | dict]:
        messages = list(state["messages"])
        for index in range(len(messages) - 1, -1, -1):
            if self._is_human_message(messages[index]):
                return messages[index:]
        return messages

    def _tool_rounds_in_current_call(self, state: AgentState) -> int:
        messages = self._current_call_messages(state)
        rounds = 0
        previous_was_tool = False
        for message in messages:
            is_tool = self._is_tool_message(message)
            if is_tool and not previous_was_tool:
                rounds += 1
            previous_was_tool = is_tool
        return rounds

    def _trim_recent_messages(
        self,
        messages: list[AnyMessage | dict],
    ) -> tuple[list[AnyMessage | dict], list[AnyMessage | dict]]:
        if self.max_recent_messages is None or len(messages) <= self.max_recent_messages:
            return [], messages

        anchor = messages[0] if messages and self._is_human_message(messages[0]) else None
        if anchor is None:
            split_at = max(0, len(messages) - self.max_recent_messages)
            return messages[:split_at], messages[split_at:]

        tail_budget = max(0, self.max_recent_messages - 1)
        tail_start = max(1, len(messages) - tail_budget)
        tail = messages[tail_start:] if tail_budget else []
        if any(message is anchor for message in tail):
            return messages[:tail_start], tail
        return messages[1:tail_start], [anchor] + tail

    def _split_message_turns(
        self, messages: list[AnyMessage | dict]
    ) -> list[list[AnyMessage | dict]]:
        if not messages:
            return []

        turns: list[list[AnyMessage | dict]] = []
        current_turn: list[AnyMessage | dict] = []
        for message in messages:
            if self._is_human_message(message):
                if current_turn:
                    turns.append(current_turn)
                current_turn = [message]
                continue

            if not current_turn:
                current_turn = [message]
            else:
                current_turn.append(message)

        if current_turn:
            turns.append(current_turn)
        return turns

    def _select_recent_turns(
        self,
        messages: list[AnyMessage | dict],
    ) -> tuple[list[AnyMessage | dict], list[AnyMessage | dict]]:
        turns = self._split_message_turns(messages)
        if not turns or len(turns) <= self.recent_full_turns:
            return [], messages

        older_turns = turns[: -self.recent_full_turns]
        recent_turns = turns[-self.recent_full_turns :]
        older_messages = [message for turn in older_turns for message in turn]
        recent_messages = [message for turn in recent_turns for message in turn]
        return older_messages, recent_messages

    def _build_session_summary(self, messages: list[AnyMessage | dict]):
        return build_session_summary(
            messages,
            message_content=self.tool_policy._message_content,
            message_role=self._message_role,
            is_tool_message=self._is_tool_message,
            is_human_message=self._is_human_message,
            tool_name=self.tool_policy._tool_name,
            shorten=self._shorten,
        )

    def _build_task_state(self, messages: list[AnyMessage | dict]):
        return build_task_state(
            messages,
            message_content=self.tool_policy._message_content,
            is_tool_message=self._is_tool_message,
            is_human_message=self._is_human_message,
            tool_name=self.tool_policy._tool_name,
        )

    def _extract_question_text(self, messages: list[AnyMessage | dict]) -> str:
        for message in reversed(messages):
            if self._is_human_message(message):
                content = self.tool_policy._message_content(message).strip()
                if content:
                    return content
        return ""

    def _sync_derived_state(self, state: AgentState) -> AgentState:
        if not state.get("current_question"):
            state["current_question"] = self._extract_question_text(state["messages"])
        if not state.get("research_plan"):
            state["research_plan"] = format_research_plan(
                latest_research_plan(list(state["messages"]))
            )
        if not state.get("question_frame"):
            question_frame = self._build_question_frame(list(state["messages"]))
            if question_frame:
                state["question_frame"] = format_question_frame(question_frame)
        if not state.get("task_state"):
            task_state = self._build_task_state(list(state["messages"]))
            if task_state:
                state["task_state"] = format_task_state(
                    task_state, shorten=self._shorten, max_chars=self.max_context_chars
                )
        return state

    def _build_question_frame(self, messages: list[AnyMessage | dict]) -> QuestionFrame | None:
        question = self._extract_question_text(messages)
        if question:
            return build_question_frame(question)
        return None

    def _build_evidence_cache(self, messages: list[AnyMessage | dict]):
        return build_evidence_cache(
            messages,
            message_content=self.tool_policy._message_content,
            is_tool_message=self._is_tool_message,
            tool_name=self.tool_policy._tool_name,
        )

    def _answer_after_tool_limit(
        self,
        *,
        messages: list[Any],
        trigger_message: Any,
        limit_message: str,
        completed_rounds: int,
        scholar_title_guardrail: str | None = None,
    ) -> Any:
        return self.tool_executor.answer_after_tool_limit(
            messages=messages,
            trigger_message=trigger_message,
            limit_message=limit_message,
            completed_rounds=completed_rounds,
            final_model=self.base_model or self.model,
            scholar_title_guardrail=scholar_title_guardrail,
        )

    def exists_action(self, state: AgentState):
        result = state["messages"][-1]
        return len(getattr(result, "tool_calls", [])) > 0

    def build_context_result(self, state: AgentState) -> ContextBuildResult:
        trimmed_messages, base_messages = self._select_recent_turns(list(state["messages"]))
        evidence_cache = self._build_evidence_cache(list(state["messages"]))
        question_frame = self._build_question_frame(list(state["messages"]))
        tool_results = []
        for message in reversed(base_messages):
            if self._is_tool_message(message):
                tool_results.append(message)
                continue
            break

        summary = self._build_session_summary(trimmed_messages)
        summary_text = None
        if summary is not None:
            summary_text = format_session_summary(
                summary,
                shorten=self._shorten,
                max_chars=self.max_context_chars,
            )

        user_memory_text = format_user_memory(
            self.user_memory,
            shorten=self._shorten,
            max_chars=self.max_context_chars,
        )
        question_frame_text = None
        if question_frame is not None:
            question_frame_text = format_question_frame(question_frame)

        evidence_cache_text = format_evidence_cache(evidence_cache)
        skill_inventory_text = (
            self.skill_registry.format_inventory() if self.skill_registry is not None else None
        )
        research_plan_text = format_research_plan(latest_research_plan(list(state["messages"])))

        task_state = self._build_task_state(base_messages)
        task_state_text = None
        if task_state is not None:
            task_state_text = format_task_state(
                task_state,
                shorten=self._shorten,
                max_chars=self.max_context_chars,
            )
        reflection_text = None
        if tool_results:
            tool_results.reverse()
            reflection_text = self.build_reflection_prompt(tool_results)
            fetch_failure_guidance = self.tool_policy._format_fetch_failure_guidance(state)
            if fetch_failure_guidance:
                reflection_text = reflection_text + "\n\n" + fetch_failure_guidance
            scholar_title_guardrail = self.tool_policy._format_scholar_title_guardrail(
                list(state["messages"])
            )
            if scholar_title_guardrail:
                reflection_text = reflection_text + "\n\n" + scholar_title_guardrail

        return build_context_messages(
            base_messages=base_messages,
            system_text=self.system,
            summary_text=summary_text,
            question_frame_text=question_frame_text,
            user_memory_text=user_memory_text,
            skill_inventory_text=skill_inventory_text,
            evidence_cache_text=evidence_cache_text,
            research_plan_text=research_plan_text,
            task_state_text=task_state_text,
            reflection_text=reflection_text,
            budget=ContextBudget(
                max_chars=self.max_context_chars,
                max_tokens=self.max_context_tokens,
            ),
            shorten=self._shorten,
            message_content=self.tool_policy._message_content,
            message_role=self._message_role,
            token_estimator=self.token_estimator,
            live_messages_compression_enabled=self.live_messages_compression_enabled,
            live_messages_keep_turns=self.live_messages_keep_turns,
            live_messages_max_fetch_chars=self.live_messages_max_fetch_chars,
            live_messages_max_search_results=self.live_messages_max_search_results,
            system_message_factory=SystemMessage,
            human_message_factory=HumanMessage,
        )

    def build_llm_messages(self, state: AgentState) -> list[AnyMessage]:
        return self.build_context_result(state).messages

    def call_openai(self, state: AgentState):
        state = self._sync_derived_state(state)
        if self.model is None:
            raise RuntimeError("Model is required to run graph_rag agent.")

        last_message = state["messages"][-1]
        if ToolMessage is not None and isinstance(last_message, ToolMessage):
            self._append(
                "Reflection step\n"
                "The model will decide whether to continue using tools or answer directly."
            )
            reflection_log_prompt = self.build_reflection_prompt([last_message])
            fetch_failure_guidance = self.tool_policy._format_fetch_failure_guidance(state)
            if fetch_failure_guidance:
                reflection_log_prompt = reflection_log_prompt + "\n\n" + fetch_failure_guidance
            self._append(f"Reflection prompt\n{self._shorten(reflection_log_prompt, 1200)}")
            self.current_round += 1

        context_result = self.build_context_result(state)
        messages = context_result.messages
        self.model_call_count += 1
        role = self._message_role(last_message)
        content = self._shorten(self.tool_policy._message_content(last_message), 800)
        if self.model_call_count == 1:
            self.current_round = 1
            self.log_round_header("Start analyzing the user question")
            self._log_question_frame(self._build_question_frame(list(state["messages"])))

        if context_result.metrics is not None:
            self._append(format_context_metrics(context_result.metrics))
            self._trace(
                "context_built",
                {
                    "model_call_count": self.model_call_count,
                    "layer_names": [layer.name for layer in context_result.layers],
                    "dropped_layer_names": [layer.name for layer in context_result.dropped_layers],
                    "estimated_total_chars": context_result.estimated_total_chars,
                    "estimated_total_tokens": context_result.estimated_total_tokens,
                },
            )
        self._append(f"LLM input source: {role}\nLLM input content:\n{content}")
        message = self.model.invoke(messages)
        response_text = self.tool_policy._message_content(message)
        tool_calls = getattr(message, "tool_calls", [])
        completed_rounds = self._tool_rounds_in_current_call(state)
        if tool_calls and completed_rounds >= self.max_rounds:
            limited_content = self._tool_limit_message_text()
            self._log_reflection_record(
                state,
                llm_decision="final_after_tool_limit",
                tool_calls=tool_calls,
            )
            self._append(
                "LLM decision: stop calling tools because the limit was reached.\n"
                f"Completed rounds: {completed_rounds}\n"
                "Switching to final answer synthesis without additional tools."
            )
            return {
                "messages": [
                    self._answer_after_tool_limit(
                        messages=messages,
                        trigger_message=message,
                        limit_message=limited_content,
                        completed_rounds=completed_rounds,
                        scholar_title_guardrail=self.tool_policy._format_scholar_title_guardrail(
                            list(state["messages"])
                        ),
                    )
                ]
            }

        if tool_calls:
            message, tool_calls = self.tool_policy._apply_failed_domain_search_policy(
                state, message, tool_calls
            )
            message, tool_calls = self.tool_policy._apply_official_first_fetch_policy(
                state, message, tool_calls
            )

        if tool_calls:
            self._log_reflection_record(state, llm_decision="tool_use", tool_calls=tool_calls)
            self._trace(
                "llm_decision",
                {
                    "decision": "tool_use",
                    "model_call_count": self.model_call_count,
                    "tool_calls": [
                        {"name": call.get("name", ""), "args": dict(call.get("args", {}))}
                        for call in tool_calls
                    ],
                },
            )
            first_query = tool_calls[0].get("args", {}).get("query", "")
            self._append(
                "LLM decision: continue with tool use.\n"
                f"Intent for this round:\n{self._shorten(response_text, 500)}\n"
                f"Primary query:\n{first_query}"
            )
        else:
            self._log_reflection_record(state, llm_decision="answer", tool_calls=[])
            self._trace(
                "llm_decision",
                {
                    "decision": "answer",
                    "model_call_count": self.model_call_count,
                    "answer_preview": self._shorten(response_text, 500),
                },
            )
            self._append(
                "LLM decision: answer directly.\n"
                f"Output summary:\n{self._shorten(response_text, 800)}"
            )
        self._append(f"LLM raw output:\n{self._shorten(response_text, 1200)}")
        return {"messages": [message]}

    def take_action(self, state: AgentState):
        state = self._sync_derived_state(state)
        evidence_cache = self._build_evidence_cache(list(state["messages"]))
        return self.tool_executor.take_action(state, evidence_cache)


def build_agent(
    index_dir: str | Path,
    checkpointer: Any | None = None,
    app_config: AppConfig | None = None,
    *,
    ensure_log_file: Any | None = None,
    append_log: Any | None = None,
    shorten_text: Any | None = None,
    trace_writer: StructuredTraceWriter | None = None,
) -> Agent:
    if ChatOpenAI is None:
        raise ImportError("Missing runtime dependencies for LangGraph execution.")

    config = app_config or build_app_config(index_dir)
    permission_policy = build_permission_policy(config.permissions)
    permission_policy.validate_index_dir(index_dir)
    if not config.model.api_key:
        raise EnvironmentError(
            "Missing RAG_MODEL_API_KEY or DASHSCOPE_API_KEY environment variable."
        )

    model = ChatOpenAI(
        model=config.model.model_name,
        openai_api_key=config.model.api_key,
        openai_api_base=config.model.api_base,
    )
    tools = [
        LocalRAGRetrieveTool(
            store=load_index(
                index_dir,
                keyword_weight=config.retrieval.keyword_weight,
                metadata_rerank_config=config.metadata_rerank,
            ),
            min_evidence_score=config.generation.min_evidence_score,
        )
    ]
    skill_registry = SkillRegistry(config.harness.skills_dir)
    if skill_registry.available:
        tools.append(LoadSkillTool(registry=skill_registry))
    tools.append(ResearchPlanTool())

    if config.web.enabled:
        backend = build_configured_web_search_backend(config.web)

        def fetcher(url: str):
            return fetch_url(
                permission_policy.validate_web_fetch_url(url),
                timeout_seconds=config.web.fetch_timeout_seconds,
                max_bytes=config.web.fetch_max_bytes,
                max_chars=config.web.fetch_max_chars,
                user_agent=config.web.user_agent,
                redirect_validator=permission_policy.validate_web_fetch_url,
            )

        tools.extend(
            [
                WebSearchTool(backend=backend, default_top_k=config.web.search_top_k),
            ]
        )
        if config.scholar.enabled:
            tools.append(
                ScholarSearchTool(
                    searcher=build_scholar_search_service(config),
                    default_count=config.scholar.default_count,
                )
            )
        tools.append(WebFetchTool(fetcher=fetcher))

    return Agent(
        model,
        tools,
        checkpointer=checkpointer,
        system=PROMPT,
        ensure_log_file=ensure_log_file,
        append_log=append_log,
        shorten_text=shorten_text,
        max_rounds=config.generation.max_rounds,
        max_recent_messages=config.context.max_recent_messages,
        recent_full_turns=config.context.recent_full_turns,
        max_context_chars=config.context.max_context_chars,
        max_context_tokens=config.context.max_context_tokens,
        live_messages_compression_enabled=config.context.live_messages_compression_enabled,
        live_messages_keep_turns=config.context.live_messages_keep_turns,
        live_messages_max_fetch_chars=config.context.live_messages_max_fetch_chars,
        live_messages_max_search_results=config.context.live_messages_max_search_results,
        user_memory=load_user_memory(
            config.runtime.user_memory_path,
            user_id=config.runtime.user_id,
        ),
        token_estimator=select_token_estimator(config.model.model_name),
        skill_registry=skill_registry,
        trace_writer=trace_writer,
    )
