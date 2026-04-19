from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Type

from pydantic import BaseModel, Field, PrivateAttr

from .config import AppConfig, DEFAULT_TOP_K, build_app_config
from .context_budget import ContextBudget
from .context_builder import ContextBuildResult, build_context_messages
from .context_metrics import format_context_metrics
from .evidence_cache import build_evidence_cache, format_evidence_cache, lookup_cached_tool_result
from .indexing import load_index
from .question_frame import QuestionFrame, build_question_frame, format_question_frame
from .retrieval import Retriever, normalize_search_result
from .scholar_search import build_scholar_search_service
from .session_summary import build_session_summary, format_session_summary
from .scholar_tools import ScholarSearchTool
from .task_state import build_task_state, format_task_state
from .token_estimation import HeuristicTokenEstimator, select_token_estimator
from .user_memory import UserMemory, format_user_memory, load_user_memory
from .web_fetch import fetch_url
from .web_runtime import build_configured_web_search_backend
from .web_tools import WebFetchTool, WebSearchTool

try:
    import operator
    from typing import Annotated, TypedDict

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

REFLECTION_PROMPT_TEMPLATE = """\
You just received tool results. First decide:
1. What is already answered by the current evidence?
2. What information is still missing to complete the task?
3. What is the smallest necessary next step?

If the current evidence is sufficient, answer directly and do not call another tool.
If the evidence is insufficient, make the next query or fetch step more specific.
Do not treat unsupported facts as known facts.
If search snippets hint at needed details on a page, fetch the page before relying on those details.
If multiple answers, dates, or scenarios are plausible, explain that clearly.
"""


if LANGGRAPH_AVAILABLE:
    class AgentState(TypedDict):
        messages: Annotated[list[AnyMessage], operator.add]
else:
    AgentState = dict[str, Any]


class LocalRAGInput(BaseModel):
    query: str = Field(..., description="Query used to search the local knowledge base.")
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=8, description="Number of relevant passages to return.")


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


@dataclass(frozen=True)
class ReflectionRecord:
    stage: str
    question: str
    entities: tuple[str, ...]
    evidence_sufficiency: str
    missing_information: tuple[str, ...]
    recommended_next_action: str
    latest_tool_name: str
    latest_tool_result_count: int | None
    cached_web_queries: tuple[str, ...]
    cached_fetched_urls: tuple[str, ...]
    llm_decision: str
    tool_calls: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class LoggedQuestionFrame:
    question: str
    target_entities: tuple[str, ...]
    task_intent: str
    focus_dimensions: tuple[str, ...]
    evidence_scope: dict[str, bool]
    success_criteria: tuple[str, ...]


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
        self.tools = {tool.name: tool for tool in (tools or [])}
        self.base_model = model
        self.graph = None
        self._ensure_log_file = ensure_log_file
        self._append_log = append_log
        self._shorten_text = shorten_text

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

    @staticmethod
    def _message_role(message: AnyMessage | dict) -> str:
        if isinstance(message, dict):
            return str(message.get("role", "unknown"))
        message_type = getattr(message, "type", None)
        if message_type:
            return str(message_type)
        return message.__class__.__name__

    @staticmethod
    def _message_content(message: AnyMessage | dict) -> str:
        if isinstance(message, dict):
            content = message.get("content", "")
        else:
            content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        return str(content)

    def _append(self, text: str) -> None:
        if self._append_log is not None:
            self._append_log(self.log_path, text)

    def _append_json(self, title: str, payload: dict[str, Any]) -> None:
        self._append(f"{title}\n{json.dumps(payload, ensure_ascii=False, sort_keys=True)}")

    def _shorten(self, text: str, max_len: int) -> str:
        if self._shorten_text is None:
            compact = " ".join(text.split())
            return compact if len(compact) <= max_len else compact[: max_len - 3] + "..."
        return self._shorten_text(text, max_len)

    def log_round_header(self, title: str) -> None:
        self._append(f"===== Round {self.current_round} =====\n{title}")

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

    def _tool_name(self, message: AnyMessage | dict) -> str:
        if isinstance(message, dict):
            return str(message.get("name", "tool"))
        return str(getattr(message, "name", "tool"))

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

    def _split_message_turns(self, messages: list[AnyMessage | dict]) -> list[list[AnyMessage | dict]]:
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
            message_content=self._message_content,
            message_role=self._message_role,
            is_tool_message=self._is_tool_message,
            is_human_message=self._is_human_message,
            tool_name=self._tool_name,
            shorten=self._shorten,
        )

    def _build_task_state(self, messages: list[AnyMessage | dict]):
        return build_task_state(
            messages,
            message_content=self._message_content,
            is_tool_message=self._is_tool_message,
            is_human_message=self._is_human_message,
            tool_name=self._tool_name,
        )

    def _extract_question_text(self, messages: list[AnyMessage | dict]) -> str:
        for message in reversed(messages):
            if self._is_human_message(message):
                content = self._message_content(message).strip()
                if content:
                    return content
        return ""

    def _build_question_frame(self, messages: list[AnyMessage | dict]) -> QuestionFrame | None:
        question = self._extract_question_text(messages)
        if question:
            return build_question_frame(question)
        return None

    def _log_question_frame(self, question_frame: QuestionFrame | None) -> None:
        if question_frame is None:
            return
        payload = LoggedQuestionFrame(
            question=question_frame.question,
            target_entities=question_frame.target_entities,
            task_intent=question_frame.task_intent,
            focus_dimensions=question_frame.focus_dimensions,
            evidence_scope=asdict(question_frame.evidence_scope),
            success_criteria=question_frame.success_criteria,
        )
        self._append_json("Question frame", asdict(payload))

    def _build_evidence_cache(self, messages: list[AnyMessage | dict]):
        return build_evidence_cache(
            messages,
            message_content=self._message_content,
            is_tool_message=self._is_tool_message,
            tool_name=self._tool_name,
        )

    @staticmethod
    def _tool_limit_message_text() -> str:
        return (
            "The tool call limit has been reached. Please answer using the available evidence "
            "and clearly state any uncertainty."
        )

    def _finalize_without_tool_calls(self, message: Any, content: str) -> Any:
        try:
            setattr(message, "content", content)
            setattr(message, "tool_calls", [])
            return message
        except Exception:
            pass

        message_type = message.__class__
        try:
            return message_type(content=content)
        except Exception:
            pass

        if HumanMessage is not None:
            return HumanMessage(content=content)
        return {"role": "assistant", "content": content}

    def _answer_after_tool_limit(
        self,
        *,
        messages: list[Any],
        trigger_message: Any,
        limit_message: str,
        completed_rounds: int,
    ) -> Any:
        final_instruction = (
            f"{limit_message}\n\n"
            "Do not call any more tools. Write the final answer now using only the evidence "
            "already present in the conversation. If the evidence is incomplete, state the "
            "uncertainty clearly."
        )
        final_messages = list(messages)
        if HumanMessage is not None:
            final_messages.append(HumanMessage(content=final_instruction))
        else:
            final_messages.append({"role": "human", "content": final_instruction})

        final_model = self.base_model or self.model
        try:
            final_message = final_model.invoke(final_messages)
        except Exception as exc:
            self._append(
                "LLM decision: failed to synthesize final answer after tool limit.\n"
                f"Completed rounds: {completed_rounds}\n"
                f"Error: {exc}"
            )
            return self._finalize_without_tool_calls(trigger_message, limit_message)

        final_content = self._message_content(final_message).strip() or limit_message
        self._append(
            "LLM decision: synthesize final answer after tool limit.\n"
            f"Completed rounds: {completed_rounds}\n"
            f"Output summary:\n{self._shorten(final_content, 800)}"
        )
        self._append(f"LLM raw output:\n{self._shorten(final_content, 1200)}")
        return self._finalize_without_tool_calls(final_message, final_content)

    def _replace_tool_calls(
        self,
        message: Any,
        tool_calls: list[dict[str, Any]],
        *,
        content: str | None = None,
    ) -> Any:
        updated_content = self._message_content(message) if content is None else content
        try:
            setattr(message, "content", updated_content)
            setattr(message, "tool_calls", tool_calls)
            return message
        except Exception:
            pass

        try:
            return message.__class__(content=updated_content, tool_calls=tool_calls)
        except Exception:
            pass

        return {"role": "assistant", "content": updated_content, "tool_calls": tool_calls}

    @staticmethod
    def _parse_tool_payload(content: str) -> dict[str, Any]:
        try:
            payload = json.loads(content)
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _failed_web_fetch_urls(self, state: AgentState) -> set[str]:
        failed_urls: set[str] = set()
        for message in self._current_call_messages(state):
            if not self._is_tool_message(message):
                continue
            if self._tool_name(message) != "web_fetch":
                continue
            payload = self._parse_tool_payload(self._message_content(message))
            if payload.get("error") != "tool_execution_failed":
                continue
            tool_args = payload.get("tool_args", {})
            if not isinstance(tool_args, dict):
                continue
            url = str(tool_args.get("url", "")).strip()
            if url:
                failed_urls.add(url)
        return failed_urls

    def _official_search_urls(self, state: AgentState) -> list[str]:
        for message in reversed(self._current_call_messages(state)):
            if not self._is_tool_message(message):
                continue
            if self._tool_name(message) != "web_search":
                continue

            payload = self._parse_tool_payload(self._message_content(message))
            urls: list[str] = []
            results = payload.get("results", [])
            if isinstance(results, list):
                for result in results:
                    if not isinstance(result, dict):
                        continue
                    if not result.get("is_official"):
                        continue
                    url = str(result.get("url", "")).strip()
                    if url:
                        urls.append(url)

            debug = payload.get("debug", {})
            if isinstance(debug, dict):
                official_urls = debug.get("official_urls", [])
                if isinstance(official_urls, list):
                    for item in official_urls:
                        url = str(item).strip()
                        if url and url not in urls:
                            urls.append(url)
            return urls
        return []

    def _latest_tool_snapshot(self, state: AgentState) -> tuple[str, dict[str, Any]]:
        for message in reversed(self._current_call_messages(state)):
            if not self._is_tool_message(message):
                continue
            return self._tool_name(message), self._parse_tool_payload(self._message_content(message))
        return "", {}

    def _build_reflection_record(
        self,
        state: AgentState,
        *,
        llm_decision: str,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> ReflectionRecord | None:
        task_state = self._build_task_state(list(state["messages"]))
        if task_state is None:
            return None
        evidence_cache = self._build_evidence_cache(list(state["messages"]))
        latest_tool_name, latest_tool_payload = self._latest_tool_snapshot(state)
        latest_tool_result_count = latest_tool_payload.get("result_count")
        if not isinstance(latest_tool_result_count, int):
            latest_tool_result_count = None
        serialized_calls = tuple(
            {
                "name": str(tool_call.get("name", "")),
                "args": dict(tool_call.get("args", {})),
            }
            for tool_call in (tool_calls or [])
        )
        return ReflectionRecord(
            stage="post_tool_reflection",
            question=task_state.question,
            entities=task_state.entities,
            evidence_sufficiency=task_state.evidence_sufficiency,
            missing_information=task_state.missing_information,
            recommended_next_action=task_state.next_action,
            latest_tool_name=latest_tool_name,
            latest_tool_result_count=latest_tool_result_count,
            cached_web_queries=tuple(sorted(evidence_cache.web_results_by_query.keys())),
            cached_fetched_urls=tuple(sorted(evidence_cache.fetched_pages_by_url.keys())),
            llm_decision=llm_decision,
            tool_calls=serialized_calls,
        )

    def _log_reflection_record(
        self,
        state: AgentState,
        *,
        llm_decision: str,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        record = self._build_reflection_record(
            state,
            llm_decision=llm_decision,
            tool_calls=tool_calls,
        )
        if record is None:
            return
        self._append_json("Reflection result", asdict(record))

    def _apply_official_first_fetch_policy(
        self,
        state: AgentState,
        message: Any,
        tool_calls: list[dict[str, Any]],
    ) -> tuple[Any, list[dict[str, Any]]]:
        official_urls = self._official_search_urls(state)
        if not official_urls:
            return message, tool_calls
        failed_urls = self._failed_web_fetch_urls(state)
        preferred_url = next((url for url in official_urls if url not in failed_urls), None)
        if not preferred_url:
            return message, tool_calls

        first_fetch_index: int | None = None
        official_fetch_present = False
        updated_tool_calls: list[dict[str, Any]] = []
        for index, tool_call in enumerate(tool_calls):
            copied_call = dict(tool_call)
            copied_args = dict(tool_call.get("args", {}))
            copied_call["args"] = copied_args
            updated_tool_calls.append(copied_call)

            if copied_call.get("name") != "web_fetch":
                continue

            url = str(copied_args.get("url", "")).strip()
            if url == preferred_url:
                official_fetch_present = True
            elif first_fetch_index is None:
                first_fetch_index = index

        if official_fetch_present or first_fetch_index is None:
            return message, tool_calls

        original_url = str(updated_tool_calls[first_fetch_index]["args"].get("url", "")).strip()
        updated_tool_calls[first_fetch_index]["args"]["url"] = preferred_url
        self._append(
            "Official-first fetch policy\n"
            f"Rewrote web_fetch URL from {original_url} to {preferred_url}"
        )
        return self._replace_tool_calls(message, updated_tool_calls), updated_tool_calls

    def exists_action(self, state: AgentState):
        result = state["messages"][-1]
        return len(getattr(result, "tool_calls", [])) > 0

    @staticmethod
    def build_reflection_prompt(tool_results: list[AnyMessage | dict]) -> str:
        result_blocks = []
        for index, message in enumerate(tool_results, start=1):
            if isinstance(message, dict):
                name = str(message.get("name", "tool"))
                content = str(message.get("content", ""))
            else:
                name = str(getattr(message, "name", "tool"))
                content = str(getattr(message, "content", ""))
            result_blocks.append(f"Tool result {index} ({name}):\n{content}")
        return REFLECTION_PROMPT_TEMPLATE + "\n\n" + "\n\n".join(result_blocks)

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

        return build_context_messages(
            base_messages=base_messages,
            system_text=self.system,
            summary_text=summary_text,
            question_frame_text=question_frame_text,
            user_memory_text=user_memory_text,
            evidence_cache_text=evidence_cache_text,
            task_state_text=task_state_text,
            reflection_text=reflection_text,
            budget=ContextBudget(
                max_chars=self.max_context_chars,
                max_tokens=self.max_context_tokens,
            ),
            shorten=self._shorten,
            message_content=self._message_content,
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
        if self.model is None:
            raise RuntimeError("Model is required to run graph_rag agent.")

        last_message = state["messages"][-1]
        if ToolMessage is not None and isinstance(last_message, ToolMessage):
            self._append("Reflection step\nThe model will decide whether to continue using tools or answer directly.")
            self._append(
                f"Reflection prompt\n{self._shorten(self.build_reflection_prompt([last_message]), 1200)}"
            )
            self.current_round += 1

        context_result = self.build_context_result(state)
        messages = context_result.messages
        self.model_call_count += 1
        role = self._message_role(last_message)
        content = self._shorten(self._message_content(last_message), 800)
        if self.model_call_count == 1:
            self.current_round = 1
            self.log_round_header("Start analyzing the user question")
            self._log_question_frame(self._build_question_frame(list(state["messages"])))

        if context_result.metrics is not None:
            self._append(format_context_metrics(context_result.metrics))
        self._append(f"LLM input source: {role}\nLLM input content:\n{content}")
        message = self.model.invoke(messages)
        response_text = self._message_content(message)
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
                    )
                ]
            }

        if tool_calls:
            message, tool_calls = self._apply_official_first_fetch_policy(state, message, tool_calls)

        if tool_calls:
            self._log_reflection_record(state, llm_decision="tool_use", tool_calls=tool_calls)
            first_query = tool_calls[0].get("args", {}).get("query", "")
            self._append(
                "LLM decision: continue with tool use.\n"
                f"Intent for this round:\n{self._shorten(response_text, 500)}\n"
                f"Primary query:\n{first_query}"
            )
        else:
            self._log_reflection_record(state, llm_decision="answer", tool_calls=[])
            self._append(
                "LLM decision: answer directly.\n"
                f"Output summary:\n{self._shorten(response_text, 800)}"
            )
        self._append(f"LLM raw output:\n{self._shorten(response_text, 1200)}")
        return {"messages": [message]}

    def take_action(self, state: AgentState):
        tool_calls = state["messages"][-1].tool_calls
        evidence_cache = self._build_evidence_cache(list(state["messages"]))
        results = []
        for tool_call in tool_calls:
            self.tool_call_count += 1
            tool_args = tool_call.get("args", {})
            serialized_args = json.dumps(tool_args, ensure_ascii=False, sort_keys=True)
            self._append(f"Tool call\nName: {tool_call['name']}\nArgs: {serialized_args}")
            cached_result = lookup_cached_tool_result(
                evidence_cache,
                tool_name=tool_call["name"],
                tool_args=tool_args,
            )
            if cached_result is not None:
                self._append(f"Tool cache hit\nName: {tool_call['name']}")
                result = cached_result
            elif tool_call["name"] not in self.tools:
                result = json.dumps(
                    {
                        "error": "invalid_tool",
                        "tool_name": tool_call["name"],
                        "tool_args": tool_args,
                    },
                    ensure_ascii=False,
                )
            else:
                try:
                    result = self.tools[tool_call["name"]].invoke(tool_args)
                except Exception as exc:
                    result = json.dumps(
                        {
                            "error": "tool_execution_failed",
                            "tool_name": tool_call["name"],
                            "tool_args": tool_args,
                            "error_type": exc.__class__.__name__,
                            "message": str(exc),
                        },
                        ensure_ascii=False,
                    )
            self._append(f"Tool result\n{str(result)}")
            results.append(
                ToolMessage(tool_call_id=tool_call["id"], name=tool_call["name"], content=str(result))
            )
        return {"messages": results}


def build_agent(
    index_dir: str | Path,
    checkpointer: Any | None = None,
    app_config: AppConfig | None = None,
    *,
    ensure_log_file: Any | None = None,
    append_log: Any | None = None,
    shorten_text: Any | None = None,
) -> Agent:
    if ChatOpenAI is None:
        raise ImportError("Missing runtime dependencies for LangGraph execution.")

    config = app_config or build_app_config(index_dir)
    if not config.model.api_key:
        raise EnvironmentError("Missing DASHSCOPE_API_KEY environment variable.")

    model = ChatOpenAI(
        model=config.model.model_name,
        openai_api_key=config.model.api_key,
        openai_api_base=config.model.api_base,
    )
    tools = [
        LocalRAGRetrieveTool(
            store=load_index(index_dir, keyword_weight=config.retrieval.keyword_weight),
            min_evidence_score=config.generation.min_evidence_score,
        )
    ]

    if config.web.enabled:
        backend = build_configured_web_search_backend(config.web)

        def fetcher(url: str):
            return fetch_url(
                url,
                timeout_seconds=config.web.fetch_timeout_seconds,
                max_bytes=config.web.fetch_max_bytes,
                max_chars=config.web.fetch_max_chars,
                user_agent=config.web.user_agent,
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
    )
