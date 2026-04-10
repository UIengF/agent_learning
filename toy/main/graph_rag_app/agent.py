from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Type

from pydantic import BaseModel, Field, PrivateAttr

from .config import AppConfig, DEFAULT_TOP_K, build_app_config
from .indexing import load_index
from .retrieval import Retriever, normalize_search_result
from .web_fetch import fetch_url
from .web_search import DuckDuckGoHtmlSearchBackend, MultiQuerySearchBackend
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
Do not answer detailed factual questions from search snippets alone. If a snippet suggests the needed evidence is on a page, call `web_fetch` before relying on it.

Do not invent facts that are not supported by tool results. If evidence is incomplete, say what is missing.
If results contain multiple plausible answers, dates, or scenarios, call that out clearly before answering.
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
    ):
        self.system = system
        self.log_path = log_path or Path("logs") / "graph_rag.log"
        self.model_call_count = 0
        self.tool_call_count = 0
        self.current_round = 0
        self.max_rounds = max(1, int(max_rounds))
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

    def exists_action(self, state: AgentState):
        result = state["messages"][-1]
        return len(getattr(result, "tool_calls", [])) > 0

    @staticmethod
    def build_reflection_prompt(tool_results: list[ToolMessage]) -> str:
        result_blocks = []
        for index, message in enumerate(tool_results, start=1):
            result_blocks.append(f"Tool result {index} ({message.name}):\n{message.content}")
        return REFLECTION_PROMPT_TEMPLATE + "\n\n" + "\n\n".join(result_blocks)

    def build_llm_messages(self, state: AgentState) -> list[AnyMessage]:
        messages = list(state["messages"])
        tool_results = []
        for message in reversed(messages):
            if ToolMessage is not None and isinstance(message, ToolMessage):
                tool_results.append(message)
                continue
            break

        if tool_results and HumanMessage is not None:
            tool_results.reverse()
            messages.append(HumanMessage(content=self.build_reflection_prompt(tool_results)))

        if self.system and SystemMessage is not None:
            messages = [SystemMessage(content=self.system)] + messages
        return messages

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

        messages = self.build_llm_messages(state)
        self.model_call_count += 1
        role = self._message_role(last_message)
        content = self._shorten(self._message_content(last_message), 800)
        if self.model_call_count == 1:
            self.current_round = 1
            self.log_round_header("Start analyzing the user question")

        self._append(f"LLM input source: {role}\nLLM input content:\n{content}")
        message = self.model.invoke(messages)
        response_text = self._message_content(message)
        tool_calls = getattr(message, "tool_calls", [])
        completed_rounds = self._tool_rounds_in_current_call(state)
        if tool_calls and completed_rounds >= self.max_rounds:
            limited_content = self._tool_limit_message_text()
            self._append(
                "LLM decision: stop calling tools because the limit was reached.\n"
                f"Completed rounds: {completed_rounds}\n"
                f"Limited output:\n{limited_content}"
            )
            return {"messages": [self._finalize_without_tool_calls(message, limited_content)]}

        if tool_calls:
            first_query = tool_calls[0].get("args", {}).get("query", "")
            self._append(
                "LLM decision: continue with tool use.\n"
                f"Intent for this round:\n{self._shorten(response_text, 500)}\n"
                f"Primary query:\n{first_query}"
            )
        else:
            self._append(
                "LLM decision: answer directly.\n"
                f"Output summary:\n{self._shorten(response_text, 800)}"
            )
        self._append(f"LLM raw output:\n{self._shorten(response_text, 1200)}")
        return {"messages": [message]}

    def take_action(self, state: AgentState):
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for tool_call in tool_calls:
            self.tool_call_count += 1
            tool_args = tool_call.get("args", {})
            serialized_args = json.dumps(tool_args, ensure_ascii=False, sort_keys=True)
            self._append(f"Tool call\nName: {tool_call['name']}\nArgs: {serialized_args}")
            if tool_call["name"] not in self.tools:
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
        if config.web.search_provider != "duckduckgo_html":
            raise ValueError(f"Unsupported web search provider: {config.web.search_provider}")

        backend = MultiQuerySearchBackend(
            DuckDuckGoHtmlSearchBackend(
                timeout_seconds=config.web.fetch_timeout_seconds,
                user_agent=config.web.user_agent,
            )
        )

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
                WebFetchTool(fetcher=fetcher),
            ]
        )

    return Agent(
        model,
        tools,
        checkpointer=checkpointer,
        system=PROMPT,
        ensure_log_file=ensure_log_file,
        append_log=append_log,
        shorten_text=shorten_text,
        max_rounds=config.generation.max_rounds,
    )
