from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Type

from pydantic import BaseModel, Field, PrivateAttr

from .config import AppConfig, DEFAULT_TOP_K, build_app_config
from .indexing import ensure_index_for_kb, load_index
from .retrieval import DashScopeEmbeddingClient, Retriever, normalize_search_result

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
    HumanMessage = None
    SystemMessage = None
    ToolMessage = None
    BaseTool = object  # type: ignore[assignment]
    ChatOpenAI = None
    StateGraph = None
    END = None
    Annotated = list  # type: ignore[assignment]
    TypedDict = dict  # type: ignore[assignment]
    operator = None
    LANGGRAPH_AVAILABLE = False


PROMPT = """\
你是一个基于本地知识库回答问题的研究助手。你可以调用本地检索工具查找相关片段，
但只有在当前证据不足时才继续检索。回答时只能依据检索结果中的内容，不要编造知识库中
没有出现的事实。如果证据不足，请明确说明缺少什么信息。如果检索结果里存在多个可能答案、
多个时间点或多个场景，请先指出歧义，再分别说明，并尽量标注引用的片段编号。"""

REFLECTION_PROMPT_TEMPLATE = """\
你刚收到一轮本地知识库检索结果。请先判断：
1. 当前结果已经回答了什么；
2. 还缺少哪些完成任务必须的信息；
3. 下一步最小必要动作是什么。

如果当前证据已经足够，请直接基于检索结果回答，不要继续调用工具。
如果证据不足且必须继续检索，请改写出更具体的新查询，再调用工具。
不要把检索结果中没有出现的事实当作已知事实。
如果结果中存在多个可能答案、多个时间点或多个场景，请先指出歧义，再分别说明，
并尽量标注引用的片段编号。"""


if LANGGRAPH_AVAILABLE:
    class AgentState(TypedDict):
        messages: Annotated[list[AnyMessage], operator.add]
else:
    AgentState = dict[str, Any]


class LocalRAGInput(BaseModel):
    query: str = Field(..., description="用于检索本地知识库的查询")
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=8, description="返回的相关片段数量")


class LocalRAGRetrieveTool(BaseTool):
    name: str = "local_rag_retrieve"
    description: str = "从本地知识库中检索相关片段"
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
        self._append(f"===== 第 {self.current_round} 轮 =====\n{title}")

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

    @staticmethod
    def _tool_message_name(message: AnyMessage | dict) -> str:
        if isinstance(message, dict):
            return str(message.get("name", ""))
        return str(getattr(message, "name", ""))

    def _current_call_messages(self, state: AgentState) -> list[AnyMessage | dict]:
        messages = list(state["messages"])
        for index in range(len(messages) - 1, -1, -1):
            if self._is_human_message(messages[index]):
                return messages[index:]
        return messages

    def _consecutive_tool_rounds_in_current_call(
        self,
        state: AgentState,
        tool_name: str,
    ) -> int:
        if not tool_name:
            return 0

        messages = self._current_call_messages(state)
        index = len(messages) - 1
        rounds = 0

        while index >= 0:
            if not self._is_tool_message(messages[index]):
                break

            block_names: set[str] = set()
            while index >= 0 and self._is_tool_message(messages[index]):
                name = self._tool_message_name(messages[index])
                if name:
                    block_names.add(name)
                index -= 1

            if block_names != {tool_name}:
                break

            rounds += 1
            while index >= 0 and not self._is_tool_message(messages[index]):
                index -= 1

        return rounds

    @staticmethod
    def _tool_limit_message_text() -> str:
        return "已达到最大检索轮次，当前证据仍不足，无法继续检索。请基于现有证据回答，并明确说明不确定性。"

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
            result_blocks.append(f"检索结果 {index}（{message.name}）:\n{message.content}")
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
            self._append("反思阶段\n模型将基于当前检索结果判断是继续检索还是直接回答。")
            self._append(
                f"反思提示:\n{self._shorten(self.build_reflection_prompt([last_message]), 1200)}"
            )
            self.current_round += 1

        messages = self.build_llm_messages(state)
        self.model_call_count += 1
        role = self._message_role(last_message)
        content = self._shorten(self._message_content(last_message), 800)
        if self.model_call_count == 1:
            self.current_round = 1
            self.log_round_header("开始分析用户问题")

        self._append(f"LLM 输入来源: {role}\nLLM 输入内容:\n{content}")
        message = self.model.invoke(messages)
        response_text = self._message_content(message)
        tool_calls = getattr(message, "tool_calls", [])
        blocked_tools = []
        allowed_tool_calls = []
        for tool_call in tool_calls:
            tool_name = str(tool_call.get("name", ""))
            consecutive_rounds = self._consecutive_tool_rounds_in_current_call(state, tool_name)
            if consecutive_rounds >= self.max_rounds:
                blocked_tools.append((tool_name, consecutive_rounds))
            else:
                allowed_tool_calls.append(tool_call)
        completed_rounds = blocked_tools[0][1] if blocked_tools else 0
        if tool_calls and not allowed_tool_calls:
            limited_content = self._tool_limit_message_text()
            self._append(
                "LLM 决策: 达到最大检索轮次，停止继续检索\n"
                f"已完成轮次: {completed_rounds}\n"
                f"限制后输出:\n{limited_content}"
            )
            return {"messages": [self._finalize_without_tool_calls(message, limited_content)]}

        if blocked_tools:
            message = self._replace_tool_calls(message, allowed_tool_calls)
            tool_calls = allowed_tool_calls

        if tool_calls:
            first_query = tool_calls[0].get("args", {}).get("query", "")
            self._append(
                "LLM 决策: 继续检索本地知识库\n"
                f"本轮检索意图:\n{self._shorten(response_text, 500)}\n"
                f"生成查询:\n{first_query}"
            )
        else:
            self._append(
                "LLM 决策: 直接给出答案\n"
                f"输出摘要:\n{self._shorten(response_text, 800)}"
            )
        self._append(f"LLM 原始输出:\n{self._shorten(response_text, 1200)}")
        return {"messages": [message]}

    def take_action(self, state: AgentState):
        if ToolMessage is None:
            raise RuntimeError("langchain_core is required for tool execution.")

        tool_calls = state["messages"][-1].tool_calls
        results = []
        for tool_call in tool_calls:
            self.tool_call_count += 1
            query = tool_call.get("args", {}).get("query", "")
            top_k = tool_call.get("args", {}).get("top_k", DEFAULT_TOP_K)
            self._append(f"检索调用\n名称: {tool_call['name']}\n查询: {query}\ntop_k: {top_k}")
            if tool_call["name"] not in self.tools:
                result = json.dumps(
                    {
                        "query": query,
                        "result_count": 0,
                        "reason": "invalid_tool",
                        "results": [],
                    },
                    ensure_ascii=False,
                )
            else:
                result = self.tools[tool_call["name"]].invoke(tool_call["args"])
            self._append(f"原始检索结果:\n{str(result)}")
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
    tool = LocalRAGRetrieveTool(
        store=load_index(index_dir, keyword_weight=config.retrieval.keyword_weight),
        min_evidence_score=config.generation.min_evidence_score,
    )
    return Agent(
        model,
        [tool],
        checkpointer=checkpointer,
        system=PROMPT,
        ensure_log_file=ensure_log_file,
        append_log=append_log,
        shorten_text=shorten_text,
        max_rounds=config.generation.max_rounds,
    )
