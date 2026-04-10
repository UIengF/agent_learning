import operator
import os
import sqlite3
from pathlib import Path
from typing import Annotated, Any, Type, TypedDict

import requests
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, PrivateAttr


class BoChaSearchInput(BaseModel):
    query: str = Field(..., description="搜索查询内容")


class BoChaSearchResults(BaseTool):
    name: str = "bocha_web_search"
    description: str = "使用博查 API 进行网络搜索，可用于查找实时信息或新闻"
    args_schema: Type[BaseModel] = BoChaSearchInput

    _api_key: str = PrivateAttr()
    _count: int = PrivateAttr()
    _summary: bool = PrivateAttr()
    _freshness: str = PrivateAttr()

    def __init__(self, api_key: str, count: int = 5, summary: bool = True, freshness: str = "noLimit", **kwargs):
        super().__init__(**kwargs)
        self._api_key = api_key
        self._count = count
        self._summary = summary
        self._freshness = freshness

    def _run(self, query: str) -> str:
        url = "https://api.bochaai.com/v1/web-search"
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "query": query,
            "summary": self._summary,
            "freshness": self._freshness,
            "count": self._count,
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()

            results = data.get("data", {}).get("webPages", {}).get("value", [])
            if not results:
                return f"未找到相关内容。\n[DEBUG] 返回数据：{data}"

            output = ""
            for i, item in enumerate(results[:self._count]):
                title = item.get("name", "无标题")
                snippet = item.get("snippet", "无摘要")
                result_url = item.get("url", "")
                output += f"{i + 1}. {title}\n{snippet}\n链接: {result_url}\n\n"

            return output.strip()
        except Exception as e:
            return f"搜索失败: {e}"


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


DEFAULT_THREAD_ID = "graph_begin_default"
DEFAULT_CHECKPOINT_DB = "checkpoints.db"


def ensure_log_file(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        log_path.write_text("", encoding="utf-8")


def append_log(log_path: Path, text: str) -> None:
    with log_path.open("a", encoding="utf-8") as file:
        file.write(text.rstrip() + "\n\n")


def build_sqlite_checkpointer(db_path: str | Path) -> Any:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path.resolve(), check_same_thread=False)
    checkpointer = SqliteSaver(connection)
    # Keep a strong reference to avoid the sqlite connection being garbage-collected.
    setattr(checkpointer, "_connection", connection)
    return checkpointer


def get_thread_config(thread_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": thread_id}}


def run_graph_invoke(graph: Any, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    try:
        return graph.invoke(state, config=config)
    except TypeError as exc:
        if "config" not in str(exc):
            raise
        return graph.invoke(state, config)


class Agent:
    def __init__(self, model, tools, checkpointer=None, system=""):
        self.system = system
        self.log_path = Path("logs") / "agent_run.log"
        self.model_call_count = 0
        self.tool_call_count = 0
        self.current_round = 0
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
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)
        ensure_log_file(self.log_path)

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

    @classmethod
    def format_messages_for_log(cls, messages: list[AnyMessage | dict]) -> str:
        blocks = []
        for index, message in enumerate(messages, start=1):
            role = cls._message_role(message)
            content = cls._message_content(message)
            blocks.append(f"[{index}] {role}\n{content}")
        return "\n\n".join(blocks)

    @staticmethod
    def shorten_text(text: str, max_len: int = 600) -> str:
        compact = " ".join(text.split())
        if len(compact) <= max_len:
            return compact
        return compact[: max_len - 3] + "..."

    def log_round_header(self, title: str) -> None:
        append_log(self.log_path, f"===== 第 {self.current_round} 轮 =====\n{title}")

    def summarize_tool_result(self, result_text: str) -> str:
        blocks = [block.strip() for block in result_text.split("\n\n") if block.strip()]
        summary_blocks = []
        for block in blocks[:4]:
            lines = [line.strip() for line in block.splitlines() if line.strip()]
            if not lines:
                continue
            title = lines[0]
            snippet = lines[1] if len(lines) > 1 else ""
            link = next((line for line in lines if line.startswith("链接")), "")
            summary = f"- {title}"
            if snippet:
                summary += f"\n  摘要: {self.shorten_text(snippet, 120)}"
            if link:
                summary += f"\n  {link}"
            summary_blocks.append(summary)
        return "\n".join(summary_blocks) if summary_blocks else self.shorten_text(result_text)

    def exists_action(self, state: AgentState):
        result = state["messages"][-1]
        return len(result.tool_calls) > 0

    @staticmethod
    def build_reflection_prompt(tool_results: list[ToolMessage]) -> str:
        result_blocks = []
        for index, message in enumerate(tool_results, start=1):
            result_blocks.append(
                f"工具结果 {index}（{message.name}）:\n{message.content}"
            )

        joined_results = "\n\n".join(result_blocks)
        return (
            "你刚收到一轮工具调用结果。\n"
            "请先基于当前任务判断：\n"
            "1. 这些结果已经回答了什么；\n"
            "2. 还缺少哪些完成任务所必需的信息；\n"
            "3. 下一步最小必要动作是什么。\n"
            "如果信息已经足够，就不要继续调用工具，直接给出结论。\n"
            "如果信息还不够，并且必须再次调用工具，请先根据缺口改写出一个更具体、更聚焦的新查询，"
            "避免重复上一轮的宽泛问法，再调用工具。\n\n"
            f"{joined_results}"
        )

    def build_llm_messages(self, state: AgentState) -> list[AnyMessage]:
        messages = list(state["messages"])
        tool_results = []
        for message in reversed(messages):
            if isinstance(message, ToolMessage):
                tool_results.append(message)
                continue
            break

        if tool_results:
            tool_results.reverse()
            messages.append(HumanMessage(content=self.build_reflection_prompt(tool_results)))

        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        return messages

    def call_openai(self, state: AgentState):
        last_message = state["messages"][-1]
        if isinstance(last_message, ToolMessage):
            append_log(
                self.log_path,
                "反思阶段:\n模型在收到工具结果后，将在本次 LLM 调用中直接完成反思并决定下一步动作。",
            )
            append_log(
                self.log_path,
                f"反思提示:\n{self.shorten_text(self.build_reflection_prompt([last_message]), 1200)}",
            )
            self.current_round += 1

        messages = self.build_llm_messages(state)
        self.model_call_count += 1
        role = self._message_role(last_message)
        content = self.shorten_text(self._message_content(last_message), 800)
        if self.model_call_count == 1:
            self.current_round = 1
            self.log_round_header("开始分析用户问题")
        append_log(
            self.log_path,
            f"LLM 输入来源: {role}\nLLM 输入内容:\n{content}",
        )
        message = self.model.invoke(messages)
        response_text = self._message_content(message)
        tool_calls = getattr(message, "tool_calls", [])
        if tool_calls:
            first_query = tool_calls[0].get("args", {}).get("query", "")
            append_log(
                self.log_path,
                "LLM 决策: 继续调用工具\n"
                f"本轮查询意图:\n{self.shorten_text(response_text, 500)}\n"
                f"生成查询:\n{first_query}",
            )
        else:
            append_log(
                self.log_path,
                "LLM 决策: 直接给出结论\n"
                f"输出摘要:\n{self.shorten_text(response_text, 800)}",
            )
        append_log(
            self.log_path,
            f"LLM 原始输出:\n{self.shorten_text(response_text, 1200)}",
        )
        return {"messages": [message]}

    def take_action(self, state: AgentState):
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            self.tool_call_count += 1
            append_log(
                self.log_path,
                f"工具调用:\n名称: {t['name']}\n查询: {t['args'].get('query', '')}",
            )
            append_log(
                self.log_path,
                f"工具参数:\n{t['args']}",
            )
            if t["name"] not in self.tools:
                print("\n ....bad tool name....")
                result = "bad tool name, retry"
            else:
                result = self.tools[t["name"]].invoke(t["args"])
            append_log(
                self.log_path,
                f"工具结果摘要:\n{self.summarize_tool_result(str(result))}",
            )
            results.append(ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result)))
        print("Back to the model!")
        return {"messages": results}


prompt = """\
今年是2026年,你是一名聪明的科研助理。可以使用搜索引擎查找信息。
你可以多次调用搜索，也可以分步调用。
只有当你确定知道要找什么时，才去检索信息。
如果在提出后续问题之前需要先检索信息，你也可以这样做。
"""

def build_agent(checkpointer=None) -> Agent:
    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
    bocha_api_key = os.getenv("BOCHA_API_KEY")

    if not dashscope_api_key:
        raise EnvironmentError("Missing DASHSCOPE_API_KEY environment variable.")

    if not bocha_api_key:
        raise EnvironmentError("Missing BOCHA_API_KEY environment variable.")

    model = ChatOpenAI(
        model="qwen3.6-plus",
        openai_api_key=dashscope_api_key,
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    tool = BoChaSearchResults(api_key=bocha_api_key, count=4)
    return Agent(model, [tool], checkpointer=checkpointer, system=prompt)


def run_demo(
    question: str,
    thread_id: str = DEFAULT_THREAD_ID,
    checkpoint_db: str | Path = DEFAULT_CHECKPOINT_DB,
) -> str:
    checkpointer = build_sqlite_checkpointer(checkpoint_db)
    agent = build_agent(checkpointer=checkpointer)
    messages = [HumanMessage(content=question)]
    ensure_log_file(agent.log_path)
    append_log(
        agent.log_path,
        "===== 会话开始 =====\n"
        f"thread_id: {thread_id}\n"
        f"用户问题:\n{question}\n\n"
        f"系统提示:\n{agent.system}",
    )
    result = run_graph_invoke(
        agent.graph,
        {"messages": messages},
        config=get_thread_config(thread_id),
    )
    append_log(
        agent.log_path,
        "===== 最终答案 =====\n"
        f"{agent._message_content(result['messages'][-1])}",
    )
    return result["messages"][-1].content


if __name__ == "__main__":
    answer = run_demo(" 2025 年中国GDP总量,比2024年高了多少")
    print(answer)
