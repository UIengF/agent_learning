import operator
import os
import sqlite3
from typing import Annotated, Type, TypedDict

import requests
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field, PrivateAttr


db_path = os.path.abspath("checkpoints.db")
conn = sqlite3.connect(db_path, check_same_thread=False)
memory = SqliteSaver(conn)


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
        try:
            response = requests.post(
                "https://api.bochaai.com/v1/web-search",
                headers={"Authorization": f"Bearer {self._api_key}", "Content-Type": "application/json"},
                json={"query": query, "summary": self._summary, "freshness": self._freshness, "count": self._count},
                timeout=10,
            )
            response.raise_for_status()
            data = response.json()
            results = data.get("data", {}).get("webPages", {}).get("value", [])
            if not results:
                return f"未找到相关内容。\n[DEBUG] 返回数据：{data}"
            return "\n\n".join(
                f"{i + 1}. {item.get('name', '无标题')}\n{item.get('snippet', '无摘要')}\n链接: {item.get('url', '')}"
                for i, item in enumerate(results[:self._count])
            )
        except Exception as e:
            return f"搜索失败: {e}"


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


class Agent:
    def __init__(self, model, tools, checkpointer=None, system=""):
        self.system = system
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile(checkpointer=checkpointer)

    def exists_action(self, state: AgentState):
        return len(state["messages"][-1].tool_calls) > 0

    def call_openai(self, state: AgentState):
        messages = state["messages"]
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        return {"messages": [self.model.invoke(messages)]}

    def take_action(self, state: AgentState):
        results = []
        for t in state["messages"][-1].tool_calls:
            print(f"Calling: {t}")
            result = self.tools[t["name"]].invoke(t["args"]) if t["name"] in self.tools else "bad tool name, retry"
            results.append(ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result)))
        print("Back to the model!")
        return {"messages": results}


prompt = """你是一个科研助理。你可以使用搜索工具查找信息。可以多次调用搜索，但如果搜索结果不可靠，不要编造答案，应明确说明无法确认。"""

dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
bocha_api_key = os.getenv("BOCHA_API_KEY")

if not dashscope_api_key:
    raise EnvironmentError("Missing DASHSCOPE_API_KEY environment variable.")

if not bocha_api_key:
    raise EnvironmentError("Missing BOCHA_API_KEY environment variable.")

model = ChatOpenAI(
    model="qwen-plus",
    openai_api_key=dashscope_api_key,
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
)
tool = BoChaSearchResults(api_key=bocha_api_key, count=4)

abot = Agent(model, [tool], system=prompt, checkpointer=memory)
messages = [HumanMessage(content="第一任是谁?")]
thread = {"configurable": {"thread_id": "1"}}
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v)
while abot.graph.get_state(thread).next:
    print("\n", abot.graph.get_state(thread), "\n")
    _input = input("proceed?")
    if _input != "y":
        print("aborting")
        break
    for event in abot.graph.stream(None, thread):
        for v in event.values():
            print(v)
