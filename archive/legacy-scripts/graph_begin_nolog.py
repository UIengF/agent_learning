import operator
import os
from typing import Annotated, Type, TypedDict

import requests
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
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


class Agent:
    def __init__(self, model, tools, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState):
        result = state["messages"][-1]
        return len(result.tool_calls) > 0

    @staticmethod
    def build_reflection_prompt(tool_results: list[ToolMessage]) -> str:
        result_blocks = []
        for index, message in enumerate(tool_results, start=1):
            result_blocks.append(f"工具结果 {index}（{message.name}）:\n{message.content}")

        return (
            "你刚收到一轮工具调用结果。\n"
            "请先基于当前任务判断：\n"
            "1. 这些结果已经回答了什么；\n"
            "2. 还缺少哪些完成任务所必需的信息；\n"
            "3. 下一步最小必要动作是什么。\n"
            "如果信息已经足够，就不要继续调用工具，直接给出结论。\n"
            "如果信息还不够，并且必须再次调用工具，请先根据缺口改写出一个更具体、更聚焦的新查询，"
            "避免重复上一轮的宽泛问法，再调用工具。\n\n"
            + "\n\n".join(result_blocks)
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
        messages = self.build_llm_messages(state)
        return {"messages": [self.model.invoke(messages)]}

    def take_action(self, state: AgentState):
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if t["name"] not in self.tools:
                print("\n ....bad tool name....")
                result = "bad tool name, retry"
            else:
                result = self.tools[t["name"]].invoke(t["args"])
            results.append(ToolMessage(tool_call_id=t["id"], name=t["name"], content=str(result)))
        print("Back to the model!")
        return {"messages": results}


prompt = """\
今年是2026年,你是一名聪明的科研助理。可以使用搜索引擎查找信息。
你可以多次调用搜索，也可以分步调用。
只有当你确定知道要找什么时，才去检索信息。
如果在提出后续问题之前需要先检索信息，你也可以这样做。
"""


def build_agent() -> Agent:
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
    return Agent(model, [tool], system=prompt)


def run_demo(question: str) -> str:
    agent = build_agent()
    messages = [HumanMessage(content=question)]
    result = agent.graph.invoke({"messages": messages})
    return result["messages"][-1].content


if __name__ == "__main__":
    answer = run_demo("我要比较 2024 年和 2025 年中国GDP总量的变化,并自行判断对普通消费者购物决策的影响")
    print(answer)
