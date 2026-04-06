from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated, Type
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field, PrivateAttr
from langchain_core.tools import BaseTool
import requests

# ----------------------------- Tool 定义 -----------------------------
class BoChaSearchInput(BaseModel):
    query: str = Field(..., description="搜索的查询内容")

class BoChaSearchResults(BaseTool):
    name: str = "bocha_web_search"
    description: str = "使用博查API进行网络搜索，可用来查找实时信息或新闻"
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
            "Content-Type": "application/json"
        }
        payload = {
            "query": query,
            "summary": self._summary,
            "freshness": self._freshness,
            "count": self._count
        }

        try:
            response = requests.post(url, headers=headers, json=payload, timeout=10)
            response.raise_for_status()
            data = response.json()

            results = data.get("data", {}).get("webPages", {}).get("value", [])
            if not results:
                return f"\u672a\u627e\u5230\u76f8\u5173\u5185\u5bb9\u3002\n[DEBUG] \u8fd4\u56de\u6570\u636e\uff1a{data}"

            output = ""
            for i, item in enumerate(results[:self._count]):
                title = item.get("name", "\u65e0\u6807\u9898")
                snippet = item.get("snippet", "\u65e0\u6458\u8981")
                url = item.get("url", "")
                output += f"{i+1}. {title}\n{snippet}\n\u94fe\u63a5: {url}\n\n"

            return output.strip()

        except Exception as e:
            return f"\u641c\u7d22\u5931\u8d25: {e}"

# ----------------------------- 智能体状态定义 -----------------------------
class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

# ----------------------------- Agent 实现 -----------------------------
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
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            if not t['name'] in self.tools:
                print("\n ....bad tool name....")
                result = "bad tool name, retry"
            else:
                result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}

# ----------------------------- 模型和启动 -----------------------------
prompt = """\
你是一名聪明的科研助理。可以使用搜索引擎查找信息。
你可以多次调用搜索（可以一次性调用，也可以分步骤调用）。
只有当你确切知道要找什么时，才去检索信息。
如果在提出后续问题之前需要先检索信息，你也可以这样做！
"""

aliyun_api_key = 'sk-d18ec22172af4ad2aa8fa11e82e480c0'
model = ChatOpenAI(
    model="qwen-plus",
    openai_api_key=aliyun_api_key,
    openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

tool = BoChaSearchResults(api_key="sk-ec235200d7a2406db3d25841a6dd8958", count=4)
abot = Agent(model, [tool], system=prompt)
# 绘制当前的 Graph
# ascii_diagram = abot.graph.get_graph().draw_ascii()
# print(ascii_diagram)
# ----------------------------- 测试使用 -----------------------------
messages = [HumanMessage(content="第一任美国总统是谁?")]
result = abot.graph.invoke({"messages": messages})
print(result['messages'][-1].content)