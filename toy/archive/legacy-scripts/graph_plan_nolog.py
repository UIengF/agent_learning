from __future__ import annotations

import os
from copy import deepcopy
from typing import Any, Type

import requests
from pydantic import BaseModel, Field, PrivateAttr

try:
    import operator
    from typing import Annotated, TypedDict

    from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage
    from langchain_core.tools import BaseTool
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, StateGraph

    LANGGRAPH_AVAILABLE = True
except ImportError:  # pragma: no cover - keep helper tests importable without full runtime deps
    AnyMessage = Any
    HumanMessage = None
    ToolMessage = None
    BaseTool = object  # type: ignore[assignment]
    ChatOpenAI = None
    StateGraph = None
    END = None
    Annotated = list  # type: ignore[assignment]
    TypedDict = dict  # type: ignore[assignment]
    operator = None
    LANGGRAPH_AVAILABLE = False


if LANGGRAPH_AVAILABLE:
    class AgentState(TypedDict):
        messages: Annotated[list[AnyMessage], operator.add]
        goal: str
        plan: list[dict[str, Any]]
        current_step: int
        step_results: list[dict[str, Any]]
        tool_call_counts: dict[int, int]
        final_answer: str
else:
    AgentState = dict[str, Any]


class BoChaSearchInput(BaseModel):
    query: str = Field(..., description="搜索查询内容")


class BoChaSearchResults(BaseTool):
    name: str = "bocha_web_search"
    description: str = "使用 BoCha API 搜索网页以获取当前信息。"
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
                return f"未找到相关内容。\n[DEBUG] 原始响应：{data}"

            output = []
            for i, item in enumerate(results[:self._count], start=1):
                title = item.get("name", "未命名")
                snippet = item.get("snippet", "无摘要")
                result_url = item.get("url", "")
                output.append(f"{i}. {title}\n{snippet}\n链接：{result_url}")

            return "\n\n".join(output)
        except Exception as exc:
            return f"搜索失败：{exc}"


PROMPT = """\
你是一名可靠的研究助理。
你可以使用搜索工具查找可信信息。
先将任务拆解为简短步骤，然后一次只执行当前步骤。
在继续搜索之前，优先使用前面已完成步骤的结果。
只有在已有结果明显不足时才调用工具。
当搜索结果较弱或缺失时，不要编造事实。
"""


class PlannerAgent:
    """Minimal plan -> act -> record -> respond agent."""

    def __init__(self, model=None, tools=None, system: str = ""):
        self.system = system
        self.base_model = model
        self.tools = {tool.name: tool for tool in (tools or [])}
        self.graph = None
        self.max_tool_calls_per_step = 2

        if self.base_model is not None and tools is not None and LANGGRAPH_AVAILABLE:
            self.tool_model = self.base_model.bind_tools(tools)
            self.graph = self._build_graph()
        else:
            self.tool_model = None

    @staticmethod
    def emit_progress(message: str) -> None:
        print(f"[agent] {message}", flush=True)

    @staticmethod
    def extract_goal_from_messages(messages) -> str:
        for msg in reversed(messages):
            content = getattr(msg, "content", "")
            if content:
                return content
        return ""

    def extract_goal(self, state: AgentState) -> str:
        return self.extract_goal_from_messages(state.get("messages", []))

    def make_plan(self, state: AgentState) -> dict[str, Any]:
        goal = self.extract_goal(state)
        if not goal:
            return {"goal": "", "plan": [], "current_step": 0, "step_results": [], "tool_call_counts": {}}

        self.emit_progress("Generating plan")

        if self.base_model is None:
            return {
                "goal": goal,
                "plan": self._fallback_plan(goal),
                "current_step": 0,
                "step_results": [],
                "tool_call_counts": {},
            }

        prompt = (
            "请把下面的任务拆解为 2 到 4 个具体步骤。\n"
            "在合适的情况下，将搜索、验证和最终总结拆开。\n"
            f"任务：{goal}\n\n"
            "每行返回一个步骤，不要添加额外解释。"
        )
        response = self.base_model.invoke([HumanMessage(content=prompt)])

        plan = []
        for i, line in enumerate(str(response.content).splitlines()):
            step_text = line.strip()
            if not step_text:
                continue
            plan.append({"step_id": i, "description": step_text, "status": "pending"})

        if not plan:
            plan = self._fallback_plan(goal)

        self.emit_progress(f"Plan generated with {len(plan)} steps")
        for step in plan:
            print(self.format_plan_step(step), flush=True)

        return {"goal": goal, "plan": plan, "current_step": 0, "step_results": [], "tool_call_counts": {}}

    @staticmethod
    def _fallback_plan(goal: str) -> list[dict[str, Any]]:
        return [
            {"step_id": 0, "description": f"理解任务需求：{goal}", "status": "pending"},
            {"step_id": 1, "description": "收集回答所需的关键事实", "status": "pending"},
            {"step_id": 2, "description": "检查已收集事实是否存在冲突或缺漏", "status": "pending"},
            {"step_id": 3, "description": "比较并整理已验证的信息", "status": "pending"},
            {"step_id": 4, "description": "基于已验证的比较结果撰写最终回答", "status": "pending"},
        ]

    @staticmethod
    def get_current_step_from_state(state: AgentState) -> dict[str, Any] | None:
        current_step = state.get("current_step", 0)
        plan = state.get("plan", [])
        if current_step >= len(plan):
            return None
        return plan[current_step]

    def get_current_step(self, state: AgentState) -> dict[str, Any] | None:
        return self.get_current_step_from_state(state)

    @staticmethod
    def get_tool_call_count(state: AgentState) -> int:
        step = PlannerAgent.get_current_step_from_state(state)
        if step is None:
            return 0
        return state.get("tool_call_counts", {}).get(step["step_id"], 0)

    def has_exhausted_tool_budget(self, state: AgentState) -> bool:
        return self.get_tool_call_count(state) >= self.max_tool_calls_per_step

    @staticmethod
    def build_followup_reflection_prompt() -> str:
        return (
            "你已经拿到了一轮工具结果。"
            "先基于当前步骤目标，明确区分以下三点："
            "1. 已经确认的信息；"
            "2. 仍然缺失但对完成当前步骤必需的信息；"
            "3. 下一步最小必要动作。"
            "如果现有证据已经足够完成当前步骤，就直接给出该步骤结论，不要继续调用工具。"
            "如果现有证据不足且必须再次调用工具，你必须先根据缺失信息重写一个更具体、更窄的新查询，"
            "并确保它与第一次查询相比有明确收缩，而不是重复原来的宽泛问法。"
            "优先补缺口，避免重复搜索已经拿到的信息。"
        )

    @classmethod
    def build_step_messages(cls, state: AgentState) -> list[dict[str, str]]:
        step = cls.get_current_step_from_state(state)
        if step is None:
            return []

        messages = [
            {
                "role": "user",
                "content": (
                    f"整体任务：{state.get('goal', '')}\n"
                    f"当前仅处理这一步：{step['description']}\n"
                    "只处理当前步骤。"
                    "如果已有前序步骤结果，优先使用这些结果。"
                    "只有在这些结果对当前步骤明显不足时才调用工具。"
                    "如果当前是比较或总结步骤，优先综合前面的结果，而不是再次搜索。"
                    "如果确实需要工具，只调用一次，然后基于工具结果继续。"
                ),
            }
        ]

        prior_results = [
            entry
            for entry in state.get("step_results", [])
            if entry.get("step_id", -1) < step["step_id"]
        ]
        if prior_results:
            prior_summary = "\n\n".join(
                (
                    f"步骤 {entry['step_id'] + 1}：{entry['step_description']}\n"
                    f"结果：{entry['result']}"
                )
                for entry in prior_results
            )
            messages.append({"role": "user", "content": f"之前已完成步骤的结果：\n{prior_summary}"})

        last_message = state.get("messages", [])[-1] if state.get("messages") else None
        if last_message is not None and getattr(last_message, "__class__", type("X", (), {})).__name__ == "ToolMessage":
            tool_text = getattr(last_message, "content", "")
            messages.append({"role": "user", "content": f"当前步骤的工具结果：\n{tool_text}"})
            messages.append({"role": "user", "content": cls.build_followup_reflection_prompt()})

        return messages

    def execute_step(self, state: AgentState) -> dict[str, Any]:
        if self.base_model is None:
            raise RuntimeError("Model is required to execute plan steps.")

        step = self.get_current_step(state)
        if step is None:
            return {"messages": []}

        self.emit_progress(f"Executing step {step['step_id'] + 1}: {step['description']}")
        step_messages = self.build_step_messages(state)
        if self.has_exhausted_tool_budget(state):
            self.emit_progress(
                f"Tool call limit reached for step {step['step_id'] + 1}; forcing a text-only conclusion"
            )
            step_messages.append(
                {
                    "role": "user",
                    "content": (
                        "这一步不再允许使用工具。"
                        "请基于现有证据立即完成这一步的结论。"
                        "如果证据不足，请明确说明。"
                    ),
                }
            )
            return {"messages": [self.base_model.invoke(step_messages)]}
        return {"messages": [self.tool_model.invoke(step_messages)]}

    @staticmethod
    def record_step_result_update(state: AgentState, result_text: str) -> dict[str, Any]:
        step = PlannerAgent.get_current_step_from_state(state)
        if step is None:
            return {
                "plan": deepcopy(state.get("plan", [])),
                "step_results": deepcopy(state.get("step_results", [])),
                "tool_call_counts": deepcopy(state.get("tool_call_counts", {})),
                "current_step": state.get("current_step", 0),
            }

        plan = deepcopy(state.get("plan", []))
        step_results = deepcopy(state.get("step_results", []))
        tool_call_counts = deepcopy(state.get("tool_call_counts", {}))

        plan[step["step_id"]]["status"] = "done"
        step_results.append(
            {
                "step_id": step["step_id"],
                "step_description": step["description"],
                "result": result_text,
            }
        )

        return {
            "plan": plan,
            "step_results": step_results,
            "tool_call_counts": tool_call_counts,
            "current_step": state.get("current_step", 0) + 1,
        }

    @staticmethod
    def format_plan_step(step: dict[str, Any]) -> str:
        return f"步骤 {step['step_id'] + 1}：{step['description']}"

    @staticmethod
    def format_step_result_log(entry: dict[str, Any]) -> str:
        return (
            f"步骤 {entry['step_id'] + 1}\n"
            f"计划：{entry['step_description']}\n"
            f"结果：{entry['result']}\n"
        )

    def record_step_result(self, state: AgentState) -> dict[str, Any]:
        last_message = state["messages"][-1]
        result_text = getattr(last_message, "content", str(last_message))
        step = self.get_current_step(state)
        if step is not None:
            self.emit_progress(f"Recorded result for step {step['step_id'] + 1}")
        return self.record_step_result_update(state, result_text)

    @staticmethod
    def should_finish(state: AgentState) -> bool:
        return state.get("current_step", 0) >= len(state.get("plan", []))

    def build_final_answer(self, state: AgentState) -> dict[str, Any]:
        self.emit_progress("Building final answer")
        summary = "\n".join(
            f"步骤 {item['step_id'] + 1}：{item['step_description']}\n结果：{item['result']}"
            for item in state.get("step_results", [])
        )

        if self.base_model is None:
            return {"final_answer": summary}

        prompt = (
            f"原始任务：{state.get('goal', '')}\n\n"
            f"执行结果：\n{summary}\n\n"
            "请仅基于这些执行结果撰写最终回答。"
            "如果结果之间存在冲突，或者信息不完整，请明确说明，不要编造缺失事实。"
        )
        message = self.base_model.invoke([HumanMessage(content=prompt)])
        return {"final_answer": message.content, "messages": [message]}

    def exists_action(self, state: AgentState) -> bool:
        last_message = state["messages"][-1]
        if self.has_exhausted_tool_budget(state):
            return False
        return bool(getattr(last_message, "tool_calls", []))

    def take_action(self, state: AgentState) -> dict[str, Any]:
        if ToolMessage is None:
            raise RuntimeError("langchain_core is required for tool execution.")

        results = []
        step = self.get_current_step(state)
        tool_call_counts = deepcopy(state.get("tool_call_counts", {}))
        if step is not None:
            tool_call_counts[step["step_id"]] = tool_call_counts.get(step["step_id"], 0) + 1
        for tool_call in state["messages"][-1].tool_calls:
            self.emit_progress(f"Calling tool: {tool_call['name']}")
            if tool_call["name"] not in self.tools:
                result = "工具名称无效，请重试。"
            else:
                result = self.tools[tool_call["name"]].invoke(tool_call["args"])
            results.append(
                ToolMessage(
                    tool_call_id=tool_call["id"],
                    name=tool_call["name"],
                    content=str(result),
                )
            )
        return {"messages": results, "tool_call_counts": tool_call_counts}

    def route_after_record(self, state: AgentState) -> str:
        return "respond" if self.should_finish(state) else "llm"

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("plan", self.make_plan)
        graph.add_node("llm", self.execute_step)
        graph.add_node("action", self.take_action)
        graph.add_node("record", self.record_step_result)
        graph.add_node("respond", self.build_final_answer)

        graph.set_entry_point("plan")
        graph.add_edge("plan", "llm")
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: "record"})
        graph.add_edge("action", "llm")
        graph.add_conditional_edges("record", self.route_after_record, {"llm": "llm", "respond": "respond"})
        graph.add_edge("respond", END)
        return graph.compile()


def build_agent() -> PlannerAgent:
    if ChatOpenAI is None:
        raise ImportError("Missing runtime dependencies for LangGraph execution.")

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
    return PlannerAgent(model, [tool], system=PROMPT)


def run_demo(question: str) -> str:
    agent = build_agent()
    result = agent.graph.invoke({"messages": [HumanMessage(content=question)]})
    return result.get("final_answer", result["messages"][-1].content)


if __name__ == "__main__":
    answer = run_demo(
        "今年是2026年，我要比较 2024 年和 2025 年中国GDP总量的变化，并预测对普通消费者购物决策的影响"
    )
    print(answer)
