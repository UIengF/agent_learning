from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Type

import requests
from pydantic import BaseModel, Field, PrivateAttr

# 运行时依赖 LangGraph / LangChain。
# 这里做了降级导入处理，保证在缺少完整依赖时，这个文件仍然可以被导入或做静态检查。
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
    # AgentState 是整个图执行过程共享的状态容器。
    # `messages` 使用 operator.add 聚合，表示每个节点返回的新消息会自动追加到历史消息中。
    class AgentState(TypedDict):
        messages: Annotated[list[AnyMessage], operator.add]
        goal: str
        plan: list[dict[str, Any]]
        current_step: int
        step_results: list[dict[str, Any]]
        tool_call_counts: dict[int, int]
        final_answer: str
else:
    # 在缺少 LangGraph 依赖时，用普通 dict 兜底，避免类型声明本身导致导入失败。
    AgentState = dict[str, Any]


class BoChaSearchInput(BaseModel):
    # 工具的结构化输入定义，供模型生成 tool call 参数时参考。
    query: str = Field(..., description="搜索查询内容")


class BoChaSearchResults(BaseTool):
    # 封装 BoCha Web Search，供 LangChain 工具调用。
    name: str = "bocha_web_search"
    description: str = "使用 BoCha API 搜索网页以获取当前信息。"
    args_schema: Type[BaseModel] = BoChaSearchInput

    # PrivateAttr 不参与 pydantic 的字段序列化，仅作为运行时配置保存。
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
        # LangChain 工具的同步执行入口。
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

            # 将搜索结果压缩成简洁文本，方便直接喂给模型继续推理。
            output = []
            for i, item in enumerate(results[:self._count], start=1):
                title = item.get("name", "未命名")
                snippet = item.get("snippet", "无摘要")
                result_url = item.get("url", "")
                output.append(f"{i}. {title}\n{snippet}\n链接：{result_url}")

            return "\n\n".join(output)
        except Exception as exc:
            return f"搜索失败：{exc}"


# 这是传给大模型的系统提示，约束它先规划、少搜索、避免编造。
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
        # `base_model` 负责纯文本规划和最终总结。
        # `tool_model` 是绑定了工具后的模型实例，用于步骤执行阶段按需触发工具。
        self.system = system
        self.base_model = model
        self.tools = {tool.name: tool for tool in (tools or [])}
        self.graph = None
        self.log_path = Path("logs") / "agent_run.log"
        self.prompt_call_counts: dict[str, int] = {}
        # 限制单步最多调用两次工具，防止模型在某一步无限搜索。
        self.max_tool_calls_per_step = 2

        if self.base_model is not None and tools is not None and LANGGRAPH_AVAILABLE:
            self.tool_model = self.base_model.bind_tools(tools)
            self.graph = self._build_graph()
        else:
            self.tool_model = None

    @staticmethod
    def emit_progress(message: str) -> None:
        # 统一的终端进度输出，便于观察图执行过程。
        print(f"[agent] {message}", flush=True)

    @staticmethod
    def _message_role(message: Any) -> str:
        # 兼容 LangChain Message 对象和 dict 形式消息，提取可读角色名。
        if isinstance(message, dict):
            return str(message.get("role", "unknown"))
        message_type = getattr(message, "type", None)
        if message_type:
            return str(message_type)
        return message.__class__.__name__

    @staticmethod
    def _message_content(message: Any) -> str:
        # 提取消息文本内容；复杂内容结构统一转成字符串，避免日志丢信息。
        if isinstance(message, dict):
            content = message.get("content", "")
        else:
            content = getattr(message, "content", "")

        if isinstance(content, str):
            return content
        return str(content)

    @classmethod
    def format_messages_for_log(cls, messages: list[Any]) -> str:
        # 将一次模型调用的输入消息格式化为可读日志块。
        blocks = []
        for index, message in enumerate(messages, start=1):
            role = cls._message_role(message)
            content = cls._message_content(message)
            blocks.append(f"[{index}] {role}\n{content}")
        return "\n\n".join(blocks)

    def invoke_model(self, model: Any, messages: list[Any], label: str):
        # 所有模型调用统一走这里，确保每次送给 API 的内容都能落日志。
        ensure_log_file(self.log_path)
        self.prompt_call_counts[label] = self.prompt_call_counts.get(label, 0) + 1
        indexed_label = f"{label}_CALL_{self.prompt_call_counts[label]}"
        append_log(self.log_path, f"{indexed_label}\n{self.format_messages_for_log(messages)}")
        return model.invoke(messages)

    @staticmethod
    def extract_goal_from_messages(messages) -> str:
        # 默认把最近一条非空消息作为用户当前目标。
        for msg in reversed(messages):
            content = getattr(msg, "content", "")
            if content:
                return content
        return ""

    def extract_goal(self, state: AgentState) -> str:
        return self.extract_goal_from_messages(state.get("messages", []))

    def make_plan(self, state: AgentState) -> dict[str, Any]:
        # 第一阶段：从用户问题中提取目标，并让模型先生成步骤计划。
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
            "在合适的情况下，将搜索、验证、比较和最终总结拆开。\n"
            f"任务：{goal}\n\n"
            "每行返回一个步骤，不要添加额外解释。"
        )
        response = self.invoke_model(
            self.base_model,
            [HumanMessage(content=prompt)],
            "PLAN_PROMPT",
        )

        plan = []
        # 约定模型每行输出一个步骤，这里逐行转成结构化 plan。
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
        # 当模型不可用或规划输出为空时，使用固定模板兜底，确保流程仍能继续。
        return [
            {"step_id": 0, "description": f"理解任务需求：{goal}", "status": "pending"},
            {"step_id": 1, "description": "收集回答所需的关键事实", "status": "pending"},
            {"step_id": 2, "description": "检查已收集事实是否存在冲突或缺漏", "status": "pending"},
            {"step_id": 3, "description": "比较并整理已验证的信息", "status": "pending"},
            {"step_id": 4, "description": "基于已验证的比较结果撰写最终回答", "status": "pending"},
        ]

    @staticmethod
    def get_current_step_from_state(state: AgentState) -> dict[str, Any] | None:
        # 根据 current_step 游标获取当前要执行的计划项。
        current_step = state.get("current_step", 0)
        plan = state.get("plan", [])
        if current_step >= len(plan):
            return None
        return plan[current_step]

    def get_current_step(self, state: AgentState) -> dict[str, Any] | None:
        return self.get_current_step_from_state(state)

    @staticmethod
    def get_tool_call_count(state: AgentState) -> int:
        # 工具预算是按“步骤”维度统计，而不是整个任务维度统计。
        step = PlannerAgent.get_current_step_from_state(state)
        if step is None:
            return 0
        return state.get("tool_call_counts", {}).get(step["step_id"], 0)

    def has_exhausted_tool_budget(self, state: AgentState) -> bool:
        return self.get_tool_call_count(state) >= self.max_tool_calls_per_step

    @staticmethod
    def build_followup_reflection_prompt() -> str:
        # 在拿到一轮工具结果后，要求模型先反思当前证据覆盖情况，
        # 再决定是否需要发起下一次、更具体的工具调用。
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
        # 为当前步骤构造模型输入：
        # 1. 明确当前只允许处理这一步；
        # 2. 注入历史步骤结果，促使模型优先复用已有信息；
        # 3. 如果刚拿到工具结果，把工具结果也附加进来。
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
            # 只注入当前步骤之前的结果，避免未来步骤污染当前推理。
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
            messages.append(
                {
                    "role": "user",
                    "content": f"当前步骤的工具结果：\n{tool_text}",
                }
            )
            messages.append(
                {
                    "role": "user",
                    "content": cls.build_followup_reflection_prompt(),
                }
            )

        return messages

    def execute_step(self, state: AgentState) -> dict[str, Any]:
        # 第二阶段：执行当前步骤。
        # 这里可能产出普通文本消息，也可能产出带 tool_calls 的模型消息。
        if self.base_model is None:
            raise RuntimeError("Model is required to execute plan steps.")

        step = self.get_current_step(state)
        if step is None:
            return {"messages": []}

        self.emit_progress(f"Executing step {step['step_id'] + 1}: {step['description']}")
        step_messages = self.build_step_messages(state)
        if self.has_exhausted_tool_budget(state):
            # 预算用尽后，强制模型基于现有证据收敛，避免重复搜索。
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
            return {
                "messages": [
                    self.invoke_model(
                        self.base_model,
                        step_messages,
                        f"STEP_PROMPT_{step['step_id'] + 1}_TEXT_ONLY",
                    )
                ]
            }
        return {
            "messages": [
                self.invoke_model(
                    self.tool_model,
                    step_messages,
                    f"STEP_PROMPT_{step['step_id'] + 1}",
                )
            ]
        }

    @staticmethod
    def record_step_result_update(state: AgentState, result_text: str) -> dict[str, Any]:
        # 把当前步骤的结果写回状态，并将步骤标记为 done。
        # 这里返回的是“状态增量”，由 LangGraph 合并到全局状态中。
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

        # 更新 plan 和 step_results 时使用 deepcopy，避免原状态被原地修改。
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
        # 日志输出辅助函数。
        return f"步骤 {step['step_id'] + 1}：{step['description']}"

    @staticmethod
    def format_step_result_log(entry: dict[str, Any]) -> str:
        # 将步骤结果格式化为适合打印和写日志的块文本。
        return (
            f"步骤 {entry['step_id'] + 1}\n"
            f"计划：{entry['step_description']}\n"
            f"结果：{entry['result']}\n"
        )

    def record_step_result(self, state: AgentState) -> dict[str, Any]:
        # 第三阶段：从最新消息中提取文本结果，登记到 step_results。
        last_message = state["messages"][-1]
        result_text = getattr(last_message, "content", str(last_message))
        step = self.get_current_step(state)
        if step is not None:
            self.emit_progress(f"Recorded result for step {step['step_id'] + 1}")
        return self.record_step_result_update(state, result_text)

    @staticmethod
    def should_finish(state: AgentState) -> bool:
        # 所有计划项都执行完成后，进入最终回答阶段。
        return state.get("current_step", 0) >= len(state.get("plan", []))

    def build_final_answer(self, state: AgentState) -> dict[str, Any]:
        # 第四阶段：把各步骤结果汇总，再让模型生成最终面向用户的回答。
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
        message = self.invoke_model(
            self.base_model,
            [HumanMessage(content=prompt)],
            "FINAL_PROMPT",
        )
        return {"final_answer": message.content, "messages": [message]}

    def exists_action(self, state: AgentState) -> bool:
        # 如果模型返回了 tool_calls，图会转入 action 节点；否则直接记录当前步骤结果。
        last_message = state["messages"][-1]
        if self.has_exhausted_tool_budget(state):
            return False
        return bool(getattr(last_message, "tool_calls", []))

    def take_action(self, state: AgentState) -> dict[str, Any]:
        # 执行模型请求的工具调用，并把工具结果包装成 ToolMessage 回灌给模型。
        if ToolMessage is None:
            raise RuntimeError("langchain_core is required for tool execution.")

        results = []
        step = self.get_current_step(state)
        tool_call_counts = deepcopy(state.get("tool_call_counts", {}))
        if step is not None:
            # 每进入一次 action 节点，就记作当前步骤发生了一次工具调用。
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
        # record 节点之后，根据是否完成全部步骤决定走下一轮 llm，还是直接生成最终答案。
        return "respond" if self.should_finish(state) else "llm"

    def _build_graph(self):
        # 整个图的结构：
        # plan -> llm -> (action 或 record) -> ... -> respond -> END
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
    # 组装一个可运行的默认 agent：包含模型、搜索工具和系统提示。
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


def ensure_log_file(log_path: Path) -> None:
    # 确保日志目录和日志文件存在。
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        log_path.write_text("", encoding="utf-8")


def append_log(log_path: Path, text: str) -> None:
    # 简单的追加写日志工具，每次写入一个文本块并额外空一行。
    with log_path.open("a", encoding="utf-8") as file:
        file.write(text.rstrip() + "\n\n")


def print_and_log_plan(result: dict[str, Any], log_path: Path) -> None:
    # 把 plan 和已记录的 step_results 按步骤顺序打印，并同步写入日志文件。
    plan = result.get("plan", [])
    step_results = result.get("step_results", [])
    result_by_step = {entry["step_id"]: entry for entry in step_results}

    if plan:
        append_log(log_path, "PLAN")

    for step in plan:
        append_log(log_path, PlannerAgent.format_plan_step(step))
        entry = result_by_step.get(step["step_id"])
        if entry is not None:
            block = PlannerAgent.format_step_result_log(entry)
            print(block.rstrip())
            append_log(log_path, block)


def run_demo(question: str) -> str:
    # 一个完整演示入口：构建 agent、执行问题、记录日志，并返回最终答案。
    agent = build_agent()
    log_path = Path("logs") / "agent_run.log"
    ensure_log_file(log_path)

    # 入口消息只有一条 HumanMessage，图会从它中提取 goal。
    result = agent.graph.invoke({"messages": [HumanMessage(content=question)]})
    append_log(log_path, f"QUESTION\n{question}")
    print_and_log_plan(result, log_path)
    return result.get("final_answer", result["messages"][-1].content)


if __name__ == "__main__":
    # 本地直接运行文件时，使用一个需要规划、搜索、比较和总结的示例问题做演示。
    answer = run_demo(
        "今年是2026年，我要比较 2024 年和 2025 年中国GDP总量的变化，并预测对普通消费者购物决策的影响"
    )
    print(answer)
