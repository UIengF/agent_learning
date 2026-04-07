from __future__ import annotations

import math
import os
import re
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Type
from xml.etree import ElementTree as ET

from pydantic import BaseModel, Field, PrivateAttr

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


if LANGGRAPH_AVAILABLE:
    class AgentState(TypedDict):
        messages: Annotated[list[AnyMessage], operator.add]
else:
    AgentState = dict[str, Any]


DEFAULT_TOP_K = 3


def ensure_log_file(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        log_path.write_text("", encoding="utf-8")


def append_log(log_path: Path, text: str) -> None:
    with log_path.open("a", encoding="utf-8") as file:
        file.write(text.rstrip() + "\n\n")


def shorten_text(text: str, max_len: int = 600) -> str:
    compact = " ".join(text.split())
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3] + "..."


def tokenize(text: str) -> list[str]:
    lowered = text.lower()
    latin_tokens = re.findall(r"[a-z0-9_]+", lowered)
    cjk_chars = re.findall(r"[\u4e00-\u9fff]", lowered)
    return latin_tokens + cjk_chars


def load_docx_text(docx_path: str | Path) -> str:
    path = Path(docx_path)
    if not path.exists():
        raise FileNotFoundError(f"未找到文档：{path}")
    if path.suffix.lower() != ".docx":
        raise ValueError(f"当前仅支持 .docx 文件：{path}")

    with zipfile.ZipFile(path) as archive:
        xml_bytes = archive.read("word/document.xml")

    root = ET.fromstring(xml_bytes)
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    paragraphs = []
    for paragraph in root.findall(".//w:p", ns):
        texts = [node.text or "" for node in paragraph.findall(".//w:t", ns)]
        paragraph_text = "".join(texts).strip()
        if paragraph_text:
            paragraphs.append(paragraph_text)

    return "\n\n".join(paragraphs)


def chunk_text(text: str, chunk_size: int = 60, chunk_overlap: int = 40) -> list[str]:
    if chunk_size <= 0:
        raise ValueError("chunk_size 必须大于 0")
    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap 必须满足 0 <= overlap < chunk_size")

    paragraphs = [segment.strip() for segment in text.split("\n\n") if segment.strip()]
    if not paragraphs:
        return []

    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)
            tail = current[-chunk_overlap:] if chunk_overlap else ""
            current = f"{tail}\n\n{paragraph}".strip() if tail else paragraph
        else:
            start = 0
            step = chunk_size - chunk_overlap
            while start < len(paragraph):
                piece = paragraph[start:start + chunk_size].strip()
                if piece:
                    chunks.append(piece)
                start += step
            current = ""

    if current:
        chunks.append(current)

    return chunks


class LocalRAGStore:
    def __init__(self, docx_path: str | Path, chunk_size: int = 60, chunk_overlap: int = 40):
        self.docx_path = Path(docx_path)
        self.text = load_docx_text(self.docx_path)
        self.chunks = chunk_text(self.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not self.chunks:
            raise ValueError("文档中未提取到可检索内容。")

        self.chunk_token_counts = [Counter(tokenize(chunk)) for chunk in self.chunks]
        self.idf = self._build_idf(self.chunk_token_counts)
        self.chunk_vectors = [self._vectorize(counter) for counter in self.chunk_token_counts]

    @staticmethod
    def _build_idf(counters: list[Counter[str]]) -> dict[str, float]:
        doc_count = len(counters)
        document_frequency: Counter[str] = Counter()
        for counter in counters:
            document_frequency.update(counter.keys())

        return {
            token: math.log((1 + doc_count) / (1 + freq)) + 1.0
            for token, freq in document_frequency.items()
        }

    def _vectorize(self, token_counts: Counter[str]) -> dict[str, float]:
        total = sum(token_counts.values())
        if total == 0:
            return {}

        vector = {
            token: (count / total) * self.idf.get(token, 0.0)
            for token, count in token_counts.items()
            if token in self.idf
        }
        norm = math.sqrt(sum(value * value for value in vector.values()))
        if norm == 0:
            return {}
        return {token: value / norm for token, value in vector.items()}

    @staticmethod
    def _cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
        if not left or not right:
            return 0.0
        if len(left) > len(right):
            left, right = right, left
        return sum(value * right.get(token, 0.0) for token, value in left.items())

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        if not query.strip():
            return []

        query_vector = self._vectorize(Counter(tokenize(query)))
        scored = []
        for index, chunk_vector in enumerate(self.chunk_vectors):
            score = self._cosine_similarity(query_vector, chunk_vector)
            if score <= 0:
                continue
            scored.append(
                {
                    "chunk_id": index,
                    "score": score,
                    "text": self.chunks[index],
                }
            )

        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]


class LocalRAGInput(BaseModel):
    query: str = Field(..., description="用于检索本地知识库的查询")
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=8, description="返回的相关片段数量")


class LocalRAGRetrieveTool(BaseTool):
    name: str = "local_rag_retrieve"
    description: str = "从本地 Word 文档构建的知识库中检索相关片段。"
    args_schema: Type[BaseModel] = LocalRAGInput

    _store: LocalRAGStore = PrivateAttr()

    def __init__(self, store: LocalRAGStore, **kwargs):
        super().__init__(**kwargs)
        self._store = store

    def _run(self, query: str, top_k: int = DEFAULT_TOP_K) -> str:
        results = self._store.search(query, top_k=top_k)
        if not results:
            return "未检索到相关片段。"

        blocks = []
        for item in results:
            blocks.append(
                f"片段 {item['chunk_id']}（score={item['score']:.4f}）\n"
                f"{item['text']}"
            )
        return "\n\n".join(blocks)


class Agent:
    def __init__(self, model=None, tools=None, system: str = ""):
        self.system = system
        self.log_path = Path("logs") / "graph_rag.log"
        self.model_call_count = 0
        self.tool_call_count = 0
        self.current_round = 0
        self.tools = {tool.name: tool for tool in (tools or [])}
        self.base_model = model
        self.graph = None

        if self.base_model is not None and tools is not None and LANGGRAPH_AVAILABLE:
            graph = StateGraph(AgentState)
            graph.add_node("llm", self.call_openai)
            graph.add_node("action", self.take_action)
            graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
            graph.add_edge("action", "llm")
            graph.set_entry_point("llm")
            self.graph = graph.compile()
            self.model = model.bind_tools(tools)
        else:
            self.model = None

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

    def log_round_header(self, title: str) -> None:
        append_log(self.log_path, f"===== 第 {self.current_round} 轮 =====\n{title}")

    def exists_action(self, state: AgentState):
        result = state["messages"][-1]
        return len(getattr(result, "tool_calls", [])) > 0

    @staticmethod
    def build_reflection_prompt(tool_results: list[ToolMessage]) -> str:
        result_blocks = []
        for index, message in enumerate(tool_results, start=1):
            result_blocks.append(f"检索结果 {index}（{message.name}）:\n{message.content}")

        return (
            "你刚收到一轮本地知识库检索结果。\n"
            "请先判断：\n"
            "1. 当前结果已经回答了什么；\n"
            "2. 还缺少哪些完成任务必需的信息；\n"
            "3. 下一步最小必要动作是什么。\n"
            "如果当前证据已经足够，请直接基于检索结果回答，不要继续调用工具。\n"
            "如果证据不足且必须继续检索，请改写出更具体的新查询，再调用工具。\n"
            "不要把检索结果中没有出现的事实当成已知事实。\n\n"
            "如果检索结果中存在多个可能答案、多个时间点或多个场景，请先指出歧义，再分别说明。\n"
            "回答时请尽量标注所依据的片段编号。\n\n"
            + "\n\n".join(result_blocks)
        )

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
            append_log(
                self.log_path,
                "反思阶段:\n模型将在本次调用中基于检索结果判断是否继续检索或直接回答。",
            )
            append_log(
                self.log_path,
                f"反思提示:\n{shorten_text(self.build_reflection_prompt([last_message]), 1200)}",
            )
            self.current_round += 1

        messages = self.build_llm_messages(state)
        self.model_call_count += 1
        role = self._message_role(last_message)
        content = shorten_text(self._message_content(last_message), 800)
        if self.model_call_count == 1:
            self.current_round = 1
            self.log_round_header("开始分析用户问题")

        append_log(self.log_path, f"LLM 输入来源: {role}\nLLM 输入内容:\n{content}")
        message = self.model.invoke(messages)
        response_text = self._message_content(message)
        tool_calls = getattr(message, "tool_calls", [])
        if tool_calls:
            first_query = tool_calls[0].get("args", {}).get("query", "")
            append_log(
                self.log_path,
                "LLM 决策: 继续检索本地知识库\n"
                f"本轮检索意图:\n{shorten_text(response_text, 500)}\n"
                f"生成查询:\n{first_query}",
            )
        else:
            append_log(
                self.log_path,
                "LLM 决策: 直接给出答案\n"
                f"输出摘要:\n{shorten_text(response_text, 800)}",
            )
        append_log(self.log_path, f"LLM 原始输出:\n{shorten_text(response_text, 1200)}")
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
            append_log(
                self.log_path,
                f"检索调用:\n名称: {tool_call['name']}\n查询: {query}\ntop_k: {top_k}",
            )
            if tool_call["name"] not in self.tools:
                result = "工具名称无效，请重试。"
            else:
                result = self.tools[tool_call["name"]].invoke(tool_call["args"])
            append_log(
                self.log_path,
                f"原始检索结果:\n{str(result)}",
            )
            results.append(
                ToolMessage(tool_call_id=tool_call["id"], name=tool_call["name"], content=str(result))
            )
        return {"messages": results}


PROMPT = """\
你是一名基于本地知识库回答问题的研究助理。
你可以使用本地检索工具查找 Word 文档中的相关片段。
只有在当前证据不足时才调用检索工具。
回答时只能基于检索结果中的内容，不要编造文档里没有出现的事实。
如果证据不足，请明确说明缺失了什么。
如果检索结果中存在多个可能答案、多个时间点或多个场景，请先指出歧义，再给出分情况答案。
请尽量标注所依据的片段编号，便于核对证据来源。
"""


def build_agent(docx_path: str | Path) -> Agent:
    if ChatOpenAI is None:
        raise ImportError("Missing runtime dependencies for LangGraph execution.")

    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
    if not dashscope_api_key:
        raise EnvironmentError("Missing DASHSCOPE_API_KEY environment variable.")

    model = ChatOpenAI(
        model="qwen3.6-plus",
        openai_api_key=dashscope_api_key,
        openai_api_base="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    store = LocalRAGStore(docx_path)
    tool = LocalRAGRetrieveTool(store=store)
    return Agent(model, [tool], system=PROMPT)


def run_demo(question: str, docx_path: str | Path) -> str:
    if HumanMessage is None:
        raise ImportError("Missing runtime dependencies for LangGraph execution.")

    agent = build_agent(docx_path)
    messages = [HumanMessage(content=question)]
    ensure_log_file(agent.log_path)
    agent.log_path.write_text("", encoding="utf-8")
    append_log(
        agent.log_path,
        "===== 会话开始 =====\n"
        f"文档路径:\n{Path(docx_path).resolve()}\n\n"
        f"用户问题:\n{question}\n\n"
        f"系统提示:\n{agent.system}",
    )
    result = agent.graph.invoke({"messages": messages})
    append_log(
        agent.log_path,
        "===== 最终答案 =====\n"
        f"{agent._message_content(result['messages'][-1])}",
    )
    return result["messages"][-1].content


if __name__ == "__main__":
    docx_path = os.getenv("RAG_DOCX_PATH", "")
    if not docx_path:
        raise EnvironmentError("Missing RAG_DOCX_PATH environment variable.")

    answer = run_demo("林晓的数学和语文成绩分别变化了多少？", docx_path)
    print(answer)
