from __future__ import annotations

import argparse
import math
import os
import re
import sqlite3
import zipfile
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Type
from urllib import error as urllib_error
from urllib import request as urllib_request
from xml.etree import ElementTree as ET

from pydantic import BaseModel, Field, PrivateAttr

try:
    import operator
    from typing import Annotated, TypedDict

    from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage, ToolMessage
    from langchain_core.tools import BaseTool
    from langchain_openai import ChatOpenAI
    from langgraph.checkpoint.sqlite import SqliteSaver
    from langgraph.graph import END, StateGraph

    LANGGRAPH_AVAILABLE = True
except ImportError:  # pragma: no cover - keep helper tests importable without full runtime deps
    AnyMessage = Any
    HumanMessage = None
    SystemMessage = None
    ToolMessage = None
    BaseTool = object  # type: ignore[assignment]
    ChatOpenAI = None
    SqliteSaver = None
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
DEFAULT_KEYWORD_WEIGHT = 0.5
DEFAULT_SESSION_ID = "graph_rag_default"
DEFAULT_CHECKPOINT_DB = "checkpoints.db"


def build_sqlite_checkpointer(db_path: str | Path) -> Any | None:
    if SqliteSaver is None:
        return None

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path.resolve(), check_same_thread=False)
    checkpointer = SqliteSaver(connection)
    # Keep a strong reference so the sqlite connection is not garbage-collected.
    setattr(checkpointer, "_connection", connection)
    return checkpointer


def ensure_log_file(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        log_path.write_text("", encoding="utf-8")


def append_log(log_path: Path, text: str) -> None:
    with log_path.open("a", encoding="utf-8") as file:
        file.write(text.rstrip() + "\n\n")


def parse_bool_env(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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


class DashScopeEmbeddingClient:
    def __init__(
        self,
        api_key: str,
        api_base: str,
        model: str = "text-embedding-v3",
        timeout_seconds: int = 30,
        max_batch_size: int = 10,
    ):
        if not api_key:
            raise ValueError("api_key is required for embedding client.")
        self.api_key = api_key
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_batch_size = max(1, max_batch_size)

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        payload = json.dumps({"model": self.model, "input": texts}).encode("utf-8")
        request = urllib_request.Request(
            url=f"{self.api_base}/embeddings",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )

        try:
            with urllib_request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = response.read().decode("utf-8")
        except urllib_error.HTTPError as exc:  # pragma: no cover - network call
            detail = exc.read().decode("utf-8", errors="ignore")
            raise RuntimeError(f"Embedding request failed: {exc.code} {detail}") from exc
        except urllib_error.URLError as exc:  # pragma: no cover - network call
            raise RuntimeError(f"Embedding request failed: {exc.reason}") from exc

        body = json.loads(raw)
        data = body.get("data")
        if not isinstance(data, list):
            raise RuntimeError(f"Unexpected embedding response body: {raw}")
        data.sort(key=lambda item: item.get("index", 0))

        vectors: list[list[float]] = []
        for item in data:
            embedding = item.get("embedding")
            if not isinstance(embedding, list):
                raise RuntimeError(f"Unexpected embedding item: {item}")
            vectors.append([float(value) for value in embedding])

        if len(vectors) != len(texts):
            raise RuntimeError("Embedding response length does not match input length.")
        return vectors

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.max_batch_size):
            batch = texts[start:start + self.max_batch_size]
            vectors.extend(self._embed_batch(batch))
        return vectors

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def embed_query(self, text: str) -> list[float]:
        vectors = self._embed([text])
        return vectors[0] if vectors else []


class LocalRAGStore:
    def __init__(
        self,
        docx_path: str | Path,
        chunk_size: int = 60,
        chunk_overlap: int = 40,
        embedding_client: Any | None = None,
        keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
    ):
        self.docx_path = Path(docx_path)
        self.embedding_client = embedding_client
        self.keyword_weight = max(0.0, min(1.0, keyword_weight))
        self.text = load_docx_text(self.docx_path)
        self.chunks = chunk_text(self.text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not self.chunks:
            raise ValueError("文档中未提取到可检索内容。")

        self.chunk_token_counts = [Counter(tokenize(chunk)) for chunk in self.chunks]
        self.chunk_lengths = [sum(counter.values()) for counter in self.chunk_token_counts]
        self.avg_chunk_length = (
            sum(self.chunk_lengths) / len(self.chunk_lengths) if self.chunk_lengths else 1.0
        )
        self.bm25_idf = self._build_bm25_idf(self.chunk_token_counts)
        self.idf = self._build_idf(self.chunk_token_counts)
        self.chunk_keyword_vectors = [self._vectorize(counter) for counter in self.chunk_token_counts]
        # Keep legacy attribute name for compatibility with existing callers/tests.
        self.chunk_vectors = self.chunk_keyword_vectors
        self.chunk_dense_vectors = self._build_dense_vectors()

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

    @staticmethod
    def _build_bm25_idf(counters: list[Counter[str]]) -> dict[str, float]:
        doc_count = len(counters)
        document_frequency: Counter[str] = Counter()
        for counter in counters:
            document_frequency.update(counter.keys())

        return {
            token: math.log(1 + ((doc_count - freq + 0.5) / (freq + 0.5)))
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
    def _sparse_cosine_similarity(left: dict[str, float], right: dict[str, float]) -> float:
        if not left or not right:
            return 0.0
        if len(left) > len(right):
            left, right = right, left
        return sum(value * right.get(token, 0.0) for token, value in left.items())

    @staticmethod
    def _normalize_dense_vector(vector: list[float]) -> list[float]:
        if not vector:
            return []
        norm = math.sqrt(sum(value * value for value in vector))
        if norm == 0:
            return []
        return [value / norm for value in vector]

    def _bm25_score(
        self,
        doc_token_counts: Counter[str],
        query_token_counts: Counter[str],
        doc_length: int,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> float:
        if not doc_token_counts or not query_token_counts:
            return 0.0

        norm = k1 * (1.0 - b + b * (doc_length / max(self.avg_chunk_length, 1e-9)))
        score = 0.0
        for token in query_token_counts:
            term_freq = doc_token_counts.get(token, 0)
            if term_freq <= 0:
                continue
            idf = self.bm25_idf.get(token, 0.0)
            score += idf * ((term_freq * (k1 + 1.0)) / (term_freq + norm))
        return score

    @staticmethod
    def _dense_cosine_similarity(left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 0.0
        return sum(l_value * r_value for l_value, r_value in zip(left, right))

    @staticmethod
    def _stable_hash(token: str) -> tuple[int, float]:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        bucket = int.from_bytes(digest[:4], "big")
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        return bucket, sign

    def _hashed_dense_vector(self, token_counts: Counter[str], dim: int = 256) -> list[float]:
        vector = [0.0] * dim
        for token, count in token_counts.items():
            bucket, sign = self._stable_hash(token)
            vector[bucket % dim] += sign * float(count)
        return self._normalize_dense_vector(vector)

    def _build_dense_vectors(self) -> list[list[float]]:
        if self.embedding_client is not None:
            dense_vectors = self.embedding_client.embed_documents(self.chunks)
            return [self._normalize_dense_vector(vector) for vector in dense_vectors]
        return [self._hashed_dense_vector(token_counts) for token_counts in self.chunk_token_counts]

    @staticmethod
    def _normalize_scores(scores: dict[int, float]) -> dict[int, float]:
        if not scores:
            return {}

        values = list(scores.values())
        high = max(values)
        low = min(values)
        if math.isclose(high, low):
            if high <= 0:
                return {chunk_id: 0.0 for chunk_id in scores}
            return {chunk_id: 1.0 for chunk_id in scores}

        denominator = high - low
        return {chunk_id: (score - low) / denominator for chunk_id, score in scores.items()}

    def _keyword_scores(self, query: str) -> dict[int, float]:
        query_token_counts = Counter(tokenize(query))
        scores: dict[int, float] = {}
        for index, chunk_token_counts in enumerate(self.chunk_token_counts):
            score = self._bm25_score(
                doc_token_counts=chunk_token_counts,
                query_token_counts=query_token_counts,
                doc_length=self.chunk_lengths[index],
            )
            if score > 0:
                scores[index] = score
        return scores

    def _dense_scores(self, query: str) -> dict[int, float]:
        if self.embedding_client is not None:
            query_vector = self._normalize_dense_vector(self.embedding_client.embed_query(query))
        else:
            query_vector = self._hashed_dense_vector(Counter(tokenize(query)))

        scores: dict[int, float] = {}
        for index, chunk_vector in enumerate(self.chunk_dense_vectors):
            score = self._dense_cosine_similarity(query_vector, chunk_vector)
            if score > 0:
                scores[index] = score
        return scores

    def search(self, query: str, top_k: int = DEFAULT_TOP_K) -> list[dict[str, Any]]:
        if not query.strip():
            return []

        keyword_scores = self._keyword_scores(query)
        dense_scores = self._dense_scores(query)
        if not keyword_scores and not dense_scores:
            return []

        keyword_scores = self._normalize_scores(keyword_scores)
        dense_scores = self._normalize_scores(dense_scores)

        dense_weight = 1.0 - self.keyword_weight
        candidates = set(keyword_scores) | set(dense_scores)
        fused_scores = []
        for chunk_id in candidates:
            score = (
                self.keyword_weight * keyword_scores.get(chunk_id, 0.0)
                + dense_weight * dense_scores.get(chunk_id, 0.0)
            )
            if score <= 0:
                continue
            fused_scores.append(
                {
                    "chunk_id": chunk_id,
                    "score": score,
                    "text": self.chunks[chunk_id],
                }
            )

        fused_scores.sort(key=lambda item: item["score"], reverse=True)
        return fused_scores[:top_k]


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
    def __init__(self, model=None, tools=None, checkpointer: Any | None = None, system: str = ""):
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
            try:
                self.graph = graph.compile(checkpointer=checkpointer)
            except TypeError:
                # Compatibility fallback for lightweight graph stubs used in tests.
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


def build_agent(docx_path: str | Path, checkpointer: Any | None = None) -> Agent:
    if ChatOpenAI is None:
        raise ImportError("Missing runtime dependencies for LangGraph execution.")

    dashscope_api_key = os.getenv("DASHSCOPE_API_KEY")
    if not dashscope_api_key:
        raise EnvironmentError("Missing DASHSCOPE_API_KEY environment variable.")

    api_base = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    model = ChatOpenAI(
        model="qwen3.6-plus",
        openai_api_key=dashscope_api_key,
        openai_api_base=api_base,
    )
    embedding_model = os.getenv("DASHSCOPE_EMBEDDING_MODEL", "text-embedding-v3")
    embedding_batch_size_raw = os.getenv("DASHSCOPE_EMBEDDING_BATCH_SIZE", "10")
    keyword_weight_raw = os.getenv("RAG_KEYWORD_WEIGHT", str(DEFAULT_KEYWORD_WEIGHT))
    try:
        keyword_weight = float(keyword_weight_raw)
    except ValueError:
        keyword_weight = DEFAULT_KEYWORD_WEIGHT
    try:
        embedding_batch_size = int(embedding_batch_size_raw)
    except ValueError:
        embedding_batch_size = 10
    embedding_client = DashScopeEmbeddingClient(
        api_key=dashscope_api_key,
        api_base=api_base,
        model=embedding_model,
        max_batch_size=embedding_batch_size,
    )
    store = LocalRAGStore(
        docx_path,
        embedding_client=embedding_client,
        keyword_weight=keyword_weight,
    )
    tool = LocalRAGRetrieveTool(store=store)
    return Agent(model, [tool], checkpointer=checkpointer, system=PROMPT)


def get_thread_config(session_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": session_id}}


def get_graph_state(graph: Any, config: dict[str, Any]) -> Any | None:
    getter = getattr(graph, "get_state", None)
    if getter is None:
        return None
    return getter(config)


def run_graph_invoke(graph: Any, state: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    try:
        return graph.invoke(state, config=config)
    except TypeError as exc:
        if "config" not in str(exc):
            raise
        return graph.invoke(state)


def run_graph_stream(
    graph: Any,
    state: dict[str, Any] | None,
    config: dict[str, Any],
    interrupt_after: list[str] | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"config": config}
    if interrupt_after:
        kwargs["interrupt_after"] = interrupt_after

    try:
        events = graph.stream(state, **kwargs)
    except TypeError as exc:
        if "config" in str(exc) or "interrupt_after" in str(exc):
            if interrupt_after:
                raise RuntimeError(
                    "This LangGraph runtime does not support interrupt_after for streamed resume."
                ) from exc
            events = graph.stream(state)
        else:
            raise

    last_event: dict[str, Any] | None = None
    for event in events:
        last_event = event

    if last_event is None:
        raise RuntimeError("Graph stream completed without emitting any events.")
    return last_event


def extract_messages_from_result(result: dict[str, Any] | None) -> list[Any]:
    if not result:
        return []
    messages = result.get("messages", [])
    return messages if isinstance(messages, list) else []


def extract_final_answer(
    result: dict[str, Any] | None,
    state: Any | None,
    message_content: Any,
) -> str | None:
    messages = extract_messages_from_result(result)
    if not messages and state is not None:
        state_values = getattr(state, "values", None)
        if isinstance(state_values, dict):
            messages = state_values.get("messages", []) or []
    if not messages:
        return None
    return message_content(messages[-1])


def run_or_resume(
    question: str,
    docx_path: str | Path,
    session_id: str = DEFAULT_SESSION_ID,
    checkpointer: Any | None = None,
    resume: bool = False,
    interrupt_after: list[str] | None = None,
) -> str:
    if HumanMessage is None:
        raise ImportError("Missing runtime dependencies for LangGraph execution.")

    agent = build_agent(docx_path, checkpointer=checkpointer)
    ensure_log_file(agent.log_path)
    config = get_thread_config(session_id)
    state = get_graph_state(agent.graph, config)
    next_nodes = tuple(getattr(state, "next", ()) or ())

    if resume:
        if not next_nodes:
            raise RuntimeError(f"No resumable checkpoint found for session '{session_id}'.")
        append_log(
            agent.log_path,
            "===== 恢复会话 =====\n"
            f"session_id: {session_id}\n"
            f"待继续节点: {', '.join(next_nodes)}",
        )
        result = run_graph_stream(
            agent.graph,
            None,
            config=config,
            interrupt_after=interrupt_after,
        )
    else:
        append_log(
            agent.log_path,
            "===== 会话开始 =====\n"
            f"session_id: {session_id}\n"
            f"文档路径:\n{Path(docx_path).resolve()}\n\n"
            f"用户问题:\n{question}\n\n"
            f"系统提示:\n{agent.system}",
        )
        initial_state = {"messages": [HumanMessage(content=question)]}
        if interrupt_after:
            result = run_graph_stream(
                agent.graph,
                initial_state,
                config=config,
                interrupt_after=interrupt_after,
            )
        else:
            result = run_graph_invoke(agent.graph, initial_state, config=config)

    post_state = get_graph_state(agent.graph, config)
    post_next_nodes = tuple(getattr(post_state, "next", ()) or ())
    final_answer = extract_final_answer(result, post_state, agent._message_content)
    if post_next_nodes:
        append_log(
            agent.log_path,
            "===== 已保存断点 =====\n"
            f"session_id: {session_id}\n"
            f"待继续节点: {', '.join(post_next_nodes)}\n"
            "验证方式: 使用相同 session_id 调用恢复入口。",
        )
        return final_answer or f"[checkpoint saved] resume with session_id={session_id}"

    if final_answer is None:
        raise RuntimeError("Graph execution completed without any messages.")

    else:
        append_log(agent.log_path, f"===== 最终答案 =====\n{final_answer}")
    return final_answer


def run_demo(
    question: str,
    docx_path: str | Path,
    thread_id: str = DEFAULT_SESSION_ID,
    checkpointer: Any | None = None,
) -> str:
    return run_or_resume(
        question=question,
        docx_path=docx_path,
        session_id=thread_id,
        checkpointer=checkpointer,
        resume=False,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run or resume the graph_rag LangGraph agent.")
    parser.add_argument("--question", default=os.getenv("RAG_QUESTION", ""))
    parser.add_argument("--docx-path", default=os.getenv("RAG_DOCX_PATH", ""))
    parser.add_argument("--session-id", default=os.getenv("RAG_SESSION_ID", DEFAULT_SESSION_ID))
    parser.add_argument(
        "--checkpoint-db",
        default=os.getenv("RAG_CHECKPOINT_DB", DEFAULT_CHECKPOINT_DB),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=parse_bool_env("RAG_RESUME", False),
        help="Resume from an existing checkpoint for the same session id.",
    )
    parser.add_argument(
        "--interrupt-after",
        action="append",
        default=None,
        help="Interrupt after the given node name. Repeatable.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    docx_path = args.docx_path
    if not docx_path:
        raise EnvironmentError("Missing RAG_DOCX_PATH environment variable.")

    checkpointer = build_sqlite_checkpointer(args.checkpoint_db)
    question = args.question or "林晓的数学和语文成绩分别变化了多少？"
    answer = run_or_resume(
        question=question,
        docx_path=docx_path,
        session_id=args.session_id,
        checkpointer=checkpointer,
        resume=args.resume,
        interrupt_after=args.interrupt_after,
    )
    print(answer)
