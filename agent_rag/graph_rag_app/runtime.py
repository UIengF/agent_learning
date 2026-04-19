from __future__ import annotations

from dataclasses import dataclass
import sqlite3
from pathlib import Path
from typing import Any

try:
    from langgraph.checkpoint.sqlite import SqliteSaver
except ImportError:  # pragma: no cover - import fallback for lightweight tests
    SqliteSaver = None

from .agent import HumanMessage, build_agent
from .config import DEFAULT_SESSION_ID, AppConfig, build_app_config
from .user_memory import build_user_memory, merge_user_memory, save_user_memory


@dataclass(frozen=True)
class AgentRunTrace:
    answer: str
    messages: list[Any]
    source_messages: list[Any] | None = None


def build_sqlite_checkpointer(db_path: str | Path) -> Any | None:
    if SqliteSaver is None:
        return None

    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path.resolve(), check_same_thread=False)
    checkpointer = SqliteSaver(connection)
    setattr(checkpointer, "_connection", connection)
    return checkpointer


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


def get_thread_config(session_id: str) -> dict[str, dict[str, str]]:
    return {"configurable": {"thread_id": session_id}}


def get_graph_state(graph: Any, config: dict[str, Any]) -> Any | None:
    getter = getattr(graph, "get_state", None)
    if getter is None:
        return None
    try:
        return getter(config)
    except ValueError as exc:
        if "checkpointer" in str(exc).lower():
            return None
        raise


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


def extract_messages_from_state(state: Any | None) -> list[Any]:
    if state is None:
        return []
    state_values = getattr(state, "values", None)
    if not isinstance(state_values, dict):
        return []
    messages = state_values.get("messages", []) or []
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


def run_or_resume_with_trace(
    question: str,
    index_dir: str | Path,
    session_id: str = DEFAULT_SESSION_ID,
    checkpointer: Any | None = None,
    resume: bool = False,
    interrupt_after: list[str] | None = None,
    *,
    app_config: AppConfig | None = None,
) -> AgentRunTrace:
    if HumanMessage is None:
        raise ImportError("Missing runtime dependencies for LangGraph execution.")

    config_obj = app_config or build_app_config(
        index_dir,
        session_id=session_id,
        resume=resume,
        interrupt_after=interrupt_after,
    )
    agent = build_agent(
        index_dir,
        checkpointer=checkpointer,
        app_config=config_obj,
        ensure_log_file=ensure_log_file,
        append_log=append_log,
        shorten_text=shorten_text,
    )
    ensure_log_file(agent.log_path)
    config = get_thread_config(session_id)
    state = get_graph_state(agent.graph, config)
    prior_message_count = len(extract_messages_from_state(state))
    next_nodes = tuple(getattr(state, "next", ()) or ())

    if resume:
        if not next_nodes:
            raise RuntimeError(f"No resumable checkpoint found for session '{session_id}'.")
        append_log(
            agent.log_path,
            "===== Resume Session =====\n"
            f"session_id: {session_id}\n"
            f"pending_nodes: {', '.join(next_nodes)}",
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
            "===== Session Start =====\n"
            f"session_id: {session_id}\n"
            f"index_dir:\n{Path(index_dir).resolve()}\n\n"
            f"user_question:\n{question}\n\n"
            f"system_prompt:\n{agent.system}",
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
    persisted_messages = extract_messages_from_state(post_state) or extract_messages_from_result(result)
    source_messages = list(persisted_messages[prior_message_count:])
    observed_memory = build_user_memory(
        list(persisted_messages),
        message_content=agent._message_content,
        is_human_message=agent._is_human_message,
        shorten=agent._shorten,
    )
    save_user_memory(
        config_obj.runtime.user_memory_path,
        merge_user_memory(agent.user_memory, observed_memory),
        user_id=config_obj.runtime.user_id,
    )
    if post_next_nodes:
        append_log(
            agent.log_path,
            "===== Checkpoint Saved =====\n"
            f"session_id: {session_id}\n"
            f"pending_nodes: {', '.join(post_next_nodes)}\n"
            "resume_hint: call the same entrypoint with the same session_id and --resume",
        )
        return AgentRunTrace(
            answer=final_answer or f"[checkpoint saved] resume with session_id={session_id}",
            messages=list(persisted_messages),
            source_messages=source_messages,
        )

    if final_answer is None:
        raise RuntimeError("Graph execution completed without any messages.")

    append_log(agent.log_path, f"===== Final Answer =====\n{final_answer}")
    return AgentRunTrace(
        answer=final_answer,
        messages=list(persisted_messages),
        source_messages=source_messages,
    )


def run_or_resume(
    question: str,
    index_dir: str | Path,
    session_id: str = DEFAULT_SESSION_ID,
    checkpointer: Any | None = None,
    resume: bool = False,
    interrupt_after: list[str] | None = None,
    *,
    app_config: AppConfig | None = None,
) -> str:
    trace = run_or_resume_with_trace(
        question=question,
        index_dir=index_dir,
        session_id=session_id,
        checkpointer=checkpointer,
        resume=resume,
        interrupt_after=interrupt_after,
        app_config=app_config,
    )
    return trace.answer


def run_demo(
    question: str,
    docx_path: str | Path,
    thread_id: str = DEFAULT_SESSION_ID,
    checkpointer: Any | None = None,
) -> str:
    return run_or_resume(
        question=question,
        index_dir=docx_path,
        session_id=thread_id,
        checkpointer=checkpointer,
        resume=False,
    )
