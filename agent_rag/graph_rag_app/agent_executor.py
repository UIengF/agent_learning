from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable

from .agent_policy import AgentState, ToolPolicyEngine
from .evidence_cache import lookup_cached_tool_result

try:
    from langchain_core.messages import AnyMessage, HumanMessage, ToolMessage
except ImportError:  # pragma: no cover - keep helper tests importable without full runtime deps
    AnyMessage = Any

    @dataclass
    class _FallbackMessage:
        content: str = ""
        type: str = "assistant"

    @dataclass
    class HumanMessage(_FallbackMessage):
        type: str = "human"

    @dataclass
    class ToolMessage(_FallbackMessage):
        tool_call_id: str = ""
        name: str = ""
        type: str = "tool"


class ToolExecutor:
    def __init__(
        self,
        *,
        tools: dict[str, Any],
        tool_policy: ToolPolicyEngine,
        append_log: Callable[[str], None],
        trace: Callable[[str, dict[str, Any]], None],
        shorten: Callable[[str, int], str],
        increment_tool_call_count: Callable[[], int],
    ):
        self.tools = tools
        self.tool_policy = tool_policy
        self._append = append_log
        self._trace = trace
        self._shorten = shorten
        self._increment_tool_call_count = increment_tool_call_count

    @staticmethod
    def tool_limit_message_text() -> str:
        return (
            "The tool call limit has been reached. Please answer using the available evidence "
            "and clearly state any uncertainty."
        )

    @staticmethod
    def finalize_without_tool_calls(message: Any, content: str) -> Any:
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

    def answer_after_tool_limit(
        self,
        *,
        messages: list[Any],
        trigger_message: Any,
        limit_message: str,
        completed_rounds: int,
        final_model: Any,
        scholar_title_guardrail: str | None = None,
    ) -> Any:
        final_instruction = (
            f"{limit_message}\n\n"
            "Do not call any more tools. Write the final answer now using only the evidence "
            "already present in the conversation. If the evidence is incomplete, state the "
            "uncertainty clearly."
        )
        if scholar_title_guardrail:
            final_instruction = final_instruction + "\n\n" + scholar_title_guardrail
        final_messages = list(messages)
        if HumanMessage is not None:
            final_messages.append(HumanMessage(content=final_instruction))
        else:
            final_messages.append({"role": "human", "content": final_instruction})

        try:
            final_message = final_model.invoke(final_messages)
        except Exception as exc:
            self._append(
                "LLM decision: failed to synthesize final answer after tool limit.\n"
                f"Completed rounds: {completed_rounds}\n"
                f"Error: {exc}"
            )
            return self.finalize_without_tool_calls(trigger_message, limit_message)

        final_content = self.tool_policy._message_content(final_message).strip() or limit_message
        self._append(
            "LLM decision: synthesize final answer after tool limit.\n"
            f"Completed rounds: {completed_rounds}\n"
            f"Output summary:\n{self._shorten(final_content, 800)}"
        )
        self._append(f"LLM raw output:\n{self._shorten(final_content, 1200)}")
        return self.finalize_without_tool_calls(final_message, final_content)

    def take_action(self, state: AgentState, evidence_cache: Any) -> dict[str, list[AnyMessage]]:
        tool_calls = state["messages"][-1].tool_calls
        results = []
        for tool_call in tool_calls:
            tool_call_count = self._increment_tool_call_count()
            tool_args = tool_call.get("args", {})
            serialized_args = json.dumps(tool_args, ensure_ascii=False, sort_keys=True)
            self._append(f"Tool call\nName: {tool_call['name']}\nArgs: {serialized_args}")
            self._trace(
                "tool_call",
                {
                    "tool_call_count": tool_call_count,
                    "name": tool_call["name"],
                    "args": tool_args,
                },
            )
            cached_result = lookup_cached_tool_result(
                evidence_cache,
                tool_name=tool_call["name"],
                tool_args=tool_args,
            )
            if cached_result is not None:
                self._append(f"Tool cache hit\nName: {tool_call['name']}")
                self._trace(
                    "tool_cache_hit",
                    {
                        "tool_call_count": tool_call_count,
                        "name": tool_call["name"],
                        "args": tool_args,
                    },
                )
                result = cached_result
            elif tool_call["name"] not in self.tools:
                result = json.dumps(
                    {
                        "error": "invalid_tool",
                        "tool_name": tool_call["name"],
                        "tool_args": tool_args,
                    },
                    ensure_ascii=False,
                )
            else:
                try:
                    result = self.tools[tool_call["name"]].invoke(tool_args)
                except Exception as exc:
                    result = json.dumps(
                        {
                            "error": "tool_execution_failed",
                            "tool_name": tool_call["name"],
                            "tool_args": tool_args,
                            "error_type": exc.__class__.__name__,
                            "message": str(exc),
                        },
                        ensure_ascii=False,
                    )
                    if tool_call["name"] == "web_fetch":
                        result = self._try_web_fetch_fallbacks(
                            state=state,
                            evidence_cache=evidence_cache,
                            original_args=tool_args,
                            current_result=result,
                        )
            self._append(f"Tool result\n{str(result)}")
            self._trace(
                "tool_result",
                {
                    "tool_call_count": tool_call_count,
                    "name": tool_call["name"],
                    "result_preview": self._shorten(str(result), 800),
                },
            )
            results.append(
                ToolMessage(
                    tool_call_id=tool_call["id"], name=tool_call["name"], content=str(result)
                )
            )
        return {"messages": results}

    def _try_web_fetch_fallbacks(
        self,
        *,
        state: AgentState,
        evidence_cache: Any,
        original_args: dict[str, Any],
        current_result: str,
    ) -> str:
        original_url = str(original_args.get("url", "")).strip()
        for fallback_url in self.tool_policy._web_fetch_fallback_urls(state, original_url):
            fallback_args = dict(original_args)
            fallback_args["url"] = fallback_url
            cached_fallback = lookup_cached_tool_result(
                evidence_cache,
                tool_name="web_fetch",
                tool_args=fallback_args,
            )
            self._append(
                "Web fetch fallback\n"
                f"Original URL failed: {original_url}\n"
                f"Trying fallback URL: {fallback_url}"
            )
            if cached_fallback is not None:
                self._append("Tool cache hit\nName: web_fetch")
                return cached_fallback
            try:
                return self.tools["web_fetch"].invoke(fallback_args)
            except Exception as fallback_exc:
                self._append(
                    "Web fetch fallback failed\n"
                    f"URL: {fallback_url}\n"
                    f"{fallback_exc.__class__.__name__}: {fallback_exc}"
                )
        return current_result
