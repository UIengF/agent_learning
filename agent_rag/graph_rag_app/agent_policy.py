from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

from .sources import extract_sources_from_messages

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


AgentState = dict[str, Any]


class ToolPolicyEngine:
    def __init__(self, append_log: Any):
        self._append_log = append_log

    def _append(self, text: str) -> None:
        self._append_log(text)

    @staticmethod
    def _message_content(message: AnyMessage | dict) -> str:
        if isinstance(message, dict):
            content = message.get("content", "")
        else:
            content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        return str(content)

    @staticmethod
    def _is_human_message(message: AnyMessage | dict) -> bool:
        if HumanMessage is not None and isinstance(message, HumanMessage):
            return True
        if isinstance(message, dict):
            return str(message.get("role", "")).lower() in {"human", "user"}
        return str(getattr(message, "type", "")).lower() in {"human", "user"}

    @staticmethod
    def _is_tool_message(message: AnyMessage | dict) -> bool:
        if ToolMessage is not None and isinstance(message, ToolMessage):
            return True
        if isinstance(message, dict):
            return str(message.get("role", "")) == "tool"
        return str(getattr(message, "type", "")) == "tool"

    @staticmethod
    def _tool_name(message: AnyMessage | dict) -> str:
        if isinstance(message, dict):
            return str(message.get("name", "tool"))
        return str(getattr(message, "name", "tool"))

    @staticmethod
    def _parse_tool_payload(content: str) -> dict[str, Any]:
        try:
            payload = json.loads(content)
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    @staticmethod
    def _url_domain(url: str) -> str:
        try:
            parsed = urlparse(url)
        except Exception:
            return ""
        return (parsed.netloc or "").lower()

    @staticmethod
    def _remove_site_filter(query: str, domain: str) -> str:
        pattern = re.compile(rf"(?i)\bsite:{re.escape(domain)}\b")
        updated = pattern.sub(" ", query)
        updated = re.sub(r"\s+", " ", updated).strip()
        updated = re.sub(r"(?i)^(OR|AND)\s+", "", updated).strip()
        updated = re.sub(r"(?i)\s+(OR|AND)$", "", updated).strip()
        return updated

    def _current_call_messages(self, state: AgentState) -> list[AnyMessage | dict]:
        messages = list(state["messages"])
        for index in range(len(messages) - 1, -1, -1):
            if self._is_human_message(messages[index]):
                return messages[index:]
        return messages

    def _replace_tool_calls(
        self,
        message: Any,
        tool_calls: list[dict[str, Any]],
        *,
        content: str | None = None,
    ) -> Any:
        updated_content = self._message_content(message) if content is None else content
        try:
            setattr(message, "content", updated_content)
            setattr(message, "tool_calls", tool_calls)
            return message
        except Exception:
            pass

        try:
            return message.__class__(content=updated_content, tool_calls=tool_calls)
        except Exception:
            pass

        return {"role": "assistant", "content": updated_content, "tool_calls": tool_calls}

    def _failed_web_fetch_urls(self, state: AgentState) -> set[str]:
        failed_urls: set[str] = set()
        for message in self._current_call_messages(state):
            if not self._is_tool_message(message):
                continue
            if self._tool_name(message) != "web_fetch":
                continue
            payload = self._parse_tool_payload(self._message_content(message))
            if payload.get("error") != "tool_execution_failed":
                continue
            tool_args = payload.get("tool_args", {})
            if not isinstance(tool_args, dict):
                continue
            url = str(tool_args.get("url", "")).strip()
            if url:
                failed_urls.add(url)
        return failed_urls

    def _failed_web_fetch_domains(self, state: AgentState) -> list[str]:
        domains: list[str] = []
        for url in sorted(self._failed_web_fetch_urls(state)):
            domain = self._url_domain(url)
            if domain and domain not in domains:
                domains.append(domain)
        return domains

    def _format_fetch_failure_guidance(self, state: AgentState) -> str | None:
        failed_domains = self._failed_web_fetch_domains(state)
        if not failed_domains:
            return None

        lines = [
            "Fetch failure guidance:",
            "failed_fetch_domains: " + ", ".join(failed_domains),
            (
                "If evidence is still missing, broaden the next web_search to official or primary "
                "sources from the same organization or adjacent documentation ecosystem."
            ),
            "Do not keep searching only within a domain whose pages failed to fetch.",
        ]
        for domain in failed_domains:
            lines.append(
                f"avoid repeating site:{domain} unless the user explicitly requires that domain"
            )
        lines.append("Keep the target entities and focus terms in the query.")
        return "\n".join(lines)

    def _official_search_urls(self, state: AgentState) -> list[str]:
        for message in reversed(self._current_call_messages(state)):
            if not self._is_tool_message(message):
                continue
            if self._tool_name(message) != "web_search":
                continue

            payload = self._parse_tool_payload(self._message_content(message))
            urls: list[str] = []
            results = payload.get("results", [])
            if isinstance(results, list):
                for result in results:
                    if not isinstance(result, dict):
                        continue
                    if not result.get("is_official"):
                        continue
                    url = str(result.get("url", "")).strip()
                    if url:
                        urls.append(url)

            debug = payload.get("debug", {})
            if isinstance(debug, dict):
                official_urls = debug.get("official_urls", [])
                if isinstance(official_urls, list):
                    for item in official_urls:
                        url = str(item).strip()
                        if url and url not in urls:
                            urls.append(url)
            return urls
        return []

    def _ranked_web_search_urls(self, state: AgentState) -> list[str]:
        ranked: list[tuple[int, int, int, str]] = []
        seen: set[str] = set()
        for search_order, message in enumerate(reversed(self._current_call_messages(state))):
            if not self._is_tool_message(message):
                continue
            if self._tool_name(message) != "web_search":
                continue

            payload = self._parse_tool_payload(self._message_content(message))
            results = payload.get("results", [])
            if not isinstance(results, list):
                continue

            for index, result in enumerate(results):
                if not isinstance(result, dict):
                    continue
                url = str(result.get("url", "")).strip()
                if not url or url in seen:
                    continue
                seen.add(url)
                try:
                    rank = int(result.get("rank", index + 1))
                except (TypeError, ValueError):
                    rank = index + 1
                official_priority = 0 if result.get("is_official") else 1
                ranked.append((official_priority, search_order, rank, url))

        ranked.sort(key=lambda item: (item[0], item[1], item[2]))
        return [url for _, _, _, url in ranked]

    def _web_fetch_fallback_urls(self, state: AgentState, original_url: str) -> list[str]:
        failed_urls = self._failed_web_fetch_urls(state) | {original_url}
        urls: list[str] = []
        for url in self._ranked_web_search_urls(state):
            if url in failed_urls or url in urls:
                continue
            urls.append(url)
        return urls

    def _latest_tool_snapshot(self, state: AgentState) -> tuple[str, dict[str, Any]]:
        for message in reversed(self._current_call_messages(state)):
            if not self._is_tool_message(message):
                continue
            return self._tool_name(message), self._parse_tool_payload(
                self._message_content(message)
            )
        return "", {}

    def _current_question(self, state: AgentState) -> str:
        question = ""
        for message in self._current_call_messages(state):
            if self._is_human_message(message):
                content = self._message_content(message).strip()
                if content:
                    question = content
        return question

    def _format_scholar_title_guardrail(
        self, messages: list[AnyMessage | dict]
    ) -> str | None:
        sources = extract_sources_from_messages(messages)
        scholar_titles: list[str] = []
        for source in sources:
            if str(source.get("source_type", "") or "") != "scholar":
                continue
            title = str(source.get("title", "") or "").strip()
            if title:
                scholar_titles.append(title)

        if not scholar_titles:
            return None

        lines = [
            "Scholar title guardrail:",
            "Only cite or discuss papers whose titles appear in the final source list below.",
            "If a paper is not in this list, do not mention it by title, author, venue, or year.",
            "Allowed paper titles:",
        ]
        lines.extend(f"- {title}" for title in scholar_titles)
        return "\n".join(lines)

    def _apply_official_first_fetch_policy(
        self,
        state: AgentState,
        message: Any,
        tool_calls: list[dict[str, Any]],
    ) -> tuple[Any, list[dict[str, Any]]]:
        official_urls = self._official_search_urls(state)
        if not official_urls:
            return message, tool_calls
        failed_urls = self._failed_web_fetch_urls(state)
        preferred_url = next((url for url in official_urls if url not in failed_urls), None)
        if not preferred_url:
            return message, tool_calls

        first_fetch_index: int | None = None
        official_fetch_present = False
        updated_tool_calls: list[dict[str, Any]] = []
        for index, tool_call in enumerate(tool_calls):
            copied_call = dict(tool_call)
            copied_args = dict(tool_call.get("args", {}))
            copied_call["args"] = copied_args
            updated_tool_calls.append(copied_call)

            if copied_call.get("name") != "web_fetch":
                continue

            url = str(copied_args.get("url", "")).strip()
            if url == preferred_url:
                official_fetch_present = True
            elif first_fetch_index is None:
                first_fetch_index = index

        if official_fetch_present or first_fetch_index is None:
            return message, tool_calls

        original_url = str(updated_tool_calls[first_fetch_index]["args"].get("url", "")).strip()
        updated_tool_calls[first_fetch_index]["args"]["url"] = preferred_url
        self._append(
            "Official-first fetch policy\n"
            f"Rewrote web_fetch URL from {original_url} to {preferred_url}"
        )
        return self._replace_tool_calls(message, updated_tool_calls), updated_tool_calls

    def _apply_failed_domain_search_policy(
        self,
        state: AgentState,
        message: Any,
        tool_calls: list[dict[str, Any]],
    ) -> tuple[Any, list[dict[str, Any]]]:
        failed_domains = self._failed_web_fetch_domains(state)
        if not failed_domains:
            return message, tool_calls

        question = self._current_question(state).lower()
        updated_tool_calls: list[dict[str, Any]] = []
        changed = False
        for tool_call in tool_calls:
            copied_call = dict(tool_call)
            copied_args = dict(tool_call.get("args", {}))
            copied_call["args"] = copied_args
            updated_tool_calls.append(copied_call)

            if copied_call.get("name") != "web_search":
                continue
            query = str(copied_args.get("query", "")).strip()
            if not query:
                continue
            updated_query = query
            removed_domains: list[str] = []
            for domain in failed_domains:
                if domain.lower() in question:
                    continue
                next_query = self._remove_site_filter(updated_query, domain)
                if next_query != updated_query:
                    updated_query = next_query
                    removed_domains.append(domain)
            if removed_domains and updated_query:
                copied_args["query"] = updated_query
                changed = True
                self._append(
                    "Failed-domain search policy\n"
                    f"Removed site filters for failed fetch domains: {', '.join(removed_domains)}\n"
                    f"Original query: {query}\n"
                    f"Updated query: {updated_query}"
                )

        if not changed:
            return message, tool_calls
        return self._replace_tool_calls(message, updated_tool_calls), updated_tool_calls
