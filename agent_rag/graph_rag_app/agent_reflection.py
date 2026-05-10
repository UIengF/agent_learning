from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Callable

from .agent_policy import AgentState, ToolPolicyEngine
from .question_frame import QuestionFrame

try:
    from langchain_core.messages import AnyMessage
except ImportError:  # pragma: no cover - keep helper tests importable without full runtime deps
    AnyMessage = Any


REFLECTION_PROMPT_TEMPLATE = """\
You just received tool results. First decide:
1. What is already answered by the current evidence?
2. What information is still missing to complete the task?
3. What is the smallest necessary next step?

If the current evidence is sufficient, answer directly and do not call another tool.
If the evidence is insufficient, make the next query or fetch step more specific.
Do not treat unsupported facts as known facts.
If search snippets hint at needed details on a page, fetch the page before relying on those details.
If multiple answers, dates, or scenarios are plausible, explain that clearly.
"""


@dataclass(frozen=True)
class ReflectionRecord:
    stage: str
    question: str
    entities: tuple[str, ...]
    evidence_sufficiency: str
    missing_information: tuple[str, ...]
    recommended_next_action: str
    latest_tool_name: str
    latest_tool_result_count: int | None
    cached_web_queries: tuple[str, ...]
    cached_fetched_urls: tuple[str, ...]
    failed_web_fetch_domains: tuple[str, ...]
    llm_decision: str
    tool_calls: tuple[dict[str, Any], ...]


@dataclass(frozen=True)
class LoggedQuestionFrame:
    question: str
    target_entities: tuple[str, ...]
    task_intent: str
    focus_dimensions: tuple[str, ...]
    evidence_scope: dict[str, bool]
    success_criteria: tuple[str, ...]


class ReflectionRecorder:
    def __init__(
        self,
        *,
        tool_policy: ToolPolicyEngine,
        append: Callable[[str], None],
        append_json: Callable[[str, dict[str, Any]], None],
        shorten: Callable[[str, int], str],
        build_task_state: Callable[[list[AnyMessage | dict]], Any],
        build_evidence_cache: Callable[[list[AnyMessage | dict]], Any],
        get_current_round: Callable[[], int],
    ):
        self.tool_policy = tool_policy
        self._append = append
        self._append_json = append_json
        self._shorten = shorten
        self._build_task_state = build_task_state
        self._build_evidence_cache = build_evidence_cache
        self._get_current_round = get_current_round

    @staticmethod
    def build_reflection_prompt(tool_results: list[AnyMessage | dict]) -> str:
        result_blocks = []
        for index, message in enumerate(tool_results, start=1):
            if isinstance(message, dict):
                name = str(message.get("name", "tool"))
                content = str(message.get("content", ""))
            else:
                name = str(getattr(message, "name", "tool"))
                content = str(getattr(message, "content", ""))
            result_blocks.append(f"Tool result {index} ({name}):\n{content}")
        return REFLECTION_PROMPT_TEMPLATE + "\n\n" + "\n\n".join(result_blocks)

    def log_round_header(self, title: str) -> None:
        self._append(f"===== Round {self._get_current_round()} =====\n{title}")

    def log_question_frame(self, question_frame: QuestionFrame | None) -> None:
        if question_frame is None:
            return
        payload = LoggedQuestionFrame(
            question=question_frame.question,
            target_entities=question_frame.target_entities,
            task_intent=question_frame.task_intent,
            focus_dimensions=question_frame.focus_dimensions,
            evidence_scope=asdict(question_frame.evidence_scope),
            success_criteria=question_frame.success_criteria,
        )
        self._append_json("Question frame", asdict(payload))

    def build_reflection_record(
        self,
        state: AgentState,
        *,
        llm_decision: str,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> ReflectionRecord | None:
        task_state = self._build_task_state(list(state["messages"]))
        if task_state is None:
            return None
        evidence_cache = self._build_evidence_cache(list(state["messages"]))
        latest_tool_name, latest_tool_payload = self.tool_policy._latest_tool_snapshot(state)
        latest_tool_result_count = latest_tool_payload.get("result_count")
        if not isinstance(latest_tool_result_count, int):
            latest_tool_result_count = None
        serialized_calls = tuple(
            {
                "name": str(tool_call.get("name", "")),
                "args": dict(tool_call.get("args", {})),
            }
            for tool_call in (tool_calls or [])
        )
        return ReflectionRecord(
            stage="post_tool_reflection",
            question=task_state.question,
            entities=task_state.entities,
            evidence_sufficiency=task_state.evidence_sufficiency,
            missing_information=task_state.missing_information,
            recommended_next_action=task_state.next_action,
            latest_tool_name=latest_tool_name,
            latest_tool_result_count=latest_tool_result_count,
            cached_web_queries=tuple(sorted(evidence_cache.web_results_by_query.keys())),
            cached_fetched_urls=tuple(sorted(evidence_cache.fetched_pages_by_url.keys())),
            failed_web_fetch_domains=tuple(self.tool_policy._failed_web_fetch_domains(state)),
            llm_decision=llm_decision,
            tool_calls=serialized_calls,
        )

    def log_reflection_record(
        self,
        state: AgentState,
        *,
        llm_decision: str,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> None:
        record = self.build_reflection_record(
            state,
            llm_decision=llm_decision,
            tool_calls=tool_calls,
        )
        if record is None:
            return
        self._append_json("Reflection result", asdict(record))
