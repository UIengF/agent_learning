from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Callable
from urllib.parse import urlparse

from .config import AppConfig, build_app_config
from .eval_datasets import EvaluationCase, EvaluationDataset
from .eval_judge import build_answer_judge_prompt, parse_answer_judge_payload
from .langsmith_runtime import (
    build_langsmith_run_name,
    build_langsmith_tags,
    langsmith_run_context,
    write_langsmith_feedback,
)
from .question_frame import QuestionFrame, build_question_frame
from .runtime import AgentRunTrace, run_or_resume_with_trace
from .sources import extract_sources_from_messages

try:
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover
    ChatOpenAI = None


AgentRunner = Callable[..., AgentRunTrace]


@dataclass(frozen=True)
class LayerScore:
    name: str
    status: str
    score: float | None
    max_score: float | None
    reasons: list[str] = field(default_factory=list)
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AdaptedTrace:
    answer: str
    tool_names: list[str]
    sources: list[dict[str, Any]]
    local_results: list[dict[str, Any]]
    web_searches: list[dict[str, Any]]
    question_frame: QuestionFrame


@dataclass(frozen=True)
class CaseEvaluationResult:
    case_id: str
    question: str
    answer: str
    passed: bool
    layer_scores: dict[str, LayerScore]
    trace: AdaptedTrace
    judge_artifacts: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class EvaluationRunReport:
    dataset_name: str
    dataset_version: str
    case_count: int
    passed_case_count: int
    failed_case_count: int
    pass_rate: float
    layer_pass_rates: dict[str, float | None]
    results: list[CaseEvaluationResult]
    langsmith_run_id: str | None = None


def build_eval_judge(app_config: AppConfig) -> Any | None:
    if ChatOpenAI is None:
        return None
    if not app_config.eval_judge.enabled or not app_config.eval_judge.api_key:
        return None
    return ChatOpenAI(
        model=app_config.eval_judge.model_name,
        openai_api_key=app_config.eval_judge.api_key,
        openai_api_base=app_config.eval_judge.api_base,
        temperature=0,
    )


def _message_name(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("name", "") or "")
    return str(getattr(message, "name", "") or "")


def _message_content(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("content", "") or "")
    return str(getattr(message, "content", "") or "")


def _response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    return str(content)


def _json_payload(message: Any) -> dict[str, Any]:

    try:
        payload = json.loads(_message_content(message))
    except (TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _collect_local_results(messages: list[Any]) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for message in messages:
        if _message_name(message) != "local_rag_retrieve":
            continue
        payload = _json_payload(message)
        raw_results = payload.get("results", [])
        if not isinstance(raw_results, list):
            continue
        for item in raw_results:
            if isinstance(item, dict):
                results.append(item)
    return results


def _collect_web_searches(messages: list[Any]) -> list[dict[str, Any]]:
    searches: list[dict[str, Any]] = []
    for message in messages:
        if _message_name(message) != "web_search":
            continue
        payload = _json_payload(message)
        if payload:
            searches.append(payload)
    return searches


def adapt_trace(question: str, trace: AgentRunTrace) -> AdaptedTrace:
    messages = trace.source_messages if trace.source_messages is not None else trace.messages
    tool_names = [_message_name(message) for message in messages if _message_name(message)]
    return AdaptedTrace(
        answer=trace.answer,
        tool_names=tool_names,
        sources=extract_sources_from_messages(messages),
        local_results=_collect_local_results(messages),
        web_searches=_collect_web_searches(messages),
        question_frame=build_question_frame(question),
    )


def _invoke_agent_runner(
    runner: AgentRunner,
    *,
    question: str,
    index_dir: str,
    run_metadata: dict[str, Any],
) -> AgentRunTrace:
    try:
        return runner(
            question=question,
            index_dir=index_dir,
            resume=False,
            run_metadata=run_metadata,
        )
    except TypeError as exc:
        if "run_metadata" not in str(exc):
            raise
        return runner(question=question, index_dir=index_dir, resume=False)


def _sync_case_feedback_to_langsmith(
    *,
    app_config: AppConfig,
    trace: AgentRunTrace,
    layer_scores: dict[str, LayerScore],
    passed: bool,
) -> None:
    if not trace.langsmith_run_id:
        return

    write_langsmith_feedback(
        config=app_config.langsmith,
        run_id=trace.langsmith_run_id,
        key="case_passed",
        score=1.0 if passed else 0.0,
        value={"passed": passed},
        comment="Overall case evaluation result.",
    )

    for layer_name, layer_score in layer_scores.items():
        if layer_score.status == "not_applicable":
            continue
        score = layer_score.score
        value = {
            "status": layer_score.status,
            "score": layer_score.score,
            "max_score": layer_score.max_score,
            "reasons": list(layer_score.reasons),
            "details": dict(layer_score.details),
        }
        comment = "; ".join(layer_score.reasons) if layer_score.reasons else f"{layer_name} passed."
        write_langsmith_feedback(
            config=app_config.langsmith,
            run_id=trace.langsmith_run_id,
            key=f"layer_{layer_name}",
            score=score,
            value=value,
            comment=comment,
        )


def _sync_trace_metrics_to_langsmith(
    *,
    app_config: AppConfig,
    trace: AgentRunTrace,
    adapted_trace: AdaptedTrace,
) -> None:
    if not trace.langsmith_run_id:
        return

    write_langsmith_feedback(
        config=app_config.langsmith,
        run_id=trace.langsmith_run_id,
        key="trajectory_summary",
        score=None,
        value={
            "tool_count": len(adapted_trace.tool_names),
            "tool_names": list(adapted_trace.tool_names),
            "source_count": len(adapted_trace.sources),
            "local_result_count": len(adapted_trace.local_results),
            "web_search_count": len(adapted_trace.web_searches),
        },
        comment="Observed tool trajectory and evidence counts.",
    )


def _sync_answer_artifact_to_langsmith(
    *,
    app_config: AppConfig,
    trace: AgentRunTrace,
    answer_artifact: dict[str, Any] | None,
) -> None:
    if not trace.langsmith_run_id or not answer_artifact:
        return

    judge_response = answer_artifact.get("judge_response")
    if not isinstance(judge_response, dict):
        return

    if "score" in judge_response:
        write_langsmith_feedback(
            config=app_config.langsmith,
            run_id=trace.langsmith_run_id,
            key="answer_judge_score",
            score=judge_response.get("score"),
            value=judge_response,
            comment="Answer judge score and artifact.",
        )
    if "grounded" in judge_response:
        grounded = bool(judge_response.get("grounded"))
        write_langsmith_feedback(
            config=app_config.langsmith,
            run_id=trace.langsmith_run_id,
            key="answer_grounded",
            score=1.0 if grounded else 0.0,
            value={"grounded": grounded},
            comment="Whether the answer judge considered the answer grounded.",
        )
    if "complete" in judge_response:
        complete = bool(judge_response.get("complete"))
        write_langsmith_feedback(
            config=app_config.langsmith,
            run_id=trace.langsmith_run_id,
            key="answer_complete",
            score=1.0 if complete else 0.0,
            value={"complete": complete},
            comment="Whether the answer judge considered the answer complete.",
        )


def _sync_report_feedback_to_langsmith(
    *,
    app_config: AppConfig,
    run_id: str | None,
    report: EvaluationRunReport,
) -> None:
    if not run_id:
        return

    write_langsmith_feedback(
        config=app_config.langsmith,
        run_id=run_id,
        key="dataset_pass_rate",
        score=report.pass_rate,
        value={
            "dataset_name": report.dataset_name,
            "dataset_version": report.dataset_version,
            "case_count": report.case_count,
            "passed_case_count": report.passed_case_count,
            "failed_case_count": report.failed_case_count,
            "pass_rate": report.pass_rate,
        },
        comment="Dataset-level evaluation summary.",
    )
    for layer_name, layer_pass_rate in report.layer_pass_rates.items():
        if layer_pass_rate is None:
            continue
        write_langsmith_feedback(
            config=app_config.langsmith,
            run_id=run_id,
            key=f"dataset_layer_{layer_name}",
            score=layer_pass_rate,
            value={"layer_name": layer_name, "pass_rate": layer_pass_rate},
            comment=f"Dataset pass rate for layer '{layer_name}'.",
        )


def _pass(name: str, score: float, *, details: dict[str, Any] | None = None) -> LayerScore:
    return LayerScore(name=name, status="passed", score=score, max_score=1.0, details=details or {})


def _fail(
    name: str,
    reasons: list[str],
    *,
    score: float = 0.0,
    details: dict[str, Any] | None = None,
) -> LayerScore:
    return LayerScore(
        name=name,
        status="failed",
        score=score,
        max_score=1.0,
        reasons=reasons,
        details=details or {},
    )


def _na(name: str) -> LayerScore:
    return LayerScore(name=name, status="not_applicable", score=None, max_score=None)


def _evaluate_question_layer(case: EvaluationCase, trace: AdaptedTrace) -> LayerScore:
    assertions = case.assertions.question
    if assertions is None and not case.expected_entities:
        return _na("question")

    reasons: list[str] = []
    checks = 0
    passed = 0
    expected_entities = list(case.expected_entities)
    if expected_entities:
        checks += 1
        actual_entities = set(trace.question_frame.target_entities)
        if all(entity in actual_entities for entity in expected_entities):
            passed += 1
        else:
            reasons.append(
                "Missing expected entities in question frame: "
                + ", ".join(entity for entity in expected_entities if entity not in actual_entities)
            )
    if assertions and assertions.expected_intent:
        checks += 1
        if trace.question_frame.task_intent == assertions.expected_intent:
            passed += 1
        else:
            reasons.append(
                f"Expected intent '{assertions.expected_intent}' but got '{trace.question_frame.task_intent}'."
            )
    if assertions and assertions.expected_focus_dimensions:
        checks += 1
        actual_dimensions = set(trace.question_frame.focus_dimensions)
        missing = [
            item for item in assertions.expected_focus_dimensions if item not in actual_dimensions
        ]
        if not missing:
            passed += 1
        else:
            reasons.append("Missing focus dimensions: " + ", ".join(missing))

    if checks == 0:
        return _na("question")
    score = passed / checks
    if reasons:
        return _fail("question", reasons, score=score, details={"frame": trace.question_frame})
    return _pass("question", score, details={"frame": trace.question_frame})


def _evaluate_retrieval_layer(case: EvaluationCase, trace: AdaptedTrace) -> LayerScore:
    assertions = case.assertions.retrieval
    if assertions is None:
        return _na("retrieval")

    reasons: list[str] = []
    checks = 0
    passed = 0
    if assertions.min_result_count is not None:
        checks += 1
        if len(trace.local_results) >= assertions.min_result_count:
            passed += 1
        else:
            reasons.append(
                f"Expected at least {assertions.min_result_count} local results but got {len(trace.local_results)}."
            )
    if assertions.expected_source_paths:
        checks += 1
        actual_paths = {str(item.get("source_path", "")) for item in trace.local_results}
        if any(path in actual_paths for path in assertions.expected_source_paths):
            passed += 1
        else:
            reasons.append("Expected source paths were not retrieved.")
    if assertions.expected_entities:
        checks += 1
        haystacks = []
        for item in trace.local_results:
            haystacks.append(str(item.get("source_path", "")))
            haystacks.append(str(item.get("section_title", "")))
            haystacks.append(str(item.get("text", "")))
        joined = "\n".join(haystacks).lower()
        missing = [
            entity for entity in assertions.expected_entities if entity.lower() not in joined
        ]
        if not missing:
            passed += 1
        else:
            reasons.append(
                "Retrieved local evidence misses expected entities: " + ", ".join(missing)
            )

    if checks == 0:
        return _na("retrieval")
    score = passed / checks
    if reasons:
        return _fail(
            "retrieval",
            reasons,
            score=score,
            details={"local_result_count": len(trace.local_results)},
        )
    return _pass("retrieval", score, details={"local_result_count": len(trace.local_results)})


def _evaluate_trajectory_layer(case: EvaluationCase, trace: AdaptedTrace) -> LayerScore:
    assertions = case.assertions.trajectory
    if assertions is None:
        return _na("trajectory")

    reasons: list[str] = []
    checks = 0
    passed = 0
    actual_tools = set(trace.tool_names)
    if assertions.must_use_tools:
        checks += 1
        missing = [tool for tool in assertions.must_use_tools if tool not in actual_tools]
        if not missing:
            passed += 1
        else:
            reasons.append("Missing required tools: " + ", ".join(missing))
    if assertions.must_not_use_tools:
        checks += 1
        unexpected = [tool for tool in assertions.must_not_use_tools if tool in actual_tools]
        if not unexpected:
            passed += 1
        else:
            reasons.append("Unexpected forbidden tools: " + ", ".join(unexpected))

    if checks == 0:
        return _na("trajectory")
    score = passed / checks
    if reasons:
        return _fail("trajectory", reasons, score=score, details={"tool_names": trace.tool_names})
    return _pass("trajectory", score, details={"tool_names": trace.tool_names})


def _host_matches_required_domain(url: str, required_domain: str) -> bool:
    host = (urlparse(url).hostname or "").lower()
    expected = required_domain.lower()
    return host == expected or host.endswith("." + expected)


def _source_matches_entity(source: dict[str, Any], entity: str) -> bool:
    haystack = " ".join(
        str(source.get(key, ""))
        for key in ("url", "title", "snippet", "text", "source_path", "section_title")
    ).lower()
    return entity.lower() in haystack


def _urls_contain_forbidden_domain(urls: list[str], forbidden_domains: list[str]) -> list[str]:
    matched: list[str] = []
    for domain in forbidden_domains:
        if any(_host_matches_required_domain(url, domain) for url in urls):
            matched.append(domain)
    return matched


def _local_covered_entities(trace: AdaptedTrace, expected_entities: list[str]) -> set[str]:
    covered: set[str] = set()
    for entity in expected_entities:
        for item in trace.local_results:
            haystack = " ".join(
                str(item.get(key, ""))
                for key in ("source_path", "section_title", "text", "document_id")
            ).lower()
            if entity.lower() in haystack:
                covered.add(entity)
                break
    return covered


def _web_search_covered_entities(trace: AdaptedTrace, expected_entities: list[str]) -> set[str]:
    haystacks: list[str] = []
    for search in trace.web_searches:
        haystacks.append(str(search.get("query", "")))
        raw_results = search.get("results", [])
        if isinstance(raw_results, list):
            for item in raw_results:
                if isinstance(item, dict):
                    haystacks.append(
                        " ".join(str(item.get(key, "")) for key in ("title", "snippet", "url"))
                    )
    joined = "\n".join(haystacks).lower()
    return {entity for entity in expected_entities if entity.lower() in joined}


def _source_covered_entities(trace: AdaptedTrace, expected_entities: list[str]) -> set[str]:
    return {
        entity
        for entity in expected_entities
        if any(_source_matches_entity(source, entity) for source in trace.sources)
    }


def _evaluate_sources_layer(case: EvaluationCase, trace: AdaptedTrace) -> LayerScore:
    assertions = case.assertions.sources
    if assertions is None:
        return _na("sources")

    reasons: list[str] = []
    checks = 0
    passed = 0
    sources = trace.sources
    if assertions.min_source_count is not None:
        checks += 1
        if len(sources) >= assertions.min_source_count:
            passed += 1
        else:
            reasons.append(
                f"Expected at least {assertions.min_source_count} sources but got {len(sources)}."
            )
    if assertions.required_source_types:
        checks += 1
        source_types = {str(item.get("source_type", "")) for item in sources}
        missing_types = [
            item for item in assertions.required_source_types if item not in source_types
        ]
        if not missing_types:
            passed += 1
        else:
            reasons.append("Missing required source types: " + ", ".join(missing_types))
    if assertions.require_entity_coverage and case.expected_entities:
        checks += 1
        covered_entities = _local_covered_entities(
            trace, case.expected_entities
        ) | _source_covered_entities(
            trace,
            case.expected_entities,
        )
        missing_entities = [
            entity for entity in case.expected_entities if entity not in covered_entities
        ]
        if not missing_entities:
            passed += 1
        else:
            reasons.append(
                "Evidence union misses expected entities: " + ", ".join(missing_entities)
            )

    if checks == 0:
        return _na("sources")
    score = passed / checks
    if reasons:
        return _fail("sources", reasons, score=score, details={"source_count": len(sources)})
    return _pass("sources", score, details={"source_count": len(sources)})


def _evaluate_web_search_quality_layer(case: EvaluationCase, trace: AdaptedTrace) -> LayerScore:
    assertions = case.assertions.web_search
    if assertions is None:
        return _na("web_search_quality")

    reasons: list[str] = []
    checks = 0
    passed = 0
    searches = trace.web_searches
    result_items: list[dict[str, Any]] = []
    queries: list[str] = []
    for search in searches:
        queries.append(str(search.get("query", "")))
        raw_results = search.get("results", [])
        if isinstance(raw_results, list):
            result_items.extend(item for item in raw_results if isinstance(item, dict))

    if assertions.min_result_count is not None:
        checks += 1
        if len(result_items) >= assertions.min_result_count:
            passed += 1
        else:
            reasons.append(
                f"Expected at least {assertions.min_result_count} web search results but got {len(result_items)}."
            )
    if assertions.required_query_terms:
        checks += 1
        evidence_text = " ".join(queries)
        for item in trace.local_results:
            evidence_text += " " + " ".join(
                str(item.get(key, "")) for key in ("source_path", "section_title", "text")
            )
        evidence_text = evidence_text.lower()
        missing_terms = [
            term for term in assertions.required_query_terms if term.lower() not in evidence_text
        ]
        if not missing_terms:
            passed += 1
        else:
            reasons.append(
                "Local and web search evidence miss required terms: " + ", ".join(missing_terms)
            )
    if case.expected_entities:
        checks += 1
        covered_entities = _local_covered_entities(
            trace, case.expected_entities
        ) | _web_search_covered_entities(
            trace,
            case.expected_entities,
        )
        missing_entities = [
            entity for entity in case.expected_entities if entity not in covered_entities
        ]
        if not missing_entities:
            passed += 1
        else:
            reasons.append(
                "Local and web search evidence miss expected entities: "
                + ", ".join(missing_entities)
            )

    if checks == 0:
        return _na("web_search_quality")
    score = passed / checks
    details = {"query_count": len(queries), "result_count": len(result_items), "queries": queries}
    if reasons:
        return _fail("web_search_quality", reasons, score=score, details=details)
    return _pass("web_search_quality", score, details=details)


def _evaluate_source_quality_layer(case: EvaluationCase, trace: AdaptedTrace) -> LayerScore:
    source_quality = case.assertions.source_quality
    if source_quality is None:
        return _na("source_quality")
    score = _evaluate_sources_layer(
        case.model_copy(
            update={"assertions": case.assertions.model_copy(update={"sources": source_quality})}
        ),
        trace,
    )
    return LayerScore(
        name="source_quality",
        status=score.status,
        score=score.score,
        max_score=score.max_score,
        reasons=score.reasons,
        details=score.details,
    )


def _evaluate_answer_layer(
    case: EvaluationCase,
    trace: AdaptedTrace,
    *,
    app_config: AppConfig | None = None,
    judge_runner: Any | None = None,
) -> tuple[LayerScore, dict[str, Any] | None]:
    assertions = case.assertions.answer
    if assertions is None:
        return _na("answer"), None

    if app_config and app_config.eval_judge.enabled and judge_runner is not None:
        prompt = build_answer_judge_prompt(
            question=case.question,
            expected_entities=list(case.expected_entities),
            must_cover_points=list(assertions.must_cover_points),
            reference_answer=assertions.reference_answer or case.reference_answer or "",
            sources=trace.sources,
            answer=trace.answer,
        )
        try:
            response = judge_runner.invoke(prompt)
            raw_response = _response_text(response)
            payload = parse_answer_judge_payload(raw_response)
            artifact = {
                "judge_prompt": prompt,
                "judge_response": {
                    "passed": payload.passed,
                    "score": payload.score,
                    "reasons": payload.reasons,
                    "grounded": payload.grounded,
                    "complete": payload.complete,
                    "raw": raw_response,
                },
            }
            details = {
                "judge_model": app_config.eval_judge.model_name,
                "grounded": payload.grounded,
                "complete": payload.complete,
            }
            if payload.passed:
                return _pass("answer", payload.score, details=details), artifact
            return (
                _fail(
                    "answer",
                    payload.reasons or ["Judge marked answer as failed."],
                    score=payload.score,
                    details=details,
                ),
                artifact,
            )
        except Exception as exc:
            return (
                _fail(
                    "answer",
                    [f"Judge invocation failed: {exc}"],
                    score=0.0,
                    details={"judge_model": app_config.eval_judge.model_name},
                ),
                {"judge_prompt": prompt, "judge_response": {"error": str(exc)}},
            )

    reasons: list[str] = []
    checks = 0
    passed = 0
    answer_lower = trace.answer.lower()
    if assertions.must_cover_points:
        checks += 1
        missing = [
            point for point in assertions.must_cover_points if point.lower() not in answer_lower
        ]
        if not missing:
            passed += 1
        else:
            reasons.append("Answer misses required points: " + ", ".join(missing))

    if checks == 0:
        return _na("answer"), None
    score = passed / checks
    if reasons:
        return _fail("answer", reasons, score=score), None
    return _pass("answer", score), None


def evaluate_case(
    case: EvaluationCase,
    *,
    index_dir: str,
    agent_runner: AgentRunner | None = None,
    app_config: AppConfig | None = None,
    judge_runner: Any | None = None,
    dataset_name: str | None = None,
) -> CaseEvaluationResult:
    runner = agent_runner or run_or_resume_with_trace
    trace = _invoke_agent_runner(
        runner,
        question=case.question,
        index_dir=index_dir,
        run_metadata={
            "mode": "eval",
            "dataset_name": dataset_name or "",
            "case_id": case.id,
            "group": case.group or "",
            "tags": list(case.tags),
        },
    )
    adapted = adapt_trace(case.question, trace)
    config = app_config or build_app_config(index_dir)
    resolved_judge = judge_runner if judge_runner is not None else build_eval_judge(config)
    answer_score, answer_artifact = _evaluate_answer_layer(
        case,
        adapted,
        app_config=config,
        judge_runner=resolved_judge,
    )
    layer_scores = {
        "question": _evaluate_question_layer(case, adapted),
        "retrieval": _evaluate_retrieval_layer(case, adapted),
        "trajectory": _evaluate_trajectory_layer(case, adapted),
        "sources": _evaluate_sources_layer(case, adapted),
        "web_search_quality": _evaluate_web_search_quality_layer(case, adapted),
        "source_quality": _evaluate_source_quality_layer(case, adapted),
        "answer": answer_score,
    }
    passed = all(score.status != "failed" for score in layer_scores.values())
    judge_artifacts = {"answer": answer_artifact} if answer_artifact is not None else {}
    _sync_trace_metrics_to_langsmith(
        app_config=config,
        trace=trace,
        adapted_trace=adapted,
    )
    _sync_answer_artifact_to_langsmith(
        app_config=config,
        trace=trace,
        answer_artifact=answer_artifact,
    )
    _sync_case_feedback_to_langsmith(
        app_config=config,
        trace=trace,
        layer_scores=layer_scores,
        passed=passed,
    )
    return CaseEvaluationResult(
        case_id=case.id,
        question=case.question,
        answer=trace.answer,
        passed=passed,
        layer_scores=layer_scores,
        trace=adapted,
        judge_artifacts=judge_artifacts,
    )


def run_evaluation_dataset(
    dataset: EvaluationDataset,
    *,
    index_dir: str,
    agent_runner: AgentRunner | None = None,
    app_config: AppConfig | None = None,
    judge_runner: Any | None = None,
) -> EvaluationRunReport:
    resolved_config = app_config or build_app_config(index_dir)
    dataset_metadata = {
        "mode": "eval_dataset",
        "dataset_name": dataset.metadata.name,
        "dataset_version": dataset.metadata.version,
        "case_count": len(dataset.cases),
    }
    dataset_run_name = build_langsmith_run_name("graph_rag.eval.dataset", dataset_metadata)
    dataset_tags = build_langsmith_tags(dataset_metadata, list(dataset.metadata.tags))
    with langsmith_run_context(
        config=resolved_config.langsmith,
        run_name=dataset_run_name,
        metadata=dataset_metadata,
        inputs={"index_dir": index_dir, "dataset_name": dataset.metadata.name},
        tags=dataset_tags,
    ) as dataset_run:
        results = [
            evaluate_case(
                case,
                index_dir=index_dir,
                agent_runner=agent_runner,
                app_config=resolved_config,
                judge_runner=judge_runner,
                dataset_name=dataset.metadata.name,
            )
            for case in dataset.cases
        ]
    case_count = len(results)
    passed_case_count = sum(1 for item in results if item.passed)
    layer_names = (
        "question",
        "retrieval",
        "trajectory",
        "sources",
        "web_search_quality",
        "source_quality",
        "answer",
    )
    layer_pass_rates: dict[str, float | None] = {}
    for layer_name in layer_names:
        applicable = [
            item.layer_scores[layer_name]
            for item in results
            if item.layer_scores[layer_name].status != "not_applicable"
        ]
        if not applicable:
            layer_pass_rates[layer_name] = None
            continue
        passed = sum(1 for score in applicable if score.status == "passed")
        layer_pass_rates[layer_name] = passed / len(applicable)

    report = EvaluationRunReport(
        dataset_name=dataset.metadata.name,
        dataset_version=dataset.metadata.version,
        case_count=case_count,
        passed_case_count=passed_case_count,
        failed_case_count=case_count - passed_case_count,
        pass_rate=(passed_case_count / case_count) if case_count else 0.0,
        layer_pass_rates=layer_pass_rates,
        results=results,
        langsmith_run_id=str(getattr(dataset_run, "id", "")) or None,
    )
    _sync_report_feedback_to_langsmith(
        app_config=resolved_config,
        run_id=report.langsmith_run_id,
        report=report,
    )
    return report
