from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
import re
from typing import Any

from pydantic import BaseModel, Field

from .config import build_app_config
from .report_schemas import CitationAudit, ReportBrief, ReportSection, ReportSource
from .research_plan import latest_research_plan
from .sources import extract_sources_from_messages

try:
    from langchain_core.messages import HumanMessage
    from langchain_openai import ChatOpenAI
except ImportError:  # pragma: no cover
    HumanMessage = None
    ChatOpenAI = None


REPORT_SECTION_ORDER = [
    "Abstract",
    "Introduction",
    "Methods",
    "Findings",
    "Discussion",
    "Conclusion",
]
MAX_EVIDENCE_CHARS = 10000


class _SectionOutput(BaseModel):
    heading: str = Field(..., description="Section heading.")
    content: str = Field(..., description="Markdown body with [S1] citations.")
    source_ids: list[str] = Field(
        default_factory=list,
        description="Source ids cited by the section, such as S1 or [S1].",
    )


def _message_content(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("content", "") or "")
    return str(getattr(message, "content", "") or "")


def _message_name(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("name", "") or "")
    return str(getattr(message, "name", "") or "")


def _message_role(message: Any) -> str:
    if isinstance(message, dict):
        return str(message.get("role", "") or "")
    message_type = getattr(message, "type", None)
    if message_type:
        return str(message_type)
    return message.__class__.__name__.lower()


def _compact_text(value: Any, limit: int = 1200) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _parse_json(content: str) -> dict[str, Any]:
    try:
        payload = json.loads(content)
    except (TypeError, ValueError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _payload_list(payload: dict[str, Any], key: str) -> list[Any]:
    value = payload.get(key, [])
    return value if isinstance(value, list) else []


def _extract_question(messages: list[Any], plan_question: str = "") -> str:
    if plan_question.strip():
        return plan_question.strip()
    for message in reversed(messages):
        role = _message_role(message).lower()
        if role in {"human", "user", "humanmessage"}:
            content = _message_content(message).strip()
            if content:
                return content
    return "Research question not found in the session trace."


def _source_title(source: dict[str, Any]) -> str:
    title = str(source.get("title") or source.get("section_title") or "").strip()
    if title:
        return title
    path = str(source.get("source_path") or source.get("url") or "").strip()
    return Path(path).name if path else "Untitled source"


def _source_location(source: dict[str, Any]) -> str:
    return str(source.get("url") or source.get("source_path") or "").strip()


def _source_kind(source: dict[str, Any]) -> str:
    source_type = str(source.get("source_type", "") or "").strip()
    if source_type == "web":
        return "web_fetch"
    if source_type in {"local", "scholar", "web_fetch"}:
        return source_type
    return source_type or "local"


def _build_sources(sources: list[dict[str, Any]]) -> list[ReportSource]:
    report_sources: list[ReportSource] = []
    for index, source in enumerate(sources, start=1):
        snippet = source.get("snippet") or source.get("text") or source.get("publication_summary")
        report_sources.append(
            ReportSource(
                id=f"[S{index}]",
                source_type=_source_kind(source),
                title=_source_title(source),
                url_or_path=_source_location(source),
                snippet=_compact_text(snippet, 700),
                year=str(source.get("year") or ""),
            )
        )
    return report_sources


def _extract_tool_evidence(messages: list[Any]) -> list[str]:
    evidence: list[str] = []
    for message in messages:
        name = _message_name(message)
        if name not in {"local_rag_retrieve", "web_fetch", "scholar_search"}:
            continue
        payload = _parse_json(_message_content(message))
        if not payload:
            continue
        if name == "local_rag_retrieve":
            for item in _payload_list(payload, "results"):
                if isinstance(item, dict):
                    label = item.get("section_title") or item.get("source_path") or "local source"
                    evidence.append(f"{label}: {_compact_text(item.get('text'), 900)}")
        elif name == "web_fetch":
            label = payload.get("title") or payload.get("final_url") or payload.get("url")
            evidence.append(f"{label}: {_compact_text(payload.get('text'), 1200)}")
        elif name == "scholar_search":
            for item in _payload_list(payload, "results"):
                if isinstance(item, dict):
                    label = item.get("title") or "scholar result"
                    detail = item.get("publication_summary") or item.get("snippet")
                    evidence.append(f"{label}: {_compact_text(detail, 900)}")
    return evidence


def _format_source_table(sources: list[ReportSource]) -> str:
    if not sources:
        return "No sources were extracted from the session."
    lines = []
    for source in sources:
        parts = [
            source.id,
            source.source_type,
            source.title,
            source.url_or_path,
        ]
        if source.year:
            parts.append(f"year: {source.year}")
        if source.snippet:
            parts.append(f"snippet: {source.snippet}")
        lines.append(" | ".join(part for part in parts if part))
    return "\n".join(lines)


def _research_plan_dict(messages: list[Any]) -> dict[str, Any] | None:
    plan = latest_research_plan(messages)
    if plan is None:
        return None
    return asdict(plan)


def build_report_brief(
    messages: list,
    sources: list[dict],
    session_id: str = "",
    model: str = "",
) -> ReportBrief:
    """Build a unified evidence/source table from raw agent messages."""

    plan = _research_plan_dict(messages)
    question = _extract_question(messages, str((plan or {}).get("question", "")))
    evidence_items = []
    if plan and plan.get("evidence"):
        evidence_items.extend(str(item) for item in plan["evidence"])
    evidence_items.extend(_extract_tool_evidence(messages))
    evidence_summary = "\n".join(f"- {_compact_text(item, 1200)}" for item in evidence_items)
    if len(evidence_summary) > MAX_EVIDENCE_CHARS:
        evidence_summary = evidence_summary[: MAX_EVIDENCE_CHARS - 3] + "..."
    gaps = [str(item) for item in (plan or {}).get("gaps", [])]
    if not sources:
        gaps.append("No grounded sources were extracted from the completed session.")

    return ReportBrief(
        question=question,
        research_plan=plan,
        evidence_summary=evidence_summary or "No tool evidence was extracted from the session.",
        sources=_build_sources(sources),
        gaps=gaps,
        generated_at=datetime.now().astimezone().isoformat(timespec="seconds"),
        session_id=session_id,
        model=model,
    )


def _section_prompt(brief: ReportBrief, section_name: str) -> str:
    return (
        "Generate one section of an academic Markdown research report.\n"
        "Use only the evidence and sources below. Do not invent sources.\n"
        "Every non-method factual claim should include inline citations like [S1].\n"
        "If evidence is missing, state the gap explicitly.\n\n"
        f"Section: {section_name}\n"
        f"Question: {brief.question}\n"
        f"Research plan: {json.dumps(brief.research_plan, ensure_ascii=False)}\n"
        f"Gaps: {json.dumps(brief.gaps, ensure_ascii=False)}\n\n"
        f"Sources:\n{_format_source_table(brief.sources)}\n\n"
        f"Evidence summary:\n{brief.evidence_summary}\n"
    )


def _invoke_plain(llm: Any, prompt: str) -> str:
    message = (
        HumanMessage(content=prompt)
        if HumanMessage is not None
        else {"role": "user", "content": prompt}
    )
    response = llm.invoke([message])
    return _message_content(response)


def _normalize_source_ids(source_ids: list[str], content: str) -> list[str]:
    ids = {item.strip().strip("[]") for item in source_ids if str(item).strip()}
    ids.update(match.strip("[]") for match in re.findall(r"\[S\d+\]", content))
    sorted_ids = sorted(
        ids,
        key=lambda item: int(item[1:]) if item.startswith("S") and item[1:].isdigit() else 0,
    )
    return [f"[{item}]" for item in sorted_ids]


def generate_section(
    brief: ReportBrief,
    section_name: str,
    llm: Any,
    max_tokens: int = 2000,
) -> ReportSection:
    """Generate one report section using structured output."""

    prompt = _section_prompt(brief, section_name)
    try:
        structured_llm = llm.with_structured_output(_SectionOutput)
        result = structured_llm.invoke(
            [HumanMessage(content=prompt)] if HumanMessage is not None else prompt
        )
        if isinstance(result, dict):
            heading = str(result.get("heading") or section_name)
            content = str(result.get("content") or "")
            source_ids = [str(item) for item in result.get("source_ids", [])]
        else:
            heading = result.heading
            content = result.content
            source_ids = result.source_ids
    except Exception:
        content = _invoke_plain(llm, prompt)
        heading = section_name
        source_ids = []
    content = _compact_text(content, max_tokens * 5)
    # Strip duplicated heading from content (LLM often includes it)
    for prefix in (f"## {heading}", f"# {heading}", heading):
        if content.lstrip().startswith(prefix):
            content = content.lstrip()[len(prefix):].lstrip()
            break
    return ReportSection(
        heading=heading.strip() or section_name,
        content=content.strip(),
        source_ids=_normalize_source_ids(source_ids, content),
    )


def _build_llm(index_dir: str, model_name: str = "") -> Any:
    if ChatOpenAI is None:
        raise ImportError("Missing langchain-openai dependency required for report generation.")
    config = build_app_config(index_dir or ".")
    if not config.model.api_key:
        raise EnvironmentError("Missing RAG_MODEL_API_KEY or DASHSCOPE_API_KEY environment variable.")
    return ChatOpenAI(
        model=model_name or config.model.model_name,
        openai_api_key=config.model.api_key,
        openai_api_base=config.model.api_base,
    )


def _references_section(brief: ReportBrief) -> ReportSection:
    if not brief.sources:
        return ReportSection(
            heading="References",
            content="No references were extracted from the completed session.",
            source_ids=[],
        )
    lines = []
    for source in brief.sources:
        year = f" ({source.year})." if source.year else "."
        location = f" {source.url_or_path}" if source.url_or_path else ""
        lines.append(f"- {source.id} {source.title}{year}{location}")
    return ReportSection(
        heading="References",
        content="\n".join(lines),
        source_ids=[source.id for source in brief.sources],
    )


def audit_citations(
    sections: list[ReportSection],
    brief: ReportBrief,
) -> CitationAudit:
    """Verify all citations are valid and flag likely unsupported claims."""

    valid_ids = {source.id.strip("[]") for source in brief.sources}
    cited_ids: set[str] = set()
    missing_refs: set[str] = set()
    unsupported_claims: list[str] = []
    for section in sections:
        if section.heading == "References":
            continue
        citations = {match.strip("[]") for match in re.findall(r"\[S\d+\]", section.content)}
        cited_ids.update(citation for citation in citations if citation in valid_ids)
        missing_refs.update(citation for citation in citations if citation not in valid_ids)
        if valid_ids and section.heading in {"Findings", "Discussion"}:
            for paragraph in re.split(r"\n\s*\n", section.content):
                compact = _compact_text(paragraph, 180)
                if len(compact) > 80 and not re.search(r"\[S\d+\]", compact):
                    unsupported_claims.append(f"{section.heading}: {compact}")
    unused_sources = [
        f"[{item}]"
        for item in sorted(valid_ids - cited_ids, key=lambda item: int(item[1:]))
    ]
    return CitationAudit(
        unused_sources=unused_sources,
        missing_refs=[
            f"[{item}]"
            for item in sorted(
                missing_refs,
                key=lambda item: int(item[1:]) if item[1:].isdigit() else 0,
            )
        ],
        unsupported_claims=unsupported_claims,
        passed=not missing_refs and not unsupported_claims,
    )


def _format_audit(audit: CitationAudit) -> str:
    if audit.passed:
        return "<!-- Citation audit: passed -->"
    lines = ["<!-- Citation audit: issues found"]
    if audit.missing_refs:
        lines.append(f"missing_refs: {', '.join(audit.missing_refs)}")
    if audit.unused_sources:
        lines.append(f"unused_sources: {', '.join(audit.unused_sources)}")
    if audit.unsupported_claims:
        lines.append("unsupported_claims:")
        lines.extend(f"- {item}" for item in audit.unsupported_claims[:10])
    lines.append("-->")
    return "\n".join(lines)


def generate_report(
    messages: list,
    source_messages: list,
    session_id: str = "",
    index_dir: str = "",
    model_name: str = "",
    llm: Any | None = None,
) -> str:
    """Generate a complete Markdown academic report from an agent session."""

    extracted_sources = extract_sources_from_messages(source_messages or messages)
    brief = build_report_brief(
        messages,
        extracted_sources,
        session_id=session_id,
        model=model_name,
    )
    report_llm = llm or _build_llm(index_dir, model_name=model_name)
    # Generate all sections in parallel — they share the brief but not each other.
    section_map: dict[str, ReportSection] = {}
    with ThreadPoolExecutor(max_workers=len(REPORT_SECTION_ORDER)) as executor:
        futures = {
            executor.submit(generate_section, brief, name, report_llm): name
            for name in REPORT_SECTION_ORDER
        }
        for future in as_completed(futures):
            section = future.result()
            section_map[futures[future]] = section
    sections = [section_map[name] for name in REPORT_SECTION_ORDER]
    sections.append(_references_section(brief))
    audit = audit_citations(sections, brief)

    lines = [
        f"# Academic Research Report: {brief.question}",
        "",
        f"- Session: {brief.session_id or 'unknown'}",
        f"- Generated: {brief.generated_at}",
    ]
    if brief.model:
        lines.append(f"- Model: {brief.model}")
    if brief.gaps:
        lines.extend(["", "## Evidence Gaps", *[f"- {gap}" for gap in brief.gaps]])
    for section in sections:
        lines.extend(["", f"## {section.heading}", "", section.content])
    lines.extend(["", _format_audit(audit), ""])
    return "\n".join(lines)


def save_report(markdown: str, output_dir: str | Path, slug: str = "") -> Path:
    """Save a report to reports/YYYY-MM-DD-{slug}.md and return the path."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    safe_slug = re.sub(r"[^A-Za-z0-9_-]+", "-", slug.strip()).strip("-").lower()
    name = datetime.now().astimezone().date().isoformat()
    if safe_slug:
        name = f"{name}-{safe_slug}"
    path = output_path / f"{name}.md"
    counter = 2
    while path.exists():
        path = output_path / f"{name}-{counter}.md"
        counter += 1
    path.write_text(markdown, encoding="utf-8")
    return path
